import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses

from Data.dataloaders_joint import get_loaders_from_manifests
import json


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, 
                ce_loss=None, dice_loss_mc=None, num_classes=1):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        if num_classes > 1:
            # AIRA (4 classes): use CE on logits + multiclass Dice
            target_long = target.long()           # (B,H,W) int64
            loss = dice_loss_mc(output, target_long) + ce_loss(output, target_long)
        else:
            # Binary path (Kvasir/CVC): keep original behavior
            loss = Dice_loss(output, target) + BCE_loss(output, target)

        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())

        # (optional) pretty printing kept as-is...
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(), time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator), time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def iou_per_class(logits, targets, num_classes=4):
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        targ_c = (targets == c)
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        ious.append(inter / union if union > 0 else 0.0)
    return ious, sum(ious)/len(ious)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure=None, num_classes=4):
    t = time.time()
    model.eval()
    perf_accumulator = []
    all_ious = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        if perf_measure is not None and num_classes == 1:
            perf_accumulator.append(perf_measure(output, target).item())

        if num_classes > 1:
            ious, miou = iou_per_class(output, target, num_classes=num_classes)
            all_ious.append(ious)

    if num_classes > 1 and all_ious:
        mean_ious = np.mean(all_ious, axis=0).tolist()
        miou = float(np.mean(mean_ious))
        print(f"[Val IoU] per-class: {mean_ious}, mIoU: {miou:.4f}")
        # Return mIoU so scheduler/checkpoint use it
        return miou, 0.0

    # binary fallback
    return (
        float(np.mean(perf_accumulator)) if perf_accumulator else 0.0,
        float(np.std(perf_accumulator)) if perf_accumulator else 0.0,
    )



def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset in ["Kvasir", "CVC"]:
        # existing logic
        if args.dataset == "Kvasir":
            img_path = args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))
        elif args.dataset == "CVC":
            img_path = args.root + "Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))
        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size
        )
        num_classes = 1  # original binary
        ce_loss = nn.BCEWithLogitsLoss()
        dice_loss = losses.SoftDiceLoss()


    elif args.dataset == "AIRA":
        train_csv = os.path.join(args.root, "processed", "manifests", "train.csv")
        val_csv   = os.path.join(args.root, "processed", "manifests", "val.csv")
        stats_json = os.path.join(args.root, "processed", "stats.json")

        train_dataloader, val_dataloader = get_loaders_from_manifests(
            train_csv, val_csv, stats_json,
            img_size=args.img_size, batch_size=args.batch_size,
            num_workers=0
        )
        num_classes = 4

        # ---- NEW: class-weighted CE from TRAIN stats ----
        with open(stats_json) as f:
            stats = json.load(f)
        cc = stats["class_counts"]
        freq = torch.tensor(
            [cc["background"], cc["stroma"], cc["benign"], cc["tumor"]],
            dtype=torch.float
        )
        # log-balancing, then normalize to mean=1
        weights = 1.0 / torch.log(1.02 + freq / freq.sum())
        weights = weights / weights.mean()
        weights = weights.to(device)

        ce_loss = nn.CrossEntropyLoss(weight=weights)
        dice_loss = losses.MultiClassDiceLoss(num_classes=num_classes)
        
    else:
        raise ValueError("Unknown dataset")

    # model
    model = models.FCBFormer(size=args.img_size, num_classes=num_classes)  # ensure your FCBFormer accepts num_classes
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # perf metric (DiceScore in your repo is probably binary; for AIRA use CE + Dice and report IoU in test())
    perf = performance_metrics.DiceScore() if num_classes == 1 else None

    return device, train_dataloader, val_dataloader, dice_loss, ce_loss, perf, model, optimizer, num_classes



def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        CE_loss,
        perf,
        model,
        optimizer,
        num_classes, 
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch,
                Dice_loss, CE_loss,
                ce_loss=CE_loss if num_classes > 1 else None,
                dice_loss_mc=Dice_loss if num_classes > 1 else None,
                num_classes=num_classes,
            )

            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch,
                perf_measure=perf,
                num_classes=num_classes,
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/FCBFormer_" + args.dataset + ".pt",
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir","CVC","AIRA"])
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
