# 🧬 HighRes-Histopathology-WSI-Transformer-Segmentation

> **A high-resolution transformer-based framework for efficient and precise segmentation of histopathology whole-slide images (WSIs)** - packaged with full Docker reproducibility for seamless dataset preparation, training, and evaluation 🧠💻

---

## 🧪 Overview

**HighRes-Histopathology-WSI-Transformer-Segmentation** provides an end-to-end deep learning pipeline for histopathology WSI segmentation.  
It leverages **transformer-based contextual encoding** and **boundary-aware learning** to achieve high accuracy and robust generalization across tissue slides.

The project is **containerized with Docker** to ensure fully reproducible experiments from dataset preparation to model evaluation, and supports **GPU acceleration** via CUDA.

---

## 🚀 Features

✅ End-to-end pipeline for WSI segmentation  
✅ Transformer backbone with boundary & dilation-aware training  
✅ Docker-based reproducibility (no dependency headaches 🐳)  
✅ Configurable training hyperparameters and flexible dataset options  
✅ Visual sanity checks for colored mask generation  
✅ Evaluation and overlay visualization for predictions  

---

## 🧰 Prerequisites

Make sure you have the following installed on your system:

- 🐋 **Docker** (≥ 20.10)
- 🧠 **NVIDIA Container Toolkit** for GPU support  
  ```bash
  sudo apt install -y nvidia-container-toolkit
  sudo systemctl restart docker
```
---

## 📦 Pull Docker Image

Pull the prebuilt Docker image from Docker Hub:

```bash
docker pull khanm2004/fcbformer:conda-cu111
```

Check that it’s available:
```bash
docker images | grep fcbformer
```

---

## 🧑‍💻 Quick Start

### 1️⃣ Verify PyTorch GPU Setup

Run this quick check inside the container:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

Expected output:
```bash
    2.3.0+cu111 True
```

### 2️⃣ Prepare Dataset for Training

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/prepare_dataset.py \
    --root /workspace/datasets \
    --out-name tiles_512_o64 \
    --patch-size 256 \
    --overlap 64
```
This will generate processed tiles and manifests for model training.

### 3️⃣ Visualize Colored Mask Previews 🎨

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/make_colored_previews.py \
    --mask-dir /workspace/datasets/processed/tiles_512_o64/train/masks \
    --limit 50
```

### 4️⃣ Sanity Check the Dataset 🩺

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/check_manifests.py \
    --manifests-dir /workspace/datasets/processed/manifests \
    --samples 5
```

---

## 🏋️‍♂️ Training the Model

Train your transformer-based segmentation model on the prepared dataset:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python train.py \
    --dataset AIRA \
    --data-root /workspace/datasets \
    --epochs 60 \
    --batch-size 4 \
    --learning-rate 3e-4 \
    --img-size 256 \
    --boundary-weight 0.7 \
    --tumor-dil-weight 0.7 \
    --benign-dil-weight 0.7 \
    --dil-iters 5 \
    --dil-kernel 3 \
    --prob-power 1.0
```

Training logs and checkpoints are automatically saved in:
```bash
/workspace/outputs/<timestamp>/
```

---

## 📈 Model Evaluation (on validation data)

Evaluate the trained model on validation/test sets:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python eval.py \
    --data-root /workspace/datasets \
    --checkpoint /workspace/outputs/20251004-183051/best_FCBFormer.pt \
    --img-size 256 \
    --batch-size 4 \
    --save-png \
    --out-dir /workspace/outputs/EvalWSI
```

---

## 🧩 Evaluation on Extra Slides

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python eval.py \
    --data-root /workspace/datasets \
    --checkpoint /workspace/outputs/20251004-183051/best_FCBFormer.pt \
    --img-size 256 \
    --batch-size 4 \
    --tiles-manifest /workspace/datasets/processed/manifests/extra.csv \
    --out-dir /workspace/outputs/extra_preds \
    --overlay-alpha 0.4
```

This generates overlayed predictions for unseen WSIs.

---

## 📂 Directory Structure

```bash
FCBFormer/
├── Data_prep/                     # Dataset preparation utilities
│   ├── prepare_dataset.py
│   ├── make_colored_previews.py
│   └── check_manifests.py
├── train.py                       # Main training entry point
├── eval.py                        # Evaluation script
├── datasets/
│   ├── raw/                       # Original WSIs or tiles
│   └── processed/                 # Preprocessed and tiled data
└── outputs/
    ├── <timestamp>/               # Model checkpoints and logs
    ├── EvalWSI/                   # Evaluation results on validation/test data
    └── extra_preds/               # Predictions for extra/unseen slides
...
```

