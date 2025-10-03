import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
        

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    def forward(self, logits, targets):
        # logits: (B,C,H,W), targets: (B,H,W) long
        probs = torch.softmax(logits, dim=1)
        target_1h = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

