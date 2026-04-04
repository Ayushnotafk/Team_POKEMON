"""Combined CrossEntropy and multi-class Dice loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Softmax Dice: 1 - mean over classes of Dice (excluding empty classes in denom).
    logits: [B, C, H, W], targets: [B, H, W] int64.
    """
    probs = F.softmax(logits, dim=1)
    oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    p_sum = probs.sum(dims)
    t_sum = oh.sum(dims)
    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    return 1.0 - dice.mean()


class SegmentationLoss(torch.nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        dl = dice_loss(logits, targets, self.num_classes)
        return self.ce_weight * ce + self.dice_weight * dl
