"""Multi-class IoU (Jaccard) per batch."""
from __future__ import annotations

import torch


def mean_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """
    Mean IoU across classes present in the union of pred/gt (common eval convention).
    logits: [B, C, H, W], targets: [B, H, W].
    """
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = preds == c
        tgt_c = targets == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union == 0:
            continue
        ious.append((inter / union).item())
    if not ious:
        return 0.0
    return sum(ious) / len(ious)
