"""Colorize class masks and overlay on RGB images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config import NUM_CLASSES


def default_class_colors(num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Fixed palette [C, 3] uint8 for reproducible visuals."""
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([20, 20, 20], dtype=np.uint8)
    return colors


def label_to_rgb(
    labels: np.ndarray,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """labels: [H, W] int -> RGB uint8 [H, W, 3]."""
    if colors is None:
        colors = default_class_colors()
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(colors.shape[0]):
        out[labels == c] = colors[c]
    return out


def overlay(
    image_rgb_uint8: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.45,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Blend segmentation colors onto image. image_rgb_uint8: [H,W,3] uint8."""
    seg = label_to_rgb(labels, colors=colors).astype(np.float32)
    base = image_rgb_uint8.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * seg
    return np.clip(blended, 0, 255).astype(np.uint8)


def logits_to_label(logits: torch.Tensor) -> np.ndarray:
    """logits [1,C,H,W] or [C,H,W] -> argmax [H,W] numpy."""
    if logits.dim() == 4:
        logits = logits[0]
    pred = logits.argmax(dim=0).detach().cpu().numpy().astype(np.int64)
    return pred


def save_visualization(
    image_tensor_chw: torch.Tensor,
    logits: torch.Tensor,
    out_path: Path,
    mean=None,
    std=None,
) -> None:
    """Denormalize image tensor, save overlay PNG."""
    from PIL import Image

    from config import IMAGENET_MEAN, IMAGENET_STD

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    t = image_tensor_chw.detach().cpu()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    t = torch.clamp(t, 0, 1)
    rgb = (t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    lab = logits_to_label(logits)
    vis = overlay(rgb, lab)
    Image.fromarray(vis).save(out_path)
