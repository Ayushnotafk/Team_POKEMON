"""Shared preprocessing for train/test/API."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def pil_to_tensor(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    chw = (chw - mean) / std
    return torch.from_numpy(chw)
