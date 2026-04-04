"""Load off-road images and segmentation masks; resize, normalize, augment."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, MASK_VALUE_TO_CLASS


def mask_to_class_indices(mask_arr: np.ndarray) -> np.ndarray:
    """Map raw mask pixel values to class indices [0, NUM_CLASSES)."""
    out = np.zeros(mask_arr.shape, dtype=np.int64)
    for raw_val, cls in MASK_VALUE_TO_CLASS.items():
        out[mask_arr == raw_val] = cls
    return out


class OffroadSegmentationDataset(Dataset):
    """
    Expects:
      root/train/images, train/masks
      root/val/images, val/masks
    Filenames aligned between images and masks.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"

        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {img_dir}")

        self.image_paths = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        self.mask_paths = [mask_dir / (p.stem + ".png") for p in self.image_paths]
        for mp in self.mask_paths:
            if not mp.is_file():
                raise FileNotFoundError(f"Missing mask for {mp.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        label = mask_to_class_indices(mask_arr)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        label_img = Image.fromarray(label.astype(np.uint8))
        label_img = label_img.resize((self.image_size, self.image_size), Image.NEAREST)
        label = np.array(label_img, dtype=np.int64)

        img_arr = np.array(image, dtype=np.float32) / 255.0
        return img_arr, label

    def _maybe_augment(
        self, img: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return img, label
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            label = np.fliplr(label).copy()
        angle = random.uniform(-15.0, 15.0)
        if abs(angle) < 1e-3:
            return img, label
        img_p = Image.fromarray((img * 255).astype(np.uint8))
        lab_p = Image.fromarray(label.astype(np.uint8))
        img_p = img_p.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        lab_p = lab_p.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        img = np.array(img_p, dtype=np.float32) / 255.0
        label = np.array(lab_p, dtype=np.int64)
        return img, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr, label = self._load_pair(idx)
        img_arr, label = self._maybe_augment(img_arr, label)

        chw = np.transpose(img_arr, (2, 0, 1))
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        chw = (chw - mean) / std

        image_t = torch.from_numpy(chw)
        label_t = torch.from_numpy(label).long()
        return image_t, label_t


def build_default_dataloaders(
    train_root: Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    from torch.utils.data import DataLoader

    train_ds = OffroadSegmentationDataset(train_root, split="train", augment=True)
    val_ds = OffroadSegmentationDataset(train_root, split="val", augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
