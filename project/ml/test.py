"""Run inference on held-out test images; save masks and overlays; report mIoU."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    NUM_CLASSES,
    OUTPUTS_DIR,
    TEST_COLOR_DIR,
    TEST_MASK_DIR,
)
from dataset import mask_to_class_indices
from inference_utils import pil_to_tensor
from metrics import mean_iou
from model import SkipBridgedEncoderDecoder
from visualize import label_to_rgb, logits_to_label, overlay


def load_model(ckpt: Path, device: torch.device) -> SkipBridgedEncoderDecoder:
    ck = torch.load(ckpt, map_location=device)
    nc = int(ck.get("num_classes", NUM_CLASSES))
    model = SkipBridgedEncoderDecoder(nc).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "test_predictions")
    parser.add_argument("--images-dir", type=Path, default=TEST_COLOR_DIR)
    parser.add_argument("--masks-dir", type=Path, default=TEST_MASK_DIR)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N images (smoke test).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "pred_masks"
    overlay_dir = args.out_dir / "overlays"
    pred_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    image_paths = sorted(
        [p for p in args.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images in {args.images_dir}")
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    ious = []
    for img_path in tqdm(image_paths, desc="test"):
        pil = Image.open(img_path).convert("RGB")
        orig_hw = pil.size  # W, H
        inp = pil_to_tensor(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
        gt_tensor = None
        mask_path = args.masks_dir / img_path.name
        if mask_path.is_file():
            gt = np.array(Image.open(mask_path))
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt = mask_to_class_indices(gt)
            gt_img = Image.fromarray(gt.astype(np.uint8))
            gt_img = gt_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            gt_small = np.array(gt_img, dtype=np.int64)
            gt_tensor = torch.from_numpy(gt_small).unsqueeze(0).to(device)

        lab = logits_to_label(logits)
        pred_rgb = label_to_rgb(lab)
        Image.fromarray(pred_rgb).save(pred_dir / f"{img_path.stem}_mask.png")

        up = Image.fromarray(lab.astype(np.uint8))
        up = up.resize(orig_hw, Image.NEAREST)
        lab_full = np.array(up, dtype=np.int64)
        base = np.array(pil.resize(orig_hw, Image.BILINEAR))
        ov = overlay(base, lab_full)
        Image.fromarray(ov).save(overlay_dir / f"{img_path.stem}_overlay.png")

        if gt_tensor is not None:
            ious.append(mean_iou(logits, gt_tensor, NUM_CLASSES))

    if ious:
        print(f"Test mIoU (mean over images with GT): {float(np.mean(ious)):.4f}")
    else:
        print("No ground-truth masks found; skipped mIoU.")


if __name__ == "__main__":
    main()
