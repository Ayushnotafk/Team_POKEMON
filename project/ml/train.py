"""Train custom segmentation model; validate with mIoU; save best checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from config import CHECKPOINT_PATH, NUM_CLASSES, OUTPUTS_DIR, TRAIN_ROOT
from dataset import build_default_dataloaders
from losses import SegmentationLoss
from metrics import mean_iou
from model import SkipBridgedEncoderDecoder


def run_epoch(
    model,
    loader,
    criterion,
    device,
    train: bool,
    optimizer=None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (images, masks) in enumerate(
            tqdm(loader, desc="train" if train else "val", leave=False)
        ):
            if max_batches is not None and bi >= max_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_iou += mean_iou(logits, masks, NUM_CLASSES) * bs
            n += bs
    return total_loss / max(n, 1), total_iou / max(n, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop after N batches per epoch (smoke test).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Stop validation after N batches.",
    )
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_default_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = SkipBridgedEncoderDecoder(NUM_CLASSES).to(device)
    criterion = SegmentationLoss(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        va_loss, va_iou = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            train=False,
            max_batches=args.max_val_batches,
        )
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} train_mIoU={tr_iou:.4f}  "
            f"val_loss={va_loss:.4f} val_mIoU={va_iou:.4f}"
        )
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_mIoU": va_iou,
                    "num_classes": NUM_CLASSES,
                },
                CHECKPOINT_PATH,
            )
            print(f"  saved best to {CHECKPOINT_PATH}")

    print(f"Done. Best val mIoU={best_iou:.4f}")
