from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from config import NUM_CLASSES, OUTPUTS_DIR, TRAIN_ROOT
from dataset import build_default_dataloaders
from losses import SegmentationLoss
from metrics import mean_iou
from model import SkipBridgedEncoderDecoder


def run_epoch(model, loader, criterion, device, train, optimizer=None,
              scheduler=None, scaler=None, use_amp=False):

    model.train() if train else model.eval()

    total_loss, total_iou, n = 0.0, 0.0, 0
    amp_enabled = use_amp and device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

                if amp_enabled:
                    with autocast(device_type="cuda"):
                        logits = model(images)
                        loss = criterion(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images)
                    loss = criterion(logits, masks)
                    loss.backward()
                    optimizer.step()

                if scheduler:
                    scheduler.step()

            else:
                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, masks)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_iou += mean_iou(logits.float(), masks, NUM_CLASSES) * bs
            n += bs

    return total_loss / n, total_iou / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=3e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    train_loader, val_loader = build_default_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = SkipBridgedEncoderDecoder(NUM_CLASSES).to(device)
    criterion = SegmentationLoss(NUM_CLASSES)

    optimizer = optim.Adam(model.parameters(), lr=args.max_lr / 25)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.15,
    )

    scaler = GradScaler(enabled=use_amp)

    best_iou = -1.0
    best_path = OUTPUTS_DIR / "best_model.pt"

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_iou = run_epoch(
            model, train_loader, criterion, device, True,
            optimizer, scheduler, scaler, use_amp
        )

        va_loss, va_iou = run_epoch(
            model, val_loader, criterion, device, False,
            scaler=scaler, use_amp=use_amp
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_mIoU={tr_iou:.4f} | val_mIoU={va_iou:.4f}"
        )

        if va_iou > best_iou:
            best_iou = va_iou

            torch.save({
                "model_state": model.state_dict(),
                "val_mIoU": va_iou,
            }, best_path)

            print(f"✅ Saved best → {best_path} (IoU={va_iou:.4f})")

    print(f"\n🔥 FINAL BEST IoU = {best_iou:.4f}")