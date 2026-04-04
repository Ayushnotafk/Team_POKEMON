import torch
from model import SkipBridgedEncoderDecoder
from config import NUM_CLASSES

CHECKPOINT_PATH = "outputs/best_model.pt"


def load_model(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model = SkipBridgedEncoderDecoder(NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"🔥 Loaded checkpoint IoU: {ckpt['val_mIoU']:.4f}")

    return model, ckpt["val_mIoU"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, best_iou = load_model(device)

    print(f"\n✅ MODEL READY | BEST IoU = {best_iou:.4f}")