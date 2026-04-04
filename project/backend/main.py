"""FastAPI service: load trained custom CNN and return segmentation for an uploaded image."""
from __future__ import annotations

import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import torch

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

from config import CHECKPOINT_PATH, IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from dataset import mask_to_class_indices  # noqa: E402
from inference_utils import pil_to_tensor  # noqa: E402
from metrics import mean_iou  # noqa: E402
from model import SkipBridgedEncoderDecoder  # noqa: E402
from visualize import label_to_rgb, logits_to_label, overlay  # noqa: E402


class ModelHolder:
    def __init__(self) -> None:
        self.model: SkipBridgedEncoderDecoder | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_val_miou: float | None = None
        self.checkpoint_epoch: int | None = None


holder = ModelHolder()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ck = CHECKPOINT_PATH
    if ck.is_file():
        ckpt = torch.load(ck, map_location=holder.device)
        nc = int(ckpt.get("num_classes", NUM_CLASSES))
        m = SkipBridgedEncoderDecoder(nc).to(holder.device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        holder.model = m
        v = ckpt.get("val_mIoU", ckpt.get("val_miou"))
        holder.checkpoint_val_miou = float(v) if v is not None else None
        ep = ckpt.get("epoch")
        holder.checkpoint_epoch = int(ep) if ep is not None else None
    yield
    holder.model = None
    holder.checkpoint_val_miou = None
    holder.checkpoint_epoch = None


app = FastAPI(title="Off-road segmentation", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": holder.model is not None,
        "device": str(holder.device),
        "checkpoint_val_miou": holder.checkpoint_val_miou,
        "checkpoint_epoch": holder.checkpoint_epoch,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mask: UploadFile | None = File(None, description="Optional GT mask PNG for mIoU vs prediction"),
) -> dict:
    if holder.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No checkpoint at {CHECKPOINT_PATH}. Train the model first.",
        )
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    orig = pil.copy()
    inp = pil_to_tensor(pil).unsqueeze(0).to(holder.device)
    with torch.no_grad():
        logits = holder.model(inp)

    mask_miou = None
    mask_iou_error: str | None = None
    # Do not rely on mask.filename — browsers often omit it unless FormData passes a filename.
    if mask is not None:
        raw_m = await mask.read()
        if len(raw_m) > 0:
            try:
                gt_pil = Image.open(io.BytesIO(raw_m))
                # Match dataset.py: 2D or take first channel of RGB/RGBA
                gt_arr = np.array(gt_pil)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[..., 0]
                gt = mask_to_class_indices(gt_arr)
                gt_img = Image.fromarray(gt.astype(np.uint8))
                gt_img = gt_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
                gt_small = np.array(gt_img, dtype=np.int64)
                gt_t = torch.from_numpy(gt_small).unsqueeze(0).to(holder.device)
                mask_miou = float(mean_iou(logits, gt_t, NUM_CLASSES))
            except Exception as e:
                mask_iou_error = str(e)[:300]

    lab = logits_to_label(logits)

    base = orig.resize((lab.shape[1], lab.shape[0]), Image.BILINEAR)
    base_np = np.array(base)
    ov = overlay(base_np, lab)
    mask_rgb = label_to_rgb(lab)

    def _png_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    out = {
        "mask_png_base64": _png_b64(mask_rgb),
        "overlay_png_base64": _png_b64(ov),
        "width": lab.shape[1],
        "height": lab.shape[0],
        "checkpoint_val_miou": holder.checkpoint_val_miou,
        "checkpoint_epoch": holder.checkpoint_epoch,
    }
    if mask_miou is not None:
        out["mask_miou"] = mask_miou
    if mask_iou_error is not None:
        out["mask_iou_error"] = mask_iou_error
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
