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

from config import CHECKPOINT_PATH, NUM_CLASSES  # noqa: E402
from inference_utils import pil_to_tensor  # noqa: E402
from model import SkipBridgedEncoderDecoder  # noqa: E402
from visualize import label_to_rgb, logits_to_label, overlay  # noqa: E402


class ModelHolder:
    def __init__(self) -> None:
        self.model: SkipBridgedEncoderDecoder | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    yield
    holder.model = None


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
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
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
    lab = logits_to_label(logits)

    base = orig.resize((lab.shape[1], lab.shape[0]), Image.BILINEAR)
    base_np = np.array(base)
    ov = overlay(base_np, lab)
    mask_rgb = label_to_rgb(lab)

    def _png_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "mask_png_base64": _png_b64(mask_rgb),
        "overlay_png_base64": _png_b64(ov),
        "width": lab.shape[1],
        "height": lab.shape[0],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
