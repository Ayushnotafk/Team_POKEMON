# Off-road semantic segmentation — runbook

Layout:

- `ml/` — dataset, custom `SkipBridgedEncoderDecoder` model, losses (CE + Dice), mIoU, training, testing, visualization
- `backend/` — FastAPI `/predict` (loads `outputs/best_model.pt`)
- `frontend/` — React + Vite (dark UI, upload → original + overlay + mask)
- `outputs/` — checkpoints and test exports

Dataset paths (next to the `project/` folder):

- `../Offroad_Segmentation_Training_Dataset/` — `train/images`, `train/masks`, `val/images`, `val/masks`
- `../Offroad_Segmentation_testImages/Color_Images` and `Segmentation/`

## 1. Environment

From `project/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

Install PyTorch for your platform from [pytorch.org](https://pytorch.org) if the default wheel is unsuitable.

## 2. Train

```bash
cd ml
python train.py --epochs 30 --batch-size 4
```

Optional smoke run (few batches per epoch, faster sanity check):

```bash
python train.py --epochs 1 --max-train-batches 20 --max-val-batches 10 --batch-size 4
```

Best weights are written to `outputs/best_model.pt` (highest validation mIoU).

## 3. Test / evaluation

```bash
cd ml
python test.py
```

Limit images for a quick check:

```bash
python test.py --max-images 10
```

Predictions go to `outputs/test_predictions/pred_masks/` and overlays to `outputs/test_predictions/overlays/`. If `Segmentation/` masks exist, a mean IoU is printed.

## 4. Backend API

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — model loaded or not
- `POST /predict` — form field `file`: image → JSON with `mask_png_base64`, `overlay_png_base64`

## 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (e.g. `http://127.0.0.1:5173`). Requests go to `/api/*`, which proxies to `http://127.0.0.1:8000`.

Train at least once so `outputs/best_model.pt` exists; otherwise `/predict` returns HTTP 503.
