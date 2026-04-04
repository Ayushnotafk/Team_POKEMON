import torch
import cv2
import numpy as np
import os

from model import SkipBridgedEncoderDecoder
from config import NUM_CLASSES

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
CHECKPOINT_PATH = "../outputs/best_model.pt"

if not os.path.exists(CHECKPOINT_PATH):
    print("❌ No checkpoint found at:", CHECKPOINT_PATH)
    exit()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = SkipBridgedEncoderDecoder(NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"🔥 Loaded model IoU: {checkpoint['val_mIoU']:.4f}")

# ---------------- LOAD IMAGE ----------------
img_path = "../../Offroad_Segmentation_testImages/Color_Images/0000088.png"

img = cv2.imread(img_path)
if img is None:
    print("❌ Image not found. Check path.")
    exit()

img_resized = cv2.resize(img, (256, 256))

# ---------------- PREPROCESS ----------------
img_norm = img_resized / 255.0
img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# ---------------- PREDICT ----------------
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

print("Unique classes:", np.unique(pred))

# ---------------- COLOR MAP ----------------
colors = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [0, 255, 255],
    7: [128, 0, 128],
    8: [128, 128, 0],
    9: [0, 128, 128],
}

pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

for cls, color in colors.items():
    pred_color[pred == cls] = color

# ---------------- SAVE OUTPUT ----------------
cv2.imwrite("prediction_color.png", pred_color)

# Overlay (BEST for demo)
overlay = cv2.addWeighted(img_resized, 0.6, pred_color, 0.4, 0)
cv2.imwrite("overlay.png", overlay)

print("✅ Saved prediction_color.png and overlay.png")

# ---------------- EXTRA INFO ----------------
print("Class counts:", {i: int(np.sum(pred == i)) for i in np.unique(pred)})

# ---------------- IMPORTANT (IoU SHOW FIX) ----------------
print(f"\n🎯 CHECKPOINT mIoU: {checkpoint['val_mIoU']:.4f}")