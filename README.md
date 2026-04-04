# 🌍 Off-Road Scene Semantic Segmentation 🚀

> A custom deep learning pipeline for pixel-level understanding of off-road environments using a **from-scratch encoder–decoder CNN**.

---

## 📌 Overview

This project builds a **semantic segmentation model** that classifies each pixel in an off-road scene into meaningful categories such as:

* 🌾 Grass
* 🪨 Rocks
* 🌳 Vegetation
* 🌌 Sky
* 🌍 Ground

The model is designed for **autonomous navigation in off-road environments**, where understanding terrain is critical.

---

## 🎯 Key Features

* 🧠 **Custom CNN Architecture** (no pre-trained models used)
* 🔄 **Encoder–Decoder with Skip Connections**
* 📊 **mIoU-based evaluation**
* 🎨 **Color-coded segmentation outputs**
* 🖼️ **Overlay visualization for easy interpretation**
* ⚡ **Fast inference with PyTorch**
* 🌐 **Frontend UI for real-time prediction**
* 🔌 **Backend API for model inference**

---

## 🧠 Model Architecture

We implemented a custom **encoder–decoder segmentation network** inspired by U-Net:

* Encoder → extracts spatial features
* Decoder → reconstructs pixel-level predictions
* Skip connections → preserve fine details

---

## 📊 Results

| Metric              | Value      |
| ------------------- | ---------- |
| **Validation mIoU** | **0.3352** |
| mIoU vs GT Mask     | 0.1852     |

✔️ Achieved without using any pre-trained backbone
✔️ Optimized for generalization on unseen environments

---

## 🖼️ Sample Output

<img width="446" height="842" alt="image" src="https://github.com/user-attachments/assets/7d7dc6f2-c5ef-4d34-ab81-829cd105700f" />

> Left: Original Image
> Right: Segmentation Overlay

---

## ⚙️ Tech Stack

* 🐍 Python
* 🔥 PyTorch
* 🎯 OpenCV
* 📊 NumPy
* ⚡ Vite + Frontend UI

---

## 📁 Project Structure

```
project/
│
├── frontend/              # Web UI
├── ml/
│   ├── model.py           # Custom CNN architecture
│   ├── dataset.py         # Data loading
│   ├── train.py           # Training pipeline
│   ├── test.py            # Inference script
│   ├── inference_utils.py # Backend inference logic
│   ├── metrics.py         # mIoU calculation
│   ├── losses.py          # Loss functions
│
├── outputs/
│   └── best_model.pt      # Best trained model
│
└── requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train the model

```bash
python train.py
```

This will save the best model in:

```
outputs/best_model.pt
```

---

### 3️⃣ Run inference (offline)

```bash
python test.py
```

Output:

* `prediction_color.png`
* `overlay.png`

---

## 🔌 4️⃣ Start Backend Server (IMPORTANT)

The backend serves the model via an API used by the frontend.

```bash
cd ml
python final_test.py
```

👉 Backend will start on:

```
http://localhost:8000
```

---

### 📡 API Endpoint

```
POST /predict
```

**Input:**

* `file` → image
* `mask` (optional) → ground truth mask

**Output JSON:**

```
{
  "checkpoint_val_miou": 0.3352,
  "mask_miou": 0.1852,
  "prediction": "...",
  "overlay": "..."
}
```

---

## 🌐 5️⃣ Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```
http://localhost:5173
```

---

## 📥 Input & Output

### Input:

* RGB image (PNG/JPG)
* Optional Ground Truth mask

### Output:

* Segmentation mask
* Overlay visualization
* mIoU score

---

## 📈 Evaluation Metric

We use **Mean Intersection over Union (mIoU)**:

```
IoU = Intersection / Union
```

Final score = average across all classes

---

## 🧪 Experimental Strategy

* Data augmentation (flip, rotation)
* Learning rate scheduling (OneCycleLR)
* Mixed precision training (AMP)
* Best checkpoint selection based on validation mIoU

---

## 💡 Future Improvements

* 🚀 Attention modules
* 🧠 Multi-scale feature extraction
* 📊 Class imbalance handling
* ⚡ Real-time optimization

---

## 🏆 Hackathon Notes

* No pre-trained models were used
* Entire architecture implemented from scratch
* Focus on generalization and efficiency

---

## 👥 Team Pokémon

* **Ayush Kumar Thakur**
* **Swarna Champa Murmu**
* **Muntazir Ali**

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
