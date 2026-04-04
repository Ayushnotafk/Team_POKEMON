"""Paths and constants for off-road segmentation."""
from pathlib import Path

# Project root: .../yolo/project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Dataset lives next to project/: .../yolo/Offroad_Segmentation_*
YOLO_ROOT = PROJECT_ROOT.parent

TRAIN_ROOT = YOLO_ROOT / "Offroad_Segmentation_Training_Dataset"
TEST_IMAGES_DIR = YOLO_ROOT / "Offroad_Segmentation_testImages"
TEST_COLOR_DIR = TEST_IMAGES_DIR / "Color_Images"
TEST_MASK_DIR = TEST_IMAGES_DIR / "Segmentation"

# Raw mask pixel values -> class index (0 .. NUM_CLASSES-1)
MASK_VALUE_TO_CLASS = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}

NUM_CLASSES = len(MASK_VALUE_TO_CLASS)
IMAGE_SIZE = 256

# ImageNet-style normalization (common default)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = OUTPUTS_DIR / "best_model.pt"
