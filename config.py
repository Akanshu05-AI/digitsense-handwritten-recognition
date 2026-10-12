"""
DigitSense: Configuration Module
Defines directory paths, hyperparameters, dataset mappings, and hardware settings.
"""

from pathlib import Path
import torch

# Base Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Hardware & Compute Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
RANDOM_SEED = 42

# Image Processing Standards
IMAGE_SIZE = (28, 28)
POLARITY_STROKE = 255  # White ink on black background (standard MNIST/EMNIST representation)
POLARITY_BACKGROUND = 0
GAUSSIAN_BLUR_KERNEL = (3, 3)

# Segmentation Defaults
SEGMENTATION_MIN_AREA = 25  # Minimum contour area in pixels to ignore noise dots
SEGMENTATION_MAX_AREA_RATIO = 0.95  # Ignore bounding boxes encompassing the full image border
SEGMENTATION_PADDING = 4  # Pixel padding around cropped character bounding boxes

# Dataset Class Mappings
MNIST_CLASSES = [str(i) for i in range(10)]

# EMNIST Balanced Split: 47 classes (0-9, 10-35: A-Z, 36-46: distinctive lowercase letters)
# Note: EMNIST merges uppercase/lowercase for letters with identical shapes (C, I, J, K, L, M, O, P, S, U, V, W, X, Y, Z).
# The 11 lowercase classes that remain distinct are: a, b, d, e, f, g, h, n, q, r, t
EMNIST_BALANCED_CLASSES = [
    # 0-9: Digits
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # 10-35: Uppercase Letters (A-Z)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    # 36-46: Distinctive Lowercase Letters
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# EMNIST Letters Split: 26 classes (1-26 in dataset mapped to 0-25 indexing for A-Z)
EMNIST_LETTERS_CLASSES = [chr(ord('A') + i) for i in range(26)]

# Model Checkpoint Filenames
CHECKPOINT_CNN_MNIST = CHECKPOINTS_DIR / "cnn_mnist.pt"
CHECKPOINT_CNN_EMNIST = CHECKPOINTS_DIR / "cnn_emnist_balanced.pt"
CHECKPOINT_CLASSICAL_MNIST = CHECKPOINTS_DIR / "classical_mnist.joblib"
CHECKPOINT_CLASSICAL_EMNIST = CHECKPOINTS_DIR / "classical_emnist.joblib"

# Evaluation Results Filenames
RESULTS_CNN_MNIST = RESULTS_DIR / "metrics_cnn_mnist.json"
RESULTS_CNN_EMNIST = RESULTS_DIR / "metrics_cnn_emnist_balanced.json"
RESULTS_CLASSICAL_MNIST = RESULTS_DIR / "metrics_classical_mnist.json"
RESULTS_CLASSICAL_EMNIST = RESULTS_DIR / "metrics_classical_emnist.json"

# Training Hyperparameters
CNN_BATCH_SIZE = 128
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 12
CNN_WEIGHT_DECAY = 1e-4
CNN_PATIENCE = 3

# Classical HOG Configuration
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)

# Confidence & Calibration Settings
LOW_CONFIDENCE_THRESHOLD = 0.60
TOP_K_DEFAULT = 5
