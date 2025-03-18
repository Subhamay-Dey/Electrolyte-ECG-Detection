"""
Configuration parameters for the ECG Analyzer project.
"""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30
FINETUNE_EPOCHS = 15

# Classes for classification
NUM_CLASSES = {
    "potassium": 3,  # Normal, Hypokalemia, Hyperkalemia
    "calcium": 3     # Normal, Hypocalcemia, Hypercalcemia
}

# Random seed for reproducibility
RANDOM_SEED = 42