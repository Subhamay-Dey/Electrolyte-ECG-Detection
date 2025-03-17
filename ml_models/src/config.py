import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../ecg_model")

CONFIG = {
    "potassium_model": os.path.join(MODEL_PATH, "potassium_model"),
    "calcium_model": os.path.join(MODEL_PATH, "calcium_model"),
    "potassium_metadata": os.path.join(MODEL_PATH, "potassium_metadata.json"),
    "calcium_metadata": os.path.join(MODEL_PATH, "calcium_metadata.json"),
}