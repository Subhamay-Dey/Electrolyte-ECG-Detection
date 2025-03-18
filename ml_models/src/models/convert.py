"""
Model conversion utilities for deployment.
"""

import os
import json
import tensorflow as tf
from src.config import MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH

def convert_to_tflite(model, validation_generator, electrolyte_type):
    """
    Convert the trained model to TFLite format for deployment
    
    Args:
        model (tf.keras.Model): Trained model to convert
        validation_generator: Validation data generator (for class names)
        electrolyte_type (str): Type of electrolyte ('potassium' or 'calcium')
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(os.path.join(MODEL_SAVE_PATH, f"{electrolyte_type}_model.tflite"), "wb") as f:
        f.write(tflite_model)
    
    # Also save metadata about model input/output requirements
    model_metadata = {
        "input_shape": [IMG_HEIGHT, IMG_WIDTH],
        "class_names": list(validation_generator.class_indices.keys()),
        "preprocessing": "rescale 1./255"
    }
    
    # Save as JSON
    with open(os.path.join(MODEL_SAVE_PATH, f"{electrolyte_type}_metadata.json"), "w") as f:
        json.dump(model_metadata, f)