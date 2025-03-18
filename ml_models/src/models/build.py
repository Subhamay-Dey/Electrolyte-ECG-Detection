"""
Model architecture definition for ECG analysis.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from src.config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

def build_model(electrolyte_type):
    """
    Build a transfer learning model based on EfficientNetB0
    
    Args:
        electrolyte_type (str): Type of electrolyte to analyze ('potassium' or 'calcium')
        
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Start with a pre-trained model
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES[electrolyte_type], activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model