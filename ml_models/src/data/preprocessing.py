"""
Data preprocessing functions for ECG images.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def preprocess_data(data_dir):
    """
    Set up data generators for training and validation.
    
    Args:
        data_dir (str): Directory containing the image data organized in class folders
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator