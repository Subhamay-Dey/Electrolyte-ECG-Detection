"""
Model training functions for ECG analysis.
"""

import os
import tensorflow as tf
from src.config import EPOCHS, FINETUNE_EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH

def train_model(model, train_generator, validation_generator, electrolyte_type):
    """
    Train the model with callbacks for early stopping and model checkpointing
    
    Args:
        model (tf.keras.Model): The model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        electrolyte_type (str): Type of electrolyte ('potassium' or 'calcium')
        
    Returns:
        tuple: (history, trained_model)
    """
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, f"{electrolyte_type}_model.h5"),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history, model

def fine_tune_model(model, train_generator, validation_generator, electrolyte_type):
    """
    Fine-tune the model by unfreezing some layers
    
    Args:
        model (tf.keras.Model): The trained model to fine-tune
        train_generator: Training data generator
        validation_generator: Validation data generator
        electrolyte_type (str): Type of electrolyte ('potassium' or 'calcium')
        
    Returns:
        tuple: (history, fine_tuned_model)
    """
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all the layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, f"{electrolyte_type}_model_finetuned.h5"),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=FINETUNE_EPOCHS,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history, model