"""
Model evaluation functions for ECG analysis.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import BATCH_SIZE, MODEL_SAVE_PATH

def evaluate_model(model, validation_generator, electrolyte_type):
    """
    Evaluate the model and create visualizations for performance
    
    Args:
        model (tf.keras.Model): The model to evaluate
        validation_generator: Validation data generator
        electrolyte_type (str): Type of electrolyte ('potassium' or 'calcium')
        
    Returns:
        pd.DataFrame: Classification report as a DataFrame
    """
    # Get class names
    class_names = list(validation_generator.class_indices.keys())
    
    # Make predictions
    validation_generator.reset()
    y_pred = []
    y_true = []
    
    for i in range(len(validation_generator)):
        x, y = validation_generator.next()
        y_pred_batch = model.predict(x)
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
        y_true.extend(np.argmax(y, axis=1))
        
        if (i+1) * BATCH_SIZE >= validation_generator.samples:
            break
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {electrolyte_type.capitalize()} Levels')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{electrolyte_type}_confusion_matrix.png'))
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(MODEL_SAVE_PATH, f'{electrolyte_type}_classification_report.csv'))
    
    return report_df