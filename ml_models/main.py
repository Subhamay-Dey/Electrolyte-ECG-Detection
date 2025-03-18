"""
Entry point script for ECG Analyzer project.
"""

import os
import numpy as np
import tensorflow as tf
import logging
from src.config import DATA_DIR, RANDOM_SEED
from src.data.preprocessing import preprocess_data
from src.models.build import build_model
from src.models.train import train_model, fine_tune_model
from src.models.evaluate import evaluate_model
from src.models.convert import convert_to_tflite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def train_ecg_analyzer():
    """
    Main function to train models for both electrolyte types
    """
    electrolyte_types = ["potassium", "calcium"]
    
    for electrolyte_type in electrolyte_types:
        logger.info(f"Training model for {electrolyte_type} detection...")
        
        # For this example, we assume data is organized by electrolyte type
        data_dir = os.path.join(DATA_DIR, electrolyte_type)
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found! Please ensure your data is properly organized.")
            continue
        
        # Prepare data
        train_generator, validation_generator = preprocess_data(data_dir)
        logger.info(f"Found {train_generator.samples} training samples and {validation_generator.samples} validation samples")
        
        # Build and train model
        model = build_model(electrolyte_type)
        logger.info(f"Model built for {electrolyte_type} with {model.count_params()} parameters")
        
        logger.info(f"Starting initial training for {electrolyte_type}...")
        history, model = train_model(model, train_generator, validation_generator, electrolyte_type)
        
        # Evaluate
        logger.info(f"Evaluating initial model for {electrolyte_type}...")
        report = evaluate_model(model, validation_generator, electrolyte_type)
        logger.info(f"Initial model performance for {electrolyte_type}:")
        logger.info(f"Accuracy: {report.loc['accuracy', 'f1-score']:.4f}")
        
        # Fine-tune
        logger.info(f"Fine-tuning model for {electrolyte_type}...")
        history, model = fine_tune_model(model, train_generator, validation_generator, electrolyte_type)
        
        # Re-evaluate
        logger.info(f"Evaluating fine-tuned model for {electrolyte_type}...")
        report = evaluate_model(model, validation_generator, electrolyte_type)
        logger.info(f"Fine-tuned model performance for {electrolyte_type}:")
        logger.info(f"Accuracy: {report.loc['accuracy', 'f1-score']:.4f}")
        
        # Convert for deployment
        logger.info(f"Converting model for {electrolyte_type} to TFLite format...")
        convert_to_tflite(model, validation_generator, electrolyte_type)
        
        logger.info(f"Model for {electrolyte_type} saved successfully!")

if __name__ == "__main__":
    train_ecg_analyzer()