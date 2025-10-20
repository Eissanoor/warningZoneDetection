#!/usr/bin/env python3
"""
Training script for YOLOv8 safety detection models
Trains models for helmet, glove, and warning zone detection
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_helmet_model():
    """Train helmet detection model"""
    logger.info("🚀 Starting helmet model training...")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use nano model for speed
    
    # Train the model
    results = model.train(
        data='dataset/helmet/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',  # Change to 'cuda' if you have GPU
        project='models',
        name='helmet',
        save=True,
        patience=10,
        verbose=True
    )
    
    logger.info("✅ Helmet model training completed!")
    return results

def train_glove_model():
    """Train glove detection model"""
    logger.info("🚀 Starting glove model training...")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='dataset/glove/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',  # Change to 'cuda' if you have GPU
        project='models',
        name='glove',
        save=True,
        patience=10,
        verbose=True
    )
    
    logger.info("✅ Glove model training completed!")
    return results

def train_warning_zone_model():
    """Train warning zone detection model"""
    logger.info("🚀 Starting warning zone model training...")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='dataset/warningZone/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',  # Change to 'cuda' if you have GPU
        project='models',
        name='warningzone',
        save=True,
        patience=10,
        verbose=True
    )
    
    logger.info("✅ Warning zone model training completed!")
    return results

def copy_models_to_main_directory():
    """Copy trained models to main models directory"""
    logger.info("📁 Copying trained models to main directory...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Copy helmet model
    helmet_source = models_dir / "helmet" / "weights" / "best.pt"
    helmet_dest = models_dir / "helmet_model.pt"
    if helmet_source.exists():
        import shutil
        shutil.copy(str(helmet_source), str(helmet_dest))
        logger.info("✅ Helmet model copied")
    
    # Copy glove model
    glove_source = models_dir / "glove" / "weights" / "best.pt"
    glove_dest = models_dir / "glove_model.pt"
    if glove_source.exists():
        import shutil
        shutil.copy(str(glove_source), str(glove_dest))
        logger.info("✅ Glove model copied")
    
    # Copy warning zone model
    warning_source = models_dir / "warningzone" / "weights" / "best.pt"
    warning_dest = models_dir / "warningzone_model.pt"
    if warning_source.exists():
        import shutil
        shutil.copy(str(warning_source), str(warning_dest))
        logger.info("✅ Warning zone model copied")

def main():
    """Main training function"""
    print("🏭 YOLOv8 Safety Detection Model Training")
    print("=" * 50)
    
    # Check if datasets exist
    required_datasets = [
        "dataset/helmet/data.yaml",
        "dataset/glove/data.yaml", 
        "dataset/warningZone/data.yaml"
    ]
    
    for dataset in required_datasets:
        if not Path(dataset).exists():
            logger.error(f"❌ Dataset not found: {dataset}")
            return
    
    logger.info("✅ All datasets found!")
    
    try:
        # Train all models
        logger.info("Starting model training...")
        
        # Train helmet model
        train_helmet_model()
        
        # Train glove model  
        train_glove_model()
        
        # Train warning zone model
        train_warning_zone_model()
        
        # Copy models to main directory
        copy_models_to_main_directory()
        
        logger.info("🎉 All models trained successfully!")
        logger.info("You can now use the SafetyDetector with trained YOLOv8 models!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
