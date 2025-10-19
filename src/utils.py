"""
Utility functions for the warehouse safety monitoring system
"""

import logging
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "warehouse_safety.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def create_directories():
    """Create necessary directories for the project"""
    
    directories = [
        "models",
        "logs", 
        "temp",
        "output",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Project directories created successfully")

def validate_dataset_structure():
    """Validate that all required datasets are present"""
    
    required_datasets = [
        "dataset/helmet/data.yaml",
        "dataset/glove/data.yaml", 
        "dataset/warningZone/data.yaml"
    ]
    
    missing_datasets = []
    
    for dataset_path in required_datasets:
        if not Path(dataset_path).exists():
            missing_datasets.append(dataset_path)
    
    if missing_datasets:
        print("⚠️ Missing datasets:")
        for dataset in missing_datasets:
            print(f"  - {dataset}")
        return False
    else:
        print("✅ All required datasets found")
        return True

def resize_image(image, max_width=1280, max_height=720):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image (numpy.ndarray): Input image
        max_width (int): Maximum width
        max_height (int): Maximum height
        
    Returns:
        numpy.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def save_annotated_image(image, results, filename=None):
    """
    Save annotated image with detection results
    
    Args:
        image (numpy.ndarray): Input image
        results (list): Detection results
        filename (str): Output filename
        
    Returns:
        str: Path to saved image
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save image
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), image)
    
    return str(output_path)

def calculate_detection_metrics(results):
    """
    Calculate detection metrics
    
    Args:
        results (list): Detection results
        
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'total_detections': len(results),
        'helmet_violations': 0,
        'glove_violations': 0,
        'warning_zone_violations': 0,
        'people_in_warning_zones': 0
    }
    
    for result in results:
        violation_type = result.get('type', '')
        
        if 'helmet_violation' in violation_type:
            metrics['helmet_violations'] += 1
        elif 'glove_violation' in violation_type:
            metrics['glove_violations'] += 1
        elif 'warning_zone' in violation_type:
            metrics['warning_zone_violations'] += 1
        elif 'person_in_warning_zone' in violation_type:
            metrics['people_in_warning_zones'] += 1
    
    return metrics

def format_violation_message(violation_type, confidence):
    """
    Format violation message for display
    
    Args:
        violation_type (str): Type of violation
        confidence (float): Detection confidence
        
    Returns:
        str: Formatted message
    """
    messages = {
        'helmet_violation': f"🚨 No helmet detected! (Confidence: {confidence:.2f})",
        'glove_violation': f"🚨 No gloves detected! (Confidence: {confidence:.2f})",
        'warning_zone': f"⚠️ Warning zone detected (Confidence: {confidence:.2f})",
        'person_in_warning_zone': f"🚨 Person in warning zone! (Confidence: {confidence:.2f})"
    }
    
    return messages.get(violation_type, f"⚠️ Unknown violation detected (Confidence: {confidence:.2f})")

def get_camera_list():
    """
    Get list of available cameras
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    # Check first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def validate_image_file(file_path):
    """
    Validate image file
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        image = cv2.imread(file_path)
        return image is not None
    except:
        return False

def validate_video_file(file_path):
    """
    Validate video file
    
    Args:
        file_path (str): Path to video file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except:
        return False

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_dir = Path("temp")
    
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass

def get_system_info():
    """
    Get system information
    
    Returns:
        dict: System information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    }
    
    return info
