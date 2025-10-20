#!/usr/bin/env python3
"""
Quick test to verify YOLOv8 is working
"""

try:
    from ultralytics import YOLO
    print("✅ ultralytics imported successfully")
    
    # Test loading a pre-trained model
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 model loaded successfully")
    
    # Test prediction on a simple image
    import cv2
    import numpy as np
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gray image
    
    # Run prediction
    results = model(test_image)
    print("✅ YOLOv8 prediction completed successfully")
    print(f"Found {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying to install ultralytics...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    print("✅ ultralytics installed, please run the script again")
    
except Exception as e:
    print(f"❌ Error: {e}")
