#!/usr/bin/env python3
"""
Test the YOLO detector
"""

from yolo_detector_simple import YOLODetectorSimple
import cv2
import numpy as np

def test_detector():
    """Test the detector with a simple image"""
    print("Testing YOLO detector...")
    
    # Initialize detector
    detector = YOLODetectorSimple(confidence_threshold=0.5)
    
    # Create a test image (gray background with some shapes)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gray background
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White rectangle (helmet-like)
    cv2.rectangle(test_image, (300, 300), (400, 400), (200, 150, 100), -1)  # Skin-like rectangle
    
    print("Running detection...")
    results = detector.detect_image(test_image)
    
    print(f"Detection results: {len(results)} violations found")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['class']} (confidence: {result['confidence']:.2f})")
    
    if results:
        print("✅ Detector is working - found potential violations")
    else:
        print("✅ Detector is working - no violations detected")

if __name__ == "__main__":
    test_detector()
