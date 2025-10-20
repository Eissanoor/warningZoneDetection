#!/usr/bin/env python3
"""
Test the improved safety detector
"""

from simple_app import SimpleSafetyDetector
from PIL import Image
import numpy as np

def test_detector():
    """Test the improved detector with sample images"""
    print("Testing improved safety detector...")
    
    # Initialize detector
    detector = SimpleSafetyDetector()
    
    # Test with existing images if available
    test_images = ["test_image.jpg", "test_hand.png"]
    
    for image_path in test_images:
        try:
            print(f"\nTesting with {image_path}...")
            violations = detector.detect_safety_violations(image_path)
            
            print(f"Found {len(violations)} violations:")
            for i, violation in enumerate(violations):
                print(f"  {i+1}. {violation['message']} (confidence: {violation['confidence']:.2f})")
                
        except Exception as e:
            print(f"Error testing {image_path}: {e}")
    
    # Test with a synthetic image
    print("\nTesting with synthetic image...")
    synthetic_image = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic_image[:] = (128, 128, 128)  # Gray background
    
    # Add some skin-like colors (simulate person without helmet)
    synthetic_image[100:200, 200:300] = (180, 140, 120)  # Skin color
    
    # Test detection
    violations = detector.detect_safety_violations_from_array(synthetic_image)
    
    print(f"Found {len(violations)} violations in synthetic image:")
    for i, violation in enumerate(violations):
        print(f"  {i+1}. {violation['message']} (confidence: {violation['confidence']:.2f})")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_detector()
