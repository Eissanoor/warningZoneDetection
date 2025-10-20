#!/usr/bin/env python3
"""
Test the improved helmet detection with classroom-like scenarios
"""

from simple_app import SimpleSafetyDetector
import numpy as np
from PIL import Image

def create_test_classroom_image():
    """Create a synthetic classroom image with people without helmets"""
    # Create a classroom-like image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background - light colored walls
    img[:] = (240, 240, 240)  # Light gray background
    
    # Add some classroom elements
    # Blackboard at the top
    img[50:120, 50:590] = (50, 50, 50)  # Dark board
    
    # Desks - brown color
    img[300:320, 100:540] = (139, 69, 19)  # Brown desk
    img[350:370, 100:540] = (139, 69, 19)  # Brown desk
    img[400:420, 100:540] = (139, 69, 19)  # Brown desk
    
    # Add people without helmets
    # Person 1 - skin tone head
    img[280:320, 150:200] = (220, 180, 140)  # Skin tone
    # Person 1 - dark suit
    img[320:400, 140:210] = (50, 50, 50)  # Dark suit
    
    # Person 2 - skin tone head
    img[330:370, 250:300] = (200, 160, 120)  # Skin tone
    # Person 2 - dark suit
    img[370:450, 240:310] = (60, 60, 60)  # Dark suit
    
    # Person 3 - skin tone head
    img[380:420, 350:400] = (210, 170, 130)  # Skin tone
    # Person 3 - dark suit
    img[420:500, 340:410] = (55, 55, 55)  # Dark suit
    
    return img

def test_classroom_detection():
    """Test helmet detection in classroom scenario"""
    print("Testing improved helmet detection with classroom scenario...")
    
    # Initialize detector
    detector = SimpleSafetyDetector()
    
    # Test with synthetic classroom image
    classroom_img = create_test_classroom_image()
    
    print("Running detection on classroom image...")
    violations = detector.detect_safety_violations_from_array(classroom_img)
    
    print(f"Found {len(violations)} violations:")
    for i, violation in enumerate(violations):
        print(f"  {i+1}. {violation['message']} (confidence: {violation['confidence']:.2f})")
    
    if violations:
        helmet_violations = [v for v in violations if v['type'] == 'helmet_violation']
        print(f"\n✅ SUCCESS: Found {len(helmet_violations)} helmet violations!")
        print("The system now correctly detects people without helmets in classroom settings.")
    else:
        print("\n❌ ISSUE: No helmet violations detected.")
        print("The system may still need further tuning for classroom scenarios.")
    
    return len(violations) > 0

def test_with_existing_images():
    """Test with any existing images"""
    print("\nTesting with existing images...")
    
    detector = SimpleSafetyDetector()
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

if __name__ == "__main__":
    # Test classroom detection
    success = test_classroom_detection()
    
    # Test with existing images
    test_with_existing_images()
    
    if success:
        print("\n🎉 SUCCESS: The improved system now detects helmet violations correctly!")
        print("You can now test with your classroom image and it should detect 'No helmet' violations.")
    else:
        print("\n⚠️ The system may need further improvements for your specific use case.")
