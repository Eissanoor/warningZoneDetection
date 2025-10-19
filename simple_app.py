#!/usr/bin/env python3
"""
Simple Warehouse Safety Monitoring System
A basic version that works with available packages
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import logging
from datetime import datetime
import winsound
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSafetyDetector:
    """Simple safety detection using basic computer vision"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_safety_violations(self, image_path):
        """
        Simple safety violation detection
        This is a placeholder that simulates detection
        """
        violations = []
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # First, analyze the image to understand what we're looking at
            image_analysis = self._analyze_image_content(img_array)
            
            # Only run specific detections based on what we see
            self.logger.info(f"Image analysis: {image_analysis}")
            
            # Check for helmet violations only if we detect a person/head area
            if image_analysis['has_person_like_content'] and self._simulate_helmet_violation(img_array):
                violations.append({
                    'type': 'helmet_violation',
                    'message': 'No helmet detected!',
                    'confidence': 0.85,
                    'bbox': [50, 50, 200, 150]
                })
            
            # Check for glove violations only if we detect hand-like content
            if image_analysis['has_hand_like_content'] and self._simulate_glove_violation(img_array):
                violations.append({
                    'type': 'glove_violation', 
                    'message': 'No gloves detected!',
                    'confidence': 0.78,
                    'bbox': [300, 100, 450, 200]
                })
            
            # Check for warning zone violations only if we detect people
            if image_analysis['has_person_like_content'] and self._simulate_warning_zone_violation(img_array):
                violations.append({
                    'type': 'warning_zone_violation',
                    'message': 'Person detected in warning zone!',
                    'confidence': 0.92,
                    'bbox': [100, 300, 250, 400]
                })
                
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            
        return violations
    
    def _analyze_image_content(self, img_array):
        """Analyze image to understand what content is present"""
        analysis = {
            'has_person_like_content': False,
            'has_hand_like_content': False,
            'has_warehouse_content': False,
            'image_complexity': 'low'
        }
        
        if len(img_array.shape) != 3:
            return analysis
            
        # Analyze image characteristics
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Check for skin-tone regions (indicating people/hands)
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # Look for skin-like colors
        skin_like = (red_channel > 100) & (red_channel > green_channel) & (red_channel > blue_channel)
        skin_percentage = np.sum(skin_like) / total_pixels
        
        # Determine if this looks like a person or hand
        if skin_percentage > 0.05:  # At least 5% skin-like pixels
            analysis['has_hand_like_content'] = True
            if skin_percentage > 0.15:  # More skin suggests full person
                analysis['has_person_like_content'] = True
        
        # Check for warehouse-like content (industrial colors, etc.)
        # This is a very basic heuristic
        avg_brightness = np.mean(img_array)
        color_variance = np.var(img_array)
        
        if avg_brightness > 100 and color_variance > 1000:
            analysis['has_warehouse_content'] = True
            analysis['image_complexity'] = 'medium'
        
        if color_variance > 2000:
            analysis['image_complexity'] = 'high'
            
        return analysis
    
    def _simulate_helmet_violation(self, img_array):
        """Simulate helmet detection based on image properties"""
        # More sophisticated heuristic: look for human-like shapes and check for helmet-like features
        # This is still a placeholder - in production, use trained YOLO models
        
        # Check if image has reasonable dimensions and color distribution
        if len(img_array.shape) != 3:
            return False
            
        # Look for skin-tone colors (indicating exposed head/face)
        # Skin tones typically have higher red values and moderate green/blue
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1] 
        blue_channel = img_array[:, :, 2]
        
        # Check for skin-like regions (this is a very basic heuristic)
        skin_like = (red_channel > 100) & (red_channel > green_channel) & (red_channel > blue_channel)
        skin_percentage = np.sum(skin_like) / (img_array.shape[0] * img_array.shape[1])
        
        # Only trigger if there's a significant amount of skin-like pixels
        # This helps avoid false positives on simple hand images
        # Make it much more conservative - only trigger on complex warehouse scenes
        return skin_percentage > 0.3 and np.mean(red_channel) > 150 and np.var(img_array) > 2000
    
    def _simulate_glove_violation(self, img_array):
        """Simulate glove detection"""
        # Look for hand-like shapes and check if they appear to be bare hands
        if len(img_array.shape) != 3:
            return False
            
        # Check for skin-tone regions that might be bare hands
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # Look for skin-like colors
        skin_like = (red_channel > 100) & (red_channel > green_channel) & (red_channel > blue_channel)
        skin_percentage = np.sum(skin_like) / (img_array.shape[0] * img_array.shape[1])
        
        # Only trigger if there's significant skin exposure (indicating no gloves)
        # Make it much more conservative - only trigger on complex warehouse scenes
        return skin_percentage > 0.2 and np.mean(green_channel) > 150 and np.var(img_array) > 1500
    
    def _simulate_warning_zone_violation(self, img_array):
        """Simulate warning zone detection"""
        # This should only trigger if there are people in restricted areas
        # For now, make it much more conservative to avoid false positives
        if len(img_array.shape) != 3:
            return False
            
        # Look for human-like shapes (this is very basic)
        # In reality, this would use person detection + zone mapping
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # Check for human-like color patterns
        human_like = (red_channel > 80) & (green_channel > 80) & (blue_channel > 80)
        human_percentage = np.sum(human_like) / (img_array.shape[0] * img_array.shape[1])
        
        # Only trigger if there's a significant human-like presence
        # Make it much more conservative - only trigger on complex warehouse scenes
        return human_percentage > 0.25 and np.mean(blue_channel) > 150 and np.var(img_array) > 2500

class SimpleAlarmSystem:
    """Simple alarm system using winsound"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def trigger_alarm(self, violations):
        """Trigger alarm for violations"""
        if violations:
            self.logger.warning(f"ALERT: Safety violations detected: {len(violations)} violations")
            
            # Play alarm sound
            self._play_alarm_sound()
            
            # Log violations
            self._log_violations(violations)
    
    def _play_alarm_sound(self):
        """Play alarm sound"""
        try:
            # Play beep sequence
            for _ in range(3):
                winsound.Beep(1000, 300)  # High frequency beep
                time.sleep(0.2)
                winsound.Beep(800, 300)   # Lower frequency beep
                time.sleep(0.2)
        except Exception as e:
            self.logger.error(f"Error playing alarm: {str(e)}")
    
    def _log_violations(self, violations):
        """Log violations to file"""
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            log_file = logs_dir / "safety_violations.log"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] Safety Violations:\n")
                for violation in violations:
                    f.write(f"  - {violation['message']} (Confidence: {violation['confidence']:.2f})\n")
                f.write("-" * 50 + "\n")
                
        except Exception as e:
            self.logger.error(f"Error logging violations: {str(e)}")

def create_visualization(image_path, violations):
    """Create visualization of detection results"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes for violations
        colors = {'helmet_violation': 'red', 'glove_violation': 'orange', 'warning_zone_violation': 'yellow'}
        
        for violation in violations:
            bbox = violation['bbox']
            color = colors.get(violation['type'], 'red')
            
            # Create rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=3, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                bbox[0], bbox[1] - 10, 
                violation['message'], 
                color=color, 
                fontsize=12, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        ax.set_title("Warehouse Safety Detection Results", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save result
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"detection_result_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(result_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(result_file)
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def main():
    """Main function for simple safety monitoring"""
    print("Simple Warehouse Safety Monitoring System")
    print("=" * 50)
    
    # Initialize detector and alarm system
    detector = SimpleSafetyDetector()
    alarm_system = SimpleAlarmSystem()
    
    while True:
        print("\nOptions:")
        print("1. Analyze image file")
        print("2. View detection results")
        print("3. Test alarm system")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Analyze image
            image_path = input("Enter image file path: ").strip()
            
            if not Path(image_path).exists():
                print("ERROR: Image file not found!")
                continue
            
            print("Analyzing image...")
            violations = detector.detect_safety_violations(image_path)
            
            if violations:
                print(f"\nALERT: Found {len(violations)} safety violations:")
                for i, violation in enumerate(violations, 1):
                    print(f"  {i}. {violation['message']} (Confidence: {violation['confidence']:.2f})")
                
                # Create visualization
                result_file = create_visualization(image_path, violations)
                if result_file:
                    print(f"Results saved to: {result_file}")
                
                # Trigger alarm
                alarm_system.trigger_alarm(violations)
            else:
                print("SUCCESS: No safety violations detected!")
        
        elif choice == "2":
            # View detection results
            output_dir = Path("output")
            if output_dir.exists():
                result_files = list(output_dir.glob("detection_result_*.png"))
                if result_files:
                    print(f"\nFound {len(result_files)} detection results:")
                    for i, file in enumerate(result_files, 1):
                        print(f"  {i}. {file.name}")
                else:
                    print("No detection results found.")
            else:
                print("No output directory found.")
        
        elif choice == "3":
            # Test alarm system
            print("Testing alarm system...")
            test_violations = [
                {'type': 'helmet_violation', 'message': 'Test: No helmet detected!', 'confidence': 0.85}
            ]
            alarm_system.trigger_alarm(test_violations)
            print("SUCCESS: Alarm test completed!")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("ERROR: Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()
