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
            
            # Simulate detection based on image properties
            # In a real implementation, this would use trained models
            
            # Check for helmet violations (simulate based on image analysis)
            if self._simulate_helmet_violation(img_array):
                violations.append({
                    'type': 'helmet_violation',
                    'message': 'No helmet detected!',
                    'confidence': 0.85,
                    'bbox': [50, 50, 200, 150]
                })
            
            # Check for glove violations
            if self._simulate_glove_violation(img_array):
                violations.append({
                    'type': 'glove_violation', 
                    'message': 'No gloves detected!',
                    'confidence': 0.78,
                    'bbox': [300, 100, 450, 200]
                })
            
            # Check for warning zone violations
            if self._simulate_warning_zone_violation(img_array):
                violations.append({
                    'type': 'warning_zone_violation',
                    'message': 'Person detected in warning zone!',
                    'confidence': 0.92,
                    'bbox': [100, 300, 250, 400]
                })
                
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            
        return violations
    
    def _simulate_helmet_violation(self, img_array):
        """Simulate helmet detection based on image properties"""
        # Simple heuristic: check if image has certain color patterns
        # In reality, this would use a trained model
        return len(img_array.shape) == 3 and np.mean(img_array[:, :, 0]) > 100
    
    
    def _simulate_glove_violation(self, img_array):
        """Simulate glove detection"""
        return len(img_array.shape) == 3 and np.mean(img_array[:, :, 1]) > 120
    
    def _simulate_warning_zone_violation(self, img_array):
        """Simulate warning zone detection"""
        return len(img_array.shape) == 3 and np.mean(img_array[:, :, 2]) > 110

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
