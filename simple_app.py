#!/usr/bin/env python3
"""
Simple Warehouse Safety Monitoring System
A basic version that works with available packages
"""

import numpy as np
try:
    import cv2  # Optional, used for face-based heuristics
except Exception:  # pragma: no cover
    cv2 = None
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
        # Initialize face detector if OpenCV is available
        self.face_cascade = None
        if cv2 is not None:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    getattr(cv2, 'data').haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception:
                self.face_cascade = None
        
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
            
            # Detect faces (for improved helmet and glove logic)
            face_boxes = self._detect_faces(img_array)
            
            # Only run specific detections based on what we see
            self.logger.info(f"Image analysis: {image_analysis}")
            
            # Check for helmet violations using faces when available
            if face_boxes:
                for (x1, y1, x2, y2) in face_boxes:
                    if self._helmet_missing_near_face(img_array, (x1, y1, x2, y2)):
                        violations.append({
                            'type': 'helmet_violation',
                            'message': 'No helmet detected!',
                            'confidence': 0.85,
                            'bbox': [x1, max(0, y1 - (y2 - y1)), x2, y2]
                        })
            else:
                if image_analysis['has_person_like_content'] and self._simulate_helmet_violation(img_array):
                    height, width = img_array.shape[:2]
                    bbox = self._calculate_head_bbox(width, height)
                    violations.append({
                        'type': 'helmet_violation',
                        'message': 'No helmet detected!',
                        'confidence': 0.85,
                        'bbox': bbox
                    })
            
            # Check for glove violations only if we detect hand-like content
            if image_analysis['has_hand_like_content'] and self._simulate_glove_violation(img_array, face_boxes):
                # Calculate better bounding box for hands
                height, width = img_array.shape[:2]
                bbox = self._calculate_hand_bbox(width, height)
                violations.append({
                    'type': 'glove_violation', 
                    'message': 'No gloves detected!',
                    'confidence': 0.78,
                    'bbox': bbox
                })
            
            # Check for warning zone violations only if we detect people
            if image_analysis['has_person_like_content'] and self._simulate_warning_zone_violation(img_array):
                # Calculate better bounding box for person
                height, width = img_array.shape[:2]
                bbox = self._calculate_person_bbox(width, height)
                violations.append({
                    'type': 'warning_zone_violation',
                    'message': 'Person detected in warning zone!',
                    'confidence': 0.92,
                    'bbox': bbox
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
        if skin_percentage > 0.03:  # At least 3% skin-like pixels (lowered for better detection)
            analysis['has_hand_like_content'] = True
            if skin_percentage > 0.10:  # More skin suggests full person (lowered threshold)
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
        # Look for human-like shapes and check for helmet-like features
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
        
        # Calculate image complexity
        color_variance = np.var(img_array)
        height, width = img_array.shape[:2]
        
        # Look for bright/white regions that might be helmets
        # Helmets are often white, yellow, or bright colors
        bright_regions = (red_channel > 200) | (green_channel > 200) | (blue_channel > 200)
        bright_percentage = np.sum(bright_regions) / (img_array.shape[0] * img_array.shape[1])
        
        # Check for high-visibility colors (yellow, orange, bright green)
        high_vis_colors = (
            (red_channel > 180) & (green_channel > 180) & (blue_channel < 100) |  # Yellow
            (red_channel > 200) & (green_channel > 150) & (blue_channel < 100) |  # Orange
            (red_channel < 100) & (green_channel > 200) & (blue_channel < 100)   # Bright green
        )
        high_vis_percentage = np.sum(high_vis_colors) / (img_array.shape[0] * img_array.shape[1])
        
        # For helmet detection, we want to detect when someone is NOT wearing a helmet
        # This should trigger when we see skin-like regions that could be a head/face
        # BUT NOT when we see bright regions that could be helmets
        has_skin_content = skin_percentage > 0.05  # At least 5% skin-like pixels
        has_complexity = color_variance > 500  # Some color variation
        has_reasonable_size = height > 100 and width > 100  # Not too small
        
        # Check if there are bright regions that might be helmets
        has_potential_helmet = bright_percentage > 0.02 or high_vis_percentage > 0.01
        
        # Only trigger helmet violation if we see skin but NO potential helmet
        has_head_like_features = (
            has_skin_content and 
            has_complexity and 
            has_reasonable_size and
            np.mean(red_channel) > 120 and  # Reasonable skin tone
            not has_potential_helmet  # No bright regions that could be helmets
        )
        
        return has_head_like_features
    
    def _simulate_glove_violation(self, img_array, face_boxes=None):
        """Simulate glove detection"""
        # Look for hand-like shapes and check if they appear to be bare hands
        if len(img_array.shape) != 3:
            return False
            
        # Check for skin-tone regions that might be bare hands
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # If faces are detected, mask out face regions to avoid mislabeling faces as hands
        if face_boxes:
            mask = np.ones(red_channel.shape, dtype=bool)
            for (x1, y1, x2, y2) in face_boxes:
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(red_channel.shape[1]-1, x2), min(red_channel.shape[0]-1, y2)
                mask[y1c:y2c, x1c:x2c] = False
            red_channel = red_channel.copy(); red_channel[~mask] = 0
            green_channel = green_channel.copy(); green_channel[~mask] = 0
            blue_channel = blue_channel.copy(); blue_channel[~mask] = 0
        
        # Look for skin-like colors (more sophisticated detection)
        # Skin tones typically have: red > green > blue, and red > 100
        skin_like = (red_channel > 100) & (red_channel > green_channel) & (red_channel > blue_channel)
        skin_percentage = np.sum(skin_like) / (img_array.shape[0] * img_array.shape[1])
        
        # Check for hand-like features
        # Look for finger-like structures (vertical lines in skin regions)
        height, width = img_array.shape[:2]
        
        # Calculate image complexity
        color_variance = np.var(img_array)
        
        # For glove detection, we want to detect bare hands
        # This should trigger when we see skin-like regions that could be hands
        # But not on very simple images (like solid colors)
        has_skin_content = skin_percentage > 0.05  # At least 5% skin-like pixels
        has_complexity = color_variance > 500  # Some color variation
        has_reasonable_size = height > 100 and width > 100  # Not too small
        
        # Additional check: look for hand-like shapes
        # This is a simple heuristic - in reality, use trained models
        has_hand_like_features = (
            has_skin_content and 
            has_complexity and 
            has_reasonable_size and
            np.mean(red_channel) > 120  # Reasonable skin tone
        )
        
        return has_hand_like_features
    
    def _simulate_warning_zone_violation(self, img_array):
        """Simulate warning zone detection"""
        # This should only trigger if there are people in restricted areas
        # For now, make it more conservative to avoid false positives
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
        
        # Calculate image complexity
        color_variance = np.var(img_array)
        height, width = img_array.shape[:2]
        
        # For warning zone detection, we want to detect people in restricted areas
        # This should trigger when we see human-like content in what might be a restricted zone
        has_human_content = human_percentage > 0.10  # At least 10% human-like pixels
        has_complexity = color_variance > 800  # Some color variation
        has_reasonable_size = height > 100 and width > 100  # Not too small
        
        # Additional check: look for person-like features
        has_person_like_features = (
            has_human_content and 
            has_complexity and 
            has_reasonable_size and
            np.mean(blue_channel) > 100  # Reasonable color distribution
        )
        
        return has_person_like_features
    
    def _calculate_head_bbox(self, width, height):
        """Calculate bounding box for head area"""
        # Place box in upper portion of image where heads typically are
        box_width = min(width // 4, 150)
        box_height = min(height // 6, 120)
        x = width // 4  # Slightly left of center
        y = height // 8  # Upper portion
        return [x, y, x + box_width, y + box_height]
    
    def _calculate_hand_bbox(self, width, height):
        """Calculate bounding box for hand area"""
        # Place box in middle-right area where hands might be
        box_width = min(width // 5, 120)
        box_height = min(height // 8, 100)
        x = width * 3 // 5  # Right side
        y = height // 3  # Middle area
        return [x, y, x + box_width, y + box_height]
    
    def _calculate_person_bbox(self, width, height):
        """Calculate bounding box for person"""
        # Place box in center area where people typically are
        box_width = min(width // 3, 200)
        box_height = min(height // 2, 300)
        x = width // 3  # Left side
        y = height // 4  # Upper-middle
        return [x, y, x + box_width, y + box_height]

    # ----- OpenCV-assisted helpers -----
    def _detect_faces(self, img_array):
        """Detect faces using OpenCV Haar cascades; returns list of [x1,y1,x2,y2]"""
        if self.face_cascade is None:
            return []
        try:
            # Convert RGB (PIL order) to grayscale for detection
            gray = None
            if img_array.ndim == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            boxes = []
            for (x, y, w, h) in faces:
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
            return boxes
        except Exception:
            return []

    def _helmet_missing_near_face(self, img_array, face_box):
        """Heuristic: check area above face for bright/high-vis helmet; return True if absent."""
        x1, y1, x2, y2 = face_box
        h = y2 - y1
        # Region above face (possible helmet area)
        top_y1 = max(0, y1 - int(0.8 * h))
        top_y2 = y1
        left = max(0, x1 - int(0.1 * (x2 - x1)))
        right = min(img_array.shape[1], x2 + int(0.1 * (x2 - x1)))
        if top_y2 <= top_y1 or right <= left:
            return True
        region = img_array[top_y1:top_y2, left:right]
        if region.size == 0:
            return True
        r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
        bright = (r > 200) | (g > 200) | (b > 200)
        high_vis = ((r > 180) & (g > 180) & (b < 110)) | ((r > 200) & (g > 150) & (b < 110)) | ((r < 110) & (g > 200) & (b < 110))
        helmet_ratio = (np.sum(bright | high_vis) / (region.shape[0] * region.shape[1])) if region.size else 0.0
        # If enough bright/high-vis pixels are above the face, assume helmet present
        return helmet_ratio < 0.06

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
