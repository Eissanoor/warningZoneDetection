#!/usr/bin/env python3
"""
Simplified YOLOv8 detector that works with basic setup
This version provides better detection logic than the simple heuristic approach
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLODetectorSimple:
    """Simplified YOLOv8-like detector using OpenCV"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Try to load YOLOv8 if available, otherwise use OpenCV
        self.use_yolo = False
        self.yolo_model = None
        
        try:
            from ultralytics import YOLO
            # Try to load pre-trained YOLO model
            self.yolo_model = YOLO('yolov8n.pt')
            self.use_yolo = True
            self.logger.info("✅ Using YOLOv8 for detection")
        except ImportError:
            self.logger.warning("⚠️ YOLOv8 not available, using OpenCV-based detection")
            # Initialize OpenCV person detector
            try:
                self.person_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_fullbody.xml'
                )
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to load OpenCV cascades: {e}")
    
    def detect_image(self, image):
        """
        Detect safety violations in image
        
        Args:
            image: RGB numpy array
            
        Returns:
            list: Detection results
        """
        results = []
        
        try:
            if self.use_yolo and self.yolo_model:
                results = self._detect_with_yolo(image)
            else:
                results = self._detect_with_opencv(image)
                
        except Exception as e:
            self.logger.error(f"❌ Detection error: {e}")
            
        return results
    
    def _detect_with_yolo(self, image):
        """Detect using YOLOv8"""
        results = []
        
        try:
            # Run YOLO prediction
            yolo_results = self.yolo_model(image, conf=self.confidence_threshold)
            
            for prediction in yolo_results:
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = prediction.names[class_id]
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Map COCO classes to safety violations
                        if class_name == 'person':
                            # Check for helmet violation (person without helmet)
                            if not self._check_helmet_present(image, [int(x1), int(y1), int(x2), int(y2)]):
                                results.append({
                                    'class': 'No helmet',
                                    'confidence': confidence,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'type': 'helmet_violation'
                                })
                            
                            # Check for glove violation (person without gloves)
                            if self._check_hands_visible(image, [int(x1), int(y1), int(x2), int(y2)]):
                                results.append({
                                    'class': 'No gloves',
                                    'confidence': confidence * 0.8,
                                    'bbox': self._get_hand_bbox([int(x1), int(y1), int(x2), int(y2)]),
                                    'type': 'glove_violation'
                                })
                        
        except Exception as e:
            self.logger.error(f"❌ YOLO detection error: {e}")
            
        return results
    
    def _detect_with_opencv(self, image):
        """Detect using OpenCV cascades"""
        results = []
        
        try:
            # Convert RGB to grayscale for OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Check for helmet violation
                if not self._check_helmet_present(image, [x, y, x+w, y+h]):
                    results.append({
                        'class': 'No helmet',
                        'confidence': 0.75,
                        'bbox': [x, max(0, y-h//2), x+w, y+h],
                        'type': 'helmet_violation'
                    })
                
                # Check for glove violation
                if self._check_hands_visible(image, [x, y, x+w, y+h]):
                    hand_bbox = self._get_hand_bbox([x, y, x+w, y+h])
                    results.append({
                        'class': 'No gloves',
                        'confidence': 0.70,
                        'bbox': hand_bbox,
                        'type': 'glove_violation'
                    })
                    
        except Exception as e:
            self.logger.error(f"❌ OpenCV detection error: {e}")
            
        return results
    
    def _check_helmet_present(self, image, face_bbox):
        """Check if helmet is present above face"""
        x1, y1, x2, y2 = face_bbox
        h = y2 - y1
        
        # Region above face (helmet area)
        helmet_y1 = max(0, y1 - int(0.8 * h))
        helmet_y2 = y1
        helmet_x1 = max(0, x1 - int(0.1 * (x2 - x1)))
        helmet_x2 = min(image.shape[1], x2 + int(0.1 * (x2 - x1)))
        
        if helmet_y2 <= helmet_y1 or helmet_x2 <= helmet_x1:
            return False
            
        helmet_region = image[helmet_y1:helmet_y2, helmet_x1:helmet_x2]
        
        if helmet_region.size == 0:
            return False
        
        # Check for bright colors (typical of safety helmets)
        r, g, b = helmet_region[:, :, 0], helmet_region[:, :, 1], helmet_region[:, :, 2]
        
        # Look for white, yellow, or bright colors
        bright_regions = (r > 200) | (g > 200) | (b > 200)
        yellow_regions = (r > 180) & (g > 180) & (b < 120)
        
        helmet_ratio = (np.sum(bright_regions | yellow_regions) / helmet_region.size)
        
        return helmet_ratio > 0.1  # At least 10% bright pixels
    
    def _check_hands_visible(self, image, face_bbox):
        """Check if hands are visible and likely bare"""
        x1, y1, x2, y2 = face_bbox
        
        # Look for hand-like regions below and to the sides of face
        h = y2 - y1
        w = x2 - x1
        
        # Define potential hand regions
        left_hand_region = [
            max(0, x1 - w//2), y1 + h//2,
            x1, min(image.shape[0], y2 + h//2)
        ]
        right_hand_region = [
            x2, y1 + h//2,
            min(image.shape[1], x2 + w//2), min(image.shape[0], y2 + h//2)
        ]
        
        # Check for skin-like colors in hand regions
        for region in [left_hand_region, right_hand_region]:
            if self._has_skin_like_colors(image, region):
                return True
                
        return False
    
    def _has_skin_like_colors(self, image, region):
        """Check if region has skin-like colors"""
        x1, y1, x2, y2 = region
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        region_img = image[y1:y2, x1:x2]
        
        if region_img.size == 0:
            return False
        
        r, g, b = region_img[:, :, 0], region_img[:, :, 1], region_img[:, :, 2]
        
        # Skin color detection (simplified)
        skin_like = (r > 100) & (r > g) & (r > b) & (g > 80) & (b > 60)
        skin_ratio = np.sum(skin_like) / region_img.size
        
        return skin_ratio > 0.15  # At least 15% skin-like pixels
    
    def _get_hand_bbox(self, face_bbox):
        """Get bounding box for hands"""
        x1, y1, x2, y2 = face_bbox
        h = y2 - y1
        w = x2 - x1
        
        # Place hand bbox to the right of face
        hand_x1 = x2
        hand_y1 = y1 + h//3
        hand_x2 = min(image.shape[1], x2 + w//2)
        hand_y2 = min(image.shape[0], y1 + h)
        
        return [hand_x1, hand_y1, hand_x2, hand_y2]
