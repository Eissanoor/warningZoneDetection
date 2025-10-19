"""
Safety Detection Module
Handles helmet, glove, and warning zone detection using YOLO models
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from pathlib import Path

class SafetyDetector:
    """Main safety detection class"""
    
    def __init__(self, helmet_enabled=True, glove_enabled=True, warning_zone_enabled=True, confidence_threshold=0.5):
        """
        Initialize the safety detector
        
        Args:
            helmet_enabled (bool): Enable helmet detection
            glove_enabled (bool): Enable glove detection  
            warning_zone_enabled (bool): Enable warning zone detection
            confidence_threshold (float): Detection confidence threshold
        """
        self.helmet_enabled = helmet_enabled
        self.glove_enabled = glove_enabled
        self.warning_zone_enabled = warning_zone_enabled
        self.confidence_threshold = confidence_threshold
        
        self.helmet_model = None
        self.glove_model = None
        self.warning_zone_model = None
        
        self.logger = logging.getLogger(__name__)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load YOLO models for each detection type"""
        try:
            # Path to dataset directory
            dataset_path = Path("dataset")
            
            # Load helmet detection model
            if self.helmet_enabled:
                helmet_config = dataset_path / "helmet" / "data.yaml"
                if helmet_config.exists():
                    # Train a model if it doesn't exist, or load existing model
                    self.helmet_model = self._get_or_train_model("helmet", helmet_config)
                    self.logger.info("✅ Helmet detection model loaded")
                else:
                    self.logger.warning("⚠️ Helmet dataset config not found")
            
            # Load glove detection model
            if self.glove_enabled:
                glove_config = dataset_path / "glove" / "data.yaml"
                if glove_config.exists():
                    self.glove_model = self._get_or_train_model("glove", glove_config)
                    self.logger.info("✅ Glove detection model loaded")
                else:
                    self.logger.warning("⚠️ Glove dataset config not found")
            
            # Load warning zone detection model
            if self.warning_zone_enabled:
                warning_config = dataset_path / "warningZone" / "data.yaml"
                if warning_config.exists():
                    self.warning_zone_model = self._get_or_train_model("warningzone", warning_config)
                    self.logger.info("✅ Warning zone detection model loaded")
                else:
                    self.logger.warning("⚠️ Warning zone dataset config not found")
                    
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {str(e)}")
            raise
    
    def _get_or_train_model(self, model_name, config_path):
        """
        Get existing model or train a new one
        
        Args:
            model_name (str): Name of the model
            config_path (Path): Path to dataset config
            
        Returns:
            YOLO: Trained model
        """
        model_path = Path(f"models/{model_name}_model.pt")
        
        if model_path.exists():
            # Load existing model
            return YOLO(str(model_path))
        else:
            # Train new model
            self.logger.info(f"🚀 Training {model_name} detection model...")
            
            # Create models directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Train the model
            model = YOLO('yolov8n.pt')  # Start with nano model for speed
            results = model.train(
                data=str(config_path),
                epochs=50,  # Adjust based on your needs
                imgsz=640,
                batch=16,
                device='cpu',  # Change to 'cuda' if you have GPU
                project='models',
                name=model_name,
                save=True
            )
            
            # Save the trained model
            trained_model_path = model_path.parent / model_name / "weights" / "best.pt"
            if trained_model_path.exists():
                model = YOLO(str(trained_model_path))
                # Copy to main models directory
                import shutil
                shutil.copy(str(trained_model_path), str(model_path))
            
            self.logger.info(f"✅ {model_name} model training completed")
            return model
    
    def detect_image(self, image):
        """
        Detect safety violations in an image
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of detection results
        """
        results = []
        
        try:
            # Helmet detection
            if self.helmet_enabled and self.helmet_model:
                helmet_results = self._detect_helmet(image)
                results.extend(helmet_results)
            
            # Glove detection
            if self.glove_enabled and self.glove_model:
                glove_results = self._detect_glove(image)
                results.extend(glove_results)
            
            # Warning zone detection
            if self.warning_zone_enabled and self.warning_zone_model:
                warning_results = self._detect_warning_zone(image)
                results.extend(warning_results)
            
            # Also detect people in warning zones
            people_results = self._detect_people_in_warning_zones(image, results)
            results.extend(people_results)
            
        except Exception as e:
            self.logger.error(f"❌ Error in detection: {str(e)}")
        
        return results
    
    def _detect_helmet(self, image):
        """Detect helmet violations"""
        results = []
        
        try:
            helmet_predictions = self.helmet_model(image, conf=self.confidence_threshold)
            
            for prediction in helmet_predictions:
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = prediction.names[class_id]
                        
                        # Only flag violations (no helmet)
                        if class_name == "No helmet":
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            results.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'type': 'helmet_violation'
                            })
        
        except Exception as e:
            self.logger.error(f"❌ Error in helmet detection: {str(e)}")
        
        return results
    
    def _detect_glove(self, image):
        """Detect glove violations"""
        results = []
        
        try:
            glove_predictions = self.glove_model(image, conf=self.confidence_threshold)
            
            for prediction in glove_predictions:
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = prediction.names[class_id]
                        
                        # Only flag violations (no glove)
                        if class_name == "no_glove":
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            results.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'type': 'glove_violation'
                            })
        
        except Exception as e:
            self.logger.error(f"❌ Error in glove detection: {str(e)}")
        
        return results
    
    def _detect_warning_zone(self, image):
        """Detect warning zones"""
        results = []
        
        try:
            warning_predictions = self.warning_zone_model(image, conf=self.confidence_threshold)
            
            for prediction in warning_predictions:
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = prediction.names[class_id]
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        results.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'type': 'warning_zone'
                        })
        
        except Exception as e:
            self.logger.error(f"❌ Error in warning zone detection: {str(e)}")
        
        return results
    
    def _detect_people_in_warning_zones(self, image, existing_results):
        """Detect people in warning zones using YOLO person detection"""
        results = []
        
        try:
            # Use YOLO to detect people
            person_model = YOLO('yolov8n.pt')  # Use pre-trained model for person detection
            person_predictions = person_model(image, conf=self.confidence_threshold)
            
            # Get warning zones from existing results
            warning_zones = [r for r in existing_results if r['type'] == 'warning_zone']
            
            for prediction in person_predictions:
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = prediction.names[class_id]
                        
                        # Only process person detections
                        if class_name == 'person':
                            person_bbox = box.xyxy[0].cpu().numpy()
                            
                            # Check if person is in any warning zone
                            for warning_zone in warning_zones:
                                if self._bbox_overlap(person_bbox, warning_zone['bbox']):
                                    results.append({
                                        'class': 'Person in warning zone',
                                        'confidence': float(box.conf[0]),
                                        'bbox': [int(person_bbox[0]), int(person_bbox[1]), 
                                                int(person_bbox[2]), int(person_bbox[3])],
                                        'type': 'person_in_warning_zone'
                                    })
                                    break  # Don't duplicate for multiple warning zones
        
        except Exception as e:
            self.logger.error(f"❌ Error in person detection: {str(e)}")
        
        return results
    
    def _bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Check if boxes overlap
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def draw_detections(self, image, results):
        """
        Draw detection results on image
        
        Args:
            image (numpy.ndarray): Input image
            results (list): Detection results
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        annotated_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            class_name = result['class']
            confidence = result['confidence']
            violation_type = result['type']
            
            # Choose color based on violation type
            if 'violation' in violation_type or 'warning' in violation_type:
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for normal detections
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
