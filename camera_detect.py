#!/usr/bin/env python3
"""
Live camera detection demo using SimpleSafetyDetector.
Press 'q' to quit.
"""

import cv2
import numpy as np
from src.detectors import SafetyDetector


def draw_boxes(frame_bgr, violations):
    # frame_bgr is BGR; draw colored boxes
    color_map = {
        'helmet_violation': (0, 0, 255),      # Red
        'glove_violation': (0, 165, 255),     # Orange
        'warning_zone_violation': (0, 255, 255)  # Yellow
    }
    for v in violations:
        x1, y1, x2, y2 = v.get('bbox', [0, 0, 0, 0])
        c = color_map.get(v.get('type'), (0, 0, 255))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
        label = v.get('message', 'violation')
        cv2.putText(frame_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return frame_bgr


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Cannot open camera')
        return

    detector = SafetyDetector(
        helmet_enabled=True,
        glove_enabled=True,
        warning_zone_enabled=True,
        confidence_threshold=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for detector
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = detector.detect_image(rgb)
        
        # Convert detection results to violation format
        violations = []
        for result in detection_results:
            violation = {
                'type': result['type'],
                'message': f"{result['class']} detected!",
                'confidence': result['confidence'],
                'bbox': result['bbox']
            }
            violations.append(violation)

        # Overlay results
        frame = draw_boxes(frame, violations)
        status = 'OK'
        if violations:
            status = f'Violations: {len(violations)}'
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not violations else (0, 0, 255), 2)

        cv2.imshow('Warehouse Safety (press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


