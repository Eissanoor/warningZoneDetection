#!/usr/bin/env python3
"""
Command Line Interface for Warehouse Safety Monitoring System
Alternative to the web interface for command-line usage
"""

import argparse
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detectors import SafetyDetector
from alarm_system import AlarmSystem
from utils import setup_logging, create_directories, validate_dataset_structure

def main():
    """Main CLI function"""
    
    # Setup logging
    setup_logging()
    
    # Create directories
    create_directories()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Warehouse Safety Monitoring System")
    parser.add_argument("--mode", choices=["image", "video", "camera"], required=True,
                       help="Detection mode")
    parser.add_argument("--input", type=str, help="Input file path or camera index")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--helmet", action="store_true", default=True,
                       help="Enable helmet detection")
    parser.add_argument("--glove", action="store_true", default=True,
                       help="Enable glove detection")
    parser.add_argument("--warning-zone", action="store_true", default=True,
                       help="Enable warning zone detection")
    parser.add_argument("--no-alarm", action="store_true",
                       help="Disable alarm system")
    
    args = parser.parse_args()
    
    # Validate datasets
    if not validate_dataset_structure():
        print("❌ Please ensure all datasets are properly configured")
        return
    
    try:
        # Initialize detector
        print("🚀 Initializing safety detection system...")
        detector = SafetyDetector(
            helmet_enabled=args.helmet,
            glove_enabled=args.glove,
            warning_zone_enabled=args.warning_zone,
            confidence_threshold=args.confidence
        )
        
        # Initialize alarm system
        alarm_system = None if args.no_alarm else AlarmSystem()
        
        print("✅ Detection system initialized successfully!")
        
        # Process based on mode
        if args.mode == "image":
            process_image(args.input, detector, alarm_system, args.output)
        elif args.mode == "video":
            process_video(args.input, detector, alarm_system, args.output)
        elif args.mode == "camera":
            camera_index = int(args.input) if args.input else 0
            process_camera(camera_index, detector, alarm_system)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return

def process_image(input_path, detector, alarm_system, output_dir=None):
    """Process single image"""
    
    if not input_path or not Path(input_path).exists():
        print("❌ Please provide a valid image file path")
        return
    
    print(f"📸 Processing image: {input_path}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    # Perform detection
    results = detector.detect_image(image)
    
    # Draw results
    annotated_image = detector.draw_detections(image, results)
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(result_file), annotated_image)
        print(f"💾 Results saved to: {result_file}")
    
    # Display results
    display_results(results, alarm_system)
    
    # Show image
    cv2.imshow("Safety Detection Results", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(input_path, detector, alarm_system, output_dir=None):
    """Process video file"""
    
    if not input_path or not Path(input_path).exists():
        print("❌ Please provide a valid video file path")
        return
    
    print(f"🎥 Processing video: {input_path}")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output directory specified
    out_writer = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    frame_count = 0
    violation_detected = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection
            results = detector.detect_image(frame)
            
            # Draw results
            annotated_frame = detector.draw_detections(frame, results)
            
            # Check for violations
            if results and not violation_detected:
                display_results(results, alarm_system)
                violation_detected = True
            
            # Write frame if output writer exists
            if out_writer:
                out_writer.write(annotated_frame)
            
            # Display progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Show frame (optional)
            cv2.imshow("Video Analysis", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing stopped by user")
    
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        if output_dir:
            print(f"💾 Video results saved to: {output_file}")

def process_camera(camera_index, detector, alarm_system):
    """Process live camera feed"""
    
    print(f"📹 Starting live camera monitoring (Camera {camera_index})")
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return
    
    print("🎥 Camera opened successfully. Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Perform detection
            results = detector.detect_image(frame)
            
            # Draw results
            annotated_frame = detector.draw_detections(frame, results)
            
            # Check for violations
            if results:
                display_results(results, alarm_system)
            
            # Show frame
            cv2.imshow("Live Safety Monitoring", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshots/capture_{timestamp}.jpg"
                Path("screenshots").mkdir(exist_ok=True)
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def display_results(results, alarm_system):
    """Display detection results"""
    
    if not results:
        print("✅ No safety violations detected")
        return
    
    print("\n🚨 SAFETY VIOLATIONS DETECTED:")
    print("=" * 50)
    
    violations = []
    for result in results:
        violation_type = result.get('type', 'unknown')
        confidence = result.get('confidence', 0)
        class_name = result.get('class', 'Unknown')
        
        if 'violation' in violation_type or 'warning' in violation_type:
            message = f"🚨 {class_name} (Confidence: {confidence:.2f})"
            print(message)
            violations.append(message)
    
    # Trigger alarm
    if alarm_system and violations:
        alarm_system.trigger_alarm(violations)
    
    print("=" * 50)

if __name__ == "__main__":
    main()
