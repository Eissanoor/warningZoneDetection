#!/usr/bin/env python3
"""
Warehouse Safety Monitoring System
Main application entry point
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime
import logging

# Import our custom modules
from src.detectors import SafetyDetector
from src.alarm_system import AlarmSystem
from src.utils import setup_logging

def main():
    """Main Streamlit application"""
    
    # Setup logging
    setup_logging()
    
    # Page configuration
    st.set_page_config(
        page_title="Warehouse Safety Monitor",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #44ff44;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏭 Warehouse Safety Monitoring System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'alarm_system' not in st.session_state:
        st.session_state.alarm_system = AlarmSystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection options
        st.subheader("Detection Settings")
        detect_helmet = st.checkbox("Helmet Detection", value=True)
        detect_gloves = st.checkbox("Glove Detection", value=True)
        detect_warning_zone = st.checkbox("Warning Zone Detection", value=True)
        
        # Alarm settings
        st.subheader("Alarm Settings")
        enable_sound = st.checkbox("Enable Sound Alerts", value=True)
        enable_visual = st.checkbox("Enable Visual Alerts", value=True)
        
        # Confidence threshold
        confidence_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
        
        # Initialize detector button
        if st.button("🚀 Initialize Detection System"):
            try:
                with st.spinner("Loading detection models..."):
                    st.session_state.detector = SafetyDetector(
                        helmet_enabled=detect_helmet,
                        glove_enabled=detect_gloves,
                        warning_zone_enabled=detect_warning_zone,
                        confidence_threshold=confidence_threshold
                    )
                st.success("✅ Detection system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize detection system: {str(e)}")
    
    # Main content area
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detection system from the sidebar first.")
        return
    
    # Mode selection
    st.header("📹 Select Detection Mode")
    
    mode = st.radio(
        "Choose your input mode:",
        ["📸 Image Upload", "🎥 Video Upload", "📹 Live Camera Feed"],
        horizontal=True
    )
    
    if mode == "📸 Image Upload":
        handle_image_upload()
    elif mode == "🎥 Video Upload":
        handle_video_upload()
    elif mode == "📹 Live Camera Feed":
        handle_live_camera()

def handle_image_upload():
    """Handle image upload and analysis"""
    st.subheader("📸 Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to analyze for safety violations"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform detection
            with st.spinner("Analyzing image..."):
                results = st.session_state.detector.detect_image(cv_image)
            
            # Display results
            if results:
                annotated_image = st.session_state.detector.draw_detections(cv_image, results)
                annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Analysis Results", use_column_width=True)
                
                # Display violations
                display_violations(results)
            else:
                st.success("✅ No safety violations detected!")

def handle_video_upload():
    """Handle video upload and analysis"""
    st.subheader("🎥 Video Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to analyze for safety violations"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Video analysis
            st.info("🎬 Processing video... This may take a while for longer videos.")
            
            if st.button("▶️ Start Video Analysis"):
                # Create video analysis
                process_video_file(video_path)
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)

def handle_live_camera():
    """Handle live camera feed"""
    st.subheader("📹 Live Camera Feed")
    
    st.info("🎥 Live camera functionality will be implemented with camera integration.")
    
    # Camera selection
    camera_index = st.selectbox(
        "Select Camera:",
        [0, 1, 2, 3],
        help="Select the camera device index"
    )
    
    if st.button("📹 Start Live Monitoring"):
        process_live_camera(camera_index)

def process_video_file(video_path):
    """Process uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file")
        return
    
    # Create video analysis container
    video_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection on frame
        results = st.session_state.detector.detect_image(frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Check for violations
        if results:
            display_violations(results)
            break  # Stop on first violation for demo
    
    cap.release()
    st.success("✅ Video analysis completed!")

def process_live_camera(camera_index):
    """Process live camera feed"""
    st.info("📹 Live camera processing will be implemented with real-time detection.")
    # This will be implemented with threading for real-time processing

def display_violations(results):
    """Display detected safety violations"""
    violations = []
    
    for result in results:
        if result['class'] == 'No helmet':
            violations.append("🚨 No helmet detected!")
        elif result['class'] == 'no_glove':
            violations.append("🚨 No gloves detected!")
        elif result['class'] == 'warningzone':
            violations.append("⚠️ Person detected in warning zone!")
    
    if violations:
        st.markdown('<div class="alert-box">', unsafe_allow_html=True)
        st.subheader("🚨 Safety Violations Detected!")
        for violation in violations:
            st.write(violation)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trigger alarm
        if st.session_state.alarm_system:
            st.session_state.alarm_system.trigger_alarm(violations)

if __name__ == "__main__":
    main()
