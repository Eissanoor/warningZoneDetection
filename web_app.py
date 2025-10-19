#!/usr/bin/env python3
"""
Simple Web Interface for Warehouse Safety Monitoring System
Uses Flask instead of Streamlit to avoid PyArrow dependency
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from simple_app import SimpleSafetyDetector, SimpleAlarmSystem, create_visualization
from flask import Flask, render_template_string, request, jsonify, send_file, redirect, url_for
import base64
from PIL import Image
import io

app = Flask(__name__)

# Initialize detector and alarm system
detector = SimpleSafetyDetector()
alarm_system = SimpleAlarmSystem()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Warehouse Safety Monitoring System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            background-color: #ecf0f1;
        }
        .upload-section:hover {
            background-color: #d5dbdb;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 300px;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
        }
        .violations {
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .no-violations {
            background-color: #27ae60;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .violation-item {
            background-color: rgba(255,255,255,0.2);
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .result-image {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #3498db;
            font-size: 18px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏭 Warehouse Safety Monitoring System</h1>
        
        <div class="upload-section">
            <h3>Upload Image for Safety Analysis</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageFile" name="image" accept="image/*" required>
                <br>
                <button type="submit">Analyze Image</button>
                <button type="button" onclick="testAlarm()">Test Alarm</button>
            </form>
        </div>

        <div class="loading" id="loading">
            🔍 Analyzing image for safety violations...
        </div>

        <div id="results"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image file');
                return;
            }

            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            loading.style.display = 'block';
            results.innerHTML = '';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                loading.style.display = 'none';
                
                if (data.success) {
                    displayResults(data);
                } else {
                    results.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                loading.style.display = 'none';
                results.innerHTML = `<div class="status error">Error analyzing image: ${error.message}</div>`;
            }
        });

        function displayResults(data) {
            const results = document.getElementById('results');
            
            let html = '';
            
            if (data.violations && data.violations.length > 0) {
                html += '<div class="violations">';
                html += '<h3>🚨 Safety Violations Detected!</h3>';
                html += `<p>Found ${data.violations.length} violation(s):</p>`;
                
                data.violations.forEach((violation, index) => {
                    html += `<div class="violation-item">`;
                    html += `<strong>${index + 1}.</strong> ${violation.message}`;
                    html += ` <em>(Confidence: ${(violation.confidence * 100).toFixed(1)}%)</em>`;
                    html += `</div>`;
                });
                
                html += '</div>';
                
                // Play alarm sound
                playAlarmSound();
            } else {
                html += '<div class="no-violations">';
                html += '<h3>✅ No Safety Violations Detected</h3>';
                html += '<p>All safety protocols are being followed!</p>';
                html += '</div>';
            }

            if (data.result_image) {
                html += '<div class="image-container">';
                html += '<h3>Detection Results</h3>';
                html += `<img src="data:image/png;base64,${data.result_image}" class="result-image" alt="Detection Results">`;
                html += '</div>';
            }

            results.innerHTML = html;
        }

        function playAlarmSound() {
            // Create audio context for alarm sound
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.setValueAtTime(1000, audioContext.currentTime);
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                
                oscillator.start();
                oscillator.stop(audioContext.currentTime + 0.5);
                
                // Play second tone
                setTimeout(() => {
                    const oscillator2 = audioContext.createOscillator();
                    const gainNode2 = audioContext.createGain();
                    
                    oscillator2.connect(gainNode2);
                    gainNode2.connect(audioContext.destination);
                    
                    oscillator2.frequency.setValueAtTime(800, audioContext.currentTime);
                    gainNode2.gain.setValueAtTime(0.3, audioContext.currentTime);
                    
                    oscillator2.start();
                    oscillator2.stop(audioContext.currentTime + 0.5);
                }, 600);
            } catch (error) {
                console.log('Could not play alarm sound:', error);
            }
        }

        async function testAlarm() {
            try {
                const response = await fetch('/test_alarm', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Alarm test completed successfully!');
                    playAlarmSound();
                } else {
                    alert('❌ Alarm test failed: ' + data.error);
                }
            } catch (error) {
                alert('❌ Error testing alarm: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Analyze image
            violations = detector.detect_safety_violations(temp_path)
            
            result_image_b64 = None
            if violations:
                # Create visualization
                result_file = create_visualization(temp_path, violations)
                if result_file and os.path.exists(result_file):
                    # Convert result image to base64
                    with open(result_file, 'rb') as img_file:
                        result_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Trigger alarm
                alarm_system.trigger_alarm(violations)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'violations': violations,
                'result_image': result_image_b64
            })
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'success': False, 'error': str(e)})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_alarm', methods=['POST'])
def test_alarm():
    try:
        test_violations = [
            {'type': 'helmet_violation', 'message': 'Test: No helmet detected!', 'confidence': 0.85}
        ]
        alarm_system.trigger_alarm(test_violations)
        
        return jsonify({'success': True, 'message': 'Alarm test completed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting Warehouse Safety Monitoring Web Interface...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
