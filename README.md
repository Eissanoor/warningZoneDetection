# 🏭 Warehouse Safety Monitoring System

A comprehensive AI-powered safety monitoring system for warehouses that detects safety violations including missing helmets, missing gloves, and unauthorized entry into warning zones.

## 🚀 Features

### Detection Capabilities
- **Helmet Detection**: Identifies workers without safety helmets
- **Glove Detection**: Detects workers not wearing safety gloves  
- **Warning Zone Detection**: Monitors restricted areas for unauthorized access
- **Person Detection**: Identifies people entering warning zones

### Input Options
- **📸 Image Upload**: Analyze individual images for safety violations
- **🎥 Video Analysis**: Process video files frame by frame
- **📹 Live Camera**: Real-time monitoring from camera feeds

### Alert System
- **Audio Alerts**: Sound alarms for immediate notification
- **Visual Alerts**: On-screen warnings with violation details
- **Logging**: Comprehensive violation logging with timestamps

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam or camera device (for live monitoring)
- At least 4GB RAM (8GB recommended for optimal performance)
- GPU support optional but recommended for faster processing

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd warehouse-safety-monitoring
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset Structure
Ensure your datasets are properly organized:
```
dataset/
├── helmet/
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── glove/
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── warningZone/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

## 🚀 Usage

### Simple Application (Recommended for Python 3.14)
Run the simple application that works with available packages:
```bash
python simple_app.py
```

### Test the System
Run the test script to verify everything is working:
```bash
python test_system.py
```

### Advanced Usage (Requires Additional Setup)
For the full-featured system with YOLO models and Streamlit:

#### Web Interface
Launch the Streamlit web application:
```bash
streamlit run main.py
```

#### Command Line Interface
For command-line usage:

##### Analyze an Image
```bash
python cli.py --mode image --input path/to/image.jpg --output results/
```

##### Analyze a Video
```bash
python cli.py --mode video --input path/to/video.mp4 --output results/
```

##### Live Camera Monitoring
```bash
python cli.py --mode camera --input 0
```

### Command Line Options
- `--mode`: Detection mode (image/video/camera)
- `--input`: Input file path or camera index
- `--output`: Output directory for results
- `--confidence`: Detection confidence threshold (0.1-0.9)
- `--helmet`: Enable helmet detection
- `--glove`: Enable glove detection
- `--warning-zone`: Enable warning zone detection
- `--no-alarm`: Disable alarm system

## 📊 Dataset Information

The system uses three pre-trained models based on your datasets:

### Helmet Dataset
- **Classes**: 2 (No helmet, Safety helmet)
- **Training Images**: 945
- **Validation Images**: 271
- **Test Images**: 136

### Glove Dataset  
- **Classes**: 2 (glove, no_glove)
- **Training Images**: 1,462
- **Validation Images**: 138
- **Test Images**: 108

### Warning Zone Dataset
- **Classes**: 1 (warningzone)
- **Training Images**: 22
- **Validation Images**: 6
- **Test Images**: 3

## 🔧 Configuration

### Detection Settings
- **Confidence Threshold**: Adjust detection sensitivity (default: 0.5)
- **Model Selection**: Choose which detection types to enable
- **Alarm Settings**: Configure audio and visual alerts

### Performance Tuning
For better performance:
1. Use GPU acceleration by changing `device='cpu'` to `device='cuda'` in `src/detectors.py`
2. Adjust batch size in model training parameters
3. Reduce input image resolution for faster processing

## 📁 Project Structure

```
warehouse-safety-monitoring/
├── main.py                 # Streamlit web application
├── cli.py                  # Command line interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── detectors.py       # Core detection logic
│   ├── alarm_system.py    # Alarm and notification system
│   └── utils.py           # Utility functions
├── dataset/               # Training datasets
├── models/                # Trained model files (auto-created)
├── logs/                  # System logs (auto-created)
└── output/                # Detection results (auto-created)
```

## 🚨 Safety Violation Types

### Helmet Violations
- **Detection**: Worker without safety helmet
- **Alert**: Audio alarm + visual warning
- **Action**: Immediate safety protocol activation

### Glove Violations  
- **Detection**: Worker without safety gloves
- **Alert**: Audio alarm + visual warning
- **Action**: PPE compliance enforcement

### Warning Zone Violations
- **Detection**: Person entering restricted area
- **Alert**: High-priority audio alarm + visual alert
- **Action**: Emergency response protocol

## 📈 Performance Metrics

The system tracks:
- Detection accuracy per violation type
- Processing speed (FPS)
- False positive/negative rates
- Alert response times

## 🔍 Troubleshooting

### Common Issues

#### Models Not Loading
```bash
# Check if datasets exist
ls dataset/helmet/data.yaml
ls dataset/glove/data.yaml  
ls dataset/warningZone/data.yaml
```

#### Camera Not Working
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### Audio Issues
```bash
# Install audio dependencies
pip install pygame
# For Windows: pip install winsound
```

### Performance Issues
- Reduce confidence threshold for faster processing
- Use smaller input image sizes
- Enable GPU acceleration if available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/warehouse_safety.log`
3. Create an issue with detailed error information

## 🔮 Future Enhancements

- [ ] Real-time dashboard with multiple camera feeds
- [ ] Mobile app for remote monitoring
- [ ] Integration with warehouse management systems
- [ ] Advanced analytics and reporting
- [ ] Multi-language support
- [ ] Cloud deployment options

---

**⚠️ Safety Notice**: This system is designed to assist with safety monitoring but should not replace proper safety protocols, training, and supervision. Always maintain standard safety procedures in your warehouse.
