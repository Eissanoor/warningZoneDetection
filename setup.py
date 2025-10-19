#!/usr/bin/env python3
"""
Setup script for Warehouse Safety Monitoring System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "models",
        "logs",
        "temp", 
        "output",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def check_datasets():
    """Check if datasets are present"""
    print("🔍 Checking datasets...")
    
    required_datasets = [
        "dataset/helmet/data.yaml",
        "dataset/glove/data.yaml",
        "dataset/warningZone/data.yaml"
    ]
    
    missing_datasets = []
    
    for dataset_path in required_datasets:
        if Path(dataset_path).exists():
            print(f"  ✅ Found: {dataset_path}")
        else:
            print(f"  ❌ Missing: {dataset_path}")
            missing_datasets.append(dataset_path)
    
    if missing_datasets:
        print("⚠️ Some datasets are missing. Please ensure all datasets are properly configured.")
        return False
    
    return True

def download_yolo_weights():
    """Download YOLO weights if needed"""
    print("⬇️ Checking YOLO weights...")
    
    try:
        from ultralytics import YOLO
        # This will automatically download weights on first use
        model = YOLO('yolov8n.pt')
        print("✅ YOLO weights ready")
        return True
    except Exception as e:
        print(f"❌ Failed to setup YOLO weights: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        "cv2",
        "ultralytics", 
        "streamlit",
        "numpy",
        "PIL",
        "pygame",
        "torch"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print("⚠️ Some modules failed to import. Please check your installation.")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🏭 Warehouse Safety Monitoring System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Check datasets
    if not check_datasets():
        print("⚠️ Continuing setup despite missing datasets...")
    
    # Download YOLO weights
    if not download_yolo_weights():
        print("⚠️ YOLO weights setup failed, but continuing...")
    
    # Test imports
    if not test_imports():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Ensure all datasets are properly configured")
    print("2. Run the web interface: streamlit run main.py")
    print("3. Or use CLI: python cli.py --help")
    print("\n📚 See README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
