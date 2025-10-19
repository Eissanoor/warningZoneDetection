@echo off
echo 🏭 Starting Warehouse Safety Monitoring System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install -r requirements.txt
)

REM Start the web application
echo 🚀 Launching web interface...
echo Open your browser to http://localhost:8501
echo.
streamlit run main.py

pause
