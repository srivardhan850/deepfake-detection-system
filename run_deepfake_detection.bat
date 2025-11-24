@echo off
REM Deepfake Detection System start script for Windows

echo Installing required Python dependencies...
pip install torch torchvision facenet-pytorch timm scikit-learn numpy Pillow opencv-python

echo.
echo Starting Flask app...
python app.py

pause
