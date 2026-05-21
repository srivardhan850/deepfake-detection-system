@echo off
REM Deepfake Detection System start script for Windows

echo Installing required Python dependencies...
pip install -r requirements.txt

echo.
echo Starting Flask app...
python app.py

pause
