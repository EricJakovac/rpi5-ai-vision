#!/bin/bash
set -e

echo "=== RPi5 AI Vision Setup ==="

# System paketi
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-opencv \
    python3-gpiozero \
    python3-lgpio \
    python3-venv \
    python3-dev \
    libopenblas-dev \
    git

# Venv s pristupom system paketima (picamera2, opencv)
python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements-rpi.txt

echo "=== Provjera instalacije ==="
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import picamera2; print('Picamera2: OK')"
python3 -c "import onnxruntime as ort; print('ORT:', ort.__version__)"
python3 -c "from gpiozero import LED; print('GPIO: OK')"

echo "=== Setup završen ==="
