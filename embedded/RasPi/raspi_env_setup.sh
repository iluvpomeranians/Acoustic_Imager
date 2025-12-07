#!/bin/bash

echo "============================================"
echo " Acoustic Imager Setup Script (Raspberry Pi)"
echo "============================================"

# ---------- 1. Update and Upgrade ----------
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y


# ---------- 2. Install Python & Core Packages ----------
echo "[2/8] Installing Python, pip, and basic dev tools..."
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential


# ---------- 3. Install DSP + OpenCV Dependencies ----------
echo "[3/8] Installing NumPy, SciPy, OpenCV..."
sudo apt install -y python3-numpy python3-scipy python3-opencv


# ---------- 4. Install camera support ----------
echo "[4/8] Installing libcamera tools..."
sudo apt install -y libcamera0 libcamera-tools libcamera-apps python3-libcamera python3-kms++


# ---------- 5. Install Git ----------
echo "[5/8] Installing Git..."
sudo apt install -y git


# ---------- 6. Clone GitHub repository ----------
echo "[6/8] Cloning Acoustic Imager repository..."
cd ~
if [ -d "~/AcousticImager" ]; then
    echo "Repository folder already exists. Skipping clone."
else
    git clone https://github.com/B-S200502/Capstone_490_Software.git
fi


# ---------- 7. Create Python Virtual Environment ----------
echo "[7/8] Creating Python virtual environment..."

cd ~/Capstone_490_Software
python3 -m venv venv
source venv/bin/activate

echo "Installing project dependencies from requirements.txt (if it exists)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping Python package install."
fi


# ---------- 8. Enable camera ----------
echo "[8/8] Enabling Raspberry Pi camera..."
sudo raspi-config nonint do_camera 0


echo "============================================"
echo " Setup Complete!"
echo " Run the following commands to start working:"
echo ""
echo "   cd ~/Capstone_490_Software"
echo "   source venv/bin/activate"
echo "   python3 your_script.py"
echo ""
echo "============================================"
