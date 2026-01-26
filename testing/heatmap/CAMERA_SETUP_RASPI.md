# Raspberry Pi Camera Setup Guide

## For Raspberry Pi Camera Module (CSI connector)

### 1. Enable the Camera
```bash
sudo raspi-config
```
- Navigate to: **Interface Options** → **Camera** → **Enable**
- Reboot: `sudo reboot`

### 2. Install picamera2 (Modern Python Library)
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install picamera2
sudo apt install -y python3-picamera2 python3-libcamera

# Install additional dependencies
sudo apt install -y python3-opencv
```

### 3. Test Camera
```bash
# Test with libcamera
libcamera-hello

# List available cameras
libcamera-hello --list-cameras

# Take a test photo
libcamera-jpeg -o test.jpg
```

### 4. Run the Script
```bash
cd /Users/basemsousou/Documents/Capstone_490_Software/testing/heatmap
python3 heatmap_dbscaler_with_video.py
```

---

## For USB Webcam

### 1. Check if Camera is Detected
```bash
# List video devices
ls -l /dev/video*

# Get detailed info
v4l2-ctl --list-devices
```

### 2. Install v4l-utils (if needed)
```bash
sudo apt install -y v4l-utils
```

### 3. Add User to Video Group
```bash
sudo usermod -a -G video $USER
# Log out and log back in for changes to take effect
```

### 4. Test with OpenCV
```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f"Success: {ret}, Frame shape: {frame.shape if ret else 'N/A'}")
cap.release()
```

---

## Troubleshooting

### Camera Not Found
```bash
# Check if camera is detected by system
vcgencmd get_camera

# Should show: supported=1 detected=1

# Check camera module connection
libcamera-hello --list-cameras
```

### Permission Denied
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Check current groups
groups

# Reboot if needed
sudo reboot
```

### Camera Already in Use
```bash
# Check what's using the camera
sudo lsof /dev/video*

# Kill the process if needed
sudo kill <PID>
```

### GStreamer Issues
```bash
# Install GStreamer plugins
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav
```

---

## Configuration Options

Edit the script to change camera settings:

```python
# In heatmap_dbscaler_with_video.py, around line 63:

CAMERA_INDEX = 0          # For USB cameras (try 0, 1, 2)
CAMERA_WIDTH = 1024       # Camera resolution width
CAMERA_HEIGHT = 600       # Camera resolution height
USE_CAMERA = True         # Enable/disable camera
USE_LIBCAMERA = True      # Try libcamera first (for Pi Camera Module)
```

---

## Script Features

The updated script now supports:

1. **Picamera2** (Raspberry Pi Camera Module) - Preferred method
2. **GStreamer with libcamera** - Alternative for Pi Camera
3. **Standard OpenCV VideoCapture** - For USB webcams
4. **Automatic fallback** - Falls back to static background if camera fails

The script will automatically try all methods in order and use the first one that works.
