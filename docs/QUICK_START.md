# StrawberryPicker Quick Start Tutorial

## Overview
This tutorial will get you up and running with StrawberryPicker in under 30 minutes.

## Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- USB camera
- Arduino (optional for full functionality)

## Step 1: Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/theonegareth/strawberryPicker.git
cd strawberryPicker

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Pre-trained Models (2 minutes)

```bash
# Download the best YOLOv8s model
python scripts/download_models.py

# Or manually download from Hugging Face:
# https://huggingface.co/theonegareth/strawberry-yolov8s-detector
```

## Step 3: Test Detection (5 minutes)

```python
# Test with webcam
python webcam_inference.py

# Test with image
python -c "
from ultralytics import YOLO
model = YOLO('huggingface_models/strawberry-yolov8s-detector/best.pt')
results = model('assets/1.jpg')
results[0].show()
"
```

## Step 4: Real-time Detection (10 minutes)

```python
# Run real-time detection
python -c "
import cv2
from ultralytics import YOLO

model = YOLO('huggingface_models/strawberry-yolov8s-detector/best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.7)
    annotated = results[0].plot()
    
    cv2.imshow('Strawberry Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"
```

## Step 5: Arduino Integration (8 minutes)

### Basic Arduino Integration
```python
# Connect Arduino and test communication
from src.arduino_bridge import ArduinoBridge

arduino = ArduinoBridge(port='/dev/ttyUSB0', baudrate=115200)
arduino.test_connection()

# Move to position (if Arduino is connected)
arduino.move_to_position(x=10.0, y=5.0, z=15.0)
```

### Enhanced Strawberry Locator Integration (NEW!)
```python
# Use enhanced locator for better depth detection
from strawberrylocator import StrawberryLocator
from ultralytics import YOLO

# Initialize enhanced locator
locator = StrawberryLocator()
model = YOLO('huggingface_models/strawberry-yolov8s-detector/best.pt')

# Capture stereo frames (you'll need to implement frame capture)
left_frame = cv2.imread('left_image.jpg')
right_frame = cv2.imread('right_image.jpg')

# Process with enhanced depth detection
results = locator.process_frame_pair(left_frame, right_frame, model)

# Send high-confidence results to Arduino
for result in results:
    if result['confidence'] > 0.7:
        x, y, z = result['position_3d']  # 3D coordinates in cm
        print(f"Moving to: X={x:.1f}, Y={y:.1f}, Z={z:.1f}cm")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method used: {result['method']}")
        
        # Send to Arduino (replace with your function)
        # send_to_arduino(x, y, z)

## Next Steps
- Read the full [Technical Documentation](TECHNICAL_DOCUMENTATION.md)
- Train your own model with [Training Tutorial](TRAINING_TUTORIAL.md)
- Explore [API Reference](API_REFERENCE.md)

## Troubleshooting
- Camera not working: Check camera index in code
- Arduino not connecting: Verify port and permissions
- Model not loading: Check model file path

## Support
- GitHub Issues: https://github.com/theonegareth/strawberryPicker/issues
- Documentation: See docs/ folder