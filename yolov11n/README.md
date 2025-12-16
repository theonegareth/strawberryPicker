---
tags:
- object-detection
- yolo
- yolov11
- strawberry
- agriculture
- robotics
- computer-vision
- pytorch
- onnx
license: mit
datasets:
- theonegareth/strawberry-detect
language:
- python
pretty_name: YOLOv11n Strawberry Detection
description: YOLOv11 Nano model for strawberry detection using latest architecture
pipeline_tag: object-detection
---

# YOLOv11n Strawberry Detection Model

This directory contains the YOLOv11 Nano model for strawberry detection, utilizing the latest YOLO architecture improvements.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **mAP@50** | TBD |
| **mAP@50-95** | TBD |
| **Inference Speed** | TBD |
| **Model Size** | TBD |
| **Parameters** | TBD |

*Performance metrics will be updated after validation testing*

## 🚀 Quick Start

### Installation
```bash
pip install ultralytics opencv-python
```

### Python Inference
```python
from ultralytics import YOLO

# Load model
model = YOLO('strawberry_yolov11n.pt')

# Run inference
results = model('image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy
        print(f"Strawberry detected: {conf:.2f} confidence at {xyxy}")
```

### Command Line
```bash
# Single image
yolo predict model=strawberry_yolov11n.pt source='image.jpg'

# Webcam
yolo predict model=strawberry_yolov11n.pt source=0

# Video
yolo predict model=strawberry_yolov11n.pt source='video.mp4'
```

## 📁 Files

- `strawberry_yolov11n.pt` - PyTorch model weights
- `strawberry_yolov11n.onnx` - ONNX model for deployment

## 🎯 Use Cases

- **Latest Architecture Testing**: Evaluation of YOLOv11 improvements
- **Edge Deployment**: Optimized for modern edge devices
- **Research Applications**: Academic and industrial research
- **Future Deployment**: Next-generation robotic systems

## 🔧 Technical Details

- **Architecture**: YOLOv11n (Nano)
- **Input Size**: 640x640
- **Training Dataset**: Enhanced Strawberry Dataset
- **Training Epochs**: TBD
- **Batch Size**: TBD
- **Optimizer**: TBD
- **Learning Rate**: TBD

## 📈 Training Configuration

*Training configuration will be updated after model validation*

```yaml
model: yolov11n.pt
epochs: TBD
batch: TBD
imgsz: 640
optimizer: TBD
lr0: TBD
# Additional hyperparameters will be added after validation
```

## 🔗 Related Models

- [YOLOv8n](../yolov8n/) - Proven YOLOv8 nano model
- [YOLOv8s](../yolov8s/) - Higher accuracy YOLOv8 small model

## 📚 Documentation

- [Training Pipeline](https://github.com/theonegareth/strawberryPicker)
- [Dataset](https://universe.roboflow.com/theonegareth/strawberry-detect)
- [ROS2 Integration](https://github.com/theonegareth/strawberryPicker/blob/main/ROS2_INTEGRATION_PLAN.md)

## ⚠️ Note

This model is currently in testing phase. Performance metrics and training details will be updated after comprehensive validation. For production deployment, consider using the validated [YOLOv8n](../yolov8n/) or [YOLOv8s](../yolov8s/) models.

## 📄 License

MIT License - See main repository for details.

---

**Model Version**: 0.1.0 (Testing)  
**Training Date**: December 2025  
**Status**: Under validation - Not recommended for production use