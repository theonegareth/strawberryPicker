---
tags:
- object-detection
- yolo
- yolov8
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
pretty_name: YOLOv8s Strawberry Detection
description: YOLOv8 Small model for strawberry detection with higher accuracy
pipeline_tag: object-detection
---

# YOLOv8s Strawberry Detection Model

This directory contains the YOLOv8 Small model for strawberry detection, offering higher accuracy for applications requiring precision.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **mAP@50** | 93.7% |
| **mAP@50-95** | 64.0% |
| **Inference Speed** | 35.2 FPS |
| **Model Size** | 21MB |
| **Parameters** | 11.2M |

## 🚀 Quick Start

### Installation
```bash
pip install ultralytics opencv-python
```

### Python Inference
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

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
yolo predict model=best.pt source='image.jpg'

# Webcam
yolo predict model=best.pt source=0

# Video
yolo predict model=best.pt source='video.mp4'
```

## 📁 Files

- `best.pt` - PyTorch model weights (recommended)
- `strawberry_yolov8s_enhanced.pt` - Enhanced version with additional training

## 🎯 Use Cases

- **High-Accuracy Applications**: Quality control systems requiring precision
- **Research Applications**: Academic and industrial research projects
- **Desktop Deployment**: Systems with adequate computational resources
- **Batch Processing**: Offline image analysis and sorting

## 🔧 Technical Details

- **Architecture**: YOLOv8s (Small)
- **Input Size**: 640x640
- **Training Dataset**: Enhanced Strawberry Dataset
- **Training Epochs**: 100+
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (with decay)

## 📈 Training Configuration

```yaml
model: yolov8s.pt
epochs: 100
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

## 🔗 Related Models

- [YOLOv8n](../yolov8n/) - Faster, smaller model for edge deployment
- [YOLOv11n](../yolov11n/) - Latest architecture testing

## 📚 Documentation

- [Training Pipeline](https://github.com/theonegareth/strawberryPicker)
- [Dataset](https://universe.roboflow.com/theonegareth/strawberry-detect)
- [ROS2 Integration](https://github.com/theonegareth/strawberryPicker/blob/main/ROS2_INTEGRATION_PLAN.md)

## 📄 License

MIT License - See main repository for details.

---

**Model Version**: 1.0.0  
**Training Date**: December 2, 2025  
**Recommended for**: High-accuracy applications, research, desktop deployment