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
- ripeness-detection
license: mit
datasets:
- custom
language:
- python
pretty_name: YOLOv11n Strawberry Ripeness Detection
description: YOLOv11 Nano model for strawberry ripeness detection with 3-class classification
pipeline_tag: object-detection
---

# YOLOv11n Strawberry Ripeness Detection Model

This directory contains the YOLOv11 Nano model for strawberry ripeness detection, part of the two-stage Strawberry Picker AI system.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **mAP@50** | 84.0% |
| **mAP@50-95** | 57.0% |
| **Precision** | 74.5% |
| **Recall** | 82.1% |
| **Model Size** | 5.2MB |
| **Inference Speed** | ~35 FPS (RTX 3050 Ti) |

### Class Performance (Test Set)

| Class | Precision | Recall | mAP50 | Support |
|-------|-----------|--------|-------|---------|
| partially-ripe | 78.8% | 92.4% | 92.2% | 79 |
| ripe | 82.0% | 87.1% | 89.5% | 70 |
| unripe | 62.9% | 66.7% | 70.3% | 78 |

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
results = model('strawberry_image.jpg', conf=0.5)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy
        class_names = ['partially-ripe', 'ripe', 'unripe']
        print(f"{class_names[cls]} strawberry: {conf:.2f} confidence")
```

### Command Line
```bash
# Single image
yolo predict model=best.pt source='strawberry.jpg'

# Webcam
yolo predict model=best.pt source=0
```

## 📁 Files

- `best.pt` - PyTorch model weights (recommended)

## 🎯 Use Cases

- **Automated Harvesting**: First stage of two-stage picking system
- **Ripeness Assessment**: Initial strawberry detection and ripeness categorization
- **Quality Control**: Pre-classification for detailed ripeness analysis

## 🔧 Technical Details

- **Architecture**: YOLOv11n (Nano)
- **Input Size**: 416x416
- **Classes**: 3 (partially-ripe, ripe, unripe)
- **Training Dataset**: Custom dataset (1200+ annotated strawberries)
- **Training Epochs**: 50 (early stopping at 20)
- **Batch Size**: 8
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (cosine annealing)

## 📈 Training Configuration

```yaml
model: yolov11n.pt
epochs: 50
batch: 8
imgsz: 416
optimizer: AdamW
lr0: 0.01
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3.0
patience: 20
classes: 3
names: ['partially-ripe', 'ripe', 'unripe']
```

## 🔗 Related Components

- [Classification Model](../classification/) - Second stage for detailed ripeness classification
- [Training Repository](https://github.com/theonegareth/strawberryPicker)

## 📚 Documentation

- [Full System Documentation](https://github.com/theonegareth/strawberryPicker)
- [Two-Stage Pipeline](https://github.com/theonegareth/strawberryPicker#system-architecture)

## 📄 License

MIT License - See main repository for details.

---

**Model Version**: 1.0.0  
**Training Date**: November 2025  
**Part of**: Strawberry Picker AI System