# Strawberry Detection Models

This directory contains trained YOLOv8 models for strawberry detection in robotic picking applications.

## 📁 Directory Structure

```
detection/
├── baseline/                                   # ✅ Initial baseline model (Active)
├── kaggle_strawberry_yolov8n_20251204_115538/ # ⭐ **RECOMMENDED: Best performing model** (Active)
├── kaggle_strawberry_yolov8s_20251204_2105262/ # ✅ YOLOv8s trained model (Active)
├── optimized_yolov8n_20251204_154529/         # ✅ Optimized with enhanced augmentations (Active)
├── yolov8n/                                    # ✅ Standard YOLOv8n model (Active)
├── yolov8n_kaggle_2500images_trained_20251203_130255/ # ✅ Kaggle dataset model (Active)
├── yolov8s_enhanced/                          # ✅ Enhanced YOLOv8s model (Active)
├── yolov8s_improved_detection_v2_20251202_153433/ # ✅ Improved detection model (Active)
├── archive/                                   # 📦 Archived experiments (moved from active)
│   ├── kaggle_strawberry_yolov8s_20251204_210526/ # Empty (template)
│   ├── yolov8n_kaggle_dataset_2500images_20251203_125358/ # Empty (template)
│   ├── yolov8s_experiment_with_augmentation_20251202_153614/ # Empty (template)
│   ├── yolov8s_experiment_with_augmentation_20251202_153622/ # Empty (template)
│   └── yolov8s_quick_training_test_20251202_154202/ # Empty (template)
└── validation_comparison_report.csv           # Model performance comparison
```

### 📊 Model Status Legend:
- ✅ **Active**: Contains trained model weights (ready for use)
- ⭐ **Recommended**: Best performing model for production
- 📦 **Archive**: Moved to archive folder (empty templates for reference)

## 🏆 Recommended Model

### **kaggle_strawberry_yolov8n_20251204_115538** ⭐

**Performance Metrics:**
- **mAP@50: 0.989** (98.9% detection accuracy)
- **mAP@50-95: 0.682** 
- **Inference Speed: 44.7 FPS** (real-time capable)
- **Average Detections: 1.74 per image**

**Model Details:**
- Architecture: YOLOv8n (nano version for edge deployment)
- Training Dataset: Kaggle Strawberry Dataset (2500+ images)
- Training Epochs: 50
- Batch Size: 16
- Input Size: 640x640
- Optimized for: Raspberry Pi deployment

**Files:**
- `weights/best.pt` - Best model weights (recommended for inference)
- `weights/last.pt` - Last checkpoint
- `results.csv` - Complete training metrics
- `config/training_config.json` - Training configuration
- `args.yaml` - Ultralytics training arguments

## 🚀 Training New Models

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Dataset structure
datasets/
└── train/
    ├── RipeStrawberry/
    ├── UnripeStrawberry/
    └── RottenStrawberry/
```

### Training Steps

1. **Prepare Dataset:**
   ```bash
   # Organize images in YOLO format
   # Each image should have corresponding .txt label file
   # Format: class x_center y_center width height (normalized 0-1)
   ```

2. **Train Model:**
   ```bash
   # Train YOLOv8n on Kaggle dataset
   python train_kaggle_strawberry.py \
     --model yolov8n \
     --epochs 50 \
     --batch 16 \
     --data model/dataset_strawberry_kaggle/data.yaml
   
   # Train optimized version
   python train_optimized_yolov8n.py \
     --epochs 100 \
     --batch 32 \
     --data model/dataset_strawberry_kaggle/data.yaml
   ```

3. **Training Configuration:**
   ```python
   # Key parameters from training_config.json
   {
     "model": "yolov8n.pt",
     "epochs": 50,
     "batch": 16,
     "imgsz": 640,
     "optimizer": "SGD",
     "lr0": 0.01,
     "lrf": 0.01,
     "momentum": 0.937,
     "weight_decay": 0.0005,
     "warmup_epochs": 3.0,
     "warmup_momentum": 0.8,
     "warmup_bias_lr": 0.1,
     "box": 7.5,
     "cls": 0.5,
     "dfl": 1.5,
     "hsv_h": 0.015,
     "hsv_s": 0.7,
     "hsv_v": 0.4,
     "degrees": 0.0,
     "translate": 0.1,
     "scale": 0.5,
     "shear": 0.0,
     "perspective": 0.0,
     "flipud": 0.0,
     "fliplr": 0.5,
     "mosaic": 1.0,
     "mixup": 0.0,
     "copy_paste": 0.0
   }
   ```

## 📊 Model Comparison

| Model | mAP@50 | mAP@50-95 | FPS | Size | Use Case |
|-------|--------|-----------|-----|------|----------|
| **kaggle_strawberry_yolov8n_20251204_115538** | **0.989** | **0.682** | **44.7** | 5.7MB | **Production** ⭐ |
| optimized_yolov8n_20251204_154529 | 0.945 | 0.650 | 42.1 | 5.7MB | Enhanced training |
| yolov8s_improved_detection_v2 | 0.937 | 0.640 | 35.2 | 21MB | Higher accuracy needs |
| yolov8n_kaggle_2500images | 0.923 | 0.620 | 45.8 | 5.7MB | Balanced performance |
| baseline | 0.850 | 0.550 | 48.3 | 5.7MB | Legacy reference |

## 🔍 Inference Usage

### Python Inference
```python
from ultralytics import YOLO

# Load model
model = YOLO('model/detection/kaggle_strawberry_yolov8n_20251204_115538/weights/best.pt')

# Run inference
results = model('path/to/image.jpg', conf=0.25, iou=0.45)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        cls = int(box.cls)  # Class ID
        conf = float(box.conf)  # Confidence
        xyxy = box.xyxy  # Coordinates
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: {xyxy}")
```

### Command Line Inference
```bash
# Single image
yolo predict model=model/detection/kaggle_strawberry_yolov8n_20251204_115538/weights/best.pt source='image.jpg'

# Directory of images
yolo predict model=model/detection/kaggle_strawberry_yolov8n_20251204_115538/weights/best.pt source='data/images/'

# Video
yolo predict model=model/detection/kaggle_strawberry_yolov8n_20251204_115538/weights/best.pt source='video.mp4'
```

## 📈 Training Results Analysis

### Key Metrics from results.csv:
- **Training Losses:** Continuously decreasing (good convergence)
- **Validation mAP:** Reached 0.989 by epoch 50
- **Learning Rate:** Proper decay schedule applied
- **Overfitting:** Minimal (train/val gap is small)

### Performance Characteristics:
- **Precision:** 95.5% (low false positives)
- **Recall:** 96.4% (low false negatives)
- **Speed:** Real-time capable on Raspberry Pi
- **Model Size:** 5.7MB (suitable for edge deployment)

## 🎯 Model Selection Guide

### For Production Deployment:
Use **`kaggle_strawberry_yolov8n_20251204_115538`** - Best balance of accuracy and speed

### For Research/Experimentation:
Use **`optimized_yolov8n_20251204_154529`** - Enhanced augmentations, longer training

### For High-Accuracy Requirements:
Use **`yolov8s_improved_detection_v2`** - Larger model, higher accuracy, slower inference

### For Resource-Constrained Devices:
Use **`yolov8n`** - Smallest model, fastest inference, slightly lower accuracy

## 🔧 Validation

Run validation on all models:
```bash
python scripts/validation/validate_all_models.py
```

This generates `validation_comparison_report.csv` with detailed metrics for all models.

## 📦 Export for Deployment

Export to different formats:
```bash
# Export to ONNX
yolo export model=weights/best.pt format=onnx opset=12

# Export to TensorRT (for NVIDIA Jetson)
yolo export model=weights/best.pt format=engine device=0

# Export to CoreML (for Apple devices)
yolo export model=weights/best.pt format=coreml
```

## 🐛 Troubleshooting

**Low Accuracy:**
- Check dataset quality and annotations
- Increase training epochs
- Adjust augmentation parameters

**Slow Inference:**
- Use smaller model (yolov8n instead of yolov8s)
- Reduce input image size
- Use TensorRT/ONNX export

**Overfitting:**
- Increase augmentation strength
- Add more training data
- Use early stopping

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Training Configuration](config/training_config.json)
- [Validation Results](../validation_results/)
- [Inference Scripts](../../scripts/inference/)

---

**Last Updated:** December 4, 2025  
**Model Version:** 1.0.0  
**Recommended Model:** `kaggle_strawberry_yolov8n_20251204_115538` ⭐