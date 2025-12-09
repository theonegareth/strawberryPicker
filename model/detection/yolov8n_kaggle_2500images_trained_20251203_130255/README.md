# yolov8n_kaggle_2500images_trained_20251203_130255

## Model Information
- **Architecture**: YOLOv8
- **Model Size**: n
- **Dataset**: strawberry_kaggle_2500
- **Training Date**: 2025-12-03 13:02:55

## Training Parameters
- **Epochs**: 73
- **Batch Size**: 16
- **Image Size**: 416
- **Learning Rate**: 0.002
- **Optimizer**: AdamW

## Performance Metrics
- **mAP@0.5**: 0.99
- **mAP@0.5:0.95**: 0.688
- **Precision**: 0.945
- **Recall**: 0.982
- **Training Time**: 3.3 minutes

## Files
- `weights/best.pt` - Best model weights
- `weights/model.pt` - Model weights (copy)
- `config/training_config.json` - Training configuration
- `validation/` - Validation results and metrics

## Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO('weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
```

## Deployment
For Raspberry Pi 4B deployment, export to TensorFlow Lite:
```bash
yolo export model=weights/best.pt format=tflite imgsz=416 int8=True
```
