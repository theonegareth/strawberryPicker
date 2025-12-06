# 🚀 Multi-Model YOLOv8 Training Guide

## Overview
This guide explains how to train multiple YOLOv8 model variants (s, m, l) for different accuracy/speed trade-offs in strawberry detection.

## 🎯 Model Variants

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **YOLOv8n** | Nano | ⚡⚡⚡⚡⚡ | ⭐⭐ | Ultra-fast inference, limited hardware |
| **YOLOv8s** | Small | ⚡⚡⚡⚡ | ⭐⭐⭐ | **Recommended for most deployments** |
| **YOLOv8m** | Medium | ⚡⚡⚡ | ⭐⭐⭐⭐ | Higher accuracy for critical applications |
| **YOLOv8l** | Large | ⚡⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy for research |

## 📋 Quick Start

### Basic Usage
```bash
# Train all three variants (s, m, l)
python3 scripts/training/train_multiple_yolov8_variants.py \
  --dataset model/dataset_strawberry_kaggle \
  --output model/detection/multi_model_training

# Train only specific variants
python3 scripts/training/train_multiple_yolov8_variants.py \
  --dataset model/dataset_strawberry_kaggle \
  --output model/detection/multi_model_training \
  --models yolov8s yolov8m

# Skip existing models (resume training)
python3 scripts/training/train_multiple_yolov8_variants.py \
  --dataset model/dataset_strawberry_kaggle \
  --output model/detection/multi_model_training \
  --skip-existing
```

## ⚙️ Training Configuration

### YOLOv8s (Recommended)
- **Epochs**: 100
- **Batch Size**: 32
- **Image Size**: 640x640
- **Learning Rate**: 0.01
- **Weight Decay**: 0.0005
- **Training Time**: ~2-3 hours
- **Best For**: Production deployments

### YOLOv8m (High Accuracy)
- **Epochs**: 120
- **Batch Size**: 16
- **Image Size**: 640x640
- **Learning Rate**: 0.01
- **Weight Decay**: 0.0005
- **Training Time**: ~4-5 hours
- **Best For**: Critical applications requiring high precision

### YOLOv8l (Maximum Accuracy)
- **Epochs**: 150
- **Batch Size**: 8
- **Image Size**: 640x640
- **Learning Rate**: 0.005
- **Weight Decay**: 0.001
- **Training Time**: ~6-8 hours
- **Best For**: Research and validation

## 📊 Output Structure

```
model/detection/multi_model_training/
├── yolov8s_20251206_121500/
│   ├── runs/detect/train/
│   │   ├── weights/best.pt
│   │   ├── weights/last.pt
│   │   └── results.png
│   └── validation/
├── yolov8m_20251206_121500/
│   └── ...
├── yolov8l_20251206_121500/
│   └── ...
├── multi_model_training_report_20251206_121500.md
└── training_results_20251206_121500.json
```

## 📈 Training Report

The script generates a comprehensive report comparing all models:

### Performance Metrics
- **mAP@50**: Mean Average Precision at 0.5 IoU
- **mAP@50:95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Example Report
```markdown
# 🍓 Multi-Model YOLOv8 Training Report

## 📊 Model Performance Comparison

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | Training Time (h) |
|-------|--------|-----------|-----------|--------|----|-------------------|
| yolov8s | 0.989 | 0.756 | 0.945 | 0.923 | 0.934 | 2.3 |
| yolov8m | 0.991 | 0.782 | 0.956 | 0.934 | 0.945 | 4.1 |
| yolov8l | 0.993 | 0.801 | 0.967 | 0.941 | 0.954 | 6.7 |

## 🎯 Recommendations

- **Highest Accuracy**: yolov8l (mAP@50: 0.993)
- **Best Balance**: yolov8s (speed vs accuracy)
- **Production Choice**: yolov8s for most deployments
```

## 🔧 Advanced Usage

### Custom Training Parameters
You can modify the training configurations in the script:

```python
self.model_configs = {
    'yolov8s': {
        'model_size': 's',
        'epochs': 100,        # Adjust epochs
        'batch_size': 32,     # Adjust batch size
        'imgsz': 640,         # Adjust image size
        'lr0': 0.01,          # Learning rate
        'weight_decay': 0.0005,
        'description': 'Custom description'
    }
}
```

### Hardware Requirements
- **GPU**: Recommended for faster training (RTX 3060+ or equivalent)
- **CPU**: Minimum 8 cores, 16GB RAM
- **Storage**: 50GB+ free space for models and logs
- **Training Time**: 2-8 hours depending on model size and hardware

## 🚀 Next Steps After Training

### 1. Model Optimization
```bash
# Convert best models to ONNX
python3 scripts/optimization/convert_yolo_to_onnx.py \
  --model model/detection/multi_model_training/yolov8s_*/weights/best.pt \
  --output model/detection/multi_model_training/yolov8s_*/weights/best.onnx

# Quantize to FP16
python3 scripts/optimization/quantize_onnx_fp16.py \
  --model model/detection/multi_model_training/yolov8s_*/weights/best.onnx \
  --output model/detection/multi_model_training/yolov8s_*/weights/best_fp16.onnx
```

### 2. Performance Benchmarking
```bash
# Compare all model variants
python3 scripts/optimization/benchmark_inference_speed.py \
  --pytorch-model model/detection/multi_model_training/yolov8s_*/weights/best.pt \
  --onnx-model model/detection/multi_model_training/yolov8s_*/weights/best.onnx \
  --fp16-model model/detection/multi_model_training/yolov8s_*/weights/best_fp16.onnx
```

### 3. Raspberry Pi Testing
Transfer the optimized models to Raspberry Pi and test real-world performance.

## 🎯 Model Selection Guide

### For Raspberry Pi Deployment
- **Primary Choice**: YOLOv8s with FP16 quantization
- **Fallback**: YOLOv8n if memory is extremely limited
- **High Accuracy**: YOLOv8m if processing power allows

### For Desktop/Server Deployment
- **Real-time Applications**: YOLOv8s or YOLOv8n
- **Batch Processing**: YOLOv8m or YOLOv8l
- **Research/Validation**: YOLOv8l

## ⚠️ Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Ensure GPU is being used (`nvidia-smi`)
3. **Poor Results**: Check dataset quality and augmentation settings
4. **CUDA Errors**: Update PyTorch and CUDA drivers

### Performance Tips
- Use `--cache` flag for faster data loading
- Enable `--amp` for automatic mixed precision
- Use multiple GPUs with `device=0,1` (requires code modification)
- Monitor GPU usage with `nvidia-smi -l 1`

## 📞 Support

If you encounter issues:
1. Check the training logs in the output directory
2. Verify dataset format matches YOLO requirements
3. Ensure sufficient GPU memory and storage
4. Review the generated training report for insights

---
*Happy Training! 🍓*