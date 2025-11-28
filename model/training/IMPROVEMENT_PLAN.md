# Strawberry Model Improvement Plan

## Current Model Performance
- **Model**: YOLOv8n (nano)
- **mAP@0.5**: 0.581
- **Epochs**: 50
- **Image Size**: 416
- **Batch Size**: 8

## Target Performance
- **Model**: YOLOv8s (small)
- **Target mAP@0.5**: 0.70+ (+20% improvement)
- **Epochs**: 150
- **Image Size**: 640 (with progressive resizing)
- **Batch Size**: 8 (optimized for 4GB GPU)

## Enhancement Strategy

### 1. **Model Architecture Upgrade** (Biggest Impact)
- **From**: YOLOv8n (3.2M parameters)
- **To**: YOLOv8s (11.2M parameters)
- **Expected Gain**: +8-12% mAP
- **Trade-off**: 2x slower inference, 3x larger model

### 2. **Enhanced Data Augmentation** (+3-5% mAP)
- **Mosaic**: Combine 4 images into one
- **MixUp**: Blend 2 images together
- **Copy-Paste**: Copy strawberries onto new backgrounds
- **Color Augmentation**: Hue, saturation, brightness variations
- **Geometric Augmentation**: Rotation, scaling, translation

### 3. **Advanced Training Techniques** (+2-4% mAP)
- **Progressive Resizing**: Start at 320px → 416px → 640px
- **AdamW Optimizer**: Better than SGD for convergence
- **Cosine LR Scheduler**: Smooth learning rate decay
- **Longer Training**: 150 epochs vs 50 epochs
- **Early Stopping**: Patience of 30 epochs

### 4. **Hyperparameter Tuning** (+1-2% mAP)
- **Learning Rate**: 0.01 (AdamW)
- **Weight Decay**: 0.0005
- **Warmup**: 5 epochs
- **Dropout**: 0.1 (regularization)

## Training Configuration

### **Enhanced Settings**:
```python
train_args = {
    # Architecture
    'model': 'yolov8s.pt',
    'epochs': 150,
    'imgsz': 640,
    'batch': 8,
    
    # Augmentation
    'mosaic': 1.0,       # Enable mosaic
    'mixup': 0.1,        # Enable mixup
    'copy_paste': 0.1,   # Enable copy-paste
    'hsv_h': 0.015,      # Hue
    'hsv_s': 0.7,        # Saturation
    'hsv_v': 0.4,        # Value
    'degrees': 15.0,     # Rotation
    'scale': 0.5,        # Scaling
    
    # Optimization
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.01,
    'cos_lr': True,
    'warmup_epochs': 5,
    'patience': 30,
    
    # Regularization
    'dropout': 0.1,
    'weight_decay': 0.0005,
}
```

## Expected Timeline

### **Day 1-2**: Training
- **Duration**: ~4-6 hours on RTX 3050 Ti
- **Monitoring**: Track mAP, loss, learning rate
- **Checkpoints**: Save every 10 epochs

### **Day 3**: Validation
- **Test Set**: 90 validation images
- **Metrics**: mAP@0.5, precision, recall, F1
- **Comparison**: vs current model (0.581 mAP)

### **Day 4**: Optimization
- **Export**: TFLite with INT8 quantization
- **Benchmark**: Inference speed on GPU and Raspberry Pi
- **Optimization**: If needed, reduce size for deployment

## Success Criteria

✅ **Primary**: mAP@0.5 ≥ 0.70 (+20% improvement)  
✅ **Secondary**: Inference speed <100ms on RTX 3050 Ti  
✅ **Tertiary**: Model size <25MB (manageable for deployment)

## Next Steps

1. **Run Enhanced Training**:
   ```bash
   cd /home/user/machine-learning/GitHubRepos/strawberryPicker
   python3 models/model-training/train_enhanced.py --epochs 150 --batch-size 8 --model yolov8s.pt
   ```

2. **Monitor Training**:
   - Check TensorBoard: `tensorboard --logdir model/results`
   - Watch for overfitting (train/val gap)
   - Save best checkpoint

3. **Validate**:
   ```bash
   python3 models/validation/validate_model.py
   ```

4. **Compare**:
   - Current: 0.581 mAP
   - Target: 0.70+ mAP
   - Expected improvement: +20%