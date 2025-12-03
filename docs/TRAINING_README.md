# Strawberry Detection Model Training Guide

This guide covers training a YOLOv8 model for strawberry detection using multiple environments.

## Quick Start

### Option 1: Local/WSL Training (Recommended for initial setup)

```bash
# 1. Setup environment
python setup_training.py

# 2. Train model
python train_yolov8.py --epochs 100 --batch-size 16

# 3. Validate dataset only (without training)
python train_yolov8.py --validate-only
```

### Option 2: Google Colab Training (Recommended for faster training)

1. Open `train_yolov8_colab.ipynb` in Google Colab
2. Connect to GPU runtime: Runtime → Change runtime type → GPU
3. Run cells sequentially
4. Download trained model when complete

### Option 3: VS Code with Colab Extension

1. Install VS Code Google Colab extension
2. Open `train_yolov8_colab.ipynb`
3. Connect to Colab kernel
4. Run cells

## Environment Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Git (for version control)

### Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Validate setup:**
   ```bash
   python setup_training.py --validate-only
   ```

3. **Full setup (install + validate):**
   ```bash
   python setup_training.py
   ```

## Dataset Structure

Your dataset should be organized as follows:

```
model/dataset/straw-detect.v1-straw-detect.yolov8/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### data.yaml Format

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['strawberry']
```

## Training Parameters

### Basic Training

```bash
python train_yolov8.py \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --export-onnx
```

### Advanced Options

- `--dataset PATH`: Custom dataset path
- `--epochs N`: Number of training epochs (default: 100)
- `--batch-size N`: Batch size (default: 16)
- `--img-size N`: Image size (default: 640)
- `--export-onnx`: Export to ONNX format after training
- `--validate-only`: Only validate dataset without training

### Model Sizes

Choose different YOLOv8 models based on your needs:

| Model | Parameters | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------------|-------------|-------------|----------|
| yolov8n | 3.2M | Fastest | Fastest | Good |
| yolov8s | 11.2M | Fast | Fast | Better |
| yolov8m | 25.9M | Medium | Medium | Best |

To use a different model, edit the `MODEL_NAME` variable in the training script.

## Training on Different Environments

### Google Colab Advantages
- **Free GPU**: Tesla T4 with 16GB VRAM
- **Faster training**: 5-10x faster than CPU
- **No local setup**: Everything runs in the cloud

### WSL (Windows Subsystem for Linux)
- **Native GPU support**: If you have NVIDIA GPU
- **Persistent**: Files saved locally
- **Better for large datasets**: No upload needed

### Local Python
- **Simplest setup**: No additional configuration
- **CPU only**: Slower training
- **Good for testing**: Quick validation

## Monitoring Training

### TensorBoard (Local Training)
```bash
tensorboard --logdir model/results
```

### Colab Training
- Use the built-in progress bars
- View loss curves in real-time
- Download metrics after training

## Expected Training Time

| Environment | Epochs | Estimated Time |
|-------------|--------|----------------|
| Colab GPU | 100 | 30-60 minutes |
| WSL GPU | 100 | 30-60 minutes |
| CPU only | 100 | 5-8 hours |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch-size 8` or `--batch-size 4`
   - Use smaller model: `yolov8n` instead of `yolov8s`

2. **Dataset not found**
   - Check data.yaml paths are correct
   - Verify images are in the right directories
   - Run with `--validate-only` to debug

3. **Import errors**
   - Run `setup_training.py` again
   - Check Python version: `python --version`
   - Reinstall ultralytics: `pip install --force-reinstall ultralytics`

4. **Slow training**
   - Verify GPU is being used: Check setup output
   - Reduce image size: `--img-size 416`
   - Use smaller model: `yolov8n`

### Getting Help

1. Run validation: `python train_yolov8.py --validate-only`
2. Check setup: `python setup_training.py --validate-only`
3. Review error messages carefully
4. Check dataset structure matches expected format

## Next Steps After Training

1. **Export models**: ONNX, TensorFlow Lite formats
2. **Test inference**: Run on sample images
3. **Deploy to Raspberry Pi**: Optimize for edge deployment
4. **Integrate with robot**: Connect to robotic arm control

## Performance Optimization Tips

### For Faster Training
- Use GPU environment (Colab or WSL with GPU)
- Reduce image size (`--img-size 416`)
- Use smaller batch size if GPU memory limited
- Enable image caching (already enabled by default)

### For Better Accuracy
- Increase epochs to 150-200
- Use larger model (`yolov8s` or `yolov8m`)
- Collect more diverse training images
- Use data augmentation
- Fine-tune learning rate

### For Raspberry Pi Deployment
- Use `yolov8n` model (smallest)
- Export to TensorFlow Lite
- Apply INT8 quantization
- Reduce input resolution to 320x320 or 416x416

## Version Control

Track your training experiments:

```bash
git add model/results/
git commit -m "Training: yolov8n, 100 epochs, 640px"
git tag v1.0-yolov8n-baseline
```

## License

This training pipeline is part of the Strawberry Picker project. See main README for license information.