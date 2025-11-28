# Strawberry Picker Robot - ML Vision System

Machine learning model for detecting strawberries and determining ripeness for robotic picking automation.

## 🎯 Project Overview

This project provides a complete ML pipeline for strawberry detection optimized for Raspberry Pi 4B deployment on a robotic picker. The system uses YOLOv8 for real-time object detection with a target of 20-30 FPS.

### Features
- **Real-time detection**: YOLOv8 optimized for edge deployment
- **Multi-phase approach**: Detection → Ripeness classification → Robotic integration
- **Raspberry Pi optimized**: TensorFlow Lite with INT8 quantization
- **Multiple training environments**: Local, WSL, Google Colab

## 📁 Project Structure

```
strawberryPicker/
├── model/
│   ├── dataset/                 # YOLO format dataset
│   │   └── straw-detect.v1-straw-detect.yolov8/
│   │       ├── data.yaml
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   ├── weights/                 # Trained models (gitignored)
│   ├── results/                 # Training results (gitignored)
│   └── exports/                 # Exported models (gitignored)
├── ArduinoCode/                 # Robotic arm control code
├── assets/                      # Images and 3D models
├── requirements.txt             # Python dependencies
├── setup_training.py           # Environment setup script
├── train_yolov8.py             # Command-line training script
├── train_yolov8_colab.ipynb    # Google Colab notebook
├── TRAINING_README.md          # Detailed training guide
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies and validate setup
python setup_training.py
```

### 2. Train Model

Choose your training environment:

**Option A: Local/WSL (CPU or GPU)**
```bash
python train_yolov8.py --epochs 100 --export-onnx
```

**Option B: Google Colab (Recommended for speed)**
1. Open `train_yolov8_colab.ipynb` in Google Colab
2. Connect to GPU runtime
3. Run all cells

**Option C: VS Code with Colab Extension**
1. Open `train_yolov8_colab.ipynb` in VS Code
2. Connect to Colab kernel
3. Run cells sequentially

### 3. Validate Dataset Only
```bash
python train_yolov8.py --validate-only
```

## 📊 Dataset

Current dataset contains strawberry detection data in YOLO format:
- **1 class**: strawberry
- **Training images**: 100+ images
- **Format**: YOLOv8 with bounding box annotations

### Dataset Structure
```
model/dataset/straw-detect.v1-straw-detect.yolov8/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🎓 Training Guide

See [TRAINING_README.md](TRAINING_README.md) for detailed training instructions including:
- Environment setup for different platforms
- Hyperparameter tuning
- Troubleshooting guide
- Performance optimization tips
- Expected training times

## 🔧 Model Optimization Pipeline

### Phase 1: Training ✅ Ready
- [x] YOLOv8n model training
- [x] Multi-environment support (Colab/WSL/Local)
- [x] Automated setup and validation

### Phase 2: Raspberry Pi Optimization (Next)
- [ ] ONNX export
- [ ] TensorFlow Lite conversion
- [ ] INT8 quantization
- [ ] Inference script for Pi Camera

### Phase 3: Ripeness Classification (Future)
- [ ] Dataset collection script
- [ ] 3-class annotation (unripe/partially ripe/fully ripe)
- [ ] Multi-class YOLO training

### Phase 4: Integration (Future)
- [ ] Real-time detection demo
- [ ] Robotic arm control integration
- [ ] Performance benchmarking

## 💻 Requirements

- Python 3.8+
- pip package manager
- Git
- (Optional) GPU with CUDA for faster training
- (Optional) Google account for Colab training

### Python Dependencies
```txt
torch>=1.8.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.3.0
onnx>=1.10.0
tensorflow>=2.8.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 📈 Performance Targets

- **Detection Speed**: 20-30 FPS on Raspberry Pi 4B
- **Model Size**: < 10MB (after quantization)
- **Input Resolution**: 640x640 (configurable)
- **Classes**: 1 (strawberry) → 3 (with ripeness)

## 🛠️ Development

### Training Scripts
- `train_yolov8.py` - Command-line training with argument parsing
- `train_yolov8_colab.ipynb` - Interactive notebook for Colab
- `setup_training.py` - Environment validation and setup

### Key Features
- Auto-detection of training environment (Colab/WSL/Local)
- GPU/CPU auto-configuration
- Dataset validation
- Model export (PyTorch, ONNX, TFLite)
- Training monitoring and checkpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Train and validate your model
4. Submit a pull request with results

## 📄 License

This project is part of the Kinematics and Dynamics coursework. See LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Roboflow for dataset management
- University guidance and support

## 📞 Support

For issues and questions:
1. Check [TRAINING_README.md](TRAINING_README.md) troubleshooting section
2. Run validation: `python train_yolov8.py --validate-only`
3. Check setup: `python setup_training.py --validate-only`

## 🔄 Changelog

### v1.0.0 - Initial Release
- YOLOv8 training pipeline
- Multi-environment support
- Dataset validation
- Model export functionality
