# 🍓 Strawberry Picker AI

A comprehensive computer vision system for real-time strawberry detection and ripeness classification using YOLOv8 and deep learning.

## 📁 Repository Structure

```
strawberryPicker/
├── scripts/                    # Main scripts organized by function
│   ├── inference/             # Real-time inference scripts
│   │   ├── image_inference.py         # Single image detection
│   │   └── webcam_inference_WSL.py    # Real-time webcam inference
│   ├── training/              # Model training scripts
│   │   ├── train_and_organize.py      # Training organization
│   │   ├── train_enhanced.py          # Enhanced training
│   │   ├── train_yolov8.py            # YOLOv8 training
│   │   └── ...                        # Other training scripts
│   └── validation/            # Model validation scripts
│       ├── training_registry.py       # Training history tracking
│       ├── validate_models.py         # Model validation
│       └── view_registry.py           # View training history
├── model/                    # Trained models and configurations
│   ├── detection/            # Detection models (YOLOv8)
│   ├── classification/       # Classification models (ripeness)
│   ├── training_registry.json # Training history database
│   └── data.yaml             # Dataset configuration
├── docs/                     # Documentation and guides
│   ├── TRAINING_WORKFLOW.md  # Complete training guide
│   ├── TRAINING_README.md    # Training documentation
│   ├── IMPROVEMENT_PLAN.md   # Future improvements
│   └── *.ipynb               # Jupyter notebooks
├── legacy/                   # Archived code and datasets
│   ├── archive/              # Old scripts
│   └── datasets/             # Backup datasets
├── assets/                   # Images, STL files, SolidWorks models
├── ArduinoCode/              # Robotics integration code
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Real-time Detection
```bash
# Webcam inference (if webcam available)
python3 scripts/inference/webcam_inference_WSL.py

# Single image inference
python3 scripts/inference/image_inference.py --image path/to/image.jpg
```

### 3. Train New Models
```bash
# Follow the training workflow
python3 scripts/training/train_yolov8.py --epochs 100 --batch-size 16
```

## 🎯 Features

- **Real-time Detection**: YOLOv8-based strawberry detection with high accuracy
- **Ripeness Classification**: 4-class ripeness assessment (unripe/partially-ripe/ripe/overripe)
- **Multi-Input Support**: Webcam, IP camera, video files, and single images
- **WSL Optimized**: Special optimizations for Windows Subsystem for Linux
- **Training Registry**: Complete tracking of all training runs and metrics
- **Model Validation**: Comprehensive validation and performance monitoring
- **Robotics Ready**: Arduino integration for automated picking

## 📊 Model Performance

- **Detection mAP@50**: 0.937 (YOLOv8s enhanced)
- **Classification Accuracy**: 89.2% (4-class ripeness)
- **Inference Speed**: ~13ms per frame on GPU
- **Training Registry**: 15+ tracked training runs

## 🛠️ Development Workflow

1. **Training**: Use `scripts/training/` for model development
2. **Validation**: Use `scripts/validation/` for performance testing
3. **Inference**: Use `scripts/inference/` for deployment
4. **Documentation**: See `docs/` for detailed guides

## 📈 Training History

View all training runs and their performance metrics:
```bash
python3 scripts/validation/view_registry.py
```

## 🤖 Robotics Integration

Arduino code for automated strawberry picking is available in the `ArduinoCode/` directory.

## 📚 Documentation

- [Training Workflow](docs/TRAINING_WORKFLOW.md) - Complete training guide
- [Training README](docs/TRAINING_README.md) - Training documentation
- [Improvement Plan](docs/IMPROVEMENT_PLAN.md) - Future enhancements

## 🔗 Related Repositories

- [HuggingFace Models](https://huggingface.co/theonegareth/strawberry-models) - Pre-trained models
- [Dataset](https://universe.roboflow.com/theonegareth/strawberry-detect) - Training dataset

## 📄 License

This project is open source. See individual files for license information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ for automated agriculture and computer vision research**
