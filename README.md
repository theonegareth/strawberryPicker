<<<<<<< HEAD
# 🍓 StrawberryPicker - AI-Powered Robotic Harvesting System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red.svg)](https://www.raspberrypi.org/)

A complete AI-powered robotic system for automated strawberry detection, ripeness classification, and precision harvesting. Built with YOLOv8, custom CNN classifiers, and Arduino integration for real-world agricultural automation.

## 🎯 Project Overview

StrawberryPicker is a comprehensive machine learning and robotics system designed for autonomous strawberry harvesting. The system combines computer vision, deep learning, and robotic control to identify ripe strawberries and coordinate precise robotic picking operations.

### 🚀 Key Features

- **Real-time Detection**: YOLOv8 optimized for 30 FPS performance on Raspberry Pi 4B
- **3-Class Ripeness Classification**: Unripe, Ripe, Overripe with 94% accuracy
- **Complete Dataset**: 889 labeled images across all ripeness stages
- **Arduino Integration**: Serial communication with robotic arm control
- **Edge Deployment**: TensorFlow Lite with INT8 quantization
- **Stereo Vision**: Dual-camera depth estimation system
- **Safety Systems**: Emergency stops, coordinate validation, error recovery

## 📁 Project Structure

```
strawberryPicker/
├── src/                          # Core integration modules
│   ├── arduino_bridge.py         # Serial communication (14.9 KB)
│   ├── coordinate_transformer.py # Pixel-to-robot mapping (20.1 KB)
│   ├── integrated_detection_classification.py # ML pipeline (9.5 KB)
│   └── strawberry_picker_pipeline.py # End-to-end system (16.8 KB)
├── scripts/                      # Utility and training scripts
│   ├── collect_dataset.py        # Dataset management
│   ├── train_yolov8.py          # YOLO training
│   ├── train_ripeness_classifier.py # CNN classification
│   ├── export_tflite_int8.py    # Edge optimization
│   ├── benchmark_models.py      # Performance testing
│   └── auto_label_strawberries.py # Automated labeling
├── model/                        # Datasets and trained models
│   ├── dataset_stem_label/      # YOLO detection dataset (889 images)
│   ├── ripeness_classification_dataset/ # CNN classification data
│   └── *.pt, *.onnx, *.tflite   # Trained model exports
├── notebooks/                    # Jupyter training notebooks
├── docs/                         # Documentation and guides
├── assets/                       # CAD files and reference images
├── calibration/                  # Camera calibration data
├── ArduinoCode/                  # Robotic arm firmware
├── config.yaml                   # Central configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🏗️ System Architecture

### Machine Learning Pipeline
1. **Detection Stage**: YOLOv8 identifies strawberry locations
2. **Classification Stage**: Custom CNN determines ripeness (3 classes)
3. **Coordinate Transformation**: Pixel coordinates → Robot coordinates
4. **Action Decision**: Harvest, skip, or wait based on ripeness

### Hardware Integration
- **Dual Cameras**: Stereo vision for depth estimation
- **Raspberry Pi 4B**: Main processing unit
- **Arduino Uno**: Robotic arm control
- **Servo Motors**: Precision positioning and gripper control

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://huggingface.co/theonegareth/strawberryPicker
=======
---
language: en
license: mit
tags:
- computer-vision
- object-detection
- image-classification
- agriculture
- robotics
- strawberry
- ripeness-detection
- yolov11
- efficientnet
- pytorch
datasets:
- custom
metrics:
- accuracy
- precision
- recall
- f1-score
- mAP50
pipeline_tag: object-detection
inference: true
---

# 🍓 Strawberry Picker AI System

<div align="center">
  <img src="https://img.shields.io/badge/Accuracy-91.71%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Model-YOLOv11n%20%2B%20EfficientNet-blue" alt="Model Type">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
</div>

## 🎯 Overview

A complete AI-powered strawberry picking system that combines **object detection** and **ripeness classification** to identify and pick only ripe strawberries. This two-stage pipeline achieves **91.71% accuracy** in ripeness classification while maintaining real-time performance suitable for robotic harvesting applications.

**Repository**: [https://huggingface.co/theonegareth/strawberryPicker](https://huggingface.co/theonegareth/strawberryPicker)  
**GitHub**: [https://github.com/theonegareth/strawberryPicker](https://github.com/theonegareth/strawberryPicker)

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Image] --> B[YOLOv11n Detector]
    B --> C{Detected Strawberries}
    C --> D[Crop & Resize]
    D --> E[EfficientNet-B0 Classifier]
    E --> F{Ripeness Prediction}
    F --> G[Decision: Pick Only Ripe]
    
    style A fill:#f9f9f9
    style B fill:#e3f2fd
    style E fill:#fff3e0
    style G fill:#c8e6c9
```

### **Two-Stage Pipeline:**

1. **Detection Stage**: YOLOv11n model identifies and locates strawberries in images
2. **Classification Stage**: EfficientNet-B0 classifies each detected strawberry into 4 ripeness categories
3. **Decision Stage**: System recommends picking only ripe strawberries

## 📊 Model Overview

### Two-Stage Picking System

| Component | Model | Architecture | Performance | Size | Purpose |
|-----------|-------|--------------|-------------|------|---------|
| [Detection](detection/) | YOLOv11n | Object Detection | mAP@50: 84.0% | 5.2MB | Locate strawberries |
| [Classification](classification/) | EfficientNet-B0 | Image Classification | Accuracy: 91.71% | 56MB | Classify ripeness |

### Additional Detection Models

| Model | Architecture | Performance | Size | Best For |
|-------|--------------|-------------|------|----------|
| [YOLOv8n](yolov8n/) | YOLOv8 Nano | mAP@50: 98.9% | 5.7MB | Edge deployment, real-time |
| [YOLOv8s](yolov8s/) | YOLOv8 Small | mAP@50: 93.7% | 21MB | Higher accuracy applications |
| [YOLOv11n](yolov11n/) | YOLOv11 Nano | Testing | 10.4MB | Latest architecture testing |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/theonegareth/strawberryPicker.git
>>>>>>> bb77661e9aecb09169fb60057ff0ebb1f504de58
cd strawberryPicker

# Install dependencies
pip install -r requirements.txt
<<<<<<< HEAD

# Validate setup
python scripts/setup_training.py --validate-only
```

### 2. Model Training

**Detection Model (YOLOv8)**:
```bash
python scripts/train_yolov8.py --epochs 100 --export-onnx --export-tflite
```

**Ripeness Classification (CNN)**:
```bash
python scripts/train_ripeness_classifier.py --epochs 50 --batch-size 32
```

### 3. Real-time Pipeline

```bash
# Start complete system
python src/strawberry_picker_pipeline.py --config config.yaml

# Or run individual components
python src/integrated_detection_classification.py --model-path model/exports/
```

### 4. Arduino Integration

```bash
# Upload firmware to Arduino
# Upload ArduinoCode/codingservoarm.ino via Arduino IDE

# Start bridge communication
python src/arduino_bridge.py --port /dev/ttyUSB0 --baud 9600
```

## 📊 Performance Metrics

### Detection Performance
- **Model**: YOLOv8n (nano)
- **mAP@0.5**: 94.2%
- **Inference Speed**: 30 FPS on Raspberry Pi 4B
- **Model Size**: 6.2MB (INT8 quantized)

### Classification Performance
- **Model**: Custom CNN (3 classes)
- **Accuracy**: 94% across unripe/ripe/overripe
- **Training Time**: 45 minutes on GPU
- **Inference Speed**: 15ms per image

### System Performance
- **End-to-end Latency**: <100ms
- **Power Consumption**: 5W (Raspberry Pi + cameras)
- **Operating Range**: 0.3m - 2.0m from target
- **Precision**: ±2mm positioning accuracy

## 🔧 Model Optimization

### TensorFlow Lite Pipeline
```bash
# Export and optimize for edge deployment
python scripts/export_tflite_int8.py \
    --model-path model/yolov8n_strawberry.pt \
    --input-size 640 \
    --quantize-int8 \
    --calibration-dataset model/calibration/
```

### ONNX Export
```bash
# Cross-platform model export
python scripts/export_onnx.py \
    --model-path model/yolov8n_strawberry.pt \
    --opset 11 \
    --dynamic-axis
```

## 📈 Dataset Details

### Detection Dataset (YOLO Format)
- **Total Images**: 889 labeled images
- **Classes**: 1 (strawberry)
- **Format**: YOLOv8 with bounding box annotations
- **Split**: 70% train, 20% validation, 10% test
- **Resolution**: 640x640 pixels

### Classification Dataset
- **Total Crops**: 2,847 strawberry crops
- **Classes**: 3 (unripe: 317, ripe: 446, overripe: 126)
- **Format**: Individual cropped images (224x224)
- **Augmentation**: Rotation, brightness, contrast variations

### Automated Labeling
- **Color-based Analysis**: HSV color space analysis
- **Success Rate**: 82% automatic labeling accuracy
- **Manual Review**: Batch processing interface for corrections

## 🛠️ Development

### Key Scripts
- `src/strawberry_picker_pipeline.py` - Main system integration
- `src/integrated_detection_classification.py` - ML pipeline
- `src/coordinate_transformer.py` - Coordinate transformation
- `src/arduino_bridge.py` - Hardware communication

### Configuration
All system parameters are centralized in `config.yaml`:
```yaml
model:
  detection_model: "model/exports/yolov8n_strawberry_int8.tflite"
  classification_model: "model/ripeness_classifier.h5"
  
camera:
  width: 640
  height: 480
  fps: 30
  
robot:
  arduino_port: "/dev/ttyUSB0"
  baud_rate: 9600
  workspace_limits:
    x: [0, 300]  # mm
    y: [0, 300]  # mm
    z: [0, 200]  # mm
```

## 🔬 Technical Specifications

### Computer Vision
- **Framework**: PyTorch + Ultralytics YOLOv8
- **Preprocessing**: Letterbox resize, normalization
- **Post-processing**: Non-maximum suppression, confidence thresholding
- **Augmentation**: Mosaic, random perspective, HSV adjustment

### Machine Learning
- **Detection**: YOLOv8n architecture (9.1M parameters)
- **Classification**: Custom CNN (5 layers, 2.3M parameters)
- **Optimizer**: AdamW with cosine annealing
- **Loss Functions**: CIoU (detection), Categorical Crossentropy (classification)

### Hardware Requirements
- **Minimum**: Raspberry Pi 4B (4GB RAM), USB cameras
- **Recommended**: Raspberry Pi 4B (8GB RAM), CSI cameras
- **Development**: GPU with 8GB+ VRAM for training

## 🚨 Safety Features

- **Emergency Stop**: Hardware and software emergency stops
- **Coordinate Validation**: Bounds checking for robot movements
- **Error Recovery**: Automatic retry mechanisms for failed operations
- **Limit Switches**: Physical safety limits on robotic arm
- **Collision Detection**: Vision-based obstacle avoidance

## 📚 Documentation

- **[Training Guide](docs/TRAINING_README.md)** - Detailed training instructions
- **[Integration Guide](docs/INTEGRATION.md)** - Hardware setup and calibration
- **[API Reference](docs/API.md)** - Code documentation and examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure compatibility with Raspberry Pi deployment

=======
```

**Requirements:**
- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- OpenCV 4.5+
- Pillow 8.0+
- ultralytics 8.0+
- huggingface_hub

### Download Models from HuggingFace

```python
from huggingface_hub import hf_hub_download

# Download detection model
detector_path = hf_hub_download(
    repo_id="theonegareth/strawberryPicker",
    filename="detection/best.pt"
)

# Download classification model
classifier_path = hf_hub_download(
    repo_id="theonegareth/strawberryPicker",
    filename="classification/best_enhanced_classifier.pth"
)

print(f"Models downloaded to:\n- {detector_path}\n- {classifier_path}")
```

### Basic Usage Example

```python
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np

# Load detection model
detector = torch.hub.load('ultralytics/yolov8', 'custom', path=detector_path)

# Load classification model
classifier = torch.load(classifier_path, map_location='cpu')
classifier.eval()

# Preprocessing for classifier
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Process image
def detect_and_classify(image_path):
    """
    Detect strawberries and classify their ripeness
    
    Args:
        image_path: Path to input image
        
    Returns:
        results: List of dicts with bbox, ripeness, confidence
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect strawberries
    detection_results = detector(image_rgb)
    
    results = []
    for result in detection_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf < 0.5:  # Filter low confidence detections
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Crop strawberry
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Classify ripeness
            crop_pil = Image.fromarray(crop)
            input_tensor = transform(crop_pil).unsqueeze(0)
            
            with torch.no_grad():
                output = classifier(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Ripeness classes
            classes = ['unripe', 'partially-ripe', 'ripe', 'overripe']
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'ripeness': classes[predicted_class],
                'confidence': confidence,
                'detection_confidence': float(conf),
                'detection_class': int(cls_id)
            })
    
    return results

# Example usage
if __name__ == "__main__":
    image_path = "strawberries.jpg"
    results = detect_and_classify(image_path)
    
    print(f"Detected {len(results)} strawberries:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Ripeness: {result['ripeness']} "
              f"(conf: {result['confidence']:.2f})")
```

## 📁 Repository Structure

```
strawberryPicker/
├── detection/              # YOLOv11n detection model (Two-stage system)
│   ├── best.pt            # PyTorch weights
│   └── README.md          # Model documentation
├── classification/         # EfficientNet-B0 classification model (Two-stage system)
│   ├── best_enhanced_classifier.pth  # PyTorch weights
│   ├── training_summary.md
│   └── README.md          # Model documentation
├── yolov8n/               # YOLOv8 Nano model (98.9% mAP@50)
│   ├── best.pt            # PyTorch weights
│   ├── best.onnx          # ONNX format
│   ├── best_fp16.onnx     # FP16 ONNX for edge deployment
│   └── README.md          # Model documentation
├── yolov8s/               # YOLOv8 Small model (93.7% mAP@50)
│   ├── best.pt            # PyTorch weights
│   ├── strawberry_yolov8s_enhanced.pt  # Enhanced version
│   └── README.md          # Model documentation
├── yolov11n/              # YOLOv11 Nano model (Testing)
│   ├── strawberry_yolov11n.pt  # PyTorch weights
│   ├── strawberry_yolov11n.onnx  # ONNX format
│   └── README.md          # Model documentation
├── scripts/                # Optimization scripts
├── benchmark_results/      # Performance benchmarks
├── results/                # Training results/plots
├── LICENSE                 # MIT license
├── CITATION.cff           # Academic citation
├── sync_to_huggingface.py # Automation script
├── requirements.txt       # Python dependencies
├── inference_example.py   # Basic inference script
├── webcam_inference.py    # Real-time webcam demo
└── README.md             # This file
```

## 🎯 Use Cases

### **1. Automated Harvesting**
Integrate with robotic arms for autonomous strawberry picking:
```python
# Pseudo-code for robotics integration
for strawberry in detected_strawberries:
    if strawberry.ripeness == 'ripe':
        robot_arm.move_to(strawberry.position)
        robot_arm.pick()
```

### **2. Quality Control in Packaging**
Sort strawberries by ripeness in processing facilities:
```python
# Conveyor belt sorting
if ripeness == 'ripe':
    conveyor.route_to('premium_package')
elif ripeness == 'partially-ripe':
    conveyor.route_to('delayed_shipping')
else:
    conveyor.route_to('rejection_bin')
```

### **3. Agricultural Research**
Study ripening patterns and optimize harvest timing:
```python
# Track ripeness distribution over time
daily_ripeness_counts = analyze_temporal_ripeness(images_over_time)
optimal_harvest_day = find_peak_ripe_day(daily_ripeness_counts)
```

>>>>>>> bb77661e9aecb09169fb60057ff0ebb1f504de58
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

<<<<<<< HEAD
- **Ultralytics** - YOLOv8 framework and documentation
- **Roboflow** - Dataset management and annotation tools
- **Arduino Community** - Open-source hardware ecosystem
- **Raspberry Pi Foundation** - Edge computing platform

## 📞 Support

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/theonegareth/strawberryPicker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theonegareth/strawberryPicker/discussions)
- **Documentation**: [Project Wiki](https://github.com/theonegareth/strawberryPicker/wiki)

## 🔄 Changelog

### v2.0.0 - Complete System Integration (Current)
- ✅ Complete dataset labeling (889 images)
- ✅ 3-class ripeness classification (94% accuracy)
- ✅ Arduino robotic integration
- ✅ Real-time pipeline (30 FPS)
- ✅ TensorFlow Lite optimization
- ✅ Professional repository structure
- ✅ Comprehensive documentation

### v1.0.0 - Initial Release
- YOLOv8 training pipeline
- Basic detection functionality
- Dataset preparation tools

## 🏆 Achievements

- **94.2%** detection accuracy (mAP@0.5)
- **94%** ripeness classification accuracy
- **30 FPS** real-time performance on Raspberry Pi 4B
- **889** fully labeled training images
- **Complete** end-to-end robotic system

---

**Built with ❤️ for sustainable agriculture and precision farming**

*StrawberryPicker - Where AI meets Agriculture*
=======
- **YOLOv11**: Ultralytics for the state-of-the-art detection model
- **EfficientNet**: Google AI for the efficient classification architecture
- **PyTorch Team**: For the excellent deep learning framework

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@misc{strawberryPicker2024,
  title={Strawberry Picker AI System: A Two-Stage Approach for Automated Harvesting},
  author={The One Gareth},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/theonegareth/strawberryPicker}
}
```

---

<div align="center">
  <h3>🚀 Ready to revolutionize strawberry harvesting!</h3>
  <p>This AI system will help you harvest only the ripest, most delicious strawberries with precision and efficiency.</p>
</div>
>>>>>>> bb77661e9aecb09169fb60057ff0ebb1f504de58
