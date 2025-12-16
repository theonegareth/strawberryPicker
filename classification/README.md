---
tags:
- image-classification
- efficientnet
- strawberry
- agriculture
- robotics
- computer-vision
- pytorch
- ripeness-classification
license: mit
datasets:
- custom
language:
- python
pretty_name: EfficientNet-B0 Strawberry Ripeness Classification
description: EfficientNet-B0 model for detailed strawberry ripeness classification with 4-class output
pipeline_tag: image-classification
---

# EfficientNet-B0 Strawberry Ripeness Classification Model

This directory contains the EfficientNet-B0 model for detailed strawberry ripeness classification, the second stage of the Strawberry Picker AI system.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 91.71% |
| **Macro F1-Score** | 0.92 |
| **Weighted F1-Score** | 0.93 |
| **Model Size** | 56MB |
| **Input Size** | 128x128 |

### Class Performance (Validation Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| unripe | 0.92 | 0.89 | 0.91 | 163 |
| partially-ripe | 0.88 | 0.91 | 0.89 | 135 |
| ripe | 0.94 | 0.93 | 0.93 | 124 |
| overripe | 0.96 | 0.95 | 0.95 | 422 |

## 🎯 Ripeness Classes

| Class | Description | Pick? |
|-------|-------------|-------|
| **unripe** | Green, hard texture | ❌ |
| **partially-ripe** | Pink/red, firm | ❌ |
| **ripe** | Bright red, soft | ✅ |
| **overripe** | Dark red/brown, mushy | ❌ |

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision pillow
```

### Python Inference
```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('best_enhanced_classifier.pth', map_location='cpu')
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('strawberry_crop.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

class_names = ['unripe', 'partially-ripe', 'ripe', 'overripe']
print(f"Ripeness: {class_names[predicted_class]} ({confidence:.2f})")
```

## 📁 Files

- `best_enhanced_classifier.pth` - PyTorch model weights
- `training_summary.md` - Detailed training information

## 🎯 Use Cases

- **Automated Harvesting**: Second stage ripeness verification
- **Quality Control**: Precise ripeness assessment for sorting
- **Agricultural Research**: Ripeness pattern analysis

## 🔧 Technical Details

- **Architecture**: EfficientNet-B0
- **Input Size**: 128x128 RGB
- **Output**: 4-class probabilities
- **Training Dataset**: 844 cropped strawberry images
- **Training Epochs**: 50 (early stopping)
- **Batch Size**: 8
- **Optimizer**: AdamW
- **Learning Rate**: 0.002 (cosine annealing)

## 📈 Training Configuration

```python
# Model Architecture
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 4)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

## 🔗 Related Components

- [Detection Model](../detection/) - First stage for strawberry localization
- [Training Repository](https://github.com/theonegareth/strawberryPicker)

## 📚 Documentation

- [Full System Documentation](https://github.com/theonegareth/strawberryPicker)
- [Training Summary](training_summary.md)

## 📄 License

MIT License - See main repository for details.

---

**Model Version**: 1.0.0  
**Training Date**: November 2025  
**Part of**: Strawberry Picker AI System
