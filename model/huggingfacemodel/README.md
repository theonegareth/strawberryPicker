# Hugging Face Model Repository

Consolidated model repository for the Strawberry Picker project.

## Structure

```
huggingfacemodel/
├── classification/           # Ripeness classification models
│   ├── best_ripeness_classifier.pth
│   ├── final_ripeness_classifier.pth
│   ├── README.md
│   └── training_summary.md
├── detection/                # Strawberry detection models
│   ├── best.pt
│   └── README.md
├── models/                   # Hugging Face model repository
│   ├── publish_to_huggingface.py
│   └── strawberry-yolov8s-detector/
│       ├── args.yaml
│       ├── best.onnx
│       ├── best.pt
│       ├── config.yaml
│       ├── inference_example.py
│       ├── README.md
│       └── requirements.txt
├── datasets/                 # Dataset configurations (empty)
├── scripts/                  # Utility scripts (empty)
└── README.md                 # This file
```

## Model Details

### Detection Model (`detection/best.pt`)
- **Type**: YOLOv8s fine-tuned for strawberry detection
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes with confidence scores
- **Use Case**: Primary strawberry detection for robotic picking

### Classification Model (`classification/best_ripeness_classifier.pth`)
- **Type**: EfficientNet-B0 fine-tuned for ripeness classification
- **Input**: 128x128 RGB cropped strawberry images
- **Output**: 4-class probabilities (unripe, partially-ripe, ripe, overripe)
- **Accuracy**: 91.71% validation accuracy

### Hugging Face Repository (`models/strawberry-yolov8s-detector/`)
- Ready-to-publish model repository for Hugging Face
- Includes ONNX export for optimized inference
- Complete documentation and example usage

## Usage

### Detection
```python
from ultralytics import YOLO
model = YOLO('huggingfacemodel/detection/best.pt')
results = model('path/to/image.jpg')
```

### Classification
```python
import torch
from torchvision import transforms
from PIL import Image

model = torch.load('huggingfacemodel/classification/best_ripeness_classifier.pth', map_location='cpu')
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('strawberry_crop.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

class_names = ['unripe', 'partially-ripe', 'ripe', 'overripe']
print(f"Ripeness: {class_names[predicted_class]}")
```

## Integration with Existing Code

To maintain backward compatibility, consider updating import paths in your scripts:

**Old paths:**
- `detection_model/best.pt` → `huggingfacemodel/detection/best.pt`
- `classification_model/best_enhanced_classifier.pth` → `huggingfacemodel/classification/best_ripeness_classifier.pth`
- `huggingface_models/` → `huggingfacemodel/models/`

## Publishing to Hugging Face

```bash
cd huggingfacemodel/models
python publish_to_huggingface.py
```

## Notes

- All model files have been consolidated from scattered locations
- Original folders (`classification`, `classification_model`, `detection`, `detection_model`, `huggingface_models`) still exist but can be removed after verification
- Update your scripts to use the new paths for consistency