# Strawberry Ripeness Classification Model

## Model Description

This is a 4-class strawberry ripeness classification model trained on PyTorch with 91.71% validation accuracy. The model classifies strawberry crops into four ripeness categories:

- **unripe**: Green, hard strawberries not ready for picking
- **partially-ripe**: Pink/red, firm strawberries
- **ripe**: Bright red, soft strawberries ready for picking
- **overripe**: Dark red/brown, mushy strawberries past optimal ripeness

## Training Details

- **Architecture**: EfficientNet-B0 with custom classification head
- **Input Size**: 128x128 RGB images
- **Training Epochs**: 50 (early stopping at epoch 14)
- **Batch Size**: 8
- **Optimizer**: Adam with cosine annealing LR scheduler
- **Dataset**: 2,436 total images (889 strawberry crops + 800 Kaggle overripe images)
- **Validation Accuracy**: 91.71%
- **Training Time**: ~14 epochs with early stopping

## Usage

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("best_enhanced_classifier.pth")
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classify image
image = Image.open("strawberry_crop.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

classes = ["unripe", "partially-ripe", "ripe", "overripe"]
print(f"Predicted ripeness: {classes[predicted_class]}")
```

## Model Files

- `classification_model/best_enhanced_classifier.pth`: Trained PyTorch model (4.7MB)
- `classification_model/training_summary.md`: Detailed training metrics and results
- `classification_model/enhanced_training_curves.png`: Training/validation curves

## Integration

This model is designed to work with the strawberry detection model for a complete picking system:

1. **Detection**: YOLOv8 finds strawberries in images
2. **Classification**: This model determines ripeness of each detected strawberry
3. **Decision**: Only pick ripe strawberries (avoid unripe, partially-ripe, and overripe)

## Performance Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| unripe | 0.92 | 0.89 | 0.91 |
| partially-ripe | 0.88 | 0.91 | 0.89 |
| ripe | 0.94 | 0.93 | 0.93 |
| overripe | 0.96 | 0.95 | 0.95 |

**Overall Accuracy**: 91.71%

## Dataset

- **Source**: Mixed dataset with manual annotations + Kaggle fruit ripeness dataset
- **Classes**: 4 ripeness categories
- **Total Images**: 2,436 (train: 1,436, val: 422)
- **Preprocessing**: Cropped strawberry regions from detection model

## Requirements

- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- Pillow >= 8.0.0
- numpy >= 1.21.0

## License

MIT License - see main repository for details.

## Contact

For questions or improvements, please open an issue in the main repository.
