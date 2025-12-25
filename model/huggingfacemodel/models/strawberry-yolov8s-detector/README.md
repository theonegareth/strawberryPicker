# Strawberry YOLOv8s Detector

A fine-tuned YOLOv8s model for accurate strawberry detection in agricultural robotics applications.

## Model Details

- **Model Architecture**: YOLOv8s (Small)
- **Task**: Object Detection
- **Dataset**: Custom strawberry detection dataset
- **Training Epochs**: 150
- **Image Size**: 640x640
- **Model Size**: 22MB
- **Performance**: Optimized for real-time detection

## Intended Uses & Limitations

### Intended Uses
- Real-time strawberry detection in agricultural environments
- Integration with robotic harvesting systems
- Quality assessment and ripeness detection workflows
- Agricultural monitoring and analytics

### Limitations
- Trained specifically for strawberry detection
- Performance may vary in different lighting conditions
- Requires proper camera calibration for accurate positioning
- Optimized for outdoor agricultural settings

## Performance Metrics

Based on validation results:
- **Precision**: High accuracy in strawberry detection
- **Speed**: Optimized for real-time inference
- **Robustness**: Handles various strawberry sizes and orientations

## Usage

### Installation

```bash
pip install ultralytics
```

### Basic Inference

```python
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Perform detection
results = model('path/to/image.jpg')

# Display results
for r in results:
    print(r.boxes.xyxy)  # bounding box coordinates
    print(r.boxes.conf)  # confidence scores
    print(r.boxes.cls)   # class labels
```

### Real-time Detection

```python
import cv2
from ultralytics import YOLO

# Initialize camera
cap = cv2.VideoCapture(0)
model = YOLO('best.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    cv2.imshow('Strawberry Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Batch Processing

```python
from ultralytics import YOLO
import os

model = YOLO('best.pt')

# Process multiple images
image_dir = 'path/to/images'
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        results = model(os.path.join(image_dir, image_file))
        # Save annotated images or extract detection data
```

## Model Configuration

The model was trained with the following configuration:
- **Batch Size**: 4
- **Learning Rate**: Optimized for agricultural imagery
- **Data Augmentation**: Applied for robustness
- **Validation Split**: 20% of training data

## Integration with Robotic Systems

This model is designed for integration with robotic strawberry picking systems:

```python
# Example integration with coordinate transformation
from src.coordinate_transformer import PixelToWorldTransformer
from src.arduino_bridge import ArduinoBridge

# Initialize components
transformer = PixelToWorldTransformer(camera_matrix, distortion_coeffs)
arduino = ArduinoBridge()

# Detect and convert coordinates
results = model(image)
for detection in results[0].boxes.xyxy:
    pixel_coords = detection[:2]  # x, y
    world_coords = transformer.pixel_to_world(pixel_coords)
    arduino.move_to_position(world_coords)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{strawberry_yolov8s_detector,
  title={Strawberry YOLOv8s Detector},
  author={Gareth},
  year={2024},
  url={https://huggingface.co/theonegareth/strawberry-yolov8s-detector}
}
```

## License

This model is released under the same license as the original YOLOv8 implementation.

## Support

For issues, questions, or contributions, please visit the main project repository:
https://github.com/theonegareth/strawberryPicker