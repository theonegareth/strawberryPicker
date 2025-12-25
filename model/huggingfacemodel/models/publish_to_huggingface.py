#!/usr/bin/env python3
"""
Hugging Face Model Publication Script for Strawberry YOLOv8s Detector

This script helps publish the strawberry detection model to Hugging Face Model Hub.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed")
        return True
    except ImportError:
        print("❌ huggingface_hub is not installed")
        print("Install it with: pip install huggingface_hub")
        return False

def setup_huggingface_repo():
    """Setup and upload the model to Hugging Face."""
    
    # Import here to avoid issues if not installed
    try:
        from huggingface_hub import HfApi, HfFolder
        from huggingface_hub import ModelCard, ModelCardData
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Login check
    try:
        api.whoami()
        print("✅ Logged in to Hugging Face")
    except Exception:
        print("❌ Not logged in to Hugging Face")
        print("Please run: huggingface-cli login")
        print("Or visit: https://huggingface.co/settings/tokens")
        return False
    
    # Model repository details
    model_id = "theonegareth/strawberry-yolov8s-detector"
    local_dir = "strawberry-yolov8s-detector"
    
    # Check if local directory exists
    if not os.path.exists(local_dir):
        print(f"❌ Local directory {local_dir} not found")
        print("Make sure you're running this from the huggingface_models directory")
        return False
    
    print(f"📤 Uploading model to: {model_id}")
    
    try:
        # Upload the model
        api.upload_folder(
            folder_path=local_dir,
            repo_id=model_id,
            repo_type="model",
            commit_message="Initial upload: Strawberry YOLOv8s detector model",
            create_pr=False
        )
        
        print(f"✅ Successfully uploaded to: https://huggingface.co/{model_id}")
        print(f"🌟 Your model is now live on Hugging Face Model Hub!")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

def create_model_card():
    """Create a detailed model card for the repository."""
    
    model_card_content = """---
language: en
license: apache-2.0
datasets:
- custom-strawberry-dataset
tags:
- computer-vision
- object-detection
- agriculture
- robotics
- yolov8
- strawberry-detection
- farming
- agricultural-automation
---

# Strawberry YOLOv8s Detector

A fine-tuned YOLOv8s model for accurate strawberry detection in agricultural robotics applications.

## Model Description

This model has been specifically trained for strawberry detection in agricultural environments. It achieves high accuracy while maintaining real-time inference speed, making it ideal for robotic harvesting systems.

## Model Architecture

- **Base Model**: YOLOv8s (Small)
- **Task**: Object Detection
- **Input Size**: 640x640
- **Model Size**: 22MB (PyTorch), 43MB (ONNX)

## Performance

- Optimized for real-time detection
- High precision in strawberry identification
- Robust performance across various lighting conditions
- Handles different strawberry sizes and orientations

## Intended Uses

- Real-time strawberry detection in agricultural environments
- Integration with robotic harvesting systems
- Quality assessment and ripeness detection workflows
- Agricultural monitoring and analytics

## Limitations

- Trained specifically for strawberry detection
- Performance may vary in different lighting conditions
- Requires proper camera calibration for accurate positioning
- Optimized for outdoor agricultural settings

## Usage

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

This model is released under the Apache 2.0 license.

## Support

For issues, questions, or contributions, please visit the main project repository:
https://github.com/theonegareth/strawberryPicker
"""
    
    # Write the model card
    with open("strawberry-yolov8s-detector/README.md", "w") as f:
        f.write(model_card_content)
    
    print("✅ Model card created/updated")

def main():
    """Main publication function."""
    print("🚀 Hugging Face Model Publication Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("strawberry-yolov8s-detector"):
        print("❌ Please run this script from the huggingface_models directory")
        return
    
    # Check requirements
    if not check_requirements():
        print("\n📦 Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
    
    # Create/update model card
    print("\n📝 Creating model card...")
    create_model_card()
    
    # Upload to Hugging Face
    print("\n📤 Uploading to Hugging Face...")
    success = setup_huggingface_repo()
    
    if success:
        print("\n🎉 Publication completed successfully!")
        print("Your model is now available on Hugging Face Model Hub")
        print("URL: https://huggingface.co/theonegareth/strawberry-yolov8s-detector")
    else:
        print("\n❌ Publication failed. Please check the error messages above.")

if __name__ == "__main__":
    main()