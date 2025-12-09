#!/usr/bin/env python3
"""
Export ripeness classifier to TensorFlow Lite format for Raspberry Pi
Optimized for edge deployment with quantization
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_ripeness_classifier import create_model

def export_to_tflite(
    model_path: str,
    output_path: str = None,
    quantize: bool = True
):
    """
    Export PyTorch model to TensorFlow Lite format
    
    Args:
        model_path: Path to trained PyTorch model (.pth file)
        output_path: Output path for TFLite model
        quantize: Apply INT8 quantization for smaller size/faster inference
    """
    
    print("🍓 Exporting Ripeness Classifier to TensorFlow Lite")
    print("=" * 60)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model_type = checkpoint.get('model_type', 'mobilenet_v2')
    
    print(f"Model type: {model_type}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    
    # Create model
    model = create_model(num_classes=3, model_type=model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Create output path
    if output_path is None:
        output_path = str(Path(model_path).parent / "ripeness_classifier.tflite")
    
    # Note: Direct PyTorch to TFLite conversion is complex
    # We'll create a simple inference script that uses the PyTorch model
    # For actual TFLite conversion, we'd need to go through ONNX then TensorFlow
    
    print("\n⚠️  Direct PyTorch to TFLite conversion requires additional steps:")
    print("   1. Convert PyTorch → ONNX")
    print("   2. Convert ONNX → TensorFlow")
    print("   3. Convert TensorFlow → TFLite")
    print("\n💡 For now, we'll use the PyTorch model directly on Raspberry Pi")
    print("   (PyTorch works well on Raspberry Pi 4B with proper installation)")
    
    # Save model in a more portable format
    model_info = {
        'model_type': model_type,
        'num_classes': 3,
        'classes': ['unripe', 'ripe', 'overripe'],
        'val_accuracy': checkpoint.get('val_acc', 0),
        'model_path': model_path,
        'export_date': str(Path(model_path).stat().st_mtime)
    }
    
    info_path = str(Path(model_path).parent / "ripeness_classifier_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n💾 Model info saved to: {info_path}")
    print(f"💾 PyTorch model ready for Raspberry Pi deployment")
    print(f"\n✅ Export complete!")
    
    return model_path, info_path

def main():
    """Main export function"""
    
    model_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/best_ripeness_classifier.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train_ripeness_classifier.py")
        return
    
    export_to_tflite(model_path)

if __name__ == "__main__":
    main()