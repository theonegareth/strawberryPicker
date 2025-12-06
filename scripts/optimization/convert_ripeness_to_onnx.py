#!/usr/bin/env python3
"""
Ripeness Classifier ONNX Conversion Script
Converts PyTorch ripeness classifier to ONNX format for faster inference on Raspberry Pi
"""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import argparse
import json
import os
from pathlib import Path

# Add the training module to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

def create_mobilenet_v2_classifier(num_classes=3):
    """Create MobileNet v2 classifier for ripeness classification"""
    
    # Import the model creation function
    from train_ripeness_classifier import create_model
    
    # Create the model using the existing function
    model = create_model(num_classes=num_classes)
    
    return model

def convert_ripeness_classifier_to_onnx(model_path, output_path, input_size=(224, 224)):
    """
    Convert ripeness classifier from PyTorch to ONNX format
    
    Args:
        model_path (str): Path to PyTorch model (.pth file)
        output_path (str): Path to save ONNX model
        input_size (tuple): Input image size (width, height)
    """
    print(f"Loading ripeness classifier from: {model_path}")
    
    # Load model info
    info_path = Path(model_path).parent / "ripeness_classifier_info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        print(f"✅ Model info loaded: {model_info['model_type']} with {model_info['num_classes']} classes")
        classes = model_info['classes']
        val_accuracy = model_info['val_accuracy']
    else:
        print("⚠️ Model info file not found, using defaults")
        classes = ['unripe', 'ripe', 'overripe']
        val_accuracy = 'N/A'
    
    # Load the PyTorch model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model architecture
    num_classes = len(classes)
    model = create_mobilenet_v2_classifier(num_classes)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ PyTorch model loaded successfully")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📊 Original model size: {model_size:.2f} MB")
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
    
    # Export to ONNX
    print(f"🔄 Converting to ONNX format with input size {input_size}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validation passed")
    
    # Get ONNX model size
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"📊 ONNX model size: {onnx_size:.2f} MB")
    
    # Size comparison
    if onnx_size < model_size:
        reduction = ((model_size - onnx_size) / model_size) * 100
        print(f"🚀 Size reduction: {reduction:.1f}%")
    else:
        increase = ((onnx_size - model_size) / model_size) * 100
        print(f"📈 Size increase: {increase:.1f}%")
    
    # Test ONNX inference
    print("\n🧪 Testing ONNX inference...")
    try:
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Test with dummy input
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Run inference
        outputs = ort_session.run([output_name], {input_name: dummy_input.numpy()})
        predictions = outputs[0]
        
        print(f"✅ ONNX inference successful")
        print(f"📊 Output shape: {predictions.shape}")
        print(f"🔧 Classes: {classes}")
        print(f"📈 Validation accuracy: {val_accuracy}%")
        
        # Show sample prediction
        probabilities = torch.softmax(torch.tensor(predictions[0]), dim=0)
        print(f"🎯 Sample prediction:")
        for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
            print(f"   {class_name}: {prob:.3f}")
            
    except Exception as e:
        print(f"❌ Error testing ONNX inference: {e}")
        return
    
    return {
        'original_size': model_size,
        'onnx_size': onnx_size,
        'classes': classes,
        'val_accuracy': val_accuracy,
        'input_size': input_size
    }

def main():
    parser = argparse.ArgumentParser(description='Convert ripeness classifier to ONNX format')
    parser.add_argument('--model', required=True, help='Path to PyTorch ripeness classifier')
    parser.add_argument('--output', required=True, help='Path to save ONNX model')
    parser.add_argument('--input-size', nargs=2, type=int, default=[224, 224],
                       help='Input image size (width height)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert the model
    results = convert_ripeness_classifier_to_onnx(args.model, args.output, tuple(args.input_size))
    
    print(f"\n🎯 Ripeness Classifier ONNX Conversion Complete!")
    print(f"📁 ONNX model saved to: {args.output}")
    print(f"💾 Size: {results['original_size']:.2f} MB → {results['onnx_size']:.2f} MB")
    print(f"🔍 Classes: {', '.join(results['classes'])}")
    print(f"📈 Accuracy: {results['val_accuracy']}%")
    
    print("\n📋 Next steps:")
    print("1. Quantize ONNX model for even smaller size")
    print("2. Test inference speed on target hardware")
    print("3. Integrate into two-stage detection + classification pipeline")

if __name__ == "__main__":
    main()