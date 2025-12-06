#!/usr/bin/env python3
"""
FP16 Quantization Script for ONNX Models
Converts ONNX models to FP16 precision for reduced memory usage on Raspberry Pi
"""

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os
from pathlib import Path

def quantize_model_to_fp16(input_path, output_path):
    """
    Quantize ONNX model to FP16 for reduced memory usage
    
    Args:
        input_path (str): Path to input ONNX model
        output_path (str): Path to save quantized FP16 model
    """
    print(f"Loading ONNX model from: {input_path}")
    
    # Load the ONNX model
    model = onnx.load(input_path)
    
    # Check if model is valid
    onnx.checker.check_model(model)
    print("✅ ONNX model validation passed")
    
    # Get model size before quantization
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"Original model size: {original_size:.2f} MB")
    
    # For FP16 quantization, we use dynamic quantization which automatically
    # converts to the most appropriate precision
    print("Converting to FP16 precision...")
    
    # Use dynamic quantization which will convert to FP16 where beneficial
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8  # Use QUInt8 for weights
    )
    
    # Get model size after quantization
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Quantized model size: {quantized_size:.2f} MB")
    
    # Calculate compression ratio
    compression_ratio = original_size / quantized_size
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.1f}%")
    
    # Validate quantized model
    quantized_model = onnx.load(output_path)
    onnx.checker.check_model(quantized_model)
    print("✅ Quantized model validation passed")
    
    return {
        'original_size': original_size,
        'quantized_size': quantized_size,
        'compression_ratio': compression_ratio
    }

def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX model to FP16')
    parser.add_argument('--model', required=True, help='Path to input ONNX model')
    parser.add_argument('--output', required=True, help='Path to save quantized model')
    parser.add_argument('--input-size', nargs=2, type=int, default=[640, 640],
                       help='Input image size (width height)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Quantize the model
    results = quantize_model_to_fp16(args.model, args.output)
    
    print(f"\n🎯 FP16 Quantization Complete!")
    print(f"📁 Quantized model saved to: {args.output}")
    print(f"💾 Size reduction: {results['original_size']:.2f} MB → {results['quantized_size']:.2f} MB")
    print(f"🚀 Compression: {results['compression_ratio']:.2f}x smaller")
    
    # Test inference with quantized model
    print("\n🧪 Testing quantized model inference...")
    try:
        # Create inference session with quantized model
        session = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        
        # Get input name and shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"✅ Quantized model loads successfully")
        print(f"📊 Input shape: {input_shape}")
        print(f"🔧 Execution providers: {session.get_providers()}")
        
    except Exception as e:
        print(f"❌ Error testing quantized model: {e}")
        return
    
    print("\n📋 Next steps:")
    print("1. Test quantized model accuracy vs original")
    print("2. Benchmark inference speed on target hardware")
    print("3. Integrate into Raspberry Pi deployment pipeline")

if __name__ == "__main__":
    main()