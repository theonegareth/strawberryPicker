#!/usr/bin/env python3
"""
Convert trained model to TensorFlow Lite format with optional quantization.
Supports conversion from Keras (.h5) and PyTorch (.pt) models.
"""

import argparse
import tensorflow as tf
import os
from pathlib import Path
import sys

def convert_keras_to_tflite(input_path, output_path, quantization=None, optimize_for_size=False):
    """Convert Keras model to TFLite format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
    
    print(f"Loading Keras model from {input_path}...")
    model = tf.keras.models.load_model(input_path)
    
    # Configure converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options
    if optimize_for_size:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantization options
    if quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("Using INT8 quantization")
    elif quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Using Float16 quantization")
    elif quantization == 'dynamic_range':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Using dynamic range quantization")
    
    # Convert model
    print("Converting model to TFLite...")
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Print model size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Model size: {size_kb:.2f} KB")
    
    return output_path

def convert_pytorch_to_tflite(input_path, output_path, quantization=None):
    """Convert PyTorch model to TFLite format (placeholder for future implementation)."""
    print("PyTorch to TFLite conversion not yet implemented.")
    print("Please convert PyTorch model to ONNX first, then use TensorFlow's converter.")
    return None

def main():
    parser = argparse.ArgumentParser(description='Convert model to TensorFlow Lite format')
    parser.add_argument('--input', type=str, default='strawberry_model.h5',
                        help='Input model path (Keras .h5 or PyTorch .pt)')
    parser.add_argument('--output', type=str, default='strawberry_model.tflite',
                        help='Output TFLite model path')
    parser.add_argument('--quantization', type=str, choices=['int8', 'float16', 'dynamic_range', 'none'],
                        default='none', help='Quantization method (default: none)')
    parser.add_argument('--optimize-for-size', action='store_true',
                        help='Apply size optimization (reduces model size)')
    parser.add_argument('--model-type', type=str, choices=['keras', 'pytorch'], default='keras',
                        help='Type of input model (default: keras)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input model '{args.input}' not found.")
        print("Available model files in current directory:")
        for f in os.listdir('.'):
            if f.endswith(('.h5', '.pt', '.pth', '.onnx')):
                print(f"  - {f}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        if args.model_type == 'keras':
            convert_keras_to_tflite(
                input_path=args.input,
                output_path=args.output,
                quantization=args.quantization if args.quantization != 'none' else None,
                optimize_for_size=args.optimize_for_size
            )
        elif args.model_type == 'pytorch':
            convert_pytorch_to_tflite(
                input_path=args.input,
                output_path=args.output,
                quantization=args.quantization if args.quantization != 'none' else None
            )
        else:
            print(f"Unsupported model type: {args.model_type}")
            sys.exit(1)
            
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()