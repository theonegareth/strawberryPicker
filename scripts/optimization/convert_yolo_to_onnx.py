#!/usr/bin/env python3
"""
Convert YOLO models to ONNX format for faster inference on Raspberry Pi
"""

import sys
import argparse
from pathlib import Path

def convert_yolo_to_onnx(model_path, output_path, input_size=(640, 640)):
    """
    Convert YOLO model to ONNX format using ultralytics built-in export
    
    Args:
        model_path: Path to YOLO .pt model
        output_path: Path to save ONNX model
        input_size: Input image size (height, width)
    """
    print(f"Loading YOLO model from: {model_path}")
    
    # Load YOLO model using ultralytics
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export to ONNX using ultralytics built-in export
    print(f"Converting to ONNX format...")
    model.export(format='onnx', imgsz=input_size[0], half=False, simplify=True)
    
    # The export method saves to the same directory as the model
    # Move it to the desired output location
    import shutil
    onnx_path = Path(model_path).with_suffix('.onnx')
    
    if onnx_path.exists():
        shutil.move(str(onnx_path), output_path)
        print(f"✅ Successfully converted to ONNX: {output_path}")
        
        # Verify file size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"ONNX model size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Failed to find exported ONNX model at {onnx_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO models to ONNX format')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO .pt model')
    parser.add_argument('--output', type=str, required=True, help='Path to save ONNX model')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], 
                       help='Input image size (height width)')
    
    args = parser.parse_args()
    
    try:
        success = convert_yolo_to_onnx(args.model, args.output, tuple(args.input_size))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()