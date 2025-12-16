#!/usr/bin/env python3
"""
Export YOLOv8/v11 model to ONNX format for optimized inference.
Supports dynamic axes, batch size, and different opset versions.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def export_to_onnx(
    model_path,
    output_path=None,
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=12,
    half=False
):
    """
    Export YOLO model to ONNX format.
    
    Args:
        model_path: Path to .pt model file
        output_path: Output .onnx file path (optional, auto-generated if None)
        imgsz: Input image size
        batch: Batch size (1 for static, -1 for dynamic)
        dynamic: Enable dynamic axes (batch, height, width)
        simplify: Apply ONNX simplifier
        opset: ONNX opset version
        half: FP16 quantization
    """
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Determine output path
    if output_path is None:
        model_name = Path(model_path).stem
        output_dir = Path(model_path).parent / "exports"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{model_name}.onnx")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare export arguments
    export_args = {
        'format': 'onnx',
        'imgsz': imgsz,
        'batch': batch,
        'simplify': simplify,
        'opset': opset,
        'half': half,
    }
    
    if dynamic:
        export_args['dynamic'] = True
    
    print(f"Exporting to ONNX with args: {export_args}")
    
    try:
        # Export model
        exported_path = model.export(**export_args)
        
        # The exported file will be in the same directory as the model
        # Find the .onnx file that was just created
        exported_files = list(Path(model_path).parent.glob("*.onnx"))
        if exported_files:
            latest_onnx = max(exported_files, key=os.path.getctime)
            # Move to desired output path if different
            if str(latest_onnx) != output_path:
                import shutil
                shutil.move(str(latest_onnx), output_path)
                print(f"Model moved to {output_path}")
            else:
                print(f"Model exported to {output_path}")
        else:
            # Try to find the exported file in the current directory
            exported_files = list(Path('.').glob("*.onnx"))
            if exported_files:
                latest_onnx = max(exported_files, key=os.path.getctime)
                if str(latest_onnx) != output_path:
                    import shutil
                    shutil.move(str(latest_onnx), output_path)
                    print(f"Model moved to {output_path}")
                else:
                    print(f"Model exported to {output_path}")
            else:
                print(f"Warning: Could not locate exported ONNX file.")
                print(f"Expected at: {output_path}")
                return None
        
        # Print model info
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ ONNX export successful!")
        print(f"   Output: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Input shape: {batch if batch > 0 else 'dynamic'}x3x{imgsz}x{imgsz}")
        print(f"   Opset: {opset}")
        print(f"   Dynamic: {dynamic}")
        print(f"   FP16: {half}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLO model (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path (default: model/exports/<model_name>.onnx)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (default: 1, use -1 for dynamic)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic axes (batch, height, width)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Disable ONNX simplifier')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 quantization')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use model from config if not specified
    if args.model == 'yolov8n.pt' and config:
        models_config = config.get('models', {})
        detection_config = models_config.get('detection', {})
        default_model = detection_config.get('strawberry_yolov8n', 'yolov8n.pt')
        if os.path.exists(default_model):
            args.model = default_model
        else:
            # Check for other available models
            available_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 
                              'model/weights/strawberry_yolov11n.pt',
                              'model/weights/ripeness_detection_yolov11n.pt']
            for model in available_models:
                if os.path.exists(model):
                    args.model = model
                    print(f"Using available model: {model}")
                    break
    
    # Export model
    success = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.img_size,
        batch=args.batch,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        opset=args.opset,
        half=args.half
    )
    
    if success:
        print(f"\n✅ Export completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Test the ONNX model with ONNX Runtime:")
        print(f"   python -m onnxruntime.tools.onnx_model_test {success}")
        print(f"2. Convert to TensorFlow Lite for edge deployment:")
        print(f"   python export_tflite_int8.py --model {success}")
        print(f"3. Use in your application with ONNX Runtime")
    else:
        print("\n❌ Export failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()