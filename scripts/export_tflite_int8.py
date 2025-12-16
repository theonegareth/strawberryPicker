#!/usr/bin/env python3
"""
Export YOLOv8 model to TensorFlow Lite with INT8 quantization.
Uses Ultralytics YOLOv8 export functionality with calibration dataset.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_representative_dataset(dataset_path, num_calibration=100):
    """
    Create representative dataset for INT8 calibration.
    Returns a generator that yields normalized images.
    """
    import cv2
    from pathlib import Path
    
    # Find validation images
    val_path = Path(dataset_path) / "valid" / "images"
    if not val_path.exists():
        val_path = Path(dataset_path) / "val" / "images"
    if not val_path.exists():
        print(f"Warning: Validation images not found at {val_path}")
        return None
    
    image_files = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
    if len(image_files) == 0:
        print("No validation images found for calibration.")
        return None
    
    # Limit to num_calibration
    image_files = image_files[:num_calibration]
    
    def representative_dataset():
        for img_path in image_files:
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = np.expand_dims(img, axis=0)     # Add batch dimension
            yield [img]
    
    return representative_dataset

def export_yolov8_to_tflite_int8(
    model_path,
    output_path,
    dataset_path=None,
    img_size=640,
    int8=True,
    dynamic=False,
    half=False
):
    """
    Export YOLOv8 model to TFLite format with optional INT8 quantization.
    
    Args:
        model_path: Path to YOLOv8 .pt model
        output_path: Output .tflite file path
        dataset_path: Path to dataset for INT8 calibration
        img_size: Input image size
        int8: Enable INT8 quantization
        dynamic: Enable dynamic range quantization (alternative to INT8)
        half: Enable FP16 quantization
    """
    print(f"Loading YOLOv8 model from {model_path}")
    model = YOLO(model_path)
    
    # Check if model is a detection model
    task = model.task if hasattr(model, 'task') else 'detect'
    print(f"Model task: {task}")
    
    # Prepare export arguments
    export_args = {
        'format': 'tflite',
        'imgsz': img_size,
        'optimize': True,
        'int8': int8,
        'half': half,
        'dynamic': dynamic,
    }
    
    # If INT8 quantization is requested, provide representative dataset
    if int8 and dataset_path:
        print(f"Using dataset at {dataset_path} for INT8 calibration")
        representative_dataset = get_representative_dataset(dataset_path)
        if representative_dataset:
            # Note: Ultralytics YOLOv8 export doesn't directly accept representative_dataset
            # We'll need to use a different approach
            print("INT8 calibration with representative dataset requires custom implementation.")
            print("Falling back to Ultralytics built-in INT8 calibration...")
            # Use built-in calibration images
            export_args['int8'] = True
        else:
            print("Warning: No representative dataset available. Using default calibration.")
            export_args['int8'] = True
    elif int8:
        print("Using built-in calibration images for INT8 quantization")
        export_args['int8'] = True
    
    # Export model
    print(f"Exporting model to TFLite with args: {export_args}")
    try:
        # Use Ultralytics export
        exported_path = model.export(**export_args)
        
        # The exported file will be in the same directory as the model
        # with a .tflite extension
        exported_files = list(Path(model_path).parent.glob("*.tflite"))
        if exported_files:
            latest_tflite = max(exported_files, key=os.path.getctime)
            # Move to desired output path
            import shutil
            shutil.move(str(latest_tflite), output_path)
            print(f"Model exported to {output_path}")
            
            # Print model size
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
            
            return output_path
        else:
            print("Error: No .tflite file was generated")
            return None
            
    except Exception as e:
        print(f"Error during export: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 model to TFLite with INT8 quantization')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 model (.pt file)')
    parser.add_argument('--output', type=str, default='model/exports/strawberry_yolov8n_int8.tflite',
                        help='Output TFLite file path')
    parser.add_argument('--dataset', type=str, default='model/dataset_strawberry_detect_v3',
                        help='Path to dataset for INT8 calibration')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--no-int8', action='store_true',
                        help='Disable INT8 quantization')
    parser.add_argument('--dynamic', action='store_true',
                        help='Use dynamic range quantization')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 quantization')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use dataset path from config if not provided
    if args.dataset is None and config:
        args.dataset = config.get('dataset', {}).get('detection', {}).get('path', 'model/dataset_strawberry_detect_v3')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Export model
    success = export_yolov8_to_tflite_int8(
        model_path=args.model,
        output_path=args.output,
        dataset_path=args.dataset,
        img_size=args.img_size,
        int8=not args.no_int8,
        dynamic=args.dynamic,
        half=args.half
    )
    
    if success:
        print(f"\n✅ Successfully exported model to: {success}")
        print("\nUsage:")
        print(f"  python detect_realtime.py --model {args.output}")
        print(f"  python detect_realtime.py --model {args.output} --mode detection")
    else:
        print("\n❌ Export failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()