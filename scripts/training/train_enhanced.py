#!/usr/bin/env python3
"""
Enhanced YOLOv8 Training Script for Strawberry Detection
Implements advanced techniques for better accuracy
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

def check_environment():
    """Detect running environment and configure paths accordingly"""
    env_info = {
        'is_colab': 'COLAB_GPU' in os.environ or '/content' in os.getcwd(),
        'is_wsl': 'WSL_DISTRO_NAME' in os.environ,
        'has_gpu': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    }
    return env_info

def setup_paths(dataset_path=None):
    """Configure dataset and output paths based on environment"""
    env = check_environment()
    
    if env['is_colab']:
        # Google Colab paths
        base_path = Path('/content/strawberry-picker')
        dataset_path = dataset_path or '/content/dataset'
        weights_dir = base_path / 'model' / 'weights'
        results_dir = base_path / 'model' / 'results'
    else:
        # Local/WSL paths - updated for new structure
        base_path = Path(__file__).parent.parent.parent
        dataset_path = dataset_path or base_path / 'model' / 'dataset'
        weights_dir = base_path / 'model' / 'weights'
        results_dir = base_path / 'model' / 'results'
    
    # Create directories
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'dataset_path': Path(dataset_path),
        'weights_dir': weights_dir,
        'results_dir': results_dir,
        'base_path': base_path
    }

def validate_dataset(dataset_path):
    """Validate YOLO dataset structure"""
    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / 'data.yaml'
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    # Load and validate YAML
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in data.yaml")
    
    # Check if paths are relative and resolve them
    train_path = dataset_path / data['train']
    val_path = dataset_path / data['val']
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training images not found at {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation images not found at {val_path}")
    
    print(f"✓ Dataset validated: {data['nc']} classes - {data['names']}")
    print(f"✓ Training images: {train_path}")
    print(f"✓ Validation images: {val_path}")
    
    return data_yaml

def train_model(data_yaml, weights_dir, results_dir, epochs=150, img_size=640, batch_size=8, model_name='yolov8s.pt'):
    """Train enhanced YOLOv8 model with advanced techniques"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    env = check_environment()
    print(f"\n{'='*60}")
    print(f"Environment: {'Google Colab' if env['is_colab'] else 'Local/WSL'}")
    print(f"GPU Available: {env['has_gpu']} ({env['gpu_name']})")
    print(f"GPU Memory: {env['gpu_memory']:.1f} GB")
    print(f"{'='*60}\n")
    
    # Use GPU if available
    device = '0' if env['has_gpu'] else 'cpu'
    
    # Adjust batch size based on GPU memory
    if env['gpu_memory'] < 5:  # 4GB GPU (RTX 3050 Ti)
        batch_size = min(batch_size, 8)
        img_size = min(img_size, 640)
        print(f"Adjusted for 4GB GPU: batch_size={batch_size}, img_size={img_size}")
    
    # Load pretrained YOLOv8 model
    print(f"Loading {model_name} model...")
    
    # Change to weights directory to ensure write permissions for download
    original_cwd = os.getcwd()
    os.chdir(weights_dir)
    try:
        # Load model - will download to current directory (weights_dir)
        model = YOLO(model_name)
        # Model is now downloaded to weights_dir, use full path
        model_path = weights_dir / model_name
        model = YOLO(str(model_path))
    finally:
        os.chdir(original_cwd)
    
    # Enhanced training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': str(results_dir),
        'name': 'strawberry_enhanced',
        'exist_ok': True,
        'patience': 30,  # More patience for better convergence
        'save': True,
        'save_period': 10,
        'cache': True,
        
        # ===== Enhanced Augmentation =====
        'hsv_h': 0.015,      # Hue augmentation (color variations)
        'hsv_s': 0.7,        # Saturation
        'hsv_v': 0.4,        # Value (brightness)
        'degrees': 15.0,     # Rotation (+/- 15 degrees)
        'translate': 0.1,    # Translation
        'scale': 0.5,        # Scaling (±50%)
        'shear': 5.0,        # Shear
        'perspective': 0.0,  # Perspective (disabled for simplicity)
        'flipud': 0.0,       # Flip up-down
        'fliplr': 0.5,       # Flip left-right
        
        # ===== Advanced Augmentation =====
        'mosaic': 1.0,       # Mosaic augmentation (combine 4 images)
        'mixup': 0.1,        # MixUp (blend 2 images)
        'copy_paste': 0.1,   # Copy-paste augmentation
        
        # ===== Optimization =====
        'lr0': 0.01,         # Initial learning rate
        'lrf': 0.01,         # Final learning rate
        'momentum': 0.937,   # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 5.0,    # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,   # Warmup bias lr
        
        # ===== Regularization =====
        'dropout': 0.1,      # Dropout rate
        'box': 7.5,          # Box loss gain
        'cls': 0.5,          # Class loss gain
        'dfl': 1.5,          # DFL loss gain
        
        # ===== Training Strategy =====
        'optimizer': 'AdamW',  # AdamW optimizer (better than SGD)
        'cos_lr': True,        # Cosine LR scheduler
        'close_mosaic': 10,    # Close mosaic augmentation in last 10 epochs
        
        # ===== Validation =====
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.25,
        'iou': 0.45,
        'max_det': 300,
        
        # ===== Speed/Efficiency =====
        'workers': 8,        # Number of worker threads
        'cache': True,       # Cache images for faster training
        'rect': False,       # Rectangular training
        'resume': False,     # Resume training from checkpoint
        
        # ===== Logging =====
        'plots': True,       # Generate plots
        'verbose': True,     # Verbose output
    }
    
    print(f"\nStarting enhanced training with {model_name}:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Optimizer: AdamW")
    print(f"  Augmentation: Mosaic, MixUp, Copy-Paste")
    print(f"  Scheduler: Cosine LR")
    
    # Train the model
    print(f"\n{'='*60}")
    print("TRAINING STARTED - Enhanced Configuration")
    print(f"{'='*60}\n")
    
    results = model.train(**train_args)
    
    # Save final model
    final_model_path = weights_dir / 'strawberry_yolov8s_enhanced.pt'
    model.save(str(final_model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results saved to: {results_dir / 'strawberry_enhanced'}")
    print(f"{'='*60}\n")
    
    return results, final_model_path

def export_model(model_path, weights_dir, format='tflite'):
    """Export model to specified format"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        return None
    
    print(f"\nExporting model to {format.upper()}...")
    model = YOLO(str(model_path))
    
    if format == 'tflite':
        # Export to TensorFlow Lite
        tflite_path = weights_dir / 'strawberry_yolov8s_enhanced.tflite'
        model.export(
            format='tflite',
            imgsz=416,
            int8=True,  # INT8 quantization
            data=str(Path(model_path).parent.parent / 'dataset' / 'data.yaml')
        )
        print(f"TFLite model exported to: {tflite_path}")
        return tflite_path
    elif format == 'onnx':
        # Export to ONNX
        onnx_path = weights_dir / 'strawberry_yolov8s_enhanced.onnx'
        model.export(format='onnx', imgsz=416, dynamic=True)
        print(f"ONNX model exported to: {onnx_path}")
        return onnx_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Train enhanced YOLOv8 for strawberry detection')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model architecture (yolov8n, yolov8s, yolov8m)')
    parser.add_argument('--export', type=str, choices=['tflite', 'onnx'], help='Export format after training')
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        paths = setup_paths(args.dataset)
        print(f"Base path: {paths['base_path']}")
        print(f"Dataset path: {paths['dataset_path']}")
        print(f"Weights directory: {paths['weights_dir']}")
        print(f"Results directory: {paths['results_dir']}")
        
        # Validate dataset
        print(f"\nValidating dataset...")
        data_yaml = validate_dataset(paths['dataset_path'])
        
        # Train model
        results, model_path = train_model(
            data_yaml=data_yaml,
            weights_dir=paths['weights_dir'],
            results_dir=paths['results_dir'],
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            model_name=args.model
        )
        
        # Export if requested
        if args.export:
            export_model(model_path, paths['weights_dir'], format=args.export)
        
        print("\n✓ Enhanced training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()