#!/usr/bin/env python3
"""
YOLOv8 Training Script for Strawberry Detection
Compatible with: Local Python, WSL, Google Colab (VS Code extension)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import yaml

def check_environment():
    """Detect running environment and configure paths accordingly"""
    env_info = {
        'is_colab': 'COLAB_GPU' in os.environ or '/content' in os.getcwd(),
        'is_wsl': 'WSL_DISTRO_NAME' in os.environ,
        'has_gpu': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    return env_info

def setup_paths(dataset_path=None):
    """Configure dataset and output paths based on environment"""
    env = check_environment()
    
    if env['is_colab']:
        # Google Colab paths
        base_path = Path('/content/strawberry-picker')
        dataset_path = dataset_path or '/content/dataset'
        weights_dir = base_path / 'weights'
        results_dir = base_path / 'results'
    else:
        # Local/WSL paths
        base_path = Path(__file__).parent
        dataset_path = dataset_path or base_path / 'model' / 'dataset' / 'straw-detect.v1-straw-detect.yolov8'
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

def train_model(data_yaml, weights_dir, results_dir, epochs=100, img_size=640, batch_size=16):
    """Train YOLOv8 model"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    env = check_environment()
    print(f"\n{'='*60}")
    print(f"Environment: {'Google Colab' if env['is_colab'] else 'Local/WSL'}")
    print(f"GPU Available: {env['has_gpu']} ({env['gpu_name']})")
    print(f"{'='*60}\n")
    
    # Use GPU if available
    device = '0' if env['has_gpu'] else 'cpu'
    
    # Load pretrained YOLOv8n model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': str(results_dir),
        'name': 'strawberry_detection',
        'exist_ok': True,
        'patience': 20,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,  # Cache images for faster training
    }
    
    # Adjust batch size for Colab's limited RAM
    if env['is_colab'] and batch_size > 16:
        train_args['batch'] = 16
        print(f"Adjusted batch size to 16 for Colab environment")
    
    print(f"\nStarting training with arguments:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Train the model
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}\n")
    
    results = model.train(**train_args)
    
    # Save final model
    final_model_path = weights_dir / 'strawberry_yolov8n.pt'
    model.save(str(final_model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results saved to: {results_dir / 'strawberry_detection'}")
    print(f"{'='*60}\n")
    
    return results, final_model_path

def export_model(model_path, weights_dir):
    """Export model to ONNX format"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        return None
    
    print(f"\nExporting model to ONNX...")
    model = YOLO(str(model_path))
    
    # Export to ONNX
    onnx_path = weights_dir / 'strawberry_yolov8n.onnx'
    model.export(format='onnx', imgsz=640, dynamic=True)
    
    print(f"ONNX model exported to: {onnx_path}")
    return onnx_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for strawberry detection')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX after training')
    parser.add_argument('--validate-only', action='store_true', help='Only validate dataset without training')
    
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
        
        if args.validate_only:
            print("Dataset validation completed. Exiting without training.")
            return
        
        # Train model
        results, model_path = train_model(
            data_yaml=data_yaml,
            weights_dir=paths['weights_dir'],
            results_dir=paths['results_dir'],
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size
        )
        
        # Export to ONNX if requested
        if args.export_onnx:
            export_model(model_path, paths['weights_dir'])
        
        print("\n✓ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()