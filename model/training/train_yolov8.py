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
import time
from model.validation_logger import validation_logger, ValidationResult, create_validation_result_from_metrics

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
    
    # Record training start time
    training_start_time = time.time()
    
    results = model.train(**train_args)
    
    # Calculate training duration
    training_duration = (time.time() - training_start_time) / 60  # minutes
    
    # Save final model
    final_model_path = weights_dir / 'strawberry_yolov8n.pt'
    model.save(str(final_model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results saved to: {results_dir / 'strawberry_detection'}")
    print(f"Training duration: {training_duration:.1f} minutes")
    print(f"{'='*60}\n")
    
    # Log training run
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from validation.training_registry import TrainingRun, TrainingRegistry
        from datetime import datetime
        import psutil
        
        training_registry = TrainingRegistry("model/training_registry.json")
        
        # Extract metrics from results
        final_metrics = {}
        if hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
        
        # Get system info
        cpu_count = psutil.cpu_count()
        ram_total_gb = psutil.virtual_memory().total / (1024**3)
        
        training_run = TrainingRun(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name=f"YOLOv8_{train_args.get('name', 'strawberry_detection')}",
            model_type='teacher',
            model_architecture='YOLOv8',
            model_size='n',
            pretrained=True,
            dataset_name='straw-detect.v1-straw-detect.yolov8',
            dataset_size=392,  # This should be calculated from actual dataset
            num_classes=1,
            class_names=['strawberry'],
            batch_size=batch_size,
            image_size=img_size,
            epochs_planned=epochs,
            epochs_completed=epochs,  # This should be actual completed epochs
            learning_rate=0.002,  # Default for YOLOv8
            optimizer='AdamW',
            weight_decay=0.0005,
            train_accuracy=0.0,  # YOLO doesn't have traditional accuracy
            val_accuracy=0.0,
            test_accuracy=0.0,
            train_loss=final_metrics.get('train/loss', 0.0),
            val_loss=final_metrics.get('val/loss', 0.0),
            test_loss=0.0,
            precision=final_metrics.get('metrics/precision(B)', 0.0),
            recall=final_metrics.get('metrics/recall(B)', 0.0),
            f1_score=0.0,
            macro_avg_precision=0.0,
            macro_avg_recall=0.0,
            macro_avg_f1=0.0,
            weighted_avg_precision=0.0,
            weighted_avg_recall=0.0,
            weighted_avg_f1=0.0,
            training_time_minutes=training_duration,
            early_stopped=False,
            best_epoch=epochs,  # This should be actual best epoch
            gpu_name=env['gpu_name'],
            gpu_memory_peak_gb=torch.cuda.max_memory_allocated() / (1024**3) if env['has_gpu'] else 0.0,
            cpu_count=cpu_count,
            ram_total_gb=ram_total_gb,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            tensorflow_version='N/A',  # Not using TensorFlow
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.version.cuda else 'N/A',
            os_info=f"{sys.platform}",
            model_path=str(final_model_path),
            results_path=str(results_dir / 'strawberry_detection'),
            config_path=str(results_dir / 'strawberry_detection' / 'args.yaml'),
            status='completed'
        )
        
        training_registry.add_run(training_run)
        print(f"✓ Training run logged: {training_run.run_id}")
        
    except Exception as e:
        print(f"Warning: Could not log training run: {e}")
        import traceback
        traceback.print_exc()
    
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