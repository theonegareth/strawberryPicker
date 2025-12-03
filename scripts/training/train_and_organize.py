#!/usr/bin/env python3
"""
Training organization script for strawberry detection
Creates organized model directories and manages training runs
"""

import os
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
import argparse

def create_model_structure(model_type, architecture, description, base_dir=None):
    """
    Create organized model directory structure
    
    Args:
        model_type: 'detection' or 'classification'
        architecture: 'yolov8n', 'yolov8s', etc.
        description: Short description of the experiment
        base_dir: Base directory for models (optional)
    
    Returns:
        Path to created model directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / 'models' / model_type
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model name
    model_name = f"{architecture}_{description}_{timestamp}"
    model_dir = base_dir / model_name
    
    # Create directory structure
    dirs = [
        model_dir,
        model_dir / 'weights',
        model_dir / 'config',
        model_dir / 'validation' / 'detection_results',
        model_dir / 'validation' / 'metrics',
        model_dir / 'validation' / 'visualizations'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created model structure: {model_dir.relative_to(Path(__file__).parent.parent.parent)}")
    print(f"📁 Model name: {model_name}")
    
    return model_dir, model_name

def copy_weights(weights_path, model_dir):
    """Copy trained weights to organized structure"""
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Copy to both locations for consistency
    best_path = model_dir / 'weights' / 'best.pt'
    model_path = model_dir / 'weights' / 'model.pt'
    
    shutil.copy2(weights_path, best_path)
    shutil.copy2(weights_path, model_path)
    
    print(f"✅ Copied weights to:")
    print(f"   - {best_path.relative_to(Path(__file__).parent.parent.parent)}")
    print(f"   - {model_path.relative_to(Path(__file__).parent.parent.parent)}")

def create_training_config(model_dir, training_params):
    """Create training configuration JSON"""
    config = {
        "model_name": model_dir.name,
        "created_at": datetime.now().isoformat(),
        "training_parameters": training_params,
        "model_metadata": {
            "architecture": training_params.get('architecture', 'unknown'),
            "model_size": training_params.get('model_size', 'unknown'),
            "dataset": training_params.get('dataset_name', 'unknown'),
            "num_classes": training_params.get('num_classes', 0)
        }
    }
    
    config_path = model_dir / 'config' / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created training config: {config_path.relative_to(Path(__file__).parent.parent.parent)}")
    return config_path

def create_model_readme(model_dir, training_results):
    """Create README for the model"""
    readme_content = f"""# {model_dir.name}

## Model Information
- **Architecture**: {training_results.get('architecture', 'YOLOv8')}
- **Model Size**: {training_results.get('model_size', 'n')}
- **Dataset**: {training_results.get('dataset_name', 'Unknown')}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Parameters
- **Epochs**: {training_results.get('epochs', 'Unknown')}
- **Batch Size**: {training_results.get('batch_size', 'Unknown')}
- **Image Size**: {training_results.get('image_size', 'Unknown')}
- **Learning Rate**: {training_results.get('learning_rate', 'Unknown')}
- **Optimizer**: {training_results.get('optimizer', 'Unknown')}

## Performance Metrics
- **mAP@0.5**: {training_results.get('map50', 'Not evaluated')}
- **mAP@0.5:0.95**: {training_results.get('map50_95', 'Not evaluated')}
- **Precision**: {training_results.get('precision', 'Not evaluated')}
- **Recall**: {training_results.get('recall', 'Not evaluated')}
- **Training Time**: {training_results.get('training_time', 'Unknown')}

## Files
- `weights/best.pt` - Best model weights
- `weights/model.pt` - Model weights (copy)
- `config/training_config.json` - Training configuration
- `validation/` - Validation results and metrics

## Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO('weights/best.pt')

# Run inference
results = model('path/to/image.jpg')
```

## Deployment
For Raspberry Pi 4B deployment, export to TensorFlow Lite:
```bash
yolo export model=weights/best.pt format=tflite imgsz=416 int8=True
```
"""
    
    readme_path = model_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created model README: {readme_path.relative_to(Path(__file__).parent.parent.parent)}")

def list_models(model_type='detection', base_dir=None):
    """List all existing models"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / 'models'
    
    model_dir = base_dir / model_type
    if not model_dir.exists():
        print(f"No models found in {model_dir}")
        return
    
    print(f"\n📋 Available {model_type} models:")
    print("=" * 60)
    
    models = sorted(model_dir.iterdir(), key=os.path.getmtime, reverse=True)
    
    for i, model_path in enumerate(models, 1):
        if model_path.is_dir():
            # Get model info
            config_path = model_path / 'config' / 'training_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                desc = config.get('model_metadata', {}).get('architecture', 'unknown')
            else:
                desc = 'no config'
            
            print(f"{i}. {model_path.name}")
            print(f"   📁 {model_path.relative_to(Path(__file__).parent.parent.parent)}")
            print(f"   📝 {desc}")
            
            # Check for weights
            best_path = model_path / 'weights' / 'best.pt'
            if best_path.exists():
                size_mb = best_path.stat().st_size / (1024 * 1024)
                print(f"   ⚖️  {size_mb:.1f} MB")
            else:
                print(f"   ⚠️  No weights found")
            print()

def get_latest_model(model_type='detection', base_dir=None):
    """Get the latest model directory"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / 'models'
    
    model_dir = base_dir / model_type
    if not model_dir.exists():
        return None
    
    models = sorted(model_dir.iterdir(), key=os.path.getmtime, reverse=True)
    
    for model_path in models:
        if model_path.is_dir():
            weights_path = model_path / 'weights' / 'best.pt'
            if weights_path.exists():
                return model_path, model_path.name
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Create organized model structure')
    parser.add_argument('--type', type=str, choices=['detection', 'classification'], 
                       default='detection', help='Model type')
    parser.add_argument('--architecture', type=str, default='yolov8n',
                       help='Model architecture (yolov8n, yolov8s, etc.)')
    parser.add_argument('--description', type=str, required=True,
                       help='Experiment description')
    parser.add_argument('--weights', type=str, help='Path to trained weights')
    parser.add_argument('--training-results', type=str, help='JSON string with training results')
    parser.add_argument('--list', action='store_true', help='List all models')
    parser.add_argument('--latest', action='store_true', help='Get latest model')
    
    args = parser.parse_args()
    
    if args.list:
        list_models(args.type)
        return 0
    
    if args.latest:
        result = get_latest_model(args.type)
        if result:
            model_path, model_name = result
            print(f"Latest model: {model_name}")
            print(f"Path: {model_path}")
        else:
            print("No models found")
        return 0
    
    # Create model structure
    try:
        model_dir, model_name = create_model_structure(
            model_type=args.type,
            architecture=args.architecture,
            description=args.description
        )
        
        # Copy weights if provided
        if args.weights:
            copy_weights(args.weights, model_dir)
        
        # Create training config if results provided
        if args.training_results:
            training_results = json.loads(args.training_results)
            create_training_config(model_dir, training_results)
            create_model_readme(model_dir, training_results)
        
        print(f"\n✅ Model structure created successfully!")
        print(f"📁 Model directory: {model_dir}")
        print(f"🏷️  Model name: {model_name}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())