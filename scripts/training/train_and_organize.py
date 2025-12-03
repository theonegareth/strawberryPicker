#!/usr/bin/env python3
"""
Model Training and Organization Script
Creates unique folders for each training run with proper versioning
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import json

class ModelTrainingOrganizer:
    def __init__(self, base_dir="models"):
        """
        Initialize the training organizer
        
        Args:
            base_dir: Base directory for all models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main category folders
        (self.base_dir / "detection").mkdir(exist_ok=True)
        (self.base_dir / "classification").mkdir(exist_ok=True)
        
    def generate_model_name(self, model_type, architecture, description=None):
        """
        Generate a unique model name with timestamp
        
        Args:
            model_type: 'detection' or 'classification'
            architecture: 'yolov8s', 'yolov8n', 'efficientnet', etc.
            description: Optional description of the model/training
            
        Returns:
            Unique model name string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if description:
            # Clean description for folder name (remove spaces, special chars)
            clean_desc = "".join(c for c in description if c.isalnum() or c in ['_', '-'])
            model_name = f"{architecture}_{clean_desc}_{timestamp}"
        else:
            model_name = f"{architecture}_{timestamp}"
            
        return model_name
    
    def create_model_structure(self, model_type, model_name):
        """
        Create the complete folder structure for a new model
        
        Args:
            model_type: 'detection' or 'classification'
            model_name: Name of the model
            
        Returns:
            Path to the created model directory
        """
        model_dir = self.base_dir / model_type / model_name
        model_dir.mkdir(parents=True, exist_ok=False)  # Fail if exists
        
        # Create subdirectories
        (model_dir / "weights").mkdir()
        (model_dir / "validation").mkdir()
        (model_dir / "validation" / "detection_results").mkdir()
        (model_dir / "validation" / "classification_results").mkdir()
        (model_dir / "training_logs").mkdir()
        (model_dir / "config").mkdir()
        
        # Create README file
        readme_content = f"""# Model: {model_name}

## Training Information
- **Model Type:** {model_type}
- **Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Architecture:** {model_name.split('_')[0]}

## Folder Structure
- `weights/` - Model weights/checkpoints
- `validation/` - Validation results and metrics
- `training_logs/` - Training logs and curves
- `config/` - Training configuration files

## Usage
To use this model, refer to the weights in the `weights/` directory.

## Validation
Run validation using validate_models.py with this model's weights.
"""
        
        with open(model_dir / "README.md", "w") as f:
            f.write(readme_content)
            
        print(f"✅ Created model structure: {model_dir}")
        return model_dir
    
    def list_models(self, model_type=None):
        """
        List all available models
        
        Args:
            model_type: Filter by 'detection' or 'classification'
            
        Returns:
            List of model paths
        """
        models = []
        
        if model_type:
            search_dirs = [self.base_dir / model_type]
        else:
            search_dirs = [self.base_dir / "detection", self.base_dir / "classification"]
            
        for search_dir in search_dirs:
            if search_dir.exists():
                for model_dir in search_dir.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith('.'):
                        models.append(model_dir)
                        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return models
    
    def get_latest_model(self, model_type, architecture=None):
        """
        Get the most recent model of a specific type
        
        Args:
            model_type: 'detection' or 'classification'
            architecture: Optional architecture filter (e.g., 'yolov8s')
            
        Returns:
            Path to latest model directory or None
        """
        models = self.list_models(model_type)
        
        if architecture:
            models = [m for m in models if architecture in m.name]
            
        return models[0] if models else None
    
    def copy_weights(self, source_path, model_dir, weight_name="model.pt"):
        """
        Copy model weights to the organized structure
        
        Args:
            source_path: Path to the trained model weights
            model_dir: Destination model directory
            weight_name: Name for the weight file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source weights not found: {source_path}")
            
        dest = model_dir / "weights" / weight_name
        shutil.copy2(source, dest)
        print(f"✅ Copied weights to: {dest}")
        
        # Also copy as "best.pt" for YOLO compatibility
        best_dest = model_dir / "weights" / "best.pt"
        shutil.copy2(source, best_dest)
        print(f"✅ Copied weights to: {best_dest}")
    
    def create_training_config(self, model_dir, config_dict):
        """
        Create a training configuration file
        
        Args:
            model_dir: Model directory path
            config_dict: Dictionary of configuration parameters
        """
        config_path = model_dir / "config" / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Created training config: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Organize model training runs')
    parser.add_argument('--type', type=str,
                       choices=['detection', 'classification'],
                       help='Model type')
    parser.add_argument('--architecture', type=str,
                       help='Model architecture (e.g., yolov8s, efficientnet)')
    parser.add_argument('--description', type=str,
                       help='Optional description of the model/training')
    parser.add_argument('--weights', type=str,
                       help='Path to trained model weights to copy')
    parser.add_argument('--list', action='store_true',
                       help='List all existing models')
    parser.add_argument('--latest', action='store_true',
                       help='Show the latest model')
    
    args = parser.parse_args()
    
    # Check if --list or --latest is used without required arguments
    if not args.list and not args.latest:
        if not args.type or not args.architecture:
            parser.error("--type and --architecture are required when not using --list or --latest")
    
    organizer = ModelTrainingOrganizer()
    
    if args.list:
        print("\n📋 Available Models:")
        print("=" * 60)
        models = organizer.list_models()
        for i, model in enumerate(models, 1):
            model_type = "detection" if "detection" in str(model) else "classification"
            created = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"{i}. [{model_type}] {model.name}")
            print(f"   📁 {model}")
            print(f"   🕒 {created}")
            print()
        return
    
    if args.latest:
        latest = organizer.get_latest_model(args.type, args.architecture)
        if latest:
            print(f"\n🎯 Latest {args.type} model:")
            print(f"📁 {latest}")
            print(f"🕒 {datetime.fromtimestamp(latest.stat().st_mtime)}")
        else:
            print(f"No {args.type} models found")
        return
    
    # Create new model structure
    print(f"\n🚀 Creating new {args.type} model...")
    model_name = organizer.generate_model_name(args.type, args.architecture, args.description)
    print(f"📛 Model name: {model_name}")
    
    model_dir = organizer.create_model_structure(args.type, model_name)
    
    if args.weights:
        print(f"\n💾 Copying weights from: {args.weights}")
        organizer.copy_weights(args.weights, model_dir)
    
    # Create example config
    example_config = {
        "model_name": model_name,
        "architecture": args.architecture,
        "type": args.type,
        "description": args.description or "",
        "created": datetime.now().isoformat(),
        "training_params": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "image_size": 640
        }
    }
    organizer.create_training_config(model_dir, example_config)
    
    print(f"\n✅ Model structure created successfully!")
    print(f"📁 Location: {model_dir}")
    print(f"\nNext steps:")
    print(f"1. Copy your trained weights to: {model_dir}/weights/")
    print(f"2. Run validation: python3 validate_models.py --detector {model_dir}/weights/best.pt")
    print(f"3. Update {model_dir}/config/training_config.json with actual parameters")

if __name__ == "__main__":
    main()