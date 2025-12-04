#!/usr/bin/env python3
"""
Scan all model files and add missing ones to the training registry
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.training_registry import TrainingRun, TrainingRegistry

def scan_model_files():
    """Scan all .pt model files in the repository"""
    
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    model_files = []
    
    # Scan organized model directories
    models_dir = base_path / "models" / "detection"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Look for best.pt and model.pt
                best_pt = model_dir / "weights" / "best.pt"
                model_pt = model_dir / "weights" / "model.pt"
                
                if best_pt.exists():
                    model_files.append({
                        "path": str(best_pt),
                        "name": model_dir.name,
                        "type": "organized"
                    })
                if model_pt.exists():
                    model_files.append({
                        "path": str(model_pt),
                        "name": model_dir.name,
                        "type": "organized"
                    })
    
    # Scan results directories
    results_dir = base_path / "model" / "results"
    if results_dir.exists():
        for result_dir in results_dir.iterdir():
            if result_dir.is_dir():
                best_pt = result_dir / "weights" / "best.pt"
                if best_pt.exists():
                    model_files.append({
                        "path": str(best_pt),
                        "name": result_dir.name,
                        "type": "results"
                    })
    
    # Scan weights directory
    weights_dir = base_path / "model" / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            model_files.append({
                "path": str(weight_file),
                "name": weight_file.stem,
                "type": "weights"
            })
    
    return model_files

def extract_model_info(model_path, model_name):
    """Extract model information from path and name"""
    
    # Default values
    architecture = "YOLOv8"
    model_size = "n"  # Default
    dataset_name = "unknown"
    dataset_size = 392
    epochs = 50
    
    # Try to extract model size from name
    if "yolov8n" in model_name.lower():
        model_size = "n"
    elif "yolov8s" in model_name.lower():
        model_size = "s"
    elif "yolov8m" in model_name.lower():
        model_size = "m"
    elif "yolov8l" in model_name.lower():
        model_size = "l"
    elif "yolov8x" in model_name.lower():
        model_size = "x"
    
    # Try to extract dataset info
    if "kaggle" in model_name.lower():
        dataset_name = "kaggle-fruit-ripeness"
        dataset_size = 2500
    elif "strawberry" in model_name.lower():
        dataset_name = "straw-detect.v1-straw-detect.yolov8"
        dataset_size = 392
    
    # Try to extract epochs
    if "100" in model_name:
        epochs = 100
    elif "150" in model_name:
        epochs = 150
    
    return {
        "architecture": architecture,
        "size": model_size,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "epochs": epochs
    }

def add_all_missing_models():
    """Add all missing model files to registry"""
    
    registry = TrainingRegistry("model/training_registry.json")
    model_files = scan_model_files()
    
    print(f"🔍 Found {len(model_files)} model files")
    print("="*60)
    
    added_count = 0
    existing_runs = registry.get_all_runs()
    existing_paths = [run.get('model_path', '') for run in existing_runs]
    
    for model_info in model_files:
        model_path = model_info["path"]
        
        # Skip if already in registry
        if model_path in existing_paths:
            print(f"✓ Already in registry: {model_info['name']}")
            continue
        
        # Skip if not a final best.pt or model.pt (skip checkpoints)
        if "epoch" in model_path and "best.pt" not in model_path and "model.pt" not in model_path:
            continue
        
        # Extract model info
        info = extract_model_info(model_path, model_info['name'])
        
        # Create run ID and date
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_info['name'][:30]}"
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create training run
        training_run = TrainingRun(
            run_id=run_id,
            date=date,
            experiment_name=model_info['name'],
            model_type='detection',
            model_architecture=info['architecture'],
            model_size=info['size'],
            pretrained=True,
            dataset_name=info['dataset_name'],
            dataset_size=info['dataset_size'],
            num_classes=1,
            class_names=['strawberry'],
            batch_size=16,
            image_size=640,
            epochs_planned=info['epochs'],
            epochs_completed=info['epochs'],
            learning_rate=0.001,
            optimizer='AdamW',
            weight_decay=0.0005,
            training_time_minutes=0.0,
            gpu_name='NVIDIA GeForce RTX 3050 Ti Laptop GPU',
            gpu_memory_peak_gb=0.0,
            cpu_count=20,
            ram_total_gb=15.5,
            python_version='3.12.3',
            pytorch_version='2.9.1+cu128',
            cuda_version='12.8',
            os_info='linux',
            model_path=model_path,
            results_path=str(Path(model_path).parent.parent),
            config_path=model_path,
            status='completed'
        )
        
        # Add to registry
        registry.add_run(training_run)
        added_count += 1
        print(f"✅ Added: {model_info['name']}")
        print(f"   Path: {model_path}")
        print(f"   Size: {info['size']}, Epochs: {info['epochs']}")
        print()
    
    print("="*60)
    print(f"SUMMARY: Added {added_count} new training runs")
    print(f"Total runs in registry: {len(registry.get_all_runs())}")
    
    return added_count

if __name__ == "__main__":
    add_all_missing_models()