#!/usr/bin/env python3
"""
Add Previous Training Runs to Registry
Manually log past training runs that weren't automatically recorded
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import psutil
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.training_registry import TrainingRun, TrainingRegistry

def add_manual_training_run(
    model_path,
    results_path,
    config_path,
    experiment_name,
    model_architecture="YOLOv8",
    model_size="n",
    epochs=100,
    batch_size=16,
    image_size=640,
    learning_rate=0.002,
    dataset_size=392,
    num_classes=1,
    class_names=None,
    val_precision=0.0,
    val_recall=0.0,
    val_map50=0.0,
    val_map50_95=0.0,
    training_time_minutes=0.0,
    date=None,
    run_id=None
):
    """
    Manually add a previous training run to the registry
    
    Args:
        model_path: Path to the trained model weights
        results_path: Path to training results directory
        config_path: Path to training config file
        experiment_name: Name of the experiment
        model_architecture: Model architecture (default: YOLOv8)
        model_size: Model size (default: n)
        epochs: Number of epochs (default: 100)
        batch_size: Batch size (default: 16)
        image_size: Image size (default: 640)
        learning_rate: Learning rate (default: 0.002)
        dataset_size: Dataset size (default: 392)
        num_classes: Number of classes (default: 1)
        class_names: List of class names (default: ['strawberry'])
        val_precision: Validation precision (default: 0.0)
        val_recall: Validation recall (default: 0.0)
        val_map50: Validation mAP@50 (default: 0.0)
        val_map50_95: Validation mAP@50-95 (default: 0.0)
        training_time_minutes: Training time in minutes (default: 0.0)
        date: Date string (default: current date)
        run_id: Custom run ID (default: auto-generated)
    """
    
    if class_names is None:
        class_names = ['strawberry']
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_manual"
    
    # Get system info
    env_info = get_system_info()
    
    # Create TrainingRun object
    training_run = TrainingRun(
        run_id=run_id,
        date=date,
        experiment_name=experiment_name,
        model_type='teacher',
        model_architecture=model_architecture,
        model_size=model_size,
        pretrained=True,
        dataset_name='straw-detect.v1-straw-detect.yolov8',
        dataset_size=dataset_size,
        num_classes=num_classes,
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        epochs_planned=epochs,
        epochs_completed=epochs,
        learning_rate=learning_rate,
        optimizer='AdamW',
        weight_decay=0.0005,
        train_accuracy=0.0,
        val_accuracy=val_map50,
        test_accuracy=0.0,
        train_loss=0.0,
        val_loss=0.0,
        test_loss=0.0,
        precision=val_precision,
        recall=val_recall,
        f1_score=0.0,
        macro_avg_precision=0.0,
        macro_avg_recall=0.0,
        macro_avg_f1=0.0,
        weighted_avg_precision=0.0,
        weighted_avg_recall=0.0,
        weighted_avg_f1=0.0,
        training_time_minutes=training_time_minutes,
        early_stopped=False,
        best_epoch=epochs,
        gpu_name=env_info['gpu_name'],
        gpu_memory_peak_gb=env_info['gpu_memory_peak_gb'],
        cpu_count=env_info['cpu_count'],
        ram_total_gb=env_info['ram_total_gb'],
        python_version=env_info['python_version'],
        tensorflow_version='N/A',
        pytorch_version=env_info['pytorch_version'],
        cuda_version=env_info['cuda_version'],
        os_info=env_info['os_info'],
        model_path=str(Path(model_path).absolute()),
        results_path=str(Path(results_path).absolute()),
        config_path=str(Path(config_path).absolute()),
        status='completed'
    )
    
    # Add to registry
    registry = TrainingRegistry("model/training_registry.json")
    registry.add_run(training_run)
    
    print(f"✅ Successfully added training run: {run_id}")
    print(f"📅 Date: {date}")
    print(f"🎯 Experiment: {experiment_name}")
    print(f"📊 mAP@50: {val_map50:.3f}")
    print(f"📁 Model: {model_path}")
    
    return training_run

def get_system_info():
    """Get current system information"""
    return {
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'gpu_memory_peak_gb': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
        'cpu_count': psutil.cpu_count(),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.version.cuda else 'N/A',
        'os_info': f"{sys.platform}"
    }

def auto_detect_training_runs():
    """
    Automatically detect and add training runs from the models directory
    Looks for model weights and attempts to extract information
    """
    base_path = Path(__file__).parent.parent
    models_dir = base_path / "model"
    
    if not models_dir.exists():
        print("❌ No models directory found")
        return
    
    print("🔍 Scanning for training runs...")
    
    # Look for model weight files
    model_files = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("best.pt"))
    
    if not model_files:
        print("❌ No model files found")
        return
    
    print(f"Found {len(model_files)} potential model files\n")
    
    for i, model_file in enumerate(model_files, 1):
        print(f"📁 Processing model {i}/{len(model_files)}: {model_file.relative_to(base_path)}")
        
        # Try to extract information from path
        parts = model_file.parts
        model_type = None
        experiment_name = None
        
        # Parse path structure: model/{type}/{name}/weights/best.pt
        if 'model' in parts and len(parts) > parts.index('model') + 2:
            model_type_idx = parts.index('models') + 1
            model_name_idx = parts.index('models') + 2
            
            if model_type_idx < len(parts):
                model_type = parts[model_type_idx]  # detection or classification
            if model_name_idx < len(parts):
                experiment_name = parts[model_name_idx]
        
        if not experiment_name:
            experiment_name = model_file.stem
        
        # Check if already in registry
        registry = TrainingRegistry("model/training_registry.json")
        existing_runs = registry.get_all_runs()
        
        already_logged = any(
            run.get('model_path', '') == str(model_file.absolute()) 
            for run in existing_runs
        )
        
        if already_logged:
            print(f"   ⏭️  Already in registry, skipping")
            continue
        
        # Try to find results directory
        results_dir = model_file.parent.parent / "training_run"
        if not results_dir.exists():
            results_dir = model_file.parent.parent
        
        # Try to find config
        config_file = model_file.parent.parent / "config" / "training_config.json"
        if not config_file.exists():
            config_file = model_file.parent.parent / "args.yaml"
        if not config_file.exists():
            config_file = model_file
        
        # Get file stats for date
        file_stat = model_file.stat()
        file_date = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Auto-detect architecture from path
        model_architecture = "YOLOv8"
        model_size = "n"
        if "yolov8s" in str(model_file):
            model_size = "s"
        elif "yolov8m" in str(model_file):
            model_size = "m"
        elif "yolov8l" in str(model_file):
            model_size = "l"
        
        # Ask user for missing info
        print(f"   📊 Model Type: {model_type or 'Unknown'}")
        print(f"   🎯 Experiment: {experiment_name}")
        print(f"   🗓️  Date: {file_date}")
        
        # Try to extract metrics from results if available
        val_map50 = 0.0
        val_precision = 0.0
        val_recall = 0.0
        
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                if not df.empty:
                    last_row = df.iloc[-1]
                    val_map50 = last_row.get('metrics/mAP50(B)', 0.0)
                    val_precision = last_row.get('metrics/precision(B)', 0.0)
                    val_recall = last_row.get('metrics/recall(B)', 0.0)
                    print(f"   📈 mAP@50: {val_map50:.3f}")
            except:
                pass
        
        # Ask if user wants to add this run
        response = input(f"   Add this run to registry? (y/n): ").strip().lower()
        
        if response == 'y':
            add_manual_training_run(
                model_path=str(model_file),
                results_path=str(results_dir),
                config_path=str(config_file),
                experiment_name=experiment_name,
                model_architecture=model_architecture,
                model_size=model_size,
                val_precision=val_precision,
                val_recall=val_recall,
                val_map50=val_map50,
                date=file_date,
                run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
            )
            print(f"   ✅ Added to registry\n")
        else:
            print(f"   ⏭️  Skipped\n")

def main():
    parser = argparse.ArgumentParser(description='Add previous training runs to registry')
    parser.add_argument('--auto', action='store_true', help='Auto-detect and add all training runs')
    parser.add_argument('--model-path', type=str, help='Path to model weights')
    parser.add_argument('--results-path', type=str, help='Path to results directory')
    parser.add_argument('--config-path', type=str, help='Path to config file')
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--image-size', type=int, default=640, help='Image size')
    parser.add_argument('--val-map50', type=float, default=0.0, help='Validation mAP@50')
    parser.add_argument('--val-precision', type=float, default=0.0, help='Validation precision')
    parser.add_argument('--val-recall', type=float, default=0.0, help='Validation recall')
    parser.add_argument('--date', type=str, help='Date string (YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    if args.auto:
        auto_detect_training_runs()
    elif args.model_path and args.experiment_name:
        # Manual single run addition
        model_path = args.model_path
        results_path = args.results_path or model_path
        config_path = args.config_path or model_path
        
        add_manual_training_run(
            model_path=model_path,
            results_path=results_path,
            config_path=config_path,
            experiment_name=args.experiment_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            val_map50=args.val_map50,
            val_precision=args.val_precision,
            val_recall=args.val_recall,
            date=args.date
        )
    else:
        print("Usage:")
        print("  Auto-detect all runs: python add_previous_run.py --auto")
        print("  Add single run: python add_previous_run.py --model-path PATH --experiment-name NAME")
        print("\nExample:")
        print("  python add_previous_run.py --model-path model/detection/yolov8s_enhanced/weights/best.pt --experiment-name 'YOLOv8s_Enhanced' --val-map50 0.937")

if __name__ == '__main__':
    main()