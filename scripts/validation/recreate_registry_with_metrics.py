#!/usr/bin/env python3
"""
Recreate training registry with actual metrics from results.csv files
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.training_registry import TrainingRegistry, TrainingRun

def extract_metrics_from_csv(csv_path):
    """Extract final metrics from results.csv file"""
    try:
        df = pd.read_csv(csv_path)

        # Get the last row (final epoch)
        last_row = df.iloc[-1]

        metrics = {
            'training_time_minutes': last_row.get('time', 0) / 60,  # Convert seconds to minutes
            'val_precision': last_row.get('metrics/precision(B)', 0.0),
            'val_recall': last_row.get('metrics/recall(B)', 0.0),
            'val_map50': last_row.get('metrics/mAP50(B)', 0.0),
            'val_map50_95': last_row.get('metrics/mAP50-95(B)', 0.0),
            'train_box_loss': last_row.get('train/box_loss', 0.0),
            'train_cls_loss': last_row.get('train/cls_loss', 0.0),
            'train_dfl_loss': last_row.get('train/dfl_loss', 0.0),
            'val_box_loss': last_row.get('val/box_loss', 0.0),
            'val_cls_loss': last_row.get('val/cls_loss', 0.0),
            'val_dfl_loss': last_row.get('val/dfl_loss', 0.0),
            'epochs_completed': len(df),
            'best_epoch': len(df)  # Assume last epoch is best
        }

        return metrics

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def find_results_csv_for_model(model_path):
    """Find the corresponding results.csv for a model"""
    model_path = Path(model_path)

    # Try different possible locations for results.csv
    possible_locations = [
        model_path.parent.parent / "results.csv",  # Same level as weights dir
        model_path.parent / "results.csv",         # In weights dir
        model_path.parent.parent.parent / "results.csv",  # Up two levels
    ]

    for csv_path in possible_locations:
        if csv_path.exists():
            return csv_path

    # Try to find results.csv in the model directory structure
    model_dir = model_path.parent
    while model_dir != model_dir.parent:  # Not root
        csv_path = model_dir / "results.csv"
        if csv_path.exists():
            return csv_path
        model_dir = model_dir.parent

    return None

def recreate_registry_with_metrics():
    """Recreate registry with actual metrics from results.csv files"""

    # Get all model files
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

    print(f"🔍 Found {len(model_files)} model files")
    print("="*60)

    # Create new registry
    registry = TrainingRegistry("model/training_registry.json")

    # Clear existing runs
    registry.runs = []
    registry._save_registry()

    added_count = 0
    temp_runs = []  # Temporary list to handle deduplication

    for model_info in model_files:
        model_path = model_info["path"]

        # Skip if not a final best.pt or model.pt (skip checkpoints)
        if "epoch" in model_path and "best.pt" not in model_path and "model.pt" not in model_path:
            continue

        # Find results.csv
        csv_path = find_results_csv_for_model(model_path)

        if not csv_path:
            print(f"⚠️  No results.csv found for {model_info['name']} - using defaults")
            # Create run with default/placeholder values
            metrics = {
                'training_time_minutes': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_map50': 0.0,
                'val_map50_95': 0.0,
                'epochs_completed': 50,
                'best_epoch': 50
            }
        else:
            # Extract metrics
            metrics = extract_metrics_from_csv(csv_path)
            if not metrics:
                print(f"❌ Failed to extract metrics for {model_info['name']}")
                continue

        # Extract model info
        model_name = model_info['name']
        architecture = "YOLOv8"
        model_size = "n"  # Default

        if "yolov8n" in model_name.lower():
            model_size = "n"
        elif "yolov8s" in model_name.lower():
            model_size = "s"
        elif "yolov8m" in model_name.lower():
            model_size = "m"

        dataset_name = "straw-detect.v1-straw-detect.yolov8"
        dataset_size = 392

        if "kaggle" in model_name.lower():
            dataset_name = "kaggle-fruit-ripeness"
            dataset_size = 2500

        # Get actual training date from model file modification time
        model_file_path = Path(model_path)
        if model_file_path.exists():
            file_mtime = model_file_path.stat().st_mtime
            training_date = datetime.fromtimestamp(file_mtime)
            date = training_date.strftime("%Y-%m-%d %H:%M:%S")
            run_id = f"run_{training_date.strftime('%Y%m%d_%H%M%S')}_{model_name[:20]}"
        else:
            # Fallback to current time if file doesn't exist
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name[:20]}"
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create training run with correct parameters
        training_run = TrainingRun(
            run_id=run_id,
            date=date,
            experiment_name=model_name,
            model_type='detection',
            model_architecture=architecture,
            model_size=model_size,
            pretrained=True,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            num_classes=1,
            class_names=['strawberry'],
            batch_size=16,
            image_size=640,
            epochs_planned=metrics['epochs_completed'],
            epochs_completed=metrics['epochs_completed'],
            learning_rate=0.001,
            optimizer='AdamW',
            weight_decay=0.0005,
            precision=metrics['val_precision'],
            recall=metrics['val_recall'],
            training_time_minutes=metrics['training_time_minutes'],
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

        # Add to temporary list
        temp_runs.append(training_run)

        status = "✅" if csv_path else "⚠️ "
        print(f"{status} Added {model_name}:")
        print(".3f")
        print(".1f")
        print()

    # Deduplicate runs by experiment name, keeping the one with real metrics
    unique_runs = {}
    for run in temp_runs:
        exp_name = run.experiment_name
        if exp_name not in unique_runs:
            unique_runs[exp_name] = run
        else:
            # Keep the run with real metrics (non-zero precision) over placeholder
            existing_run = unique_runs[exp_name]
            if run.precision > 0 and existing_run.precision == 0:
                unique_runs[exp_name] = run
            elif run.precision == 0 and existing_run.precision == 0:
                # If both have placeholder metrics, keep the more recent one
                if run.date > existing_run.date:
                    unique_runs[exp_name] = run

    # Add unique runs to registry
    for run in unique_runs.values():
        registry.add_run(run)
        added_count += 1

    print("="*60)
    print(f"SUMMARY: Recreated registry with {added_count} unique training runs")
    print(f"Original model files found: {len(model_files)}")
    print(f"Duplicates removed: {len(model_files) - len(unique_runs)}")
    print(f"Models with real metrics: {sum(1 for r in unique_runs.values() if r.precision > 0)}")
    print(f"Models with placeholder metrics: {sum(1 for r in unique_runs.values() if r.precision == 0)}")

    return added_count

if __name__ == "__main__":
    recreate_registry_with_metrics()