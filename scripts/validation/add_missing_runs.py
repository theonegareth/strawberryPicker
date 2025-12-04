#!/usr/bin/env python3
"""
Add missing training runs to the registry
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.training_registry import TrainingRun, TrainingRegistry

def add_missing_runs():
    """Add missing training runs to registry"""
    
    registry = TrainingRegistry("model/training_registry.json")
    
    # Define missing runs based on model files found
    missing_runs = [
        {
            "run_id": "run_20251202_153433_yolov8s_improved_detection",
            "date": "2025-12-02 15:34:33",
            "experiment_name": "yolov8s_improved_detection_v2_20251202_153433",
            "model_type": "detection",
            "model_architecture": "YOLOv8",
            "model_size": "s",
            "pretrained": True,
            "dataset_name": "straw-detect.v1-straw-detect.yolov8",
            "dataset_size": 392,
            "num_classes": 1,
            "class_names": ["strawberry"],
            "batch_size": 16,
            "image_size": 640,
            "epochs_planned": 100,
            "epochs_completed": 100,
            "learning_rate": 0.002,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "test_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "macro_avg_precision": 0.0,
            "macro_avg_recall": 0.0,
            "macro_avg_f1": 0.0,
            "weighted_avg_precision": 0.0,
            "weighted_avg_recall": 0.0,
            "weighted_avg_f1": 0.0,
            "confusion_matrix": None,
            "training_time_minutes": 0.0,
            "early_stopped": False,
            "best_epoch": 100,
            "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            "gpu_memory_peak_gb": 0.0,
            "cpu_count": 20,
            "ram_total_gb": 15.471839904785156,
            "python_version": "3.12.3",
            "tensorflow_version": "N/A",
            "pytorch_version": "2.9.1+cu128",
            "cuda_version": "12.8",
            "os_info": "linux",
            "model_path": "models/detection/yolov8s_improved_detection_v2_20251202_153433/weights/best.pt",
            "results_path": "models/detection/yolov8s_improved_detection_v2_20251202_153433",
            "config_path": "models/detection/yolov8s_improved_detection_v2_20251202_153433/config/training_config.json",
            "status": "completed"
        },
        {
            "run_id": "run_20251203_130255_yolov8n_kaggle_2500images",
            "date": "2025-12-03 13:02:55",
            "experiment_name": "yolov8n_kaggle_2500images_trained_20251203_130255",
            "model_type": "detection",
            "model_architecture": "YOLOv8",
            "model_size": "n",
            "pretrained": True,
            "dataset_name": "kaggle-fruit-ripeness-2500images",
            "dataset_size": 2500,
            "num_classes": 1,
            "class_names": ["strawberry"],
            "batch_size": 16,
            "image_size": 640,
            "epochs_planned": 100,
            "epochs_completed": 100,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "test_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "macro_avg_precision": 0.0,
            "macro_avg_recall": 0.0,
            "macro_avg_f1": 0.0,
            "weighted_avg_precision": 0.0,
            "weighted_avg_recall": 0.0,
            "weighted_avg_f1": 0.0,
            "confusion_matrix": None,
            "training_time_minutes": 0.0,
            "early_stopped": False,
            "best_epoch": 100,
            "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
            "gpu_memory_peak_gb": 0.0,
            "cpu_count": 20,
            "ram_total_gb": 15.471839904785156,
            "python_version": "3.12.3",
            "tensorflow_version": "N/A",
            "pytorch_version": "2.9.1+cu128",
            "cuda_version": "12.8",
            "os_info": "linux",
            "model_path": "models/detection/yolov8n_kaggle_2500images_trained_20251203_130255/weights/best.pt",
            "results_path": "models/detection/yolov8n_kaggle_2500images_trained_20251203_130255",
            "config_path": "models/detection/yolov8n_kaggle_2500images_trained_20251203_130255/config/training_config.json",
            "status": "completed"
        }
    ]
    
    # Add each missing run
    added_count = 0
    for run_data in missing_runs:
        # Check if run already exists
        existing_runs = registry.get_all_runs()
        if any(run['run_id'] == run_data['run_id'] for run in existing_runs):
            print(f"⚠️  Run {run_data['run_id']} already exists, skipping...")
            continue
        
        # Create TrainingRun object
        training_run = TrainingRun(**run_data)
        
        # Add to registry
        registry.add_run(training_run)
        added_count += 1
        print(f"✅ Added run: {run_data['run_id']}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Added {added_count} missing training runs")
    print(f"{'='*60}")
    
    # Show updated registry
    print(f"\nUpdated registry has {len(registry.get_all_runs())} total runs")
    
    return added_count

if __name__ == "__main__":
    add_missing_runs()