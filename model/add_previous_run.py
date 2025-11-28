#!/usr/bin/env python3
"""
Manually add the previous successful training run to the registry
"""

from datetime import datetime
from pathlib import Path
import torch
import platform
import psutil
from .training_registry import get_registry, TrainingRun

def add_previous_run():
    """Add the previously completed training run to the registry"""
    
    print("Adding previous successful training run to registry...")
    
    # Create TrainingRun object with data from your successful training
    run = TrainingRun(
        # Identification
        run_id=f"run_20251125_150400_manual_baseline",
        date="2025-11-25 15:04:00",
        experiment_name="Baseline_YOLOv8n",
        
        # Dataset Information
        dataset_name="straw-detect.v1-straw-detect.yolov8",
        dataset_size=392,  # 302 train + 90 val
        num_classes=1,
        class_names=['strawberry'],
        
        # Model Configuration
        model_architecture='YOLOv8',
        model_size='n',
        pretrained=True,
        
        # Training Hyperparameters
        batch_size=8,
        image_size=416,
        epochs_planned=50,
        epochs_completed=50,
        learning_rate=0.002,  # From optimizer: AdamW(lr=0.002)
        optimizer='AdamW',
        weight_decay=0.0005,
        
        # Performance Metrics (from your training output)
        train_box_loss=1.175,  # Final box loss from epoch 50
        train_cls_loss=0.5887,  # Final cls loss from epoch 50
        train_dfl_loss=0.8851,  # Final dfl loss from epoch 50
        val_precision=0.916,  # 91.6% precision
        val_recall=0.855,     # 85.5% recall
        val_map50=0.937,      # 93.7% mAP@50
        val_map50_95=0.581,   # 58.1% mAP@50-95
        
        # Training Metadata
        training_time_minutes=4.3,  # 0.079 hours = 4.3 minutes
        gpu_memory_peak_gb=1.44,  # From your GPU memory usage
        gpu_name='NVIDIA GeForce RTX 3050 Ti Laptop GPU',
        cpu_count=psutil.cpu_count(),
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        
        # System Info
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else 'N/A',
        os_info=f"{platform.system()} {platform.release()}",
        
        # Paths
        model_path=str(Path("model/weights/strawberry_yolov8n.pt")),
        results_path=str(Path("model/results/strawberry_detection")),
        config_path=str(Path("model/results/strawberry_detection/args.yaml")),
        
        # Status
        status='completed',
        early_stopped=False,
        best_epoch=50
    )
    
    # Add to registry
    registry = get_registry()
    registry.add_run(run)
    
    # Also export to CSV and generate summary
    registry.export_to_csv()
    registry.generate_summary_table()
    
    print("\n✓ Previous training run successfully added to registry!")
    print(f"✓ Registry file: {registry.registry_path}")
    print(f"✓ CSV exported: model/training_history.csv")
    print(f"✓ Markdown summary: model/training_summary.md")
    
    # Show what was added
    print(f"\n{'='*80}")
    print("ADDED RUN DETAILS:")
    print(f"{'='*80}")
    print(f"Run ID: {run.run_id}")
    print(f"Date: {run.date}")
    print(f"Experiment: {run.experiment_name}")
    print(f"Model: {run.model_architecture}-{run.model_size}")
    print(f"Performance: Precision={run.val_precision:.3f}, Recall={run.val_recall:.3f}, mAP@50={run.val_map50:.3f}")
    print(f"Training Time: {run.training_time_minutes:.1f} minutes")
    print(f"GPU: {run.gpu_name}")
    print(f"{'='*80}")

if __name__ == '__main__':
    add_previous_run()