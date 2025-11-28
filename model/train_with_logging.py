#!/usr/bin/env python3
"""
Enhanced YOLOv8 Training Script with Comprehensive Logging
Supports: TensorBoard, CSV, W&B, and custom file logging
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import yaml
from typing import Dict, Any
from .training_registry import log_training_run, get_registry

def setup_logging(results_dir: Path, experiment_name: str, enable_wandb: bool = False):
    """Setup comprehensive logging for training"""
    
    # Create logs directory
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))
        print(f"✓ TensorBoard logging enabled: {logs_dir / 'tensorboard'}")
    except ImportError:
        tb_writer = None
        print("⚠ TensorBoard not available (pip install tensorboard)")
    
    # Setup W&B if requested
    wandb_run = None
    if enable_wandb:
        try:
            import wandb
            wandb.init(
                project="strawberry-detection",
                name=experiment_name,
                config={"model": "yolov8n", "task": "detection"}
            )
            wandb_run = wandb.run
            print(f"✓ Weights & Biases logging enabled")
        except ImportError:
            print("⚠ W&B not available (pip install wandb)")
        except Exception as e:
            print(f"⚠ W&B initialization failed: {e}")
    
    # Create custom log file
    log_file = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    return tb_writer, wandb_run, log_file

def log_to_file(log_file: Path, message: str, also_print: bool = True):
    """Log message to file and optionally print"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')
    
    if also_print:
        print(message)

def log_metrics(tb_writer, wandb_run, log_file, epoch: int, metrics: Dict[str, float]):
    """Log metrics to all enabled logging systems"""
    
    # Log to TensorBoard
    if tb_writer:
        for key, value in metrics.items():
            tb_writer.add_scalar(key, value, epoch)
    
    # Log to W&B
    if wandb_run:
        wandb_run.log(metrics, step=epoch)
    
    # Log to file
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    log_to_file(log_file, f"Epoch {epoch:03d} | {metrics_str}")

def train_model_with_logging(data_yaml, weights_dir, results_dir, 
                           epochs=100, img_size=640, batch_size=16,
                           enable_wandb=False, experiment_name=None):
    """Train YOLOv8 model with comprehensive logging"""
    
    if experiment_name is None:
        experiment_name = f"strawberry_yolov8n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup logging
    tb_writer, wandb_run, log_file = setup_logging(results_dir, experiment_name, enable_wandb)
    
    log_to_file(log_file, "="*60)
    log_to_file(log_file, f"Training Experiment: {experiment_name}")
    log_to_file(log_file, f"Started at: {datetime.now()}")
    log_to_file(log_file, "="*60)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        log_to_file(log_file, "ERROR: ultralytics not installed", also_print=True)
        return None, None
    
    # Environment info
    env_info = {
        'has_gpu': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    log_to_file(log_file, f"Environment: {env_info}")
    
    # Load model
    log_to_file(log_file, "Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': '0' if env_info['has_gpu'] else 'cpu',
        'project': str(results_dir),
        'name': experiment_name,
        'exist_ok': True,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': True,
        'plots': True,
        'verbose': True
    }
    
    log_to_file(log_file, f"Training arguments: {train_args}")
    
    # Train the model
    log_to_file(log_file, "\n" + "="*60)
    log_to_file(log_file, "TRAINING STARTED")
    log_to_file(log_file, "="*60)
    
    start_time = datetime.now()
    
    # Custom callback for logging
    def on_train_epoch_end(trainer):
        """Callback at end of each epoch"""
        # Safely get learning rate (it's a dictionary in newer ultralytics versions)
        try:
            if hasattr(trainer, 'lr') and trainer.lr:
                if isinstance(trainer.lr, dict) and 'lr/pg0' in trainer.lr:
                    lr = float(trainer.lr['lr/pg0'])  # Get first parameter group LR
                elif isinstance(trainer.lr, (list, tuple)) and len(trainer.lr) > 0:
                    lr = float(trainer.lr[0])
                elif isinstance(trainer.lr, (float, int)):
                    lr = float(trainer.lr)
                else:
                    lr = 0.0
            else:
                lr = 0.0
        except Exception as e:
            print(f"Warning: Could not get learning rate: {e}")
            lr = 0.0
        
        # Get metrics with proper keys (they have (B) suffix in validation)
        # Training losses are not available in on_train_epoch_end, only validation metrics
        metrics = {
            'metrics/precision': trainer.metrics.get('metrics/precision(B)', 0),
            'metrics/recall': trainer.metrics.get('metrics/recall(B)', 0),
            'metrics/mAP50': trainer.metrics.get('metrics/mAP50(B)', 0),
            'metrics/mAP50-95': trainer.metrics.get('metrics/mAP50-95(B)', 0),
            'val/box_loss': trainer.metrics.get('val/box_loss', 0),
            'val/cls_loss': trainer.metrics.get('val/cls_loss', 0),
            'val/dfl_loss': trainer.metrics.get('val/dfl_loss', 0),
            'lr/pg0': lr
        }
        
        log_metrics(tb_writer, wandb_run, log_file, trainer.epoch + 1, metrics)
    
    # Add callback to model
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    
    # Add callback for end of training
    def on_train_end(trainer):
        """Log training to registry at end"""
        try:
            log_training_run(trainer, experiment_name)
        except Exception as e:
            print(f"Warning: Could not log to registry: {e}")
    
    model.add_callback('on_train_end', on_train_end)
    
    # Run training
    results = model.train(**train_args)
    
    # Calculate training time
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    # Save final model
    final_model_path = weights_dir / f'{experiment_name}.pt'
    model.save(str(final_model_path))
    
    # Log completion
    log_to_file(log_file, "\n" + "="*60)
    log_to_file(log_file, "TRAINING COMPLETED")
    log_to_file(log_file, "="*60)
    log_to_file(log_file, f"Duration: {training_duration:.3f} hours")
    log_to_file(log_file, f"Final model saved to: {final_model_path}")
    log_to_file(log_file, f"Results saved to: {results_dir / experiment_name}")
    log_to_file(log_file, f"Logs saved to: {log_file}")
    
    # Close loggers
    if tb_writer:
        tb_writer.close()
        log_to_file(log_file, "✓ TensorBoard writer closed")
    
    if wandb_run:
        wandb_run.finish()
        log_to_file(log_file, "✓ W&B run finished")
    
    return results, final_model_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with comprehensive logging')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--experiment-name', type=str, help='Name for this experiment')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX after training')
    parser.add_argument('--validate-only', action='store_true', help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        from train_yolov8 import setup_paths, validate_dataset, export_model
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
        
        # Train model with logging
        results, model_path = train_model_with_logging(
            data_yaml=data_yaml,
            weights_dir=paths['weights_dir'],
            results_dir=paths['results_dir'],
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            enable_wandb=args.wandb,
            experiment_name=args.experiment_name
        )
        
        # Export to ONNX if requested
        if args.export_onnx:
            export_model(model_path, paths['weights_dir'])
        
        print("\n✓ Training pipeline with logging completed successfully!")
        print(f"\nTo view TensorBoard logs, run:")
        print(f"tensorboard --logdir {paths['results_dir']}/logs/tensorboard")
        
        if args.wandb:
            print(f"\nTo view W&B logs, visit: https://wandb.ai/your-project")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()