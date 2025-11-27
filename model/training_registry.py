#!/usr/bin/env python3
"""
Training Registry System for YOLOv8 Strawberry Detection
Automatically logs all training runs with comprehensive metadata
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import platform
import psutil
import torch
from dataclasses import dataclass, asdict

@dataclass
class TrainingRun:
    """Data structure for a single training run"""
    # Identification
    run_id: str
    date: str
    experiment_name: str
    
    # Dataset Information
    dataset_name: str
    dataset_size: int
    num_classes: int
    class_names: List[str]
    
    # Model Configuration
    model_architecture: str
    model_size: str  # n, s, m, l, x
    pretrained: bool
    
    # Training Hyperparameters
    batch_size: int
    image_size: int
    epochs_planned: int
    epochs_completed: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    
    # Performance Metrics
    train_box_loss: float
    train_cls_loss: float
    train_dfl_loss: float
    val_precision: float
    val_recall: float
    val_map50: float
    val_map50_95: float
    
    # Training Metadata
    training_time_minutes: float
    gpu_memory_peak_gb: float
    gpu_name: str
    cpu_count: int
    ram_total_gb: float
    
    # System Info
    python_version: str
    pytorch_version: str
    cuda_version: str
    os_info: str
    
    # Paths
    model_path: str
    results_path: str
    config_path: str
    
    # Status
    status: str  # completed, interrupted, failed
    early_stopped: bool
    best_epoch: int

class TrainingRegistry:
    """Manages training run registry and logging"""
    
    def __init__(self, registry_path: str = "model/training_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs = self._load_registry()
    
    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load existing registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                return []
        return []
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.runs, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def add_run(self, run: TrainingRun):
        """Add a new training run to the registry"""
        run_dict = asdict(run)
        self.runs.append(run_dict)
        self._save_registry()
        print(f"✓ Training run logged: {run.run_id}")
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID"""
        for run in self.runs:
            if run['run_id'] == run_id:
                return run
        return None
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all training runs"""
        return self.runs
    
    def export_to_csv(self, csv_path: str = "model/training_history.csv"):
        """Export registry to CSV format"""
        if not self.runs:
            print("No runs to export")
            return
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        columns = [
            'run_id', 'date', 'experiment_name', 'dataset_name', 'dataset_size',
            'model_architecture', 'model_size', 'batch_size', 'image_size',
            'epochs_planned', 'epochs_completed', 'learning_rate', 'optimizer',
            'val_precision', 'val_recall', 'val_map50', 'val_map50_95',
            'training_time_minutes', 'gpu_name', 'status', 'best_epoch'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for run in self.runs:
                # Filter only the columns we want
                row = {col: run.get(col, '') for col in columns}
                writer.writerow(row)
        
        print(f"✓ Registry exported to CSV: {csv_path}")
    
    def generate_summary_table(self, output_path: str = "model/training_summary.md"):
        """Generate a markdown summary table"""
        if not self.runs:
            print("No runs to summarize")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Training History Summary\n\n")
            f.write("## Overview\n")
            f.write(f"Total training runs: {len(self.runs)}\n\n")
            
            # Group by date
            runs_by_date = {}
            for run in self.runs:
                date = run['date'].split(' ')[0]  # Get date part only
                if date not in runs_by_date:
                    runs_by_date[date] = []
                runs_by_date[date].append(run)
            
            # Write table for each date
            for date, runs in sorted(runs_by_date.items(), reverse=True):
                f.write(f"### {date}\n\n")
                f.write("| Run ID | Dataset | Model | Batch | Input Size | LR/Epochs | Optimizer | Epochs | Precision | Recall | mAP@50 | Training Time | GPU |\n")
                f.write("|--------|---------|-------|-------|------------|-----------|-----------|--------|-----------|--------|--------|---------------|-----|\n")
                
                for run in runs:
                    f.write(f"| {run['run_id'][:8]} | {run['dataset_name']} | {run['model_architecture']}-{run['model_size']} | {run['batch_size']} | {run['image_size']}x{run['image_size']} | {run['learning_rate']}/{run['epochs_planned']} | {run['optimizer']} | {run['epochs_completed']}/{run['epochs_planned']} | {run['val_precision']:.3f} | {run['val_recall']:.3f} | {run['val_map50']:.3f} | {run['training_time_minutes']:.1f} min | {run['gpu_name'].split('(')[0].strip()} |\n")
                
                f.write("\n")
            
            # Add detailed section
            f.write("## Detailed Run Information\n\n")
            for run in self.runs:
                f.write(f"### Run: {run['run_id']}\n")
                f.write(f"- **Date**: {run['date']}\n")
                f.write(f"- **Experiment**: {run['experiment_name']}\n")
                f.write(f"- **Status**: {run['status']}\n")
                f.write(f"- **Dataset**: {run['dataset_name']} ({run['dataset_size']} images, {run['num_classes']} classes)\n")
                f.write(f"- **Model**: {run['model_architecture']}-{run['model_size']} (Pretrained: {run['pretrained']})\n")
                f.write(f"- **Hyperparameters**: Batch={run['batch_size']}, Image Size={run['image_size']}, LR={run['learning_rate']}, Optimizer={run['optimizer']}\n")
                f.write(f"- **Performance**: Precision={run['val_precision']:.3f}, Recall={run['val_recall']:.3f}, mAP@50={run['val_map50']:.3f}, mAP@50-95={run['val_map50_95']:.3f}\n")
                f.write(f"- **Training**: {run['epochs_completed']}/{run['epochs_planned']} epochs, {run['training_time_minutes']:.1f} minutes\n")
                f.write(f"- **System**: {run['gpu_name']}, Peak GPU Memory: {run['gpu_memory_peak_gb']:.2f} GB\n")
                f.write(f"- **Paths**: Model={run['model_path']}, Results={run['results_path']}\n")
                f.write("\n")

def collect_system_info() -> Dict[str, Any]:
    """Collect comprehensive system information"""
    return {
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
        'cpu_count': psutil.cpu_count(),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'os_info': f"{platform.system()} {platform.release()}",
        'hostname': platform.node()
    }

def create_training_run_from_yolo(trainer, experiment_name: str = None) -> TrainingRun:
    """Create TrainingRun object from YOLO trainer object"""
    
    if experiment_name is None:
        experiment_name = f"yolov8n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Collect system info
    system_info = collect_system_info()
    
    # Extract metrics from trainer
    metrics = trainer.metrics
    
    # Create run ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
    
    return TrainingRun(
        run_id=run_id,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        experiment_name=experiment_name,
        
        # Dataset info
        dataset_name=trainer.data.get('dataset_name', 'strawberry-dataset'),
        dataset_size=len(trainer.train_loader.dataset) + len(trainer.val_loader.dataset),
        num_classes=trainer.data.get('nc', 1),
        class_names=trainer.data.get('names', ['strawberry']),
        
        # Model info
        model_architecture='YOLOv8',
        model_size='n',  # Could extract from model.yaml
        pretrained=True,
        
        # Hyperparameters
        batch_size=trainer.batch_size,
        image_size=trainer.args.imgsz,
        epochs_planned=trainer.epochs,
        epochs_completed=trainer.epoch + 1,
        learning_rate=trainer.lr[0] if trainer.lr else 0.01,
        optimizer=trainer.args.optimizer,
        weight_decay=trainer.args.weight_decay,
        
        # Performance metrics
        train_box_loss=metrics.get('train/box_loss', 0),
        train_cls_loss=metrics.get('train/cls_loss', 0),
        train_dfl_loss=metrics.get('train/dfl_loss', 0),
        val_precision=metrics.get('metrics/precision', 0),
        val_recall=metrics.get('metrics/recall', 0),
        val_map50=metrics.get('metrics/mAP50', 0),
        val_map50_95=metrics.get('metrics/mAP50-95', 0),
        
        # Training metadata
        training_time_minutes=getattr(trainer, 'training_time', 0) / 60,
        gpu_memory_peak_gb=trainer.gpu_memory_peak / (1024**3) if hasattr(trainer, 'gpu_memory_peak') else 0,
        gpu_name=system_info['gpu_name'],
        cpu_count=system_info['cpu_count'],
        ram_total_gb=system_info['ram_total_gb'],
        
        # System info
        python_version=system_info['python_version'],
        pytorch_version=system_info['pytorch_version'],
        cuda_version=system_info['cuda_version'],
        os_info=system_info['os_info'],
        
        # Paths
        model_path=str(trainer.save_dir / 'weights' / 'best.pt'),
        results_path=str(trainer.save_dir),
        config_path=str(trainer.save_dir / 'args.yaml'),
        
        # Status
        status='completed' if trainer.epoch + 1 >= trainer.epochs else 'interrupted',
        early_stopped=trainer.stopper.possible_stop if hasattr(trainer, 'stopper') else False,
        best_epoch=trainer.best_epoch if hasattr(trainer, 'best_epoch') else trainer.epoch
    )

# Global registry instance
_registry_instance = None

def get_registry() -> TrainingRegistry:
    """Get or create the global training registry"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TrainingRegistry()
    return _registry_instance

def log_training_run(trainer, experiment_name: str = None):
    """Log a completed training run to the registry"""
    registry = get_registry()
    run = create_training_run_from_yolo(trainer, experiment_name)
    registry.add_run(run)
    
    # Also export to CSV for easy viewing
    registry.export_to_csv()
    
    # Generate summary table
    registry.generate_summary_table()
    
    print(f"\n✓ Training run logged to registry: {run.run_id}")
    print(f"✓ Registry updated: {registry.registry_path}")
    print(f"✓ CSV exported: model/training_history.csv")
    print(f"✓ Summary generated: model/training_summary.md")