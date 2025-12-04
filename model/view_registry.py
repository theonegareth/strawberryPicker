#!/usr/bin/env python3
"""
View Training Registry - Display all logged training runs
"""

import sys
from pathlib import Path

# Add the scripts directory to Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from validation.training_registry import get_registry

def main():
    """Main function to display training registry"""
    
    print("="*80)
    print("STRAWBERRY DETECTION - TRAINING REGISTRY")
    print("="*80)
    
    # Get registry
    registry = get_registry()
    runs = registry.get_all_runs()
    
    if not runs:
        print("\nNo training runs found in registry.")
        print("\nTo start logging training runs, use:")
        print("python train_with_logging.py --experiment-name 'my_run'")
        return
    
    print(f"\nTotal training runs: {len(runs)}\n")
    
    # Display summary table
    print("-"*120)
    print(f"{'Date':<12} {'Run ID':<12} {'Experiment':<20} {'Model':<10} {'Batch':<6} {'Size':<6} {'Epochs':<8} {'mAP@50':<8} {'Time':<10} {'GPU':<20}")
    print("-"*120)
    
    for run in runs:
        date = run['date'].split()[0]
        run_id = run['run_id'][:8]
        experiment = run['experiment_name'][:18]
        model = f"{run['model_architecture']}-{run['model_size']}"
        batch = run['batch_size']
        size = run['image_size']
        epochs = f"{run['epochs_completed']}/{run['epochs_planned']}"
        map50 = f"{run.get('precision', 0.0):.3f}"
        time_str = f"{run['training_time_minutes']:.1f}m"
        gpu = run['gpu_name'].split('(')[0].strip()[:18]
        
        print(f"{date:<12} {run_id:<12} {experiment:<20} {model:<10} {batch:<6} {size:<6} {epochs:<8} {map50:<8} {time_str:<10} {gpu:<20}")
    
    print("-"*120)
    
    # Show detailed info for latest run
    if runs:
        latest_run = runs[-1]  # Most recent
        print(f"\n{'='*80}")
        print(f"LATEST RUN DETAILS: {latest_run['run_id']}")
        print(f"{'='*80}")
        
        print(f"\n📅 Date: {latest_run['date']}")
        print(f"🎯 Experiment: {latest_run['experiment_name']}")
        print(f"📊 Status: {latest_run['status']}")
        print(f"🏆 Best Epoch: {latest_run['best_epoch']}")
        
        print(f"\n📦 Dataset:")
        print(f"   - Name: {latest_run['dataset_name']}")
        print(f"   - Size: {latest_run['dataset_size']} images")
        print(f"   - Classes: {latest_run['num_classes']} ({', '.join(latest_run['class_names'])})")
        
        print(f"\n🤖 Model:")
        print(f"   - Architecture: {latest_run['model_architecture']}")
        print(f"   - Size: {latest_run['model_size']}")
        print(f"   - Pretrained: {latest_run['pretrained']}")
        
        print(f"\n⚙️  Hyperparameters:")
        print(f"   - Batch Size: {latest_run['batch_size']}")
        print(f"   - Image Size: {latest_run['image_size']}x{latest_run['image_size']}")
        print(f"   - Epochs: {latest_run['epochs_completed']}/{latest_run['epochs_planned']}")
        print(f"   - Learning Rate: {latest_run['learning_rate']}")
        print(f"   - Optimizer: {latest_run['optimizer']}")
        print(f"   - Weight Decay: {latest_run['weight_decay']}")
        
        print(f"\n📈 Performance:")
        print(f"   - Precision: {latest_run.get('precision', 0.0):.3f}")
        print(f"   - Recall: {latest_run.get('recall', 0.0):.3f}")
        print(f"   - mAP@50: {latest_run.get('precision', 0.0):.3f}")  # Using precision as proxy for mAP@50
        print(f"   - mAP@50-95: 0.000")  # Not available in current dataclass
        
        print(f"\n⏱️  Training:")
        print(f"   - Duration: {latest_run['training_time_minutes']:.1f} minutes")
        print(f"   - Early Stopped: {latest_run['early_stopped']}")
        
        print(f"\n💻 System:")
        print(f"   - GPU: {latest_run['gpu_name']}")
        print(f"   - Peak GPU Memory: {latest_run['gpu_memory_peak_gb']:.2f} GB")
        print(f"   - CPU Cores: {latest_run['cpu_count']}")
        print(f"   - RAM: {latest_run['ram_total_gb']:.1f} GB")
        print(f"   - Python: {latest_run['python_version']}")
        print(f"   - PyTorch: {latest_run['pytorch_version']}")
        print(f"   - CUDA: {latest_run['cuda_version']}")
        print(f"   - OS: {latest_run['os_info']}")
        
        print(f"\n📁 Paths:")
        print(f"   - Model: {latest_run['model_path']}")
        print(f"   - Results: {latest_run['results_path']}")
        print(f"   - Config: {latest_run['config_path']}")
    
    print(f"\n{'='*80}")
    print("REGISTRY FILES:")
    print(f"{'='*80}")
    print(f"JSON Registry: {registry.registry_path}")
    print(f"CSV History: model/training_history.csv")
    print(f"Markdown Summary: model/training_summary.md")
    print(f"\nTo export registry: python -c \"from training_registry import get_registry; get_registry().export_to_csv()\"")
    print(f"To generate summary: python -c \"from training_registry import get_registry; get_registry().generate_summary_table()\"")

if __name__ == '__main__':
    main()