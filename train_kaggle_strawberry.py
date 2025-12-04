#!/usr/bin/env python3
"""
Train YOLOv8n on Kaggle strawberry dataset with automatic registry logging
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import torch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from validation.training_registry import TrainingRegistry, TrainingRun

def train_kaggle_strawberry(model_size='n', epochs=100, batch_size=16, image_size=640):
    """
    Train YOLOv8 on Kaggle strawberry dataset
    
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
    """
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not found. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Setup paths
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    data_yaml = base_path / "model" / "dataset_strawberry_kaggle" / "data.yaml"
    
    # Save to models/detection/ folder structure
    results_dir = base_path / "models" / "detection" / f"kaggle_strawberry_yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    weights_dir = results_dir / "weights"
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Training YOLOv8{model_size} on Kaggle strawberry dataset")
    print(f"📊 Dataset: {data_yaml}")
    print(f"📁 Results: {results_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch: {batch_size}, Image size: {image_size}")
    print("="*60)
    
    # Load model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train
    print("🏋️  Starting training...")
    start_time = datetime.now()
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        project=str(results_dir.parent),
        name=results_dir.name,
        exist_ok=True,
        save=True,
        save_period=10,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Check if results.csv was created
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        print(f"❌ results.csv not found at {results_csv}")
        return None
    
    print(f"✅ Training completed in {training_time:.1f} minutes")
    print(f"📈 Results saved to: {results_csv}")
    
    # Add to registry
    print("\n📝 Adding to training registry...")
    try:
        registry = TrainingRegistry("model/training_registry.json")
        
        # Extract metrics from final epoch
        import pandas as pd
        df = pd.read_csv(results_csv)
        last_row = df.iloc[-1]
        
        # Create training run
        training_run = TrainingRun(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_kaggle_yolov8{model_size}",
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name=f"kaggle_strawberry_yolov8{model_size}",
            model_type='detection',
            model_architecture='YOLOv8',
            model_size=model_size,
            pretrained=True,
            dataset_name='kaggle-strawberry-dataset',
            dataset_size=2500,
            num_classes=1,
            class_names=['strawberry'],
            batch_size=batch_size,
            image_size=image_size,
            epochs_planned=epochs,
            epochs_completed=len(df),
            learning_rate=0.001,
            optimizer='AdamW',
            weight_decay=0.0005,
            precision=last_row.get('metrics/precision(B)', 0.0),
            recall=last_row.get('metrics/recall(B)', 0.0),
            training_time_minutes=training_time,
            gpu_name='NVIDIA GeForce RTX 3050 Ti Laptop GPU' if torch.cuda.is_available() else 'CPU',
            gpu_memory_peak_gb=0.0,
            cpu_count=20,
            ram_total_gb=15.5,
            python_version='3.12.3',
            pytorch_version=torch.__version__,
            cuda_version='12.8' if torch.cuda.is_available() else 'N/A',
            os_info='linux',
            model_path=str(weights_dir / "best.pt"),
            results_path=str(results_dir),
            config_path=str(results_dir / "args.yaml"),
            status='completed'
        )
        
        registry.add_run(training_run)
        print(f"✅ Added to registry: {training_run.experiment_name}")
        
        # Show final metrics
        print("\n" + "="*60)
        print("📊 FINAL METRICS:")
        print(f"   mAP@50: {last_row.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   Precision: {last_row.get('metrics/precision(B)', 0):.3f}")
        print(f"   Recall: {last_row.get('metrics/recall(B)', 0):.3f}")
        print(f"   Training Time: {training_time:.1f} minutes")
        print("="*60)
        
        return results_dir
        
    except Exception as e:
        print(f"❌ Failed to add to registry: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on Kaggle strawberry dataset')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (default: n)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    
    args = parser.parse_args()
    
    # Run training
    results_dir = train_kaggle_strawberry(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz
    )
    
    if results_dir:
        print(f"\n🎉 Training complete! Results: {results_dir}")
        print(f"📊 View registry: python model/view_registry.py")
    else:
        print("\n❌ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()