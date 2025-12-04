#!/usr/bin/env python3
"""
Optimized YOLOv8n training with enhanced augmentations and hyperparameters
Target: Improve mAP@50 from 0.989 to 0.992-0.995
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import torch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from validation.training_registry import TrainingRegistry, TrainingRun

def train_optimized_yolov8n(epochs=100, batch_size=32, image_size=640):
    """
    Train optimized YOLOv8n with enhanced augmentations and hyperparameters

    Args:
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 32)
        image_size: Input image size (default: 640)
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
    results_dir = base_path / "models" / "detection" / f"optimized_yolov8n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    weights_dir = results_dir / "weights"

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Training OPTIMIZED YOLOv8n on Kaggle strawberry dataset")
    print("="*70)
    print(f"📊 Dataset: {data_yaml}")
    print(f"📁 Results: {results_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch: {batch_size}, Image size: {image_size}")
    print()
    print("🎯 OPTIMIZATIONS APPLIED:")
    print("   ✅ Increased epochs: 100 (from 50)")
    print("   ✅ Larger batch size: 32 (from 16)")
    print("   ✅ Enhanced augmentations (rotation, translation, scale, shear)")
    print("   ✅ MixUp + Mosaic augmentations")
    print("   ✅ Cosine LR scheduling")
    print("   ✅ Label smoothing")
    print("   ✅ Dropout regularization")
    print("="*70)

    # Load model
    model = YOLO("yolov8n.pt")

    # Train with OPTIMIZED parameters
    print("🏋️  Starting optimized training...")
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
        # OPTIMIZED LEARNING PARAMETERS
        lr0=0.01,          # Higher initial LR for faster convergence
        lrf=0.01,          # Final LR
        cos_lr=True,       # Cosine LR scheduling for better convergence
        weight_decay=0.0005,
        warmup_epochs=5,   # Longer warmup
        # ENHANCED AUGMENTATIONS
        degrees=10,        # Random rotation ±10°
        translate=0.2,     # Random translation
        scale=0.5,         # Random scaling
        shear=2,           # Random shearing
        perspective=0.0001,# Perspective distortion
        hsv_h=0.1,         # HSV hue augmentation
        hsv_s=0.5,         # Saturation
        hsv_v=0.3,         # Brightness/Value
        mosaic=1.0,        # Mosaic augmentation
        mixup=0.1,         # MixUp augmentation
        # REGULARIZATION
        dropout=0.1,       # Add dropout
        label_smoothing=0.1, # Label smoothing for better generalization
        # OTHER PARAMETERS
        box=7.5,
        cls=0.5,
        dfl=1.5,
        flipud=0.0,        # No vertical flip (strawberries are orientation-sensitive)
        fliplr=0.5,        # Horizontal flip OK
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
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_optimized_yolov8n",
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name=f"optimized_yolov8n_batch{batch_size}_epochs{epochs}",
            model_type='detection',
            model_architecture='YOLOv8',
            model_size='n',
            pretrained=True,
            dataset_name='kaggle-strawberry-dataset-optimized',
            dataset_size=2500,
            num_classes=1,
            class_names=['strawberry'],
            batch_size=batch_size,
            image_size=image_size,
            epochs_planned=epochs,
            epochs_completed=len(df),
            learning_rate=0.01,  # Updated LR
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
        print("\n" + "="*70)
        print("📊 OPTIMIZED TRAINING RESULTS:")
        print(f"   mAP@50: {last_row.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   Precision: {last_row.get('metrics/precision(B)', 0):.3f}")
        print(f"   Recall: {last_row.get('metrics/recall(B)', 0):.3f}")
        print(f"   Training Time: {training_time:.1f} minutes")
        print(f"   Batch Size: {batch_size}")
        print(f"   Epochs: {len(df)}/{epochs}")
        print("="*70)

        # Compare with previous best
        print("\n📈 COMPARISON WITH PREVIOUS BEST:")
        print("   Previous: 0.989 mAP@50 (50 epochs, batch 16)")
        current_map50 = last_row.get('metrics/mAP50(B)', 0)
        improvement = current_map50 - 0.989
        print(".3f")
        if improvement > 0:
            print("   🎉 IMPROVEMENT ACHIEVED!")
        elif improvement == 0:
            print("   ➡️  Same performance (good stability)")
        else:
            print("   ⚠️  Slight decrease (may need tuning)")

        return results_dir

    except Exception as e:
        print(f"❌ Failed to add to registry: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train optimized YOLOv8n on Kaggle strawberry dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')

    args = parser.parse_args()

    # Run optimized training
    results_dir = train_optimized_yolov8n(
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz
    )

    if results_dir:
        print(f"\n🎉 Optimized training complete! Results: {results_dir}")
        print(f"📊 View registry: python model/view_registry.py")
        print(f"🔬 Validate model: python validate_kaggle_model.py")
    else:
        print("\n❌ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()