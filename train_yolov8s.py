#!/usr/bin/env python3
"""
Train YOLOv8s model for better strawberry detection accuracy
This model is larger than YOLOv8n but provides better mAP
"""

import sys
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

def train_yolov8s():
    print("=" * 70)
    print("Training YOLOv8s Model for Better Strawberry Detection")
    print("=" * 70)
    
    # Configuration
    model_size = 's'  # YOLOv8s (small) - larger than 'n' (nano)
    dataset_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset_strawberry_kaggle/data.yaml"
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the Kaggle dataset is properly set up")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Create results directory
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = base_path / "model" / "detection" / f"kaggle_strawberry_yolov8s_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}")
    
    # Load YOLOv8s model (pretrained on COCO)
    print(f"\n🤖 Loading YOLOv8{model_size} model...")
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Model info
    print(f"✅ Model loaded: YOLOv8{model_size}")
    print(f"📏 Model size: {model_size.upper()}")
    
    # Training configuration optimized for RTX 3050 Ti (4GB VRAM)
    training_config = {
        'data': str(dataset_path),
        'epochs': 150,  # More epochs for better convergence
        'batch': 4,     # Reduced batch size for 4GB VRAM
        'imgsz': 640,
        'optimizer': 'AdamW',
        'lr0': 0.001,   # Lower learning rate for stability
        'lrf': 0.0001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,     # Increased box loss weight
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,    # More rotation augmentation
        'translate': 0.2,
        'scale': 0.6,       # More scale variation
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,       # Add mixup augmentation
        'copy_paste': 0.1,  # Add copy-paste augmentation
        'name': f'kaggle_strawberry_yolov8s_{timestamp}',
        'project': str(base_path / "model" / "detection"),
        'exist_ok': False,
        'pretrained': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,
        'rect': False,
        'cos_lr': True,     # Cosine LR scheduling
        'close_mosaic': 10,
        'resume': False,
        'amp': True,        # Automatic Mixed Precision (saves VRAM)
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': True,  # Enable multi-scale training
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        'half': True,
        'dnn': False,
        'plots': True,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': None,
        'workspace': 2,     # Reduced workspace for VRAM constraints
        'nms': False,
        'lr0': 0.001,
        'lrf': 0.0001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.6,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'auto_augment': 'randaugment',  # Use RandAugment
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    print("\n⚙️  Training Configuration:")
    for key, value in training_config.items():
        if key in ['data', 'project', 'name']:
            print(f"   {key}: {value}")
        elif isinstance(value, (int, float)) and value > 0:
            print(f"   {key}: {value}")
    
    # Start training
    print(f"\n🚀 Starting YOLOv8{model_size} training...")
    print("⏱️  This will take several hours...")
    print("💡 Training improvements over YOLOv8n:")
    print("   • Larger model capacity (s vs n)")
    print("   • More training epochs (150 vs 100)")
    print("   • Enhanced augmentations (mixup, copy-paste, RandAugment)")
    print("   • Multi-scale training")
    print("   • Cosine learning rate scheduling")
    print("   • Lower learning rate for better convergence")
    
    try:
        results = model.train(**training_config)
        
        print("\n" + "=" * 70)
        print("✅ Training Completed Successfully!")
        print("=" * 70)
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\n📊 Final Metrics:")
            for metric, value in results.results_dict.items():
                print(f"   {metric}: {value:.4f}")
        
        # Get the best model path
        best_model_path = results_dir / "weights" / "best.pt"
        if best_model_path.exists():
            model_size_mb = best_model_path.stat().st_size / (1024 * 1024)
            print(f"\n🏆 Best Model: {best_model_path}")
            print(f"📏 Model Size: {model_size_mb:.1f} MB")
        
        # Print comparison with YOLOv8n
        print("\n📈 Expected Improvements over YOLOv8n:")
        print("   • Model size: 6.0 MB → ~21 MB")
        print("   • mAP@50: 0.989 → ~0.995")
        print("   • FPS: 44.7 → ~30 (still real-time)")
        print("   • Better detection of small/occluded strawberries")
        
        print(f"\n💾 Results saved to: {results_dir}")
        print("\n🎯 Next steps:")
        print(f"   1. Validate the model: python3 validate_kaggle_model.py --model-path {best_model_path}")
        print("   2. Update ROS2 node to use new model")
        print("   3. Test with webcam for real-world performance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_yolov8s()
    sys.exit(0 if success else 1)