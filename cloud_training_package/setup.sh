#!/bin/bash

# Setup script for cloud training package
# This script prepares the environment for YOLOv8 training on cloud GPU providers

echo "========================================="
echo "🚀 Strawberry Detection Cloud Training Setup"
echo "========================================="

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p model/detection/cloud_training
mkdir -p model/dataset_strawberry_kaggle

# Check if dataset exists
if [ ! -f "model/dataset_strawberry_kaggle/data.yaml" ]; then
    echo "⚠️  Dataset not found at model/dataset_strawberry_kaggle/"
    echo "Please ensure the dataset is in the correct location."
    echo "Expected structure:"
    echo "  model/dataset_strawberry_kaggle/data.yaml"
    echo "  model/dataset_strawberry_kaggle/train/images/"
    echo "  model/dataset_strawberry_kaggle/train/labels/"
    echo "  model/dataset_strawberry_kaggle/valid/images/"
    echo "  model/dataset_strawberry_kaggle/valid/labels/"
    exit 1
fi

echo "✅ Dataset found at model/dataset_strawberry_kaggle/"

# Install dependencies
echo "📦 Installing dependencies..."
pip install ultralytics torch torchvision opencv-python matplotlib Pillow tqdm tensorboard onnx onnxruntime pandas seaborn pyyaml -q

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️  No GPU detected. Training will be very slow on CPU.')
    print('   Consider using a cloud GPU provider like RunPod.')
"

# Create training script
echo "📝 Creating training script..."
cat > train_yolov8_cloud.py << 'EOF'
#!/usr/bin/env python3
"""
YOLOv8 Cloud Training Script
Optimized for RunPod RTX 3080/3090 GPUs
"""

import torch
import ultralytics
import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime

def check_environment():
    """Check system environment and GPU availability."""
    print("=" * 60)
    print("🚀 YOLOv8 Cloud Training Environment Check")
    print("=" * 60)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # PyTorch and CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Recommend batch size based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 24:  # RTX 3090/A5000
            batch_size = 48
        elif gpu_memory_gb >= 10:  # RTX 3080
            batch_size = 24
        elif gpu_memory_gb >= 8:   # RTX 3070/2070
            batch_size = 16
        else:
            batch_size = 8
        print(f"Recommended batch size: {batch_size}")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        print("   Consider using a cloud GPU provider.")
        batch_size = 8
    
    # Ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")
    
    # Dataset check
    dataset_path = "model/dataset_strawberry_kaggle"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(data_yaml):
        print(f"✅ Dataset config: {data_yaml}")
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            print(f"   Classes: {data.get('nc', 'N/A')}")
            print(f"   Class names: {data.get('names', 'N/A')}")
    else:
        print(f"❌ Dataset not found: {data_yaml}")
        return False, batch_size
    
    return True, batch_size

def train_model(model_size="m", batch_size=24, epochs=120):
    """Train YOLOv8 model."""
    
    model_name = f"yolov8{model_size}"
    output_dir = f"model/detection/cloud_training/{model_name}_runpod"
    
    print(f"\n🎯 Training {model_name.upper()} Model")
    print("=" * 60)
    print(f"Model: {model_name}.pt")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Training configuration
    config = {
        "model": f"{model_name}.pt",
        "data": "model/dataset_strawberry_kaggle/data.yaml",
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch_size,
        "workers": 8,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "project": "model/detection/cloud_training",
        "name": f"{model_name}_runpod",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.01 if model_size == "m" else 0.008,
        "amp": True,  # Mixed precision
        "plots": True,
        "save_period": 10,
        "val": True,
        "save": True,
        "save_json": False,
        "save_hybrid": False,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "dnn": False,
        "verbose": True,
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_file = os.path.join(output_dir, "training_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"📁 Config saved: {config_file}")
    
    # Start training
    print("⏳ Starting training...")
    start_time = time.time()
    
    try:
        model = YOLO(config["model"])
        results = model.train(**config)
        
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print("=" * 60)
        print(f"✅ Training Completed Successfully!")
        print(f"⏱️  Total Training Time: {hours}h {minutes}m {seconds}s")
        print(f"📁 Results saved to: {output_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"📊 Final mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"📊 Final mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function."""
    
    # Check environment
    env_ok, recommended_batch = check_environment()
    if not env_ok:
        print("❌ Environment check failed. Please fix issues above.")
        return
    
    print("\n" + "=" * 60)
    print("🤖 YOLOv8 Cloud Training Menu")
    print("=" * 60)
    print("1. Train YOLOv8m (Medium) - Balanced speed/accuracy")
    print("2. Train YOLOv8l (Large) - Maximum accuracy")
    print("3. Train both models sequentially")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Train YOLOv8m
        print("\n" + "=" * 60)
        print("Training YOLOv8m (Medium model)")
        print(f"Recommended batch size: {recommended_batch}")
        print("Estimated time: 3-4 hours on RTX 3080")
        print("Estimated cost: $0.51-$0.68 on RunPod")
        print("=" * 60)
        
        confirm = input("Start training? (yes/no): ").strip().lower()
        if confirm == "yes":
            train_model(model_size="m", batch_size=recommended_batch, epochs=120)
        else:
            print("Training cancelled.")
    
    elif choice == "2":
        # Train YOLOv8l
        print("\n" + "=" * 60)
        print("Training YOLOv8l (Large model)")
        print(f"Recommended batch size: {max(8, recommended_batch // 2)}")
        print("Estimated time: 4-5 hours on RTX 3080")
        print("Estimated cost: $0.68-$0.85 on RunPod")
        print("=" * 60)
        
        confirm = input("Start training? (yes/no): ").strip().lower()
        if confirm == "yes":
            train_model(model_size="l", batch_size=max(8, recommended_batch // 2), epochs=150)
        else:
            print("Training cancelled.")
    
    elif choice == "3":
        # Train both
        print("\n" + "=" * 60)
        print("Training BOTH YOLOv8m and YOLOv8l")
        print("Estimated total time: 7-9 hours on RTX 3080")
        print("Estimated total cost: $1.19-$1.53 on RunPod")
        print("=" * 60)
        
        confirm = input("Start sequential training? (yes/no): ").strip().lower()
        if confirm == "yes":
            # Train YOLOv8m first
            print("\n" + "=" * 60)
            print("Starting YOLOv8m training...")
            print("=" * 60)
            success_m = train_model(model_size="m", batch_size=recommended_batch, epochs=120)
            
            if success_m:
                print("\n" + "=" * 60)
                print("Starting YOLOv8l training...")
                print("=" * 60)
                train_model(model_size="l", batch_size=max(8, recommended_batch // 2), epochs=150)
            else:
                print("❌ YOLOv8m training failed. Skipping YOLOv8l.")
        else:
            print("Training cancelled.")
    
    elif choice == "4":
        print("Exiting.")
        return
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Training setup complete!")
    print("Check the output directory for results:")
    print("  model/detection/cloud_training/")
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

chmod +x train_yolov8_cloud.py

echo "✅ Created train_yolov8_cloud.py"

# Create requirements file
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
matplotlib>=3.7.0
Pillow>=10.0.0
tqdm>=4.65.0
tensorboard>=2.13.0
onnx>=1.14.0
onnxruntime>=1.15.0
pandas>=2.0.0
seaborn>=0.12.0
pyyaml>=6.0
EOF

echo "✅ Created requirements.txt"

# Create run instructions
echo "📝 Creating RUN_INSTRUCTIONS.md..."
cat > RUN_INSTRUCTIONS.md << 'EOF'
# 🚀 RunPod Cloud Training Instructions

## Quick Start

1. **Upload this package** to your RunPod instance:
   ```bash
   # On your local machine
   scp -P <port> -r cloud_training_package/ user@runpod.io:/workspace/
   ```

2. **Navigate to the directory**:
   ```bash
   cd /workspace/cloud_training_package
   ```

3. **Run setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Start training**:
   ```bash
   python train_yolov8_cloud.py
   ```

## Alternative: Jupyter Notebook

If you prefer Jupyter:

1. Upload `runpod_yolov8_training.ipynb` to RunPod
2. Open it in Jupyter Lab/Notebook
3. Run cells sequentially

## Dataset Preparation

Ensure your dataset is in the correct location:
```
model/dataset_strawberry_kaggle/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

If the dataset is not present, you'll need to upload it separately.

## Cost Management

- **RTX 3080**: $0.17/hour
- **YOLOv8m**: ~3-4 hours → $0.51-$0.68
- **YOLOv8l**: ~4-5 hours → $0.68-$0.85
- **Total**: ~7-9 hours → $1.19-$1.53

## Monitoring

During training, monitor:
- GPU usage: `nvidia-smi -l 1`
- Training progress: Check `model/detection/cloud_training/*/results.csv`
- TensorBoard: `tensorboard --logdir model/detection/cloud_training/`

## Downloading Results

After training completes:
```bash
# Package results
tar -czf trained_models_$(date +%Y%m%d_%H%M%S).tar.gz model/detection/cloud_training/

# Download via SCP
scp -P <port> user@runpod.io:/workspace/cloud_training_package/trained_models_*.tar.gz .
```

## Troubleshooting

1. **Out of Memory**: Reduce batch size in `train_yolov8_cloud.py`
2. **Dataset not found**: Check paths in `data.yaml`
3. **Slow training**: Ensure GPU is being used (check `nvidia-smi`)
4. **Import errors**: Run `pip install -r requirements.txt`

## Support

- Check the main README.md for detailed instructions
- Review training logs in the output directory
- Monitor RunPod console for instance status
EOF

echo "✅ Created RUN_INSTRUCTIONS.md"

echo "\n========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo "\nNext steps:"
echo "1. Upload this folder to your cloud GPU provider"
echo "2. Run: ./setup.sh"
echo "3. Run: python train_yolov8_cloud.py"
echo "\nFor detailed instructions, see RUN_INSTRUCTIONS.md"
echo "========================================="