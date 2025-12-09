#!/usr/bin/env python3
"""
Script to create a Jupyter notebook for RunPod YOLOv8 training.
This creates a proper .ipynb file with JSON structure.
"""

import json
import os
from datetime import datetime

def create_notebook():
    """Create a Jupyter notebook for RunPod YOLOv8 training."""
    
    # Notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Cell 1: Title and introduction
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 YOLOv8 Cloud Training on RunPod (RTX 3080/3090)\n",
            "\n",
            "This notebook provides a complete solution for training YOLOv8 models on RunPod cloud GPU instances. It's optimized for RTX 3080/3090 GPUs and includes monitoring, visualization, and automatic model saving.\n",
            "\n",
            "## 📋 Prerequisites\n",
            "1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)\n",
            "2. **Credits**: Add at least $5-10 to your account\n",
            "3. **Dataset**: Already prepared in `model/dataset_strawberry_kaggle/`\n",
            "\n",
            "## 🎯 Training Goals\n",
            "- Train **YOLOv8m** (medium) model for better accuracy than YOLOv8s\n",
            "- Train **YOLOv8l** (large) model for maximum accuracy\n",
            "- Both models trained on the strawberry detection dataset\n",
            "\n",
            "## ⚙️ Hardware Recommendations\n",
            "| GPU | VRAM | Price/hr | Training Time | Total Cost |\n",
            "|-----|------|----------|---------------|------------|\n",
            "| **RTX 3080** | 10GB | **$0.17** | 7-9 hours | **$1.19-$1.53** |\n",
            "| RTX 3090 | 24GB | $0.22 | 5-7 hours | $1.10-$1.54 |\n",
            "| RTX A5000 | 24GB | $0.16 | 6-8 hours | $0.96-$1.28 |\n",
            "\n",
            "**Recommendation**: Use **RTX 3080** for best price/performance ratio."
        ]
    }
    
    # Cell 2: Install packages
    install_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages\n",
            "!pip install ultralytics torch torchvision opencv-python matplotlib Pillow tqdm tensorboard onnx onnxruntime -q\n",
            "!pip install runpod -q  # For RunPod API if needed"
        ]
    }
    
    # Cell 3: System check
    system_check_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import ultralytics\n",
            "import sys\n",
            "import os\n",
            "import subprocess\n",
            "import json\n",
            "from datetime import datetime\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"SYSTEM CHECK FOR YOLOv8 CLOUD TRAINING\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Check Python version\n",
            "print(f\"Python version: {sys.version}\")\n",
            "\n",
            "# Check PyTorch and CUDA\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"CUDA version: {torch.version.cuda}\")\n",
            "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
            "else:\n",
            "    print(\"⚠️ WARNING: No GPU detected! Training will be very slow.\")\n",
            "\n",
            "# Check Ultralytics\n",
            "print(f\"Ultralytics version: {ultralytics.__version__}\")\n",
            "\n",
            "# Check dataset\n",
            "dataset_path = \"model/dataset_strawberry_kaggle\"\n",
            "if os.path.exists(dataset_path):\n",
            "    print(f\"✅ Dataset found at: {dataset_path}\")\n",
            "    \n",
            "    # Check data.yaml\n",
            "    data_yaml = os.path.join(dataset_path, \"data.yaml\")\n",
            "    if os.path.exists(data_yaml):\n",
            "        print(f\"✅ data.yaml found: {data_yaml}\")\n",
            "        with open(data_yaml, 'r') as f:\n",
            "            data = f.read()\n",
            "            print(f\"Dataset config:\\n{data[:500]}...\")\n",
            "    else:\n",
            "        print(f\"❌ data.yaml not found in {dataset_path}\")\n",
            "else:\n",
            "    print(f\"❌ Dataset not found at: {dataset_path}\")\n",
            "    print(\"Please ensure the dataset is in the correct location.\")\n",
            "\n",
            "print(\"=\" * 60)"
        ]
    }
    
    # Cell 4: Dataset verification
    dataset_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import yaml\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import cv2\n",
            "import random\n",
            "from PIL import Image\n",
            "import numpy as np\n",
            "\n",
            "def verify_dataset_structure():\n",
            "    \"\"\"Verify the dataset structure and sample images.\"\"\"\n",
            "    \n",
            "    data_yaml_path = \"model/dataset_strawberry_kaggle/data.yaml\"\n",
            "    \n",
            "    if not os.path.exists(data_yaml_path):\n",
            "        print(f\"❌ data.yaml not found at {data_yaml_path}\")\n",
            "        return None\n",
            "    \n",
            "    # Load YAML config\n",
            "    with open(data_yaml_path, 'r') as f:\n",
            "        config = yaml.safe_load(f)\n",
            "    \n",
            "    print(\"📊 Dataset Configuration:\")\n",
            "    print(f\"  - Train images: {config.get('train', 'Not specified')}\")\n",
            "    print(f\"  - Val images: {config.get('val', 'Not specified')}\")\n",
            "    print(f\"  - Test images: {config.get('test', 'Not specified')}\")\n",
            "    print(f\"  - Number of classes: {config.get('nc', 'Not specified')}\")\n",
            "    print(f\"  - Class names: {config.get('names', 'Not specified')}\")\n",
            "    \n",
            "    # Check if paths exist\n",
            "    train_path = config.get('train', '')\n",
            "    val_path = config.get('val', '')\n",
            "    \n",
            "    if train_path and os.path.exists(train_path):\n",
            "        train_images = list(Path(train_path).glob(\"*.jpg\")) + list(Path(train_path).glob(\"*.png\"))\n",
            "        print(f\"  - Found {len(train_images)} training images\")\n",
            "    else:\n",
            "        print(f\"  ⚠️ Training path not found: {train_path}\")\n",
            "    \n",
            "    if val_path and os.path.exists(val_path):\n",
            "        val_images = list(Path(val_path).glob(\"*.jpg\")) + list(Path(val_path).glob(\"*.png\"))\n",
            "        print(f\"  - Found {len(val_images)} validation images\")\n",
            "    else:\n",
            "        print(f\"  ⚠️ Validation path not found: {val_path}\")\n",
            "    \n",
            "    return config\n",
            "\n",
            "# Verify dataset\n",
            "config = verify_dataset_structure()"
        ]
    }
    
    # Cell 5: Visualize sample images
    visualize_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def visualize_sample_images(num_samples=3):\n",
            "    \"\"\"Display sample images from the dataset.\"\"\"\n",
            "    \n",
            "    data_yaml_path = \"model/dataset_strawberry_kaggle/data.yaml\"\n",
            "    if not os.path.exists(data_yaml_path):\n",
            "        print(\"data.yaml not found, skipping visualization\")\n",
            "        return\n",
            "    \n",
            "    with open(data_yaml_path, 'r') as f:\n",
            "        config = yaml.safe_load(f)\n",
            "    \n",
            "    train_path = config.get('train', '')\n",
            "    if not train_path or not os.path.exists(train_path):\n",
            "        print(f\"Training path not found: {train_path}\")\n",
            "        return\n",
            "    \n",
            "    # Get random sample of images\n",
            "    image_files = list(Path(train_path).glob(\"*.jpg\")) + list(Path(train_path).glob(\"*.png\"))\n",
            "    if not image_files:\n",
            "        print(\"No images found in training directory\")\n",
            "        return\n",
            "    \n",
            "    samples = random.sample(image_files, min(num_samples, len(image_files)))\n",
            "    \n",
            "    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))\n",
            "    if len(samples) == 1:\n",
            "        axes = [axes]\n",
            "    \n",
            "    for idx, (ax, img_path) in enumerate(zip(axes, samples)):\n",
            "        try:\n",
            "            img = cv2.imread(str(img_path))\n",
            "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
            "            ax.imshow(img)\n",
            "            ax.set_title(f\"Sample {idx+1}\\n{img_path.name}\")\n",
            "            ax.axis('off')\n",
            "            \n",
            "            # Get corresponding label file\n",
            "            label_path = str(img_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')\n",
            "            if os.path.exists(label_path):\n",
            "                with open(label_path, 'r') as f:\n",
            "                    labels = f.readlines()\n",
            "                ax.text(0.02, 0.98, f\"Labels: {len(labels)}\", \n",
            "                       transform=ax.transAxes, fontsize=10,\n",
            "                       verticalalignment='top', \n",
            "                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
            "        except Exception as e:\n",
            "            ax.text(0.5, 0.5, f\"Error loading\\n{str(e)[:30]}\", \n",
            "                   ha='center', va='center', transform=ax.transAxes)\n",
            "            ax.axis('off')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    print(f\"📸 Displayed {len(samples)} sample images from training set\")\n",
            "\n",
            "# Visualize samples\n",
            "visualize_sample_images(num_samples=3)"
        ]
    }
    
    # Cell 6: YOLOv8m training configuration
    config_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. YOLOv8m Training Configuration\n",
            "\n",
            "```python\n",
            "# Training configuration for YOLOv8m\n",
            "yolov8m_config = {\n",
            "    \"model\": \"yolov8m.pt\",  # Medium model\n",
            "    \"data\": \"model/dataset_strawberry_kaggle/data.yaml\",\n",
            "    \"epochs\": 120,\n",
            "    \"imgsz\": 640,\n",
            "    \"batch\": 32,  # Adjust based on GPU memory (16 for 10GB, 32 for 24GB)\n",
            "    \"workers\": 8,\n",
            "    \"device\": 0,  # Use GPU 0\n",
            "    \"project\": \"model/detection/cloud_training\",\n",
            "    \"name\": \"yolov8m_runpod\",\n",
            "    \"exist_ok\": True,\n",
            "    \"pretrained\": True,\n",
            "    \"optimizer\": \"AdamW\",\n",
            "    \"lr0\": 0.01,\n",
            "    \"lrf\": 0.01,\n",
            "    \"momentum\": 0.937,\n",
            "    \"weight_decay\": 0.0005,\n",
            "    \"warmup_epochs\": 3.0,\n",
            "    \"warmup_momentum\": 0.8,\n",
            "    \"warmup_bias_lr\": 0.1,\n",
            "    \"box\": 7.5,\n",
            "    \"cls\": 0.5,\n",
            "    \"dfl\": 1.5,\n",
            "    \"flipud\": 0.0,\n",
            "    \"fliplr\": 0.5,\n",
            "    \"mosaic\": 1.0,\n",
            "    \"mixup\": 0.0,\n",
            "    \"copy_paste\": 0.0,\n",
            "    \"hsv_h\": 0.015,\n",
            "    \"hsv_s\": 0.7,\n",
            "    \"hsv_v\": 0.4,\n",
            "    \"degrees\": 0.0,\n",
            "    \"translate\": 0.1,\n",
            "    \"scale\": 0.5,\n",
            "    \"shear\": 0.0,\n",
            "    \"perspective\": 0.0,\n",
            "    \"save_period\": 10,  # Save checkpoint every 10 epochs\n",
            "    \"seed\": 42,\n",
            "    \"deterministic\": True,\n",
            "    \"single_cls\": False,\n",
            "    \"rect\": False,\n",
            "    \"cos_lr\": False,\n",
            "    \"close_mosaic\": 10,\n",
            "    \"resume\": False,\n",
            "    \"amp\": True,  # Mixed precision training\n",
            "    \"fraction\": 1.0,\n",
            "    \"profile\": False,\n",
            "    \"freeze\": None,\n",
            "    \"multi_scale\": False,\n",
            "    \"overlap_mask\": True,\n",
            "    \"mask_ratio\": 4,\n",
            "    \"dropout\": 0.0,\n",
            "    \"val\": True,\n",
            "    \"save_json\": False,\n",
            "    \"save_hybrid\": False,\n",
            "    \"conf\": None,\n",
            "    \"iou\": 0.7,\n",
            "    \"max_det\": 300,\n",
            "    \"half\": False,\n",
            "    \"dnn\": False,\n",
            "    \"plots\": True,\n",
            "    \"source\": None,\n",
            "    \"show\": False,\n",
            "    \"save_txt\": False,\n",
            "    \"save_conf\": False,\n",
            "    \"save_crop\": False,\n",
            "    \"show_labels\": True,\n",
            "    \"show_conf\": True,\n",
            "    \"vid_stride\": 1,\n",
            "    \"line_width\": None,\n",
            "    \"visualize\": False,\n",
            "    \"augment\": False,\n",
            "    \"agnostic_nms\": False,\n",
            "    \"retina_masks\": False,\n",
            "    \"classes\": None,\n",
            "    \"boxes\": True,\n",
            "}\n",
            "```"
        ]
    }
    
    # Cell 7: Training function
    training_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from ultralytics import YOLO\n",
            "import time\n",
            "from tqdm.notebook import tqdm\n",
            "import pandas as pd\n",
            "\n",
            "def train_yolov8_model(model_size=\"m\", config=None, resume=False):\n",
            "    \"\"\"\n",
            "    Train a YOLOv8 model with the given configuration.\n",
            "    \n",
            "    Args:\n",
            "        model_size (str): 'm' for medium, 'l' for large\n",
            "        config (dict): Training configuration\n",
            "        resume (bool): Whether to resume from last checkpoint\n",
            "    \"\"\"\n",
            "    \n",
            "    if config is None:\n",
            "        # Default config\n",
            "        config = {\n",
            "            \"model\": f\"yolov8{model_size}.pt\",\n",
            "            \"data\": \"model/dataset_strawberry_kaggle/data.yaml\",\n",
            "            \"epochs\": 120 if model_size == \"m\" else 150,\n",
            "            \"imgsz\": 640,\n",
            "            \"batch\": 24 if model_size == \"m\" else 12,\n",
            "            \"workers\": 8,\n",
            "            \"device\": 0,\n",
            "            \"project\": \"model/detection/cloud_training\",\n",
            "            \"name\": f\"yolov8{model_size}_runpod\",\n",
            "            \"exist_ok\": True,\n",
            "            \"pretrained\": True,\n",
            "            \"optimizer\": \"AdamW\",\n",
            "            \"lr0\": 0.01 if model_size == \"m\" else 0.008,\n",
            "            \"amp\": True,\n",
            "            \"plots\": True,\n",
            "            \"save_period\": 10,\n",
            "            \"resume\": resume,\n",
            "        }\n",
            "    \n",
            "    print(f\"🚀 Starting YOLOv8{model_size.upper()} Training\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"Model: yolov8{model_size}.pt\")\n",
            "    print(f\"Epochs: {config['epochs']}\")\n",
            "    print(f\"Batch Size: {config['batch']}\")\n",
            "    print(f\"Image Size: {config['imgsz']}\")\n",
            "    print(f\"Output: {config['project']}/{config['name']}\")\n",
            "    print(\"=\" * 60)\n",
            "    \n",
            "    # Create output directory\n",
            "    os.makedirs(os.path.join(config['project'], config['name']), exist_ok=True)\n",
            "    \n",
            "    # Initialize model\n",
            "    model = YOLO(f\"yolov8{model_size}.pt\")\n",
            "    \n",
            "    # Start training\n",
            "    start_time = time.time()\n",
            "    \n",
            "    try:\n",
            "        results = model.train(**config)\n",
            "        \n",
            "        training_time = time.time() - start_time\n",
            "        hours = int(training_time // 3600)\n",
            "        minutes = int((training_time % 3600) // 60)\n",
            "        seconds = int(training_time % 60)\n",
            "        \n",
            "        print(\"=\" * 60)\n",
            "        print(f\"✅ Training Completed Successfully!\")\n",
            "        print(f\"⏱️  Total Training Time: {hours}h {minutes}m {seconds}s\")\n",
            "        print(f\"📁 Results saved to: {config['project']}/{config['name']}\")\n",
            "        \n",
            "        return results\n",
            "        \n",
            "    except Exception as e:\n",
            "        print(f\"❌ Training failed with error: {e}\")\n",
            "        import traceback\n",
            "        traceback.print_exc()\n",
            "        raise"
        ]
    }
    
    # Cell 8: Run training (with confirmation)
    run_training_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Uncomment and run this cell to start YOLOv8m training\n",
            "# WARNING: This will take 3-4 hours on RTX 3080\n",
            "\n",
            "print(\"⚠️  WARNING: Training will take 3-4 hours on RTX 3080\")\n",
            "print(\"💰 Estimated cost: $0.51-$0.68 (at $0.17/hr)\")\n",
            "print(\"💾 Make sure you have enough disk space for checkpoints\")\n",
            "print(\"\")\n",
            "\n",
            "response = input(\"Do you want to start YOLOv8m training? (yes/no): \")\n",
            "\n",
            "if response.lower() == 'yes':\n",
            "    print(\"Starting YOLOv8m training...\")\n",
            "    \n",
            "    # Adjust batch size based on available GPU memory\n",
            "    import torch\n",
            "    if torch.cuda.is_available():\n",
            "        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
            "        if total_memory >= 24:  # RTX 3090/A5000\n",
            "            batch_size = 48\n",
            "        elif total_memory >= 10:  # RTX 3080\n",
            "            batch_size = 24\n",
            "        else:\n",
            "            batch_size = 16\n",
            "    else:\n",
            "        batch_size = 8\n",
            "    \n",
            "    config = {\n",
            "        \"model\": \"yolov8m.pt\",\n",
            "        \"data\": \"model/dataset_strawberry_kaggle/data.yaml\",\n",
            "        \"epochs\": 120,\n",
            "        \"imgsz\": 640,\n",
            "        \"batch\": batch_size,\n",
            "        \"workers\": 8,\n",
            "        \"device\": 0,\n",
            "        \"project\": \"model/detection/cloud_training\",\n",
            "        \"name\": \"yolov8m_runpod\",\n",
            "        \"exist_ok\": True,\n",
            "        \"pretrained\": True,\n",
            "        \"optimizer\": \"AdamW\",\n",
            "        \"lr0\": 0.01,\n",
            "        \"amp\": True,\n",
            "        \"plots\": True,\n",
            "        \"save_period\": 10,\n",
            "    }\n",
            "    \n",
            "    # Start training\n",
            "    results_m = train_yolov8_model(model_size=\"m\", config=config)\n",
            "    \n",
            "    print(\"🎉 YOLOv8m training completed!\")\n",
            "    print(\"Next: Run YOLOv8l training or download the model\")\n",
            "else:\n",
            "    print(\"Skipping YOLOv8m training for now.\")"
        ]
    }
    
    # Cell 9: Monitor training progress
    monitor_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import glob\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "def monitor_training_progress(model_size=\"m\"):\n",
            "    \"\"\"Monitor training progress by reading results CSV.\"\"\"\n",
            "    \n",
            "    results_dir = f\"model/detection/cloud_training/yolov8{model_size}_runpod\"\n",
            "    results_csv = os.path.join(results_dir, \"results.csv\")\n",
            "    \n",
            "    if not os.path.exists(results_csv):\n",
            "        print(f\"❌ Results CSV not found: {results_csv}\")\n",
            "        return None\n",
            "    \n",
            "    # Read results\n",
            "    df = pd.read_csv(results_csv)\n",
            "    \n",
            "    print(f\"📊 YOLOv8{model_size.upper()} Training Progress\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"Total epochs: {len(df)}\")\n",
            "    print(f\"Latest epoch: {df['epoch'].iloc[-1] if 'epoch' in df.columns else 'N/A'}\")\n",
            "    \n",
            "    # Display latest metrics\n",
            "    if 'metrics/mAP50(B)' in df.columns:\n",
            "        latest_map = df['metrics/mAP50(B)'].iloc[-1]\n",
            "        print(f\"Latest mAP@50: {latest_map:.4f}\")\n",
            "    \n",
            "    if 'train/box_loss' in df.columns:\n",
            "        latest_box_loss = df['train/box_loss'].iloc[-1]\n",
            "        latest_cls_loss = df['train/cls_loss'].iloc[-1] if 'train/cls_loss' in df.columns else 'N/A'\n",
            "        print(f\"Latest box loss: {latest_box_loss:.4f}\")\n",
            "        print(f\"Latest cls loss: {latest_cls_loss:.4f}\")\n",
            "    \n",
            "    # Plot training curves\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "    \n",
            "    # Plot losses\n",
            "    if 'train/box_loss' in df.columns:\n",
            "        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')\n",
            "        axes[0, 0].set_xlabel('Epoch')\n",
            "        axes[0, 0].set_ylabel('Loss')\n",
            "        axes[0, 0].set_title('Box Loss')\n",
            "        axes[0, 0].grid(True)\n",
            "        axes[0, 0].legend()\n",
            "    \n",
            "    if 'train/cls_loss' in df.columns:\n",
            "        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='orange')\n",
            "        axes[0, 1].set_xlabel('Epoch')\n",
            "        axes[0, 1].set_ylabel('Loss')\n",
            "        axes[0, 1].set_title('Class Loss')\n",
            "        axes[0, 1].grid(True)\n",
            "        axes[0, 1].legend()\n",
            "    \n",
            "    # Plot mAP\n",
            "    if 'metrics/mAP50(B)' in df.columns:\n",
            "        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green')\n",
            "        axes[1, 0].set_xlabel('Epoch')\n",
            "        axes[1, 0].set_ylabel('mAP@50')\n",
            "        axes[1, 0].set_title('mAP@50 Progress')\n",
            "        axes[1, 0].grid(True)\n",
            "        axes[1, 0].legend()\n",
            "    \n",
            "    # Plot learning rate\n",
            "    if 'lr/pg0' in df.columns:\n",
            "        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='red')\n",
            "        axes[1, 1].set_xlabel('Epoch')\n",
            "        axes[1, 1].set_ylabel('Learning Rate')\n",
            "        axes[1, 1].set_title('Learning Rate Schedule')\n",
            "        axes[1, 1].grid(True)\n",
            "        axes[1, 1].legend()\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    return df\n",
            "\n",
            "# Monitor progress (run this cell periodically during training)\n",
            "try:\n",
            "    progress_df = monitor_training_progress(\"m\")\n",
            "except Exception as e:\n",
            "    print(f\"Could not monitor progress: {e}\")"
        ]
    }
    
    # Cell 10: Download models
    download_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "import zipfile\n",
            "from datetime import datetime\n",
            "\n",
            "def package_trained_models():\n",
            "    \"\"\"Package trained models for download.\"\"\"\n",
            "    \n",
            "    output_dir = \"model/detection/cloud_training\"\n",
            "    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
            "    zip_filename = f\"yolov8_cloud_trained_{timestamp}.zip\"\n",
            "    \n",
            "    # Create zip file\n",
            "    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:\n",
            "        # Add best models\n",
            "        for model_size in ['m', 'l']:\n",
            "            model_dir = os.path.join(output_dir, f\"yolov8{model_size}_runpod\")\n",
            "            best_pt = os.path.join(model_dir, \"weights\", \"best.pt\")\n",
            "            last_pt = os.path.join(model_dir, \"weights\", \"last.pt\")\n",
            "            \n",
            "            if os.path.exists(best_pt):\n",
            "                zipf.write(best_pt, f\"yolov8{model_size}/best.pt\")\n",
            "                print(f\"✅ Added yolov8{model_size}/best.pt\")\n",
            "            \n",
            "            if os.path.exists(last_pt):\n",
            "                zipf.write(last_pt, f\"yolov8{model_size}/last.pt\")\n",
            "                print(f\"✅ Added yolov8{model_size}/last.pt\")\n",
            "            \n",
            "            # Add results CSV\n",
            "            results_csv = os.path.join(model_dir, \"results.csv\")\n",
            "            if os.path.exists(results_csv):\n",
            "                zipf.write(results_csv, f\"yolov8{model_size}/results.csv\")\n",
            "                print(f\"✅ Added yolov8{model_size}/results.csv\")\n",
            "        \n",
            "        # Add training plots\n",
            "        for model_size in ['m', 'l']:\n",
            "            model_dir = os.path.join(output_dir, f\"yolov8{model_size}_runpod\")\n",
            "            for plot_file in ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n",
            "                plot_path = os.path.join(model_dir, plot_file)\n",
            "                if os.path.exists(plot_path):\n",
            "                    zipf.write(plot_path, f\"yolov8{model_size}/{plot_file}\")\n",
            "                    print(f\"✅ Added yolov8{model_size}/{plot_file}\")\n",
            "    \n",
            "    print(f\"\\n📦 Models packaged: {zip_filename}\")\n",
            "    print(f\"📁 Size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB\")\n",
            "    \n",
            "    return zip_filename\n",
            "\n",
            "# Package models for download\n",
            "print(\"📦 Packaging trained models for download...\")\n",
            "try:\n",
            "    zip_file = package_trained_models()\n",
            "    print(f\"\\n✅ Download ready: {zip_file}\")\n",
            "    print(\"\\nTo download from RunPod:\")\n",
            "    print(\"1. In the RunPod web terminal, run: `ls -lh *.zip`\")\n",
            "    print(\"2. Use RunPod's file browser to download\")\n",
            "    print(\"3. Or use SCP: `scp -P <port> user@runpod.io:/workspace/{zip_file} .`\")\n",
            "except Exception as e:\n",
            "    print(f\"❌ Error packaging models: {e}\")"
        ]
    }
    
    # Cell 11: RunPod instructions
    instructions_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## RunPod Setup Instructions\n",
            "\n",
            "### 1. Create RunPod Instance\n",
            "1. Go to [RunPod.io](https://runpod.io)\n",
            "2. Click \"Deploy\" → \"Community Cloud\"\n",
            "3. Select GPU:\n",
            "   - **Recommended**: RTX 3080 ($0.17/hr)\n",
            "   - Alternative: RTX 3090 ($0.22/hr) for faster training\n",
            "4. Choose template: `RunPod PyTorch` or `RunPod Jupyter`\n",
            "5. Deploy with at least 50GB storage\n",
            "\n",
            "### 2. Upload This Notebook\n",
            "```bash\n",
            "# On your local machine\n",
            "scp -P <port> runpod_yolov8_training.ipynb user@runpod.io:/workspace/\n",
            "\n",
            "# Or use RunPod's web file upload\n",
            "```\n",
            "\n",
            "### 3. Upload Dataset\n",
            "```bash\n",
            "# Compress dataset locally\n",
            "tar -czf dataset.tar.gz model/dataset_strawberry_kaggle/\n",
            "\n",
            "# Upload to RunPod\n",
            "scp -P <port> dataset.tar.gz user@runpod.io:/workspace/\n",
            "\n",
            "# Extract on RunPod\n",
            "tar -xzf dataset.tar.gz\n",
            "```\n",
            "\n",
            "### 4. Install Dependencies\n",
            "```bash\n",
            "pip install ultralytics torch torchvision opencv-python matplotlib Pillow tqdm tensorboard\n",
            "```\n",
            "\n",
            "### 5. Run Training\n",
            "1. Open Jupyter notebook in RunPod web interface\n",
            "2. Run all cells sequentially\n",
            "3. Monitor training progress in real-time\n",
            "4. Download models when complete\n",
            "\n",
            "### 6. Cost Management\n",
            "- **RTX 3080**: $0.17/hour\n",
            "- **YOLOv8m**: ~3-4 hours → $0.51-$0.68\n",
            "- **YOLOv8l**: ~4-5 hours → $0.68-$0.85\n",
            "- **Total**: ~7-9 hours → $1.19-$1.53\n",
            "\n",
            "### 7. Stop Instance When Done\n",
            "Don't forget to stop the RunPod instance to avoid unnecessary charges!"
        ]
    }
    
    # Add all cells to notebook
    notebook["cells"] = [
        title_cell,
        install_cell,
        system_check_cell,
        dataset_cell,
        visualize_cell,
        config_cell,
        training_cell,
        run_training_cell,
        monitor_cell,
        download_cell,
        instructions_cell
    ]
    
    return notebook

def save_notebook(notebook, output_path):
    """Save notebook to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"✅ Notebook saved to: {output_path}")

def main():
    """Main function to create the notebook."""
    output_dir = "scripts/cloud_training"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "runpod_yolov8_training.ipynb")
    
    print("Creating Jupyter notebook for RunPod YOLOv8 training...")
    notebook = create_notebook()
    save_notebook(notebook, output_path)
    
    # Also create a Python script version
    script_path = os.path.join(output_dir, "runpod_yolov8_training.py")
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Standalone Python script for RunPod YOLOv8 training.
Run with: python runpod_yolov8_training.py
"""

import torch
import ultralytics
import os
import sys
from ultralytics import YOLO

def main():
    print("🚀 YOLOv8 Cloud Training Script")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected! This script requires GPU acceleration.")
        print("   Please run on a cloud GPU instance (RunPod, Colab, etc.)")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Training configuration
    config = {
        "model": "yolov8m.pt",
        "data": "model/dataset_strawberry_kaggle/data.yaml",
        "epochs": 120,
        "imgsz": 640,
        "batch": 24 if gpu_memory >= 10 else 16,
        "workers": 8,
        "device": 0,
        "project": "model/detection/cloud_training",
        "name": "yolov8m_runpod",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "amp": True,
        "plots": True,
        "save_period": 10,
    }
    
    print(f"\\n📋 Training Configuration:")
    print(f"   Model: {config['model']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Output: {config['project']}/{config['name']}")
    
    # Confirm before starting
    response = input("\\n⚠️  Start training? This will take 3-4 hours. (yes/no): ")
    if response.lower() != "yes":
        print("Training cancelled.")
        return
    
    # Start training
    print("\\n⏳ Starting training...")
    model = YOLO(config["model"])
    results = model.train(**config)
    
    print("\\n✅ Training completed!")
    print(f"Results saved to: {config['project']}/{config['name']}")

if __name__ == "__main__":
    main()
''')
    print(f"✅ Python script saved to: {script_path}")

if __name__ == "__main__":
    main()