#!/usr/bin/env python3
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
    
    print(f"\n📋 Training Configuration:")
    print(f"   Model: {config['model']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Output: {config['project']}/{config['name']}")
    
    # Confirm before starting
    response = input("\n⚠️  Start training? This will take 3-4 hours. (yes/no): ")
    if response.lower() != "yes":
        print("Training cancelled.")
        return
    
    # Start training
    print("\n⏳ Starting training...")
    model = YOLO(config["model"])
    results = model.train(**config)
    
    print("\n✅ Training completed!")
    print(f"Results saved to: {config['project']}/{config['name']}")

if __name__ == "__main__":
    main()
