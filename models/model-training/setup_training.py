#!/usr/bin/env python3
"""
Setup script for Strawberry Picker ML Training Environment
This script installs dependencies and validates the training setup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required. Found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available"""
    print("Checking pip availability...")
    return run_command("pip --version", "Check pip version")

def install_requirements():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrade pip"):
        return False
    
    # Install requirements
    return run_command(f"pip install -r {requirements_file}", "Install requirements")

def check_ultralytics():
    """Check if ultralytics is installed correctly"""
    print("Checking ultralytics installation...")
    try:
        from ultralytics import YOLO
        print("✓ ultralytics installed successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import ultralytics: {e}")
        return False

def check_torch():
    """Check PyTorch installation and GPU availability"""
    print("Checking PyTorch installation...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠ GPU not available, will use CPU for training")
        
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import torch: {e}")
        return False

def validate_dataset():
    """Validate dataset structure"""
    print("Validating dataset structure...")
    
    dataset_path = Path(__file__).parent / "model" / "dataset" / "straw-detect.v1-straw-detect.yolov8"
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Please ensure your dataset is in the correct location")
        return False
    
    try:
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"✓ Dataset configuration loaded")
        print(f"  Classes: {data['nc']}")
        print(f"  Names: {data['names']}")
        
        # Check training images
        train_path = dataset_path / data['train']
        if train_path.exists():
            train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
            print(f"  Training images: {len(train_images)}")
        else:
            print(f"⚠ Training path not found: {train_path}")
        
        # Check validation images
        val_path = dataset_path / data['val']
        if val_path.exists():
            val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
            print(f"  Validation images: {len(val_images)}")
        else:
            print(f"⚠ Validation path not found: {val_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate dataset: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    base_path = Path(__file__).parent
    dirs = [
        base_path / "model" / "weights",
        base_path / "model" / "results",
        base_path / "model" / "exports"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup training environment for strawberry detection')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation')
    parser.add_argument('--validate-only', action='store_true', help='Only validate setup without installing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Strawberry Picker ML Training Environment Setup")
    print("="*60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check pip
    if not check_pip():
        sys.exit(1)
    
    # Step 3: Install requirements (unless skipped)
    if not args.skip_install and not args.validate_only:
        if not install_requirements():
            print("\n⚠ Installation failed. Please check the errors above.")
            response = input("Continue with validation anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Step 4: Check ultralytics
    if not check_ultralytics():
        sys.exit(1)
    
    # Step 5: Check PyTorch
    if not check_torch():
        sys.exit(1)
    
    # Step 6: Validate dataset
    if not validate_dataset():
        print("\n⚠ Dataset validation failed. Please fix the issues above.")
        if not args.validate_only:
            response = input("Continue with directory creation anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Step 7: Create directories
    if not args.validate_only:
        if not create_directories():
            sys.exit(1)
    
    print("\n" + "="*60)
    if args.validate_only:
        print("Setup validation completed!")
    else:
        print("Setup completed successfully!")
    
    print("\nNext steps:")
    print("1. Run training: python train_yolov8.py")
    print("2. Or open train_yolov8_colab.ipynb in Google Colab")
    print("3. Check README.md for detailed instructions")
    print("="*60)

if __name__ == '__main__':
    main()