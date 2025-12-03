#!/usr/bin/env python3
"""
Prepare strawberry ripeness classification dataset from Kaggle data
Organizes images into train/valid/test splits for unripe, ripe, overripe classes
"""

import os
import shutil
from pathlib import Path
import random
from typing import Dict, List

def create_ripeness_dataset(
    source_dir: str = "/home/user/Downloads/train",
    output_dir: str = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/strawberry_ripeness_classification",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_per_class: int = 600  # Limit dataset size for faster training
):
    """
    Create balanced ripeness classification dataset
    
    Args:
        source_dir: Source directory with Kaggle fruit ripeness data
        output_dir: Output directory for classification dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        max_per_class: Maximum images per class
    """
    
    print("🍓 Preparing Strawberry Ripeness Classification Dataset")
    print("=" * 60)
    
    # Define class mapping (Kaggle → Our classes)
    class_mapping = {
        "UnripeStrawberry": "unripe",
        "RipeStrawberry": "ripe", 
        "RottenStrawberry": "overripe"
    }
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ["train", "valid", "test"]:
        for class_name in class_mapping.values():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    total_images = 0
    class_counts = {}
    
    for kaggle_class, our_class in class_mapping.items():
        print(f"\n📂 Processing {kaggle_class} → {our_class}")
        
        # Get source images
        source_path = Path(source_dir) / kaggle_class
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_path}")
            continue
        
        # Get all image files
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Limit dataset size
        if len(image_files) > max_per_class:
            image_files = random.sample(image_files, max_per_class)
            print(f"   Limited to {max_per_class} images")
        
        print(f"   Found {len(image_files)} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        splits = {
            "train": train_files,
            "valid": val_files,
            "test": test_files
        }
        
        split_counts = {}
        for split_name, files in splits.items():
            split_counts[split_name] = len(files)
            for img_file in files:
                dest = output_path / split_name / our_class / img_file.name
                shutil.copy2(img_file, dest)
                total_images += 1
        
        class_counts[our_class] = split_counts
        print(f"   📊 Train: {split_counts['train']}, Val: {split_counts['valid']}, Test: {split_counts['test']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Total images: {total_images}")
    print(f"📁 Output directory: {output_path}")
    print(f"\n📊 Class distribution:")
    for class_name, splits in class_counts.items():
        print(f"   {class_name}: {splits['train']}/{splits['valid']}/{splits['test']} (train/val/test)")
    
    # Create dataset info file
    info_file = output_path / "dataset_info.json"
    dataset_info = {
        "total_images": total_images,
        "classes": list(class_mapping.values()),
        "class_distribution": class_counts,
        "splits": {
            "train": train_ratio,
            "valid": val_ratio,
            "test": test_ratio
        },
        "max_per_class": max_per_class,
        "source": str(source_dir)
    }
    
    import json
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n💾 Dataset info saved to: {info_file}")
    print(f"\n✅ Dataset ready for training!")

if __name__ == "__main__":
    create_ripeness_dataset()