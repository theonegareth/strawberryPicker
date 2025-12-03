#!/usr/bin/env python3
"""
Prepare ripeness classification dataset from 3-class detection dataset
Extracts strawberry crops and organizes by ripeness class
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def extract_strawberry_crops(dataset_path, output_path, img_size=128):
    """
    Extract strawberry crops from detection dataset
    Organize by ripeness: unripe, ripe, overripe
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        for ripeness in ['unripe', 'ripe', 'overripe']:
            (output_path / split / ripeness).mkdir(parents=True, exist_ok=True)
    
    # Class mapping (adjust based on your dataset)
    # Common mapping: 0=unripe, 1=ripe, 2=overripe (but verify!)
    class_names = ['unripe', 'ripe', 'overripe']
    
    print(f"Processing dataset from: {dataset_path}")
    print(f"Output to: {output_path}")
    print(f"Class names: {class_names}")
    
    total_crops = 0
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {split}: directories not found")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Process each image
        image_files = list(images_dir.glob("*.jpg"))
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Load labels
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Extract each strawberry
            crop_idx = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_id >= len(class_names):
                    continue
                
                # YOLO format: class x_center y_center width height (normalized)
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h
                
                # Convert to pixel coordinates
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Ensure coordinates are within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                # Extract crop
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Resize to standard size
                crop_resized = cv2.resize(crop, (img_size, img_size))
                
                # Save crop
                ripeness = class_names[class_id]
                output_file = output_path / split / ripeness / f"{img_file.stem}_crop{crop_idx}.jpg"
                cv2.imwrite(str(output_file), crop_resized)
                
                crop_idx += 1
                total_crops += 1
    
    print(f"\n✓ Extracted {total_crops} strawberry crops")
    print(f"✓ Dataset ready at: {output_path}")
    
    # Print class distribution
    print("\nClass distribution:")
    for split in ['train', 'valid', 'test']:
        print(f"\n{split}:")
        for ripeness in ['unripe', 'ripe', 'overripe']:
            count = len(list((output_path / split / ripeness).glob("*.jpg")))
            print(f"  {ripeness}: {count} images")

if __name__ == "__main__":
    # Paths
    dataset_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset_3class_backup"
    output_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/ripeness_classification_dataset"
    
    # Extract crops
    extract_strawberry_crops(dataset_path, output_path, img_size=128)
    
    print("\n" + "="*60)
    print("NEXT STEP: Train ripeness classifier")
    print("Run: python3 train_ripeness_classifier.py")
    print("="*60)