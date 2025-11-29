#!/usr/bin/env python3
"""
Convert Ripeness Detection Dataset to Classification Format
Extracts crops from bounding boxes and organizes by class
"""

import os
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

def convert_detection_to_classification(data_path, output_path):
    """
    Convert YOLO detection format to classification format
    
    Args:
        data_path: Path to detection dataset (with train/valid/test, images/labels)
        output_path: Path to save classification dataset
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Class mapping (from detection classes to classification classes)
    # Detection dataset has: unripe, partially-ripe, ripe
    # We'll map: unripe -> unripe, partially-ripe -> ripe, ripe -> ripe
    CLASS_MAPPING = {
        0: "unripe",  # unripe
        1: "ripe",    # partially-ripe -> ripe (good enough to pick)
        2: "ripe"     # ripe -> ripe
    }
    
    # Create output directories
    for split in ["train", "valid", "test"]:
        for class_name in ["unripe", "ripe"]:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        images_dir = data_path / split / "images"
        labels_dir = data_path / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  Skipping {split}: images or labels directory not found")
            continue
        
        # Get all images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print(f"⚠️  No images found in {images_dir}")
            continue
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Read corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Process each bounding box
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_width = float(parts[3]) * w
                box_height = float(parts[4]) * h
                
                # Map class
                if class_id not in CLASS_MAPPING:
                    continue
                
                class_name = CLASS_MAPPING[class_id]
                
                # Calculate crop coordinates
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                
                # Ensure coordinates are within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Crop the strawberry
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Save crop
                crop_filename = f"{img_path.stem}_box{idx}.jpg"
                crop_path = output_path / split / class_name / crop_filename
                cv2.imwrite(str(crop_path), crop)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*60}")
    
    # Print summary
    for split in ["train", "valid", "test"]:
        print(f"\n{split.upper()}:")
        for class_name in ["unripe", "ripe"]:
            class_dir = output_path / split / class_name
            count = len(list(class_dir.glob("*.jpg")))
            print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    # Convert the ripeness detection dataset
    data_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_detection"
    output_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_classification_converted"
    
    print("🔄 Converting ripeness detection dataset to classification format...")
    print("This will extract crops from bounding boxes and organize by class.")
    
    convert_detection_to_classification(data_path, output_path)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print("\nClass mapping applied:")
    print("  - Detection class 0 (unripe) -> Classification: unripe")
    print("  - Detection class 1 (partially-ripe) -> Classification: ripe")
    print("  - Detection class 2 (ripe) -> Classification: ripe")
    print("\nNote: 'partially-ripe' strawberries are considered ripe enough to pick.")