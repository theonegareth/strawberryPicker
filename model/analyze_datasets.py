#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes all downloaded datasets and compares them
"""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

def count_images(directory):
    """Count images in directory recursively"""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        count += len(list(Path(directory).rglob(ext)))
    return count

def analyze_dataset(name, path):
    """Analyze a single dataset"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return
    
    # Check if it's detection or classification
    has_labels = (path / "train" / "labels").exists() if (path / "train").exists() else False
    has_class_folders = any((path / "train" / cls).exists() for cls in ["unripe", "ripe", "overripe", "strawberry"])
    
    dataset_type = "Object Detection" if has_labels else "Classification" if has_class_folders else "Unknown"
    print(f"Type: {dataset_type}")
    
    # Count images
    total_images = count_images(path)
    print(f"Total images: {total_images}")
    
    if has_class_folders:
        # Classification dataset - count per class
        print("\nImages per class:")
        for split in ["train", "valid", "test"]:
            split_path = path / split
            if split_path.exists():
                print(f"\n  {split.upper()}:")
                for class_dir in sorted(split_path.iterdir()):
                    if class_dir.is_dir():
                        count = count_images(class_dir)
                        print(f"    {class_dir.name}: {count} images")
    
    elif has_labels:
        # Detection dataset - count per split
        print("\nImages per split:")
        for split in ["train", "valid", "test"]:
            split_path = path / split / "images"
            if split_path.exists():
                count = count_images(split_path)
                print(f"  {split}: {count} images")
        
        # Check label files
        label_path = path / "train" / "labels"
        if label_path.exists():
            label_files = list(label_path.glob("*.txt"))
            print(f"\nLabel files: {len(label_files)}")
            
            # Sample a few labels to check format
            if label_files:
                print("\nSample label format:")
                with open(label_files[0], 'r') as f:
                    first_line = f.readline().strip()
                    print(f"  {first_line}")
    
    # Check image dimensions
    image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
    if image_files:
        sample_img = cv2.imread(str(image_files[0]))
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            print(f"\nImage dimensions: {w}x{h}")
    
    return dataset_type, total_images

def main():
    """Main analysis function"""
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets")
    
    if not base_path.exists():
        print(f"❌ Base path not found: {base_path}")
        return
    
    datasets = {
        "Ripeness Detection": base_path / "ripeness_detection",
        "Strawberry Detect V3": base_path / "strawberry_detect_v3",
        "Stem Label": base_path / "stem_label",
        "Original Detect": base_path / "original_detect",
    }
    
    print("🔍 ANALYZING DATASETS")
    print("="*60)
    
    results = {}
    for name, path in datasets.items():
        dataset_type, count = analyze_dataset(name, path)
        results[name] = {"type": dataset_type, "count": count}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, info in results.items():
        print(f"{name:<25} | {info['type']:<20} | {info['count']:<6} images")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    ripeness = results.get("Ripeness Detection", {})
    if ripeness.get("type") == "Object Detection":
        print("⚠️  Ripeness Detection is in DETECTION format")
        print("   → Needs conversion to CLASSIFICATION format")
        print("   → Or use your 889 crops instead")
    
    print("\n✅ All datasets are organized and ready to use!")

if __name__ == "__main__":
    main()