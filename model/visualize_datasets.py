#!/usr/bin/env python3
"""
Visual Dataset Comparison
Shows sample images from each dataset to understand differences
"""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

def get_sample_images(dataset_path, num_samples=3):
    """Get sample images from a dataset"""
    image_files = []
    
    # Look for images in train split
    train_path = dataset_path / "train"
    if (train_path / "images").exists():
        # Detection format
        image_path = train_path / "images"
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    else:
        # Classification format or unknown
        image_files = list(train_path.rglob("*.jpg")) + list(train_path.rglob("*.png"))
    
    # Return random samples
    if len(image_files) >= num_samples:
        return random.sample(image_files, num_samples)
    elif image_files:
        return image_files
    else:
        return []

def visualize_dataset(name, dataset_path, axs):
    """Visualize a dataset"""
    samples = get_sample_images(dataset_path)
    
    if not samples:
        axs[0].text(0.5, 0.5, "No images found", ha='center', va='center', transform=axs[0].transAxes)
        for ax in axs:
            ax.axis('off')
        return
    
    # Display samples
    for i, (ax, img_path) in enumerate(zip(axs, samples)):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"Sample {i+1}\n{img_path.name[:20]}...")
        ax.axis('off')
    
    # Add dataset info
    info_text = f"Dataset: {name}\nPath: {dataset_path.relative_to(Path.cwd())}"
    axs[0].text(0.02, 0.98, info_text, transform=axs[0].transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def main():
    """Main visualization function"""
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
    
    # Create figure
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 5*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    print("🖼️  CREATING VISUALIZATION")
    print("="*60)
    
    for idx, (name, path) in enumerate(datasets.items()):
        print(f"Processing: {name}")
        visualize_dataset(name, path, axes[idx])
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: dataset_comparison.png")
    
    # Also create a simple text summary
    print(f"\n{'='*60}")
    print("DATASET COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for name, path in datasets.items():
        print(f"\n{name}:")
        print(f"  Location: {path}")
        
        # Check format
        if (path / "train" / "images").exists():
            print(f"  Format: Object Detection (images + labels)")
            # Count classes from data.yaml if exists
            data_yaml = path / "data.yaml"
            if data_yaml.exists():
                with open(data_yaml, 'r') as f:
                    content = f.read()
                    if 'nc:' in content:
                        nc_line = [line for line in content.split('\n') if 'nc:' in line]
                        if nc_line:
                            print(f"  {nc_line[0].strip()}")
                    if 'names:' in content:
                        names_line = [line for line in content.split('\n') if 'names:' in line]
                        if names_line:
                            print(f"  {names_line[0].strip()}")
        else:
            print(f"  Format: Classification (folder-based)")

if __name__ == "__main__":
    main()