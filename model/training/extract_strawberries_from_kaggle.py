#!/usr/bin/env python3
"""
Extract strawberry images from Kaggle fruit ripeness dataset
Filters for strawberries and organizes by ripeness class
"""

import shutil
from pathlib import Path
import os

def extract_strawberries(kaggle_dataset_path, output_dir):
    """
    Extract only strawberry images from the Kaggle dataset
    
    Kaggle dataset structure:
    fruit-ripeness-dataset/
    ├── train/
    │   ├── Apple/
    │   │   ├── Bad/
    │   │   ├── Average/
    │   │   └── Good/
    │   ├── Banana/
    │   │   ├── Bad/
    │   │   ├── Average/
    │   │   └── Good/
    │   ├── Strawberry/  ← We want this
    │   │   ├── Bad/     ← Maps to overripe
    │   │   ├── Average/ ← Maps to partially-ripe
    │   │   └── Good/    ← Maps to ripe
    │   └── ...
    └── test/
        └── ... (same structure)
    """
    
    kaggle_path = Path(kaggle_dataset_path)
    output_path = Path(output_dir)
    
    if not kaggle_path.exists():
        print(f"❌ Error: Kaggle dataset not found at {kaggle_path}")
        print("Please download the dataset first from:")
        print("https://www.kaggle.com/datasets/dudinurdiyansah/fruit-ripeness-dataset")
        return
    
    print("🍓 Extracting Strawberries from Kaggle Dataset")
    print("=" * 60)
    print(f"Source: {kaggle_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Mapping from Kaggle classes to our ripeness classes
    ripeness_mapping = {
        "RottenStrawberry": "overripe",      # Overripe/spoiled
        "UnripeStrawberry": "unripe",        # Unripe
        "RipeStrawberry": "ripe"             # Ripe
    }
    
    # Statistics
    stats = {"overripe": 0, "partially-ripe": 0, "ripe": 0}
    
    # Process train and test splits
    for split in ["train", "test"]:
        split_path = kaggle_path / split
        if not split_path.exists():
            continue
        
        strawberry_path = split_path / "Strawberry"
        if not strawberry_path.exists():
            print(f"⚠️  No strawberry directory found in {split}/")
            continue
        
        print(f"\n📂 Processing {split} split...")
        
        # Process each ripeness class
        for kaggle_class, our_class in ripeness_mapping.items():
            class_path = strawberry_path / kaggle_class
            if not class_path.exists():
                print(f"  ⚠️  No {kaggle_class} class found")
                continue
            
            # Get all images
            image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
            
            if not image_files:
                print(f"  ⚠️  No images in {kaggle_class}")
                continue
            
            print(f"  📸 Found {len(image_files)} images in {kaggle_class} → {our_class}")
            
            # Copy images to output
            for img_path in image_files:
                # Create filename: overripe_001.jpg, overripe_002.jpg, etc.
                count = stats[our_class] + 1
                ext = img_path.suffix
                new_filename = f"{our_class}_{count:03d}{ext}"
                
                # Create output directory
                output_class_dir = output_path / our_class
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy image
                dst_path = output_class_dir / new_filename
                shutil.copy2(img_path, dst_path)
                
                stats[our_class] += 1
    
    print(f"\n{'='*60}")
    print("✅ Extraction Complete!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count} images")
    print(f"{'='*60}")
    print(f"💾 Saved to: {output_path}")
    
    # Check if we have enough overripe images
    if stats["overripe"] >= 50:
        print(f"🎉 Great! You have {stats['overripe']} overripe images to start with.")
        print("👍 You can now train your 4-class model!")
    else:
        print(f"ℹ️  You have {stats['overripe']} overripe images.")
        print("💡 Consider collecting more from other sources or letting strawberries overripen naturally.")
    
    return stats

if __name__ == "__main__":
    # Path where you downloaded the Kaggle dataset
    # Default: ~/Downloads/fruit-ripeness-dataset
    kaggle_path = input("Enter path to Kaggle dataset (fruit-ripeness-dataset): ").strip()
    
    if not kaggle_path:
        kaggle_path = str(Path.home() / "Downloads" / "fruit-ripeness-dataset")
        print(f"Using default path: {kaggle_path}")
    
    # Output directory
    output_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/overripe_from_kaggle"
    
    # Extract strawberries
    stats = extract_strawberries(kaggle_path, output_dir)
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Check extracted images: {output_dir}")
    print(f"2. Manually review and move to: datasets/ripeness_classification_4class/")
    print(f"3. Run: python3 training/train_ripeness_classifier_enhanced.py")