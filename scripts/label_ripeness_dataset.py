#!/usr/bin/env python3
"""
Strawberry Ripeness Dataset Labeling Tool
Helps organize and label the 889 unlabeled images into 3 categories:
- Unripe (green/white/pale pink)
- Ripe (bright red) 
- Overripe (dark red/maroon/rotting)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import argparse

def create_labeling_directories(base_path):
    """Create the three labeling directories"""
    labels = ['unripe', 'ripe', 'overripe']
    dirs = {}
    
    for label in labels:
        dir_path = base_path / label
        dir_path.mkdir(exist_ok=True)
        dirs[label] = dir_path
    
    return dirs

def get_image_files(to_label_path):
    """Get all image files from the to_label directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for file_path in to_label_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)

def batch_label_images(image_files, dirs, batch_size=50):
    """Label images in batches for easier management"""
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"Found {total_images} images to label")
    print(f"Will process in {num_batches} batches of {batch_size} images each")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"\n=== BATCH {batch_num + 1}/{num_batches} ===")
        print(f"Images {start_idx + 1} to {end_idx}")
        
        # Show first few images in batch for preview
        for i, img_file in enumerate(batch_files[:5]):
            try:
                with Image.open(img_file) as img:
                    print(f"  {start_idx + i + 1}. {img_file.name} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"  {start_idx + i + 1}. {img_file.name} (Error: {e})")
        
        if len(batch_files) > 5:
            print(f"  ... and {len(batch_files) - 5} more images")
        
        # Interactive labeling for this batch
        label_batch(batch_files, dirs)

def label_batch(batch_files, dirs):
    """Interactive labeling for a batch of images"""
    print("\nLabeling Instructions:")
    print("1 = Unripe (green/white/pale pink)")
    print("2 = Ripe (bright red)")
    print("3 = Overripe (dark red/maroon/rotting)")
    print("s = Skip this image")
    print("q = Quit labeling")
    print("Enter = Next image")
    
    for img_file in batch_files:
        try:
            # Try to display image info
            with Image.open(img_file) as img:
                print(f"\nProcessing: {img_file.name}")
                print(f"Size: {img.size[0]}x{img.size[1]}, Mode: {img.mode}")
                
                while True:
                    choice = input("Label (1/2/3/s/q/Enter): ").strip().lower()
                    
                    if choice == '1' or choice == 'unripe':
                        shutil.move(str(img_file), str(dirs['unripe'] / img_file.name))
                        print(f"✓ Moved to unripe/")
                        break
                    elif choice == '2' or choice == 'ripe':
                        shutil.move(str(img_file), str(dirs['ripe'] / img_file.name))
                        print(f"✓ Moved to ripe/")
                        break
                    elif choice == '3' or choice == 'overripe':
                        shutil.move(str(img_file), str(dirs['overripe'] / img_file.name))
                        print(f"✓ Moved to overripe/")
                        break
                    elif choice == 's' or choice == 'skip':
                        print("⏭️  Skipped")
                        break
                    elif choice == 'q' or choice == 'quit':
                        print("Quitting...")
                        return
                    elif choice == '':
                        print("⏭️  Skipped (no input)")
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, s, q, or press Enter.")
                        
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue

def count_labeled_images(dirs):
    """Count images in each label directory"""
    counts = {}
    for label, dir_path in dirs.items():
        count = len([f for f in dir_path.iterdir() if f.is_file()])
        counts[label] = count
    return counts

def main():
    parser = argparse.ArgumentParser(description='Label strawberry ripeness dataset')
    parser.add_argument('--dataset-path', type=str, 
                       default='model/ripeness_manual_dataset',
                       help='Path to the ripeness dataset directory')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of images to process in each batch')
    parser.add_argument('--count-only', action='store_true',
                       help='Only count current images, do not start labeling')
    
    args = parser.parse_args()
    
    base_path = Path(args.dataset_path)
    to_label_path = base_path / 'to_label'
    
    if not to_label_path.exists():
        print(f"Error: to_label directory not found at {to_label_path}")
        return
    
    # Create labeling directories
    dirs = create_labeling_directories(base_path)
    
    # Count current state
    remaining_files = get_image_files(to_label_path)
    labeled_counts = count_labeled_images(dirs)
    
    print("=== CURRENT STATUS ===")
    print(f"Images remaining to label: {len(remaining_files)}")
    print(f"Already labeled:")
    for label, count in labeled_counts.items():
        print(f"  {label}: {count} images")
    print(f"Total labeled: {sum(labeled_counts.values())}")
    
    if args.count_only:
        return
    
    if len(remaining_files) == 0:
        print("\n✅ All images have been labeled!")
        print("You can now run: python3 train_ripeness_classifier.py")
        return
    
    # Ask user if they want to continue
    response = input(f"\nStart labeling {len(remaining_files)} remaining images? (y/n): ")
    if response.lower() != 'y':
        print("Labeling cancelled.")
        return
    
    # Start labeling process
    batch_label_images(remaining_files, dirs, args.batch_size)
    
    # Final count
    final_counts = count_labeled_images(dirs)
    print("\n=== FINAL COUNTS ===")
    for label, count in final_counts.items():
        print(f"{label}: {count} images")
    
    total_labeled = sum(final_counts.values())
    print(f"Total labeled: {total_labeled}")
    
    if total_labeled > 0:
        print("\n✅ Labeling complete!")
        print("Next steps:")
        print("1. Review the labeled images for quality")
        print("2. Run: python3 train_ripeness_classifier.py")

if __name__ == '__main__':
    main()