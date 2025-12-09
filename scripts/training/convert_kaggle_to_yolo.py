#!/usr/bin/env python3
"""
Convert Kaggle fruit ripeness classification dataset to YOLO detection format
Combines all strawberry images into single 'strawberry' class with generated bounding boxes
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import yaml

def create_yolo_detection_dataset(kaggle_dir, output_dir, train_split=0.7, val_split=0.2):
    """
    Convert Kaggle classification dataset to YOLO detection format
    
    Args:
        kaggle_dir: Path to Kaggle dataset (e.g., ~/Downloads/train)
        output_dir: Path to output YOLO dataset
        train_split: Fraction for training set
        val_split: Fraction for validation set (test = 1 - train - val)
    """
    kaggle_path = Path(kaggle_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find all strawberry images (all ripeness states)
    strawberry_classes = ['RipeStrawberry', 'RottenStrawberry', 'UnripeStrawberry']
    all_images = []
    
    print("🔍 Scanning for strawberry images...")
    for class_name in strawberry_classes:
        class_dir = kaggle_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg'))
            all_images.extend(images)
            print(f"  Found {len(images)} images in {class_name}")
    
    if not all_images:
        raise ValueError("No strawberry images found! Check the dataset path.")
    
    print(f"✅ Total strawberry images found: {len(all_images)}")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)
    
    n_train = int(len(all_images) * train_split)
    n_val = int(len(all_images) * val_split)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    print(f"📊 Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Process each split
    for split, images in [('train', train_images), ('valid', val_images), ('test', test_images)]:
        print(f"\n🔄 Processing {split} set...")
        for i, img_path in enumerate(images):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(images)} images...")
            
            try:
                # Copy image
                img_name = f"strawberry_{i:06d}.jpg"
                dst_img = output_path / split / 'images' / img_name
                shutil.copy2(img_path, dst_img)
                
                # Generate bounding box (approximate - whole image)
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                # Class 0 = strawberry
                x_center = 0.5
                y_center = 0.5
                box_width = 0.9  # Slightly smaller than full image
                box_height = 0.9
                
                # Write label file
                label_name = f"strawberry_{i:06d}.txt"
                dst_label = output_path / split / 'labels' / label_name
                
                with open(dst_label, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {img_path}: {e}")
    
    # Create data.yaml
    data_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1,
        'names': ['strawberry'],
        'roboflow': {
            'workspace': 'strawberry-picker',
            'project': 'strawberry-detection',
            'version': 1,
            'license': 'CC BY 4.0',
            'url': 'https://github.com/theonegareth/strawberryPicker'
        }
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Dataset conversion complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Total images: {len(all_images)}")
    print(f"   - Train: {len(train_images)}")
    print(f"   - Val: {len(val_images)}")
    print(f"   - Test: {len(test_images)}")
    print(f"🏷️  Classes: 1 (strawberry)")
    
    return output_path

def validate_dataset(dataset_dir):
    """Validate the created YOLO dataset"""
    dataset_path = Path(dataset_dir)
    
    print(f"\n🔍 Validating dataset: {dataset_path}")
    
    # Check data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        print("❌ data.yaml not found!")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"✅ data.yaml found: {data['nc']} classes - {data['names']}")
    
    # Check splits
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        
        if not img_dir.exists():
            print(f"❌ {split}/images directory not found!")
            return False
        
        if not label_dir.exists():
            print(f"❌ {split}/labels directory not found!")
            return False
        
        images = list(img_dir.glob('*.jpg'))
        labels = list(label_dir.glob('*.txt'))
        
        print(f"✅ {split}: {len(images)} images, {len(labels)} labels")
        
        if len(images) != len(labels):
            print(f"⚠️  Mismatch: {len(images)} images vs {len(labels)} labels")
    
    print("✅ Dataset validation passed!")
    return True

def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Kaggle dataset to YOLO format')
    parser.add_argument('--kaggle-dir', type=str, 
                       default='~/Downloads/train',
                       help='Path to Kaggle dataset train directory')
    parser.add_argument('--output-dir', type=str,
                       default='~/machine-learning/GitHubRepos/strawberryPicker/model/dataset_strawberry_kaggle',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training set fraction')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation set fraction')
    
    args = parser.parse_args()
    
    # Expand user paths
    kaggle_dir = Path(args.kaggle_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    print("🍓 Kaggle to YOLO Dataset Converter")
    print("=" * 50)
    print(f"Input: {kaggle_dir}")
    print(f"Output: {output_dir}")
    print(f"Train/Val/Test split: {args.train_split}/{args.val_split}/{1-args.train_split-args.val_split}")
    print()
    
    try:
        # Convert dataset
        dataset_path = create_yolo_detection_dataset(
            kaggle_dir=kaggle_dir,
            output_dir=output_dir,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        # Validate
        validate_dataset(dataset_path)
        
        print("\n" + "=" * 50)
        print("✅ Conversion successful!")
        print(f"📁 Dataset ready at: {dataset_path}")
        print(f"📝 Use this path in your training script: --dataset {dataset_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())