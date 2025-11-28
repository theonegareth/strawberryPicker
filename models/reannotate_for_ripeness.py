#!/usr/bin/env python3
"""
Ripeness Re-annotation Helper Script
Converts single-class strawberry dataset to 3-class ripeness dataset
"""

import shutil
from pathlib import Path
from typing import List, Dict

def setup_ripeness_directories():
    """Create backup and new directories for ripeness annotation"""
    
    base_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset")
    
    # Backup original labels
    backup_dir = base_dir / "labels_backup"
    if not backup_dir.exists():
        print("Creating backup of original labels...")
        shutil.copytree(base_dir / "train" / "labels", backup_dir / "train")
        shutil.copytree(base_dir / "valid" / "labels", backup_dir / "valid")
        print("✓ Backup created")
    
    # Create ripeness annotation directories
    ripeness_dir = base_dir / "ripeness_labels"
    ripeness_dir.mkdir(exist_ok=True)
    (ripeness_dir / "train").mkdir(exist_ok=True)
    (ripeness_dir / "valid").mkdir(exist_ok=True)
    
    print(f"\nSetup complete!")
    print(f"Original labels backed up to: {backup_dir}")
    print(f"Create ripeness labels in: {ripeness_dir}")
    print(f"\nClasses: 0=unripe, 1=ripe, 2=overripe")

def create_labelimg_config():
    """Create LabelImg configuration for ripeness classes"""
    
    config_content = """
# Ripeness classes for LabelImg
# Format: class_id class_name

0 unripe
1 ripe
2 overripe
"""
    
    config_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/models/labelimg_classes.txt")
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    print(f"✓ LabelImg config created: {config_path}")
    print("\nTo use with LabelImg:")
    print("1. Install LabelImg: pip install labelImg")
    print("2. Run: labelImg model/dataset/train/images model/labelimg_classes.txt")
    print("3. Save annotations to: model/dataset/ripeness_labels/train/")

def check_annotation_progress():
    """Check how many images have been re-annotated"""
    
    base_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset")
    
    # Count original labels
    train_original = len(list((base_dir / "train" / "labels").glob("*.txt")))
    valid_original = len(list((base_dir / "valid" / "labels").glob("*.txt")))
    
    # Count ripeness labels (if any)
    ripeness_train_dir = base_dir / "ripeness_labels" / "train"
    ripeness_valid_dir = base_dir / "ripeness_labels" / "valid"
    
    train_ripeness = len(list(ripeness_train_dir.glob("*.txt"))) if ripeness_train_dir.exists() else 0
    valid_ripeness = len(list(ripeness_valid_dir.glob("*.txt"))) if ripeness_valid_dir.exists() else 0
    
    print("\nAnnotation Progress:")
    print(f"Training: {train_ripeness}/{train_original} images ({train_ripeness/train_original*100:.1f}%)")
    print(f"Validation: {valid_ripeness}/{valid_original} images ({valid_ripeness/valid_original*100:.1f}%)")
    
    if train_ripeness == train_original and valid_ripeness == valid_original:
        print("\n✅ All images re-annotated! Ready to train ripeness model.")
        return True
    
    return False

def prepare_ripeness_dataset():
    """Prepare dataset for ripeness training once annotation is complete"""
    
    base_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset")
    
    # Check if all annotations are done
    if not check_annotation_progress():
        print("\n⚠️ Not all images have been re-annotated yet.")
        print("Complete re-annotation before preparing dataset.")
        return False
    
    # Backup original labels
    original_train = base_dir / "train" / "labels"
    original_valid = base_dir / "valid" / "labels"
    
    backup_train = base_dir / "labels_original" / "train"
    backup_valid = base_dir / "labels_original" / "valid"
    
    if not backup_train.exists():
        print("\nBacking up original labels...")
        backup_train.parent.mkdir(exist_ok=True)
        shutil.move(str(original_train), str(backup_train))
        print("✓ Training labels backed up")
    
    if not backup_valid.exists():
        shutil.move(str(original_valid), str(backup_valid))
        print("✓ Validation labels backed up")
    
    # Move ripeness labels to main location
    ripeness_train = base_dir / "ripeness_labels" / "train"
    ripeness_valid = base_dir / "ripeness_labels" / "valid"
    
    print("\nMoving ripeness labels to main dataset...")
    shutil.move(str(ripeness_train), str(original_train))
    shutil.move(str(ripeness_valid), str(original_valid))
    
    print("✓ Ripeness dataset ready!")
    print("\nNext steps:")
    print("1. Verify data.yaml has 3 classes")
    print("2. Train model: python3 train_yolov8.py --epochs 150")
    print("3. Validate: python3 model/validate_model.py")
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("STRAWBERRY RIPENESS RE-ANNOTATION HELPER")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Setup ripeness annotation directories")
        print("2. Create LabelImg config")
        print("3. Check annotation progress")
        print("4. Prepare ripeness dataset (when done)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1':
            setup_ripeness_directories()
        elif choice == '2':
            create_labelimg_config()
        elif choice == '3':
            check_annotation_progress()
        elif choice == '4':
            prepare_ripeness_dataset()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()