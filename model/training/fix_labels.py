#!/usr/bin/env python3
"""
Convert all label files to single class (0) for strawberry detection
"""

from pathlib import Path
import os

def convert_labels_to_single_class(labels_dir):
    """Convert all label files to class 0"""
    labels_path = Path(labels_dir)
    converted = 0
    errors = 0
    
    if not labels_path.exists():
        print(f"Directory not found: {labels_path}")
        return 0, 0
    
    label_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(label_files)} label files in {labels_path}")
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert all classes to 0
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class x y w h
                    # Change class to 0, keep coordinates
                    parts[0] = '0'
                    new_lines.append(' '.join(parts) + '\n')
            
            # Write back
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            
            converted += 1
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            errors += 1
    
    print(f"Converted {converted} label files, {errors} errors")
    return converted, errors

if __name__ == "__main__":
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset")
    
    # Convert train, valid, and test labels
    for split in ["train", "valid", "test"]:
        labels_dir = base_path / split / "labels"
        print(f"\nProcessing {split} labels...")
        converted, errors = convert_labels_to_single_class(labels_dir)
    
    print("\n✓ All labels converted to single class (strawberry)")
    print("✓ Model will now detect ALL strawberries regardless of ripeness")