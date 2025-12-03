#!/usr/bin/env python3
"""
Download strawberry ripeness dataset from Roboflow
Classes: unripe, ripe, overripe (properly annotated)
"""

from roboflow import Roboflow
import os
from pathlib import Path

def download_ripeness_dataset(api_key, workspace, project, version, format="folder"):
    """Download ripeness dataset from Roboflow"""
    print(f"Downloading {project} ripeness dataset from Roboflow...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(format)
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location

def setup_ripeness_dataset(base_path):
    """Setup ripeness dataset structure"""
    dataset_path = Path(base_path)
    
    # Check data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"Warning: data.yaml not found at {data_yaml}")
        return dataset_path
    
    # Read and display dataset info
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"\nDataset info:")
    print(f"  Classes: {data.get('nc', 'unknown')}")
    print(f"  Class names: {data.get('names', [])}")
    print(f"  Train path: {data.get('train', 'not specified')}")
    print(f"  Val path: {data.get('val', 'not specified')}")
    
    # Verify structure
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        if images_dir.exists():
            count = len(list(images_dir.glob("*.jpg")))
            print(f"  {split} images: {count}")
        else:
            print(f"  {split} images: directory not found")
    
    return dataset_path

if __name__ == "__main__":
    # Roboflow credentials - using your API key
    API_KEY = "x7uG1cbSLVYonaFbQdbk"  # Private API key
    
    # Try multiple ripeness datasets (in order of preference)
    DATASETS_TO_TRY = [
        # Format: (workspace, project, version, description)
        ("strawberry-ripeness-detection", "strawberry-ripeness-detection", 1, "Strawberry Ripeness Detection"),
        ("strawberries", "strawberry-ripeness", 1, "Strawberry Ripeness"),
        ("cs-7cg6r", "strawberry-ripeness-detection", 1, "CS Strawberry Ripeness"),
        ("fruit-ripeness-detection", "fruit-ripeness-detection", 1, "Fruit Ripeness Detection"),
    ]
    
    success = False
    
    for workspace, project, version, description in DATASETS_TO_TRY:
        try:
            print(f"\n{'='*60}")
            print(f"Trying: {description}")
            print(f"Workspace: {workspace}, Project: {project}, Version: {version}")
            print(f"{'='*60}")
            
            download_path = download_ripeness_dataset(API_KEY, workspace, project, version)
            dataset_path = setup_ripeness_dataset(download_path)
            
            print(f"\n✓ Successfully downloaded ripeness dataset!")
            print(f"✓ Location: {dataset_path}")
            
            success = True
            break  # Exit loop on success
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            print("Trying next dataset...")
            continue
    
    if not success:
        print("\n✗ Could not download any ripeness dataset")
        print("\nAlternative options:")
        print("1. Search for datasets at: https://universe.roboflow.com")
        print("2. Use the publishable API key: rf_loyjUPpheyQri90d4WZo6mEHEaE2")
        print("3. Create your own dataset using the detector + manual labeling")
    else:
        print("\n" + "="*60)
        print("NEXT STEP: Prepare classification dataset")
        print("Run: python3 prepare_ripeness_crops.py")
        print("="*60)