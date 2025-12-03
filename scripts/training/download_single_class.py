#!/usr/bin/env python3
"""
Download single-class strawberry dataset from Roboflow
Detects ALL strawberries regardless of ripeness
"""

from roboflow import Roboflow
import os
import shutil
from pathlib import Path

def download_dataset(api_key, workspace, project, version, format="yolov8"):
    """Download dataset from Roboflow"""
    print(f"Downloading {project} dataset from Roboflow...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(format)
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location

def setup_dataset_structure(base_path):
    """Setup proper dataset structure for YOLOv8 training"""
    dataset_path = Path(base_path)
    
    # Expected structure after Roboflow download
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    test_images = dataset_path / "test" / "images"
    test_labels = dataset_path / "test" / "labels"
    data_yaml = dataset_path / "data.yaml"
    
    # Verify structure
    print("\nVerifying dataset structure:")
    print(f"Train images: {train_images} - Exists: {train_images.exists()}")
    print(f"Train labels: {train_labels} - Exists: {train_labels.exists()}")
    print(f"Valid images: {valid_images} - Exists: {valid_images.exists()}")
    print(f"Valid labels: {valid_labels} - Exists: {valid_labels.exists()}")
    print(f"Test images: {test_images} - Exists: {test_images.exists()}")
    print(f"Data YAML: {data_yaml} - Exists: {data_yaml.exists()}")
    
    # Check data.yaml content
    if data_yaml.exists():
        print(f"\ndata.yaml content:")
        with open(data_yaml, 'r') as f:
            print(f.read())
    
    # Modify data.yaml to treat all strawberry classes as single class
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Change to single class
        data['nc'] = 1
        data['names'] = ['strawberry']
        
        # Remove class-specific names but keep them in comments
        if 'names' in data and isinstance(data['names'], list):
            original_names = data['names']
            print(f"\nOriginal classes: {original_names}")
            print(f"Consolidating to single class: strawberry")
        
        with open(data_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"\nUpdated data.yaml to single class: strawberry")
        print(f"All strawberry types (green, red, white, flower) will be detected as 'strawberry'")
    
    return dataset_path

if __name__ == "__main__":
    # Roboflow credentials
    API_KEY = "x7uG1cbSLVYonaFbQdbk"  # Private API key
    WORKSPACE = "strawberries"
    PROJECT = "strawberry-detect"
    VERSION = 1  # Adjust if needed
    
    # Download dataset
    download_path = download_dataset(API_KEY, WORKSPACE, PROJECT, VERSION)
    
    # Setup structure
    dataset_path = setup_dataset_structure(download_path)
    
    print(f"\n✓ Dataset ready at: {dataset_path}")
    print(f"✓ Single class: strawberry (detects ALL ripeness states)")
    print(f"✓ Ready for enhanced training!")