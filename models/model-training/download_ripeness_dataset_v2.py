#!/usr/bin/env python3
"""
Download strawberry ripeness dataset from Roboflow
Uses publishable API key and proper format for object detection
"""

from roboflow import Roboflow
import os
from pathlib import Path

def download_dataset(api_key, workspace, project, version, format="yolov8"):
    """Download dataset from Roboflow"""
    print(f"Downloading {project} from Roboflow...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(format)
    
    print(f"✓ Dataset downloaded to: {dataset.location}")
    return dataset.location

def setup_dataset(base_path):
    """Setup dataset structure"""
    dataset_path = Path(base_path)
    
    # Check data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"⚠ Warning: data.yaml not found at {data_yaml}")
        return dataset_path
    
    # Read dataset info
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"\nDataset Information:")
    print(f"  Classes: {data.get('nc', 'unknown')}")
    print(f"  Class names: {data.get('names', [])}")
    print(f"  Train: {data.get('train', 'not specified')}")
    print(f"  Val: {data.get('val', 'not specified')}")
    
    # Count images
    print(f"\nImage counts:")
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        if images_dir.exists():
            count = len(list(images_dir.glob("*.jpg")))
            print(f"  {split}: {count} images")
        else:
            print(f"  {split}: directory not found")
    
    return dataset_path

def search_roboflow_datasets(api_key, query="strawberry ripeness"):
    """Search for datasets on Roboflow"""
    print(f"\n{'='*60}")
    print(f"Searching Roboflow for: '{query}'")
    print(f"{'='*60}")
    
    try:
        rf = Roboflow(api_key=api_key)
        # List available datasets (this is a simplified search)
        # In practice, you'd use the Roboflow API or website
        
        print("\nPopular strawberry datasets on Roboflow:")
        print("1. strawberry-ripeness-detection (if available)")
        print("2. strawberries/strawberry-ripeness")
        print("3. Search at: https://universe.roboflow.com")
        print("\nTo find datasets:")
        print("1. Go to https://universe.roboflow.com")
        print("2. Search for 'strawberry ripeness'")
        print("3. Click on a dataset")
        print("4. Look for: workspace name, project name, version number")
        print("5. Update the script with those values")
        
    except Exception as e:
        print(f"Search error: {e}")

if __name__ == "__main__":
    # Use publishable API key (more reliable)
    API_KEY = "rf_loyjUPpheyQri90d4WZo6mEHEaE2"
    
    print("Strawberry Ripeness Dataset Downloader")
    print("="*60)
    
    # Try specific ripeness datasets
    DATASETS = [
        # (workspace, project, version, description)
        ("strawberries", "strawberry-ripeness", 2, "Strawberry Ripeness v2"),
        ("strawberry-ripeness-detection", "strawberry-ripeness-detection", 1, "Strawberry Ripeness Detection"),
        ("fruit-ripeness-detection", "fruit-ripeness-detection", 2, "Fruit Ripeness Detection"),
    ]
    
    success = False
    
    for workspace, project, version, description in DATASETS:
        try:
            print(f"\nTrying: {description}")
            print(f"Workspace: {workspace}, Project: {project}, Version: {version}")
            
            download_path = download_dataset(API_KEY, workspace, project, version)
            dataset_path = setup_dataset(download_path)
            
            print(f"\n✓ SUCCESS! Dataset downloaded and verified")
            print(f"  Location: {dataset_path}")
            
            # Move to standard location
            final_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/ripeness_detection_dataset")
            if dataset_path != final_path:
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.move(str(dataset_path), str(final_path))
                print(f"  Moved to: {final_path}")
            
            success = True
            break
            
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}...")
            continue
    
    if not success:
        print("\n" + "="*60)
        print("ALL DATASETS FAILED")
        print("="*60)
        print("\nNext steps:")
        print("1. Visit: https://universe.roboflow.com")
        print("2. Search for: 'strawberry ripeness'")
        print("3. Find a dataset with 3 classes: unripe, ripe, overripe")
        print("4. Click 'Download' and select YOLOv8 format")
        print("5. Note the workspace, project, and version numbers")
        print("6. Update this script with those values")
        print("\nOR use the manual approach:")
        print("- Run your detector on images")
        print("- Crop detected strawberries")
        print("- Manually sort into unripe/ripe/overripe folders")
        print("- Train classifier on your own data")
        
        # Offer to help with manual approach
        print("\nWould you like me to help create a manual ripeness dataset?")
        print("I can use your detector to find strawberries, then help you label them!")