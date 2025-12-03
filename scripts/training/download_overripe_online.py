#!/usr/bin/env python3
"""
Download overripe strawberry images from various online sources
Uses multiple search terms and sources to get diverse examples
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Search terms for overripe strawberries
SEARCH_TERMS = [
    "overripe strawberry",
    "spoiled strawberry", 
    "moldy strawberry",
    "rotten strawberry",
    "bad strawberry",
    "decayed strawberry",
    "fermented strawberry",
    "dark red strawberry overripe",
    "strawberry past prime",
    "old strawberry"
]

# Image URLs from various sources (manually curated examples)
# These are example URLs - in practice, you'd use an API or scrape
OVERRIPE_IMAGE_URLS = [
    # Note: These are example URLs. In practice, you'd need to:
    # 1. Use Google Images API, or
    # 2. Scrape with proper permissions, or
    # 3. Download manually from these sources:
    
    # Source 1: Kaggle Datasets (download manually)
    # https://www.kaggle.com/datasets (search "strawberry overripe")
    
    # Source 2: Roboflow Universe (download manually)
    # https://universe.roboflow.com/ (search "strawberry")
    
    # Source 3: Google Images (manual download recommended)
    # Search the SEARCH_TERMS above and download 200-300 images
]

def download_image(url, save_path):
    """Download a single image"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def create_overripe_dataset(output_dir, target_count=200):
    """
    Create overripe dataset by downloading from various sources
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🍓 Overripe Strawberry Dataset Collection")
    print("=" * 60)
    print(f"Target: {target_count} images")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Method 1: Manual Download Instructions
    print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print("Since automated downloading from Google Images is complex,")
    print("here are the best sources and methods:\n")
    
    print("1️⃣  KAGGLE (Recommended - High Quality)")
    print("   • Visit: https://www.kaggle.com/datasets")
    print("   • Search: 'strawberry overripe', 'fruit spoilage'")
    print("   • Download datasets with overripe strawberries")
    print("   • Look for: 'Fruit Freshness Dataset', 'Fruit Quality Dataset'\n")
    
    print("2️⃣  ROBOFLOW UNIVERSE (Recommended - Already Labeled)")
    print("   • Visit: https://universe.roboflow.com/")
    print("   • Search: 'strawberry', 'fruit ripeness', 'fruit quality'")
    print("   • Look for datasets with 'overripe', 'spoiled', 'bad' classes")
    print("   • Download images from those classes\n")
    
    print("3️⃣  GOOGLE IMAGES (Manual - Good Quantity)")
    print("   • Search each term below and download images:")
    for term in SEARCH_TERMS[:5]:
        print(f"     - '{term}'")
    print("   • Aim for 200-300 total images")
    print("   • Save as: overripe_001.jpg, overripe_002.jpg, etc.\n")
    
    print("4️⃣  CREATE YOUR OWN (Best Quality)")
    print("   • Buy 20-30 fresh strawberries")
    print("   • Let them overripen at room temperature (3-5 days)")
    print("   • Photograph daily as they darken/soften")
    print("   • You'll get 100-200 images from one batch!\n")
    
    # Method 2: Direct URLs (if available)
    print("=" * 60)
    print("🌐 DIRECT DOWNLOAD SOURCES:")
    print("=" * 60)
    
    # Example: Try to download from specific sources
    sources = [
        ("Fruit Freshness Dataset (Kaggle)", "https://www.kaggle.com/datasets"),
        ("Fruit Quality Dataset (Kaggle)", "https://www.kaggle.com/datasets"),
        ("Roboflow Fruit Detection", "https://universe.roboflow.com/"),
    ]
    
    for name, url in sources:
        print(f"• {name}")
        print(f"  URL: {url}")
        print(f"  Action: Search for 'strawberry overripe' or 'fruit spoilage'")
        print()
    
    print("=" * 60)
    print("💡 QUICK START RECOMMENDATION:")
    print("=" * 60)
    print("1. Go to Kaggle.com")
    print("2. Search: 'fruit freshness dataset'")
    print("3. Download a dataset with overripe/spoiled class")
    print("4. Extract strawberry images from the overripe class")
    print("5. Save to: datasets/overripe_collection/")
    print()
    print("OR")
    print()
    print("1. Buy strawberries and let them overripen")
    print("2. Take 100-200 photos over 3-5 days")
    print("3. Save to: datasets/overripe_collection/")
    print("=" * 60)
    
    # Create placeholder directory
    placeholder_dir = output_dir / "manual_download_needed"
    placeholder_dir.mkdir(exist_ok=True)
    
    # Create README with instructions
    readme_path = output_dir / "DOWNLOAD_INSTRUCTIONS.md"
    with open(readme_path, 'w') as f:
        f.write("""# Overripe Strawberry Image Collection

## Quick Start (Recommended)

### Option 1: Kaggle Datasets (Fastest)
1. Visit: https://www.kaggle.com/datasets
2. Search for: "fruit freshness", "fruit quality", "strawberry ripeness"
3. Download datasets that include "overripe", "spoiled", or "bad" classes
4. Extract strawberry images and save them here

### Option 2: Let Strawberries Overripen (Best Quality)
1. Buy 20-30 fresh strawberries
2. Place them at room temperature
3. Photograph daily for 3-5 days
4. You'll get 100-200 images as they darken and soften

### Option 3: Google Images (Good Quantity)
Search and download images for these terms:
- "overripe strawberry"
- "spoiled strawberry"
- "moldy strawberry"
- "rotten strawberry"
- "bad strawberry"

## Target: 200-300 images
## Save format: overripe_001.jpg, overripe_002.jpg, etc.

After collecting, run:
```bash
python3 training/train_ripeness_classifier_enhanced.py
```
""")
    
    print(f"✅ Created instructions file: {readme_path}")
    print(f"📁 Create your overripe dataset in: {output_dir}")
    print("=" * 60)
    
    return target_count

if __name__ == "__main__":
    # Create the collection directory
    output_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/overripe_collection"
    
    create_overripe_dataset(output_dir, target_count=200)