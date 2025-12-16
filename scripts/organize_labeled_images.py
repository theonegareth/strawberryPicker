#!/usr/bin/env python3
"""
Organize automatically labeled images from JSON results into proper directories.
This script processes the auto_labeling_results JSON files and moves images
from to_label/ to the appropriate subdirectories (unripe/ripe/overripe/).
"""

import json
import os
import shutil
from pathlib import Path
import glob

def organize_labeled_images():
    """Organize automatically labeled images based on JSON results."""
    
    # Paths
    base_dir = Path("model/ripeness_manual_dataset")
    to_label_dir = base_dir / "to_label"
    unripe_dir = base_dir / "unripe"
    ripe_dir = base_dir / "ripe"
    overripe_dir = base_dir / "overripe"
    
    # Find all auto_labeling_results JSON files
    json_files = glob.glob(str(base_dir / "auto_labeling_results_*.json"))
    json_files.sort()  # Process in chronological order
    
    print(f"Found {len(json_files)} auto_labeling_results files")
    
    total_moved = 0
    total_errors = 0
    
    for json_file in json_files:
        print(f"\nProcessing: {os.path.basename(json_file)}")
        
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            batch_moved = 0
            batch_errors = 0
            
            # Process each labeled image (results is a list)
            for label_data in results:
                if isinstance(label_data, dict) and 'image' in label_data and 'label' in label_data:
                    image_name = label_data['image']
                    label = label_data['label']
                    confidence = label_data.get('confidence', 0.0)
                    
                    # Skip if confidence is too low or label is unknown
                    if label in ['unknown', 'skip'] or confidence < 0.6:
                        continue
                    
                    # Source and destination paths
                    src_path = to_label_dir / image_name
                    
                    if not src_path.exists():
                        print(f"  ⚠️  Source file not found: {image_name}")
                        batch_errors += 1
                        continue
                    
                    # Determine destination directory
                    if label == 'unripe':
                        dst_dir = unripe_dir
                    elif label == 'ripe':
                        dst_dir = ripe_dir
                    elif label == 'overripe':
                        dst_dir = overripe_dir
                    else:
                        print(f"  ⚠️  Unknown label '{label}': {image_name}")
                        batch_errors += 1
                        continue
                    
                    dst_path = dst_dir / image_name
                    
                    # Move the file
                    try:
                        shutil.move(str(src_path), str(dst_path))
                        print(f"  ✅ {label} ({confidence:.2f}): {image_name}")
                        batch_moved += 1
                    except Exception as e:
                        print(f"  ❌ Error moving {image_name}: {e}")
                        batch_errors += 1
            
            print(f"  📦 Batch results: {batch_moved} moved, {batch_errors} errors")
            total_moved += batch_moved
            total_errors += batch_errors
            
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
            total_errors += 1
    
    # Final summary
    print(f"\n=== ORGANIZATION COMPLETE ===")
    print(f"Total images moved: {total_moved}")
    print(f"Total errors: {total_errors}")
    
    # Show final counts
    unripe_count = len(list(unripe_dir.glob("*.jpg")))
    ripe_count = len(list(ripe_dir.glob("*.jpg")))
    overripe_count = len(list(overripe_dir.glob("*.jpg")))
    remaining_count = len(list(to_label_dir.glob("*.jpg")))
    
    print(f"\nFinal counts:")
    print(f"  unripe: {unripe_count} images")
    print(f"  ripe: {ripe_count} images")
    print(f"  overripe: {overripe_count} images")
    print(f"  to_label: {remaining_count} images")
    print(f"  TOTAL: {unripe_count + ripe_count + overripe_count + remaining_count} images")
    
    completion_rate = (unripe_count + ripe_count + overripe_count) / (unripe_count + ripe_count + overripe_count + remaining_count) * 100
    print(f"  Completion: {completion_rate:.1f}%")

if __name__ == "__main__":
    organize_labeled_images()