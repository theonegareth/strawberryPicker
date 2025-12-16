#!/usr/bin/env python3
"""
Complete the final batch of ripeness labeling using conservative color analysis.
This script processes the remaining 46 images with higher confidence thresholds.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import cv2

def analyze_ripeness_conservative(image_path, confidence_threshold=0.8):
    """
    Conservative ripeness analysis with higher confidence thresholds.
    """
    try:
        # Load and convert image
        img = cv2.imread(str(image_path))
        if img is None:
            return None, 0.0
            
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for strawberry ripeness (more conservative)
        # Red ranges (ripe strawberries)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Green ranges (unripe strawberries)
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        
        # Yellow/orange ranges (overripe strawberries)
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Calculate percentages
        total_pixels = img.shape[0] * img.shape[1]
        red_percentage = np.sum(red_mask > 0) / total_pixels
        green_percentage = np.sum(green_mask > 0) / total_pixels
        yellow_percentage = np.sum(yellow_mask > 0) / total_pixels
        
        # Conservative classification logic
        if red_percentage > 0.35 and red_percentage > green_percentage and red_percentage > yellow_percentage:
            return "ripe", red_percentage
        elif green_percentage > 0.25 and green_percentage > red_percentage and green_percentage > yellow_percentage:
            return "unripe", green_percentage
        elif yellow_percentage > 0.20 and yellow_percentage > red_percentage and yellow_percentage > green_percentage:
            return "overripe", yellow_percentage
        else:
            # If no clear dominant color, use the highest percentage
            max_percentage = max(red_percentage, green_percentage, yellow_percentage)
            if max_percentage == red_percentage:
                return "ripe", red_percentage
            elif max_percentage == green_percentage:
                return "unripe", green_percentage
            else:
                return "overripe", yellow_percentage
                
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None, 0.0

def main():
    """Complete the final batch of labeling."""
    
    # Paths
    to_label_dir = Path("model/ripeness_manual_dataset/to_label")
    unripe_dir = Path("model/ripeness_manual_dataset/unripe")
    ripe_dir = Path("model/ripeness_manual_dataset/ripe")
    overripe_dir = Path("model/ripeness_manual_dataset/overripe")
    
    # Get remaining files
    remaining_files = list(to_label_dir.glob("*.jpg"))
    
    if not remaining_files:
        print("No images remaining to label!")
        return
    
    print(f"=== FINAL BATCH LABELING ===")
    print(f"Processing {len(remaining_files)} remaining images with conservative analysis...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(remaining_files),
        "unripe": 0,
        "ripe": 0,
        "overripe": 0,
        "unknown": 0,
        "images": []
    }
    
    for i, image_path in enumerate(remaining_files, 1):
        print(f"Processing {i}/{len(remaining_files)}: {image_path.name}")
        
        # Analyze with conservative threshold
        label, confidence = analyze_ripeness_conservative(image_path, confidence_threshold=0.8)
        
        if label:
            # Move to appropriate directory
            if label == "unripe":
                dest = unripe_dir / image_path.name
                results["unripe"] += 1
            elif label == "ripe":
                dest = ripe_dir / image_path.name
                results["ripe"] += 1
            elif label == "overripe":
                dest = overripe_dir / image_path.name
                results["overripe"] += 1
            
            try:
                shutil.move(str(image_path), str(dest))
                print(f"  ✅ {label} (confidence: {confidence:.2f})")
                results["images"].append({
                    "filename": image_path.name,
                    "label": label,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"  ❌ Error moving file: {e}")
                results["unknown"] += 1
        else:
            print(f"  ⚠️  Analysis failed")
            results["unknown"] += 1
    
    # Save results
    results_file = f"model/ripeness_manual_dataset/final_labeling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print(f"\n=== FINAL LABELING COMPLETE ===")
    print(f"unripe: {results['unripe']} images")
    print(f"ripe: {results['ripe']} images")
    print(f"overripe: {results['overripe']} images")
    print(f"unknown: {results['unknown']} images")
    print(f"Total processed: {results['total_processed']} images")
    
    # Calculate final dataset statistics
    total_unripe = len(list(unripe_dir.glob("*.jpg")))
    total_ripe = len(list(ripe_dir.glob("*.jpg")))
    total_overripe = len(list(overripe_dir.glob("*.jpg")))
    remaining = len(list(to_label_dir.glob("*.jpg")))
    total_dataset = total_unripe + total_ripe + total_overripe + remaining
    
    completion_percentage = (total_dataset - remaining) / total_dataset * 100
    
    print(f"\n=== FINAL DATASET STATUS ===")
    print(f"unripe: {total_unripe} images")
    print(f"ripe: {total_ripe} images")
    print(f"overripe: {total_overripe} images")
    print(f"to_label: {remaining} images")
    print(f"TOTAL: {total_dataset} images")
    print(f"Completion: {completion_percentage:.1f}%")
    
    if remaining == 0:
        print(f"\n🎉 DATASET LABELING 100% COMPLETE! 🎉")
        print(f"Total labeled images: {total_dataset}")
    else:
        print(f"\n⚠️  {remaining} images still need manual review")
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()