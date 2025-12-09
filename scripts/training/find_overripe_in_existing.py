#!/usr/bin/env python3
"""
Find potential overripe strawberries in existing dataset
Uses color analysis to identify dark/overripe strawberries
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def analyze_ripeness(image_path, threshold_darkness=60):
    """
    Analyze if strawberry is overripe based on color darkness
    Returns: (is_overripe_prob, reason)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, "Could not read image"
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Focus on red channel (overripe strawberries are very dark red)
    red_channel = img[:, :, 2]  # BGR format, so index 2 is red
    
    # Calculate average redness/darkness
    avg_red = np.mean(red_channel)
    avg_brightness = np.mean(hsv[:, :, 2])  # Value channel
    
    # Overripe strawberries are very dark (low brightness) but still red (not green)
    if avg_brightness < threshold_darkness and avg_red > 80:
        # Probability based on how dark it is
        prob = min(1.0, (threshold_darkness - avg_brightness) / threshold_darkness)
        return prob, f"Very dark (brightness: {avg_brightness:.1f})"
    
    return 0, f"Brightness: {avg_brightness:.1f}, Red: {avg_red:.1f}"

def scan_for_overripe(dataset_path, output_dir, prob_threshold=0.6):
    """
    Scan existing dataset for potential overripe strawberries
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Classes to scan (ripe strawberries are most likely to be overripe)
    scan_classes = ["ripe", "partially-ripe"]  # Focus on ripe strawberries
    
    found_overripe = []
    
    print(f"🔍 Scanning for overripe strawberries in: {dataset_path}")
    print("=" * 60)
    
    for split in ["train", "valid", "test"]:
        for class_name in scan_classes:
            class_dir = dataset_path / split / class_name
            if not class_dir.exists():
                continue
            
            image_files = list(class_dir.glob("*.jpg"))
            print(f"\nScanning {split}/{class_name}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"Scanning {split}/{class_name}"):
                prob, reason = analyze_ripeness(img_path)
                
                if prob >= prob_threshold:
                    found_overripe.append({
                        "path": img_path,
                        "prob": prob,
                        "reason": reason,
                        "original_class": class_name,
                        "split": split
                    })
    
    # Sort by probability (most likely overripe first)
    found_overripe.sort(key=lambda x: x["prob"], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Found {len(found_overripe)} potential overripe strawberries!")
    print(f"{'='*60}\n")
    
    # Copy top candidates to output directory
    if found_overripe:
        print("Copying top candidates to output directory...")
        for i, candidate in enumerate(found_overripe[:50]):  # Top 50 candidates
            src = candidate["path"]
            dst = output_dir / f"overripe_candidate_{i+1:03d}_prob{candidate['prob']:.2f}.jpg"
            shutil.copy2(src, dst)
            
            if i < 10:  # Print details for top 10
                print(f"{i+1:2d}. Probability: {candidate['prob']:.2%}")
                print(f"    Source: {candidate['split']}/{candidate['original_class']}")
                print(f"    Reason: {candidate['reason']}")
                print(f"    Saved to: {dst.name}")
                print()
    
    return found_overripe

if __name__ == "__main__":
    # Path to your existing 3-class dataset
    dataset_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_classification_converted"
    
    # Output directory for potential overripe examples
    output_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/overripe_candidates_from_existing"
    
    print("🍓 Overripe Strawberry Finder")
    print("=" * 60)
    print("Scanning your existing crops for overripe examples...")
    
    # Scan with default threshold
    candidates = scan_for_overripe(dataset_path, output_dir, prob_threshold=0.6)
    
    print(f"\n{'='*60}")
    print(f"📊 Summary: Found {len(candidates)} potential overripe strawberries")
    print(f"💾 Top candidates saved to: {output_dir}")
    print(f"{'='*60}")
    
    if len(candidates) >= 20:
        print("✅ Great! You already have many overripe examples in your dataset!")
        print("👍 You can use these to start your overripe class collection.")
    else:
        print("ℹ️  Found only a few candidates. You'll need to collect more overripe images.")
        print("💡 Try letting strawberries overripen naturally and photographing them.")