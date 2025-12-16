#!/usr/bin/env python3
"""
Automated Strawberry Ripeness Labeling System
Uses color analysis to automatically label strawberry ripeness
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from datetime import datetime

class AutoRipenessLabeler:
    def __init__(self):
        """Initialize the automatic ripeness labeler"""
        print("✅ Initialized automatic ripeness labeler")
    
    def analyze_strawberry_color(self, image_path):
        """Analyze the color of strawberries to determine ripeness"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for different ripeness stages
            # Red range (ripe strawberries)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            # Green range (unripe strawberries)
            green_lower = np.array([40, 40, 40])
            green_upper = np.array([80, 255, 255])
            
            # Dark red range (overripe strawberries)
            dark_red_lower = np.array([0, 100, 0])
            dark_red_upper = np.array([20, 255, 100])
            
            # Create masks for each color range
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            dark_red_mask = cv2.inRange(hsv, dark_red_lower, dark_red_upper)
            
            # Calculate percentages
            total_pixels = hsv.shape[0] * hsv.shape[1]
            red_pixels = np.sum(red_mask > 0)
            green_pixels = np.sum(green_mask > 0)
            dark_red_pixels = np.sum(dark_red_mask > 0)
            
            red_percentage = red_pixels / total_pixels
            green_percentage = green_pixels / total_pixels
            dark_red_percentage = dark_red_pixels / total_pixels
            
            # Calculate brightness and saturation for fallback analysis
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            avg_saturation = np.mean(hsv[:, :, 1])
            
            # Determine ripeness based on color percentages
            if green_percentage > 0.3:
                ripeness = "unripe"
                confidence = min(green_percentage * 2, 0.9)
            elif dark_red_percentage > 0.2:
                ripeness = "overripe"
                confidence = min(dark_red_percentage * 2, 0.9)
            elif red_percentage > 0.2:
                ripeness = "ripe"
                confidence = min(red_percentage * 2, 0.9)
            else:
                # Fallback: use brightness and saturation
                if avg_brightness < 80:
                    ripeness = "overripe"
                    confidence = 0.6
                elif avg_brightness > 150:
                    ripeness = "unripe"
                    confidence = 0.6
                else:
                    ripeness = "ripe"
                    confidence = 0.7
            
            return {
                'ripeness': ripeness,
                'confidence': confidence,
                'color_analysis': {
                    'red_percentage': red_percentage,
                    'green_percentage': green_percentage,
                    'dark_red_percentage': dark_red_percentage,
                    'avg_brightness': float(avg_brightness),
                    'avg_saturation': float(avg_saturation)
                }
            }
            
        except Exception as e:
            print(f"Error analyzing color in {image_path}: {e}")
            return None
    
    def batch_auto_label(self, image_files, output_dirs, confidence_threshold=0.6):
        """Automatically label a batch of images"""
        results = []
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            analysis = self.analyze_strawberry_color(image_path)
            
            if analysis and analysis['confidence'] >= confidence_threshold:
                ripeness = analysis['ripeness']
                confidence = analysis['confidence']
                
                # Copy image to appropriate directory
                dest_path = output_dirs[ripeness] / image_path.name
                try:
                    import shutil
                    shutil.copy2(image_path, dest_path)
                    print(f"  ✅ {ripeness} (confidence: {confidence:.2f})")
                    results.append({
                        'image': image_path.name,
                        'label': ripeness,
                        'confidence': confidence,
                        'analysis': analysis['color_analysis']
                    })
                except Exception as e:
                    print(f"  ❌ Error copying file: {e}")
            else:
                print(f"  ⚠️  Low confidence or analysis failed")
                results.append({
                    'image': image_path.name,
                    'label': 'unknown',
                    'confidence': analysis['confidence'] if analysis else 0.0,
                    'analysis': analysis['color_analysis'] if analysis else None
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Automatically label strawberry ripeness dataset')
    parser.add_argument('--dataset-path', type=str, 
                       default='model/ripeness_manual_dataset',
                       help='Path to the ripeness dataset directory')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence for automatic labeling')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    base_path = Path(args.dataset_path)
    to_label_path = base_path / 'to_label'
    
    if not to_label_path.exists():
        print(f"Error: to_label directory not found at {to_label_path}")
        return
    
    # Create output directories
    output_dirs = {}
    for label in ['unripe', 'ripe', 'overripe']:
        dir_path = base_path / label
        dir_path.mkdir(exist_ok=True)
        output_dirs[label] = dir_path
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for file_path in to_label_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    image_files = sorted(image_files)[:args.max_images]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    if not image_files:
        print("No images found to process.")
        return
    
    # Initialize auto labeler
    labeler = AutoRipenessLabeler()
    
    # Process images
    results = labeler.batch_auto_label(image_files, output_dirs, args.confidence_threshold)
    
    # Save results
    results_file = base_path / f'auto_labeling_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    label_counts = {'unripe': 0, 'ripe': 0, 'overripe': 0, 'unknown': 0}
    for result in results:
        label_counts[result['label']] += 1
    
    print("\n=== AUTOMATIC LABELING RESULTS ===")
    for label, count in label_counts.items():
        print(f"{label}: {count} images")
    
    print(f"\nResults saved to: {results_file}")
    
    if label_counts['unknown'] > 0:
        print(f"\n⚠️  {label_counts['unknown']} images need manual review")
        print("You can use the manual labeling tool for these:")
        print("python3 label_ripeness_dataset.py")

if __name__ == '__main__':
    main()
