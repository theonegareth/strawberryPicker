#!/usr/bin/env python3
"""
Simple test script for the trained strawberry detection model
"""

import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
from .training_registry import get_registry

def test_model(model_path, image_path, save_output=True):
    """Test the trained model on a single image"""
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Load image
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Run inference
    print("Running inference...")
    results = model(image, conf=0.25, iou=0.45)
    
    # Process results
    result = results[0]
    
    # Print detection summary
    print(f"\nDetection Summary:")
    print(f"Number of strawberries detected: {len(result.boxes)}")
    
    if len(result.boxes) > 0:
        print("\nDetections:")
        for i, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            coords = box.xyxy[0].tolist()
            print(f"  Strawberry {i+1}: Confidence={conf:.2f}, Box={coords}")
    
    # Save or display result
    if save_output:
        output_path = Path("test_output.jpg")
        result.save(str(output_path))
        print(f"\nOutput saved to: {output_path}")
        print(f"Image with bounding boxes saved successfully!")
    else:
        # Display in window
        result.show()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test trained strawberry detection model')
    parser.add_argument('--model', type=str, 
                       default='model/weights/strawberry_yolov8n.pt',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, 
                       default='assets/1.jpg',
                       help='Path to test image')
    parser.add_argument('--no-save', action='store_true',
                       help='Display instead of saving output')
    
    args = parser.parse_args()
    
    try:
        success = test_model(args.model, args.image, save_output=not args.no_save)
        if success:
            print("\n✓ Model test completed successfully!")
        else:
            print("\n✗ Model test failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())