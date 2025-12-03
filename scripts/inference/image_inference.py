#!/usr/bin/env python3
"""
Strawberry Detection on Single Image
For testing the model when webcam is not available in WSL
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Strawberry Detection on Single Image')
    parser.add_argument(
        '--detector',
        type=str,
        default='model/weights/strawberry_yolov8s_enhanced.pt',
        help='Path to YOLOv8 detection model'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output image (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  Running on CPU - this will be slower. Consider using GPU if available.")
    
    # Load detection model
    print("🍓 Loading detection model...")
    try:
        from ultralytics import YOLO
        detector = YOLO(args.detector)
        print("✅ Detection model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading detection model: {e}")
        sys.exit(1)
    
    # Load image
    print(f"📷 Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Error: Could not load image from {args.image}")
        sys.exit(1)
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Detect strawberries
    print("🔍 Running detection...")
    results = detector(image)
    
    # Process results
    detection_count = 0
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            if conf < 0.5:  # Confidence threshold
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"strawberry {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detection_count += 1
    
    print(f"✅ Detected {detection_count} strawberries")
    
    # Save result
    output_path = args.output if args.output else f"detection_result_{Path(args.image).stem}.jpg"
    cv2.imwrite(output_path, image)
    print(f"💾 Output saved to: {output_path}")

if __name__ == "__main__":
    # Check for required libraries
    try:
        import torch
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Install with: pip install torch opencv-python ultralytics")
        sys.exit(1)
    
    main()