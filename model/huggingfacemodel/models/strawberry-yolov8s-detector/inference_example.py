#!/usr/bin/env python3
"""
Simple inference example for the strawberry detection model.
"""

from ultralytics import YOLO
import cv2
import sys

def main():
    # Load the model
    print("Loading strawberry detection model...")
    model = YOLO('best.pt')
    
    # Run inference on an image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python inference_example.py <path_to_image>")
        print("Using default test - loading webcam...")
        
        # Webcam inference
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = model(frame, conf=0.5)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Display
            cv2.imshow('Strawberry Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Image inference
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    
    # Print results
    for result in results:
        boxes = result.boxes
        print(f"\nFound {len(boxes)} strawberries:")
        
        for i, box in enumerate(boxes):
            confidence = box.conf[0].item()
            print(f"  Strawberry {i+1}: {confidence:.2%} confidence")
        
        # Save annotated image
        output_path = 'output.jpg'
        result.save(output_path)
        print(f"\nSaved annotated image to {output_path}")

if __name__ == '__main__':
    main()