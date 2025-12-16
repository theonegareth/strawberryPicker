#!/usr/bin/env python3
"""
Real-time strawberry detection/classification using TFLite model.
Supports both binary classification (good/bad) and YOLOv8 detection.
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

def load_tflite_model(model_path):
    """Load TFLite model and allocate tensors."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_model_details(interpreter):
    """Get input and output details of the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def preprocess_image(image, input_shape):
    """Preprocess image for model inference."""
    height, width = input_shape[1:3] if len(input_shape) == 4 else input_shape[1:3]
    img = cv2.resize(image, (width, height))
    img = img / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def run_inference(interpreter, input_details, output_details, preprocessed_img):
    """Run inference on preprocessed image."""
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def main():
    parser = argparse.ArgumentParser(description='Real-time strawberry detection/classification')
    parser.add_argument('--model', type=str, default='strawberry_model.tflite',
                        help='Path to TFLite model (default: strawberry_model.tflite)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for binary classification (default: 0.5)')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size (width=height) for model (default: 224)')
    parser.add_argument('--mode', choices=['classification', 'detection'], default='classification',
                        help='Inference mode: classification (good/bad) or detection (YOLO)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed inference information')
    
    args = parser.parse_args()
    
    # Load model
    try:
        interpreter = load_tflite_model(args.model)
        input_details, output_details = get_model_details(interpreter)
        input_shape = input_details[0]['shape']
        if args.verbose:
            print(f"Model loaded: {args.model}")
            print(f"Input shape: {input_shape}")
            print(f"Output details: {output_details[0]}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera index {args.camera}")
        sys.exit(1)
    
    print(f"Starting real-time inference (mode: {args.mode})")
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Preprocess
        preprocessed = preprocess_image(frame, input_shape)
        
        # Inference
        predictions = run_inference(interpreter, input_details, output_details, preprocessed)
        
        # Process predictions based on mode
        if args.mode == 'classification':
            # Binary classification: single probability
            confidence = predictions[0][0]
            label = 'Good' if confidence > args.threshold else 'Bad'
            display_text = f'{label}: {confidence:.2f}'
            color = (0, 255, 0) if confidence > args.threshold else (0, 0, 255)
        else:
            # Detection mode (YOLO) - placeholder for future implementation
            display_text = 'Detection mode not yet implemented'
            color = (255, 255, 0)
        
        # Display
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Strawberry Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'capture_{cv2.getTickCount()}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")

if __name__ == '__main__':
    main()