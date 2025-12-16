#!/usr/bin/env python3
"""
Integrated Strawberry Detection and Ripeness Classification Pipeline
Combines YOLOv8 detection with 3-class ripeness classification
"""

import os
import argparse
import json
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from pathlib import Path
import yaml
from datetime import datetime
import logging

# YOLOv8
from ultralytics import YOLO

# Custom imports
from train_ripeness_classifier import create_model, get_transforms

class StrawberryDetectionClassifier:
    """Integrated detection and classification system"""
    
    def __init__(self, detection_model_path, classification_model_path, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize detection model
        print(f"Loading detection model: {detection_model_path}")
        self.detection_model = YOLO(detection_model_path)
        
        # Initialize classification model
        print(f"Loading classification model: {classification_model_path}")
        self.classification_model = self.load_classification_model(classification_model_path)
        
        # Get classification transforms
        _, self.classify_transform = get_transforms(img_size=224)
        
        # Class names for classification
        self.class_names = ['overripe', 'ripe', 'unripe']
        
        # Setup logging
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_classification_model(self, model_path):
        """Load the trained classification model"""
        model = create_model(num_classes=3, backbone='resnet18', pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('strawberry_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_strawberries(self, image):
        """Detect strawberries in image using YOLOv8"""
        results = self.detection_model(image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Only keep high-confidence detections
                    if confidence > 0.5:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': int(box.cls[0].cpu().numpy())
                        })
        
        return detections
    
    def classify_ripeness(self, image_crop):
        """Classify ripeness of strawberry crop"""
        try:
            # Apply transforms
            if isinstance(image_crop, np.ndarray):
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image_crop = Image.fromarray(image_crop)
            
            input_tensor = self.classify_transform(image_crop).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.classification_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                'class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(probabilities[0][i].item()) 
                    for i in range(len(self.class_names))
                }
            }
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {
                'class': 'unknown',
                'confidence': 0.0,
                'probabilities': {cls: 0.0 for cls in self.class_names}
            }
    
    def process_image(self, image_path, save_annotated=True, output_dir='results'):
        """Process single image with detection and classification"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        # Detect strawberries
        detections = self.detect_strawberries(image)
        
        results = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'summary': {
                'total_strawberries': len(detections),
                'ripeness_counts': {'unripe': 0, 'ripe': 0, 'overripe': 0, 'unknown': 0}
            }
        }
        
        # Process each detection
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop strawberry
            strawberry_crop = image[y1:y2, x1:x2]
            
            # Classify ripeness
            ripeness = self.classify_ripeness(strawberry_crop)
            
            # Update summary
            results['summary']['ripeness_counts'][ripeness['class']] += 1
            
            # Store result
            result = {
                'detection_id': i,
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'ripeness': ripeness
            }
            results['detections'].append(result)
            
            # Draw annotations if requested
            if save_annotated:
                color = self.get_ripeness_color(ripeness['class'])
                label = f"{ripeness['class']} ({ripeness['confidence']:.2f})"
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save annotated image
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
            output_path = Path(output_dir) / f"annotated_{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
            results['annotated_image_path'] = str(output_path)
        
        return results
    
    def get_ripeness_color(self, ripeness_class):
        """Get color for ripeness class"""
        colors = {
            'unripe': (0, 255, 0),    # Green
            'ripe': (0, 255, 255),    # Yellow
            'overripe': (0, 0, 255),  # Red
            'unknown': (128, 128, 128) # Gray
        }
        return colors.get(ripeness_class, (128, 128, 128))

def main():
    parser = argparse.ArgumentParser(description='Integrated strawberry detection and classification')
    parser.add_argument('--detection-model', default='model/weights/best_yolov8n_strawberry.pt',
                       help='Path to YOLOv8 detection model')
    parser.add_argument('--classification-model', default='model/ripeness_classifier_best.pth',
                       help='Path to ripeness classification model')
    parser.add_argument('--mode', choices=['image', 'video', 'realtime'], required=True,
                       help='Processing mode')
    parser.add_argument('--input', required=True, help='Input path (image/video/camera index)')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--save-annotated', action='store_true', help='Save annotated images')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize system
    system = StrawberryDetectionClassifier(
        args.detection_model, 
        args.classification_model, 
        args.config
    )
    
    if args.mode == 'image':
        # Process single image
        results = system.process_image(
            args.input, 
            save_annotated=args.save_annotated,
            output_dir=args.output or 'results'
        )
        
        if results:
            # Save results
            results_path = Path(args.output or 'results') / 'detection_results.json'
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {results_path}")
            print(f"Found {results['summary']['total_strawberries']} strawberries")
            print(f"Ripeness distribution: {results['summary']['ripeness_counts']}")

if __name__ == '__main__':
    main()
