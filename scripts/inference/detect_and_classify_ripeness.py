#!/usr/bin/env python3
"""
Two-stage pipeline: Detect strawberries → Classify ripeness → Pick only ripe ones
Optimized for Raspberry Pi 4B deployment
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from PIL import Image
import time
import json
import sys

class StrawberryRipenessClassifier:
    """Lightweight ripeness classifier for detected strawberries"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to trained ripeness classifier
            device: 'cpu' or 'cuda'
        """
        # Load model info
        info_path = Path(model_path).parent / "ripeness_classifier_info.json"
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Import here to avoid issues if not available
        sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
        from train_ripeness_classifier import create_model
        
        # Create and load model
        self.model = create_model(num_classes=3, model_type=self.model_info['model_type'])
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        self.device = device
        self.classes = self.model_info['classes']
        
        print(f"✅ Ripeness classifier loaded: {self.model_info['model_type']}")
        print(f"   Classes: {self.classes}")
        print(f"   Validation accuracy: {self.model_info['val_accuracy']}%")
    
    def classify(self, image):
        """
        Classify ripeness of a single strawberry image
        
        Args:
            image: PIL Image or numpy array of the strawberry crop
            
        Returns:
            dict with 'class', 'confidence', and 'all_predictions'
        """
        import torchvision.transforms as transforms
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Classify
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all predictions
            all_preds = {
                self.classes[i]: float(probabilities[0][i])
                for i in range(len(self.classes))
            }
            
            return {
                'class': self.classes[predicted.item()],
                'confidence': float(confidence.item()),
                'all_predictions': all_preds
            }

class StrawberryPickerPipeline:
    """Complete pipeline for detecting and classifying strawberries"""
    
    def __init__(self, detector_path, classifier_path, device='cpu'):
        """
        Args:
            detector_path: Path to YOLO detection model
            classifier_path: Path to ripeness classifier
            device: 'cpu' or 'cuda'
        """
        print("🍓 Initializing Strawberry Picker Pipeline")
        print("=" * 60)
        
        # Load detection model
        print("Loading detection model...")
        self.detector = YOLO(detector_path)
        self.detector.to(device)
        
        # Load ripeness classifier
        print("Loading ripeness classifier...")
        self.classifier = StrawberryRipenessClassifier(classifier_path, device)
        
        self.device = device
        
        print("✅ Pipeline ready!")
    
    def process_image(self, image, conf_threshold=0.5, ripeness_threshold=0.7):
        """
        Process image: detect strawberries, classify ripeness, identify pickable ones
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for detection
            ripeness_threshold: Minimum confidence for ripeness classification
            
        Returns:
            dict with detection and classification results
        """
        
        # Detect strawberries
        results = self.detector(image, conf=conf_threshold, imgsz=416)
        
        detections = []
        pickable_strawberries = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Extract strawberry crop
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    strawberry_crop = image[y1:y2, x1:x2]
                    
                    if strawberry_crop.size > 0:
                        # Classify ripeness
                        ripeness_result = self.classifier.classify(strawberry_crop)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'ripeness': ripeness_result['class'],
                            'ripeness_confidence': ripeness_result['confidence'],
                            'all_ripeness_probs': ripeness_result['all_predictions']
                        }
                        
                        detections.append(detection)
                        
                        # Check if it's pickable (ripe with high confidence)
                        if (ripeness_result['class'] == 'ripe' and 
                            ripeness_result['confidence'] >= ripeness_threshold):
                            pickable_strawberries.append(detection)
        
        return {
            'detections': detections,
            'pickable_strawberries': pickable_strawberries,
            'total_strawberries': len(detections),
            'pickable_count': len(pickable_strawberries)
        }
    
    def visualize_results(self, image, results, show_all=False):
        """
        Visualize detection and ripeness classification results
        
        Args:
            image: Input image
            results: Results from process_image
            show_all: If True, show all detections. If False, only show pickable ones
            
        Returns:
            Annotated image
        """
        
        annotated_image = image.copy()
        
        # Choose which detections to show
        detections_to_show = results['detections'] if show_all else results['pickable_strawberries']
        
        # Color mapping for ripeness
        ripeness_colors = {
            'unripe': (0, 255, 0),      # Green
            'ripe': (0, 165, 255),      # Orange (for pickable!)
            'overripe': (0, 0, 255)     # Red
        }
        
        for detection in detections_to_show:
            x1, y1, x2, y2 = detection['bbox']
            ripeness = detection['ripeness']
            ripeness_conf = detection['ripeness_confidence']
            det_conf = detection['confidence']
            
            # Draw bounding box
            color = ripeness_colors.get(ripeness, (255, 255, 255))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ripeness} ({ripeness_conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add summary text
        summary_text = f"Pickable: {results['pickable_count']}/{results['total_strawberries']}"
        cv2.putText(annotated_image, summary_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return annotated_image

def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description='Detect and classify strawberry ripeness')
    parser.add_argument('--detector', type=str, 
                       default='models/detection/yolov8n_baseline/weights/best.pt',
                       help='Path to YOLO detection model')
    parser.add_argument('--classifier', type=str,
                       default='model/datasets/best_ripeness_classifier.pth',
                       help='Path to ripeness classifier')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='output_ripeness.jpg',
                       help='Path to save output image')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all detections, not just pickable ones')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, or cuda')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create pipeline
    pipeline = StrawberryPickerPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier,
        device=device
    )
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image: {args.image}")
        return
    
    print(f"\n🖼️  Processing image: {args.image}")
    
    # Process image
    start_time = time.time()
    results = pipeline.process_image(image, conf_threshold=0.5, ripeness_threshold=0.7)
    processing_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Total strawberries detected: {results['total_strawberries']}")
    print(f"   Pickable (ripe) strawberries: {results['pickable_count']}")
    print(f"   Processing time: {processing_time:.3f}s")
    
    # Visualize results
    output_image = pipeline.visualize_results(image, results, show_all=args.show_all)
    
    # Save output
    cv2.imwrite(args.output, output_image)
    print(f"\n💾 Output saved to: {args.output}")
    
    # Print detailed results
    if results['pickable_strawberries']:
        print(f"\n🍓 Pickable strawberries:")
        for i, strawberry in enumerate(results['pickable_strawberries'], 1):
            print(f"   {i}. Confidence: {strawberry['confidence']:.3f}, "
                  f"Ripeness: {strawberry['ripeness']} ({strawberry['ripeness_confidence']:.3f})")

if __name__ == "__main__":
    main()