#!/usr/bin/env python3
"""
Model Validation and Visualization Script
Generates validation images for detection and classification models
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    def __init__(self, detector_path, classifier_path=None, device='cpu'):
        """
        Initialize model validator
        
        Args:
            detector_path: Path to YOLOv8 detection model
            classifier_path: Path to classification model (optional)
            device: Device to run inference on
        """
        print("🍓 Initializing Model Validator...")
        
        self.device = device
        self.results = {
            'detection': [],
            'classification': [],
            'summary': {}
        }
        
        # Load detection model
        print("Loading detection model...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO(detector_path)
            print("✅ Detection model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            sys.exit(1)
        
        # Load classification model if provided
        self.classifier = None
        if classifier_path and Path(classifier_path).exists():
            print("Loading classification model...")
            try:
                self.classifier = torch.load(classifier_path, map_location=device)
                self.classifier.eval()
                print("✅ Classification model loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load classification model: {e}")
        
        print("✅ Validator initialized!")
    
    def validate_detection(self, test_images_dir, output_dir, num_samples=20):
        """
        Validate detection model on test images
        
        Args:
            test_images_dir: Directory containing test images
            output_dir: Directory to save validation results
            num_samples: Number of images to process
        """
        print(f"\n🔍 Starting Detection Validation...")
        print(f"Test images: {test_images_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Number of samples: {num_samples}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test images
        test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                     list(Path(test_images_dir).glob("*.png")) + \
                     list(Path(test_images_dir).glob("*.jpeg"))
        
        if not test_images:
            print(f"❌ No test images found in {test_images_dir}")
            return
        
        # Limit number of samples
        test_images = test_images[:num_samples]
        print(f"Found {len(test_images)} test images")
        
        # Process each image
        detection_results = []
        for i, image_path in enumerate(tqdm(test_images, desc="Processing images")):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"⚠️  Could not load {image_path}")
                    continue
                
                # Run detection
                results = self.detector(image)
                
                # Process detections
                detections = []
                annotated_image = image.copy()
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        if conf < 0.3:  # Confidence threshold
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"strawberry {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                     (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf)
                        })
                
                # Save annotated image
                output_path = output_dir / f"detection_{i:03d}_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_image)
                
                # Store result
                detection_results.append({
                    'image': image_path.name,
                    'detections': len(detections),
                    'output_path': str(output_path)
                })
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        
        # Save summary
        self.results['detection'] = detection_results
        
        # Calculate statistics
        total_images = len(detection_results)
        total_detections = sum(r['detections'] for r in detection_results)
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        self.results['summary'] = {
            'total_images': total_images,
            'total_detections': total_detections,
            'average_detections_per_image': avg_detections,
            'detection_rate': (total_detections / total_images * 100) if total_images > 0 else 0
        }
        
        print(f"\n✅ Detection validation complete!")
        print(f"Total images processed: {total_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {avg_detections:.2f}")
        print(f"Results saved to: {output_dir}")
        
        return detection_results
    
    def create_validation_grid(self, output_dir, grid_size=5):
        """
        Create a grid of validation results
        
        Args:
            output_dir: Directory containing validation images
            grid_size: Number of images per row/column
        """
        print(f"\n🎨 Creating validation grid...")
        
        output_dir = Path(output_dir)
        validation_images = list(output_dir.glob("detection_*.jpg"))
        
        if not validation_images:
            print("❌ No validation images found")
            return
        
        # Limit number of images for the grid
        validation_images = validation_images[:grid_size*grid_size]
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle('Strawberry Detection Validation Results', fontsize=16, fontweight='bold')
        
        # Fill grid
        for i, img_path in enumerate(validation_images):
            row = i // grid_size
            col = i % grid_size
            
            if grid_size == 1:
                ax = axes
            elif grid_size > 1:
                ax = axes[row, col] if grid_size > 1 else axes[col]
            
            # Load and display image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            
            # Get detection count from filename or results
            detections = 0
            for result in self.results['detection']:
                if result['output_path'] == str(img_path):
                    detections = result['detections']
                    break
            
            ax.set_title(f"{img_path.name}\nDetections: {detections}", fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(validation_images), grid_size*grid_size):
            row = i // grid_size
            col = i % grid_size
            if grid_size > 1:
                axes[row, col].axis('off')
            else:
                axes.axis('off')
        
        # Save grid
        grid_path = output_dir / "validation_grid.jpg"
        plt.tight_layout()
        plt.savefig(str(grid_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Validation grid saved to: {grid_path}")
        return str(grid_path)
    
    def generate_report(self, output_dir):
        """
        Generate validation report
        
        Args:
            output_dir: Directory to save report
        """
        print(f"\n📊 Generating validation report...")
        
        output_dir = Path(output_dir)
        report_path = output_dir / "validation_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Validation report saved to: {report_path}")
        
        # Print summary
        summary = self.results.get('summary', {})
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total images processed: {summary.get('total_images', 0)}")
        print(f"Total detections: {summary.get('total_detections', 0)}")
        print(f"Average detections per image: {summary.get('average_detections_per_image', 0):.2f}")
        print(f"Detection rate: {summary.get('detection_rate', 0):.1f}%")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Validate strawberry detection models')
    parser.add_argument('--detector', type=str, 
                       default='model/weights/strawberry_yolov8s_enhanced.pt',
                       help='Path to detection model')
    parser.add_argument('--classifier', type=str, 
                       help='Path to classification model (optional)')
    parser.add_argument('--test-dir', type=str,
                       default='model/test/images',
                       help='Directory containing test images')
    parser.add_argument('--output-dir', type=str,
                       default='model/validation_results',
                       help='Directory to save validation results')
    parser.add_argument('--num-samples', type=int,
                       default=25,
                       help='Number of test images to process')
    parser.add_argument('--grid-size', type=int,
                       default=5,
                       help='Size of validation grid (grid_size x grid_size)')
    parser.add_argument('--device', type=str,
                       default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize validator
    validator = ModelValidator(
        detector_path=args.detector,
        classifier_path=args.classifier,
        device=device
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run detection validation
    validator.validate_detection(
        test_images_dir=args.test_dir,
        output_dir=output_dir,
        num_samples=args.num_samples
    )
    
    # Create validation grid
    validator.create_validation_grid(
        output_dir=output_dir,
        grid_size=args.grid_size
    )
    
    # Generate report
    validator.generate_report(output_dir)
    
    print(f"\n🎉 Validation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    # Check for required libraries
    try:
        import torch
        import cv2
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Install with: pip install torch opencv-python ultralytics matplotlib")
        sys.exit(1)
    
    main()