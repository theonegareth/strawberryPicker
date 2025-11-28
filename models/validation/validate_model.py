#!/usr/bin/env python3
"""
Strawberry Detection Model Validation Script
Phase 1: Model Validation & Testing
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class StrawberryValidator:
    """Validate strawberry detection model performance"""
    
    def __init__(self, model_path: str, test_dir: str, output_dir: str):
        """
        Initialize validator
        
        Args:
            model_path: Path to trained YOLO model
            test_dir: Directory with test images
            output_dir: Directory to save results
        """
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Class names (only strawberry for this model)
        self.class_names = ['strawberry']
        
        # Initialize metrics storage
        self.results = []
        self.metrics = {}
        
    def run_inference_on_test_set(self) -> List[Dict[str, Any]]:
        """
        Run inference on all test images
        
        Returns:
            List of detection results for each image
        """
        test_images = list(self.test_dir.glob("*.jpg")) + \
                     list(self.test_dir.glob("*.png")) + \
                     list(self.test_dir.glob("*.jpeg"))
        
        print(f"\nFound {len(test_images)} test images")
        print("Running inference...")
        
        results = []
        for idx, img_path in enumerate(test_images, 1):
            print(f"Processing {idx}/{len(test_images)}: {img_path.name}")
            
            # Run inference
            detections = self.model(img_path, conf=0.25, iou=0.45)
            
            # Extract results
            result = {
                'image_name': img_path.name,
                'image_path': str(img_path),
                'detections': []
            }
            
            for det in detections[0].boxes:
                box = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(det.conf[0].cpu().numpy())
                cls = int(det.cls[0].cpu().numpy())
                
                result['detections'].append({
                    'bbox': box.tolist(),
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.class_names[cls]
                })
            
            results.append(result)
            
            # Save visualization
            self._save_visualization(img_path, result)
        
        self.results = results
        return results
    
    def _save_visualization(self, img_path: Path, result: Dict[str, Any]):
        """Save detection visualization"""
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw detections
        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label = f"Strawberry {conf:.2f}"
            draw.text((x1, y1-20), label, fill="red", font=font)
        
        # Save visualization
        viz_path = self.output_dir / "visualizations" / f"{img_path.stem}_detection.jpg"
        img.save(viz_path)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score
        
        Note: Since we don't have ground truth annotations for this dataset,
        we'll calculate metrics based on detection confidence and consistency
        """
        print("\nCalculating metrics...")
        
        # For this validation, we'll use confidence-based analysis
        all_detections = []
        for result in self.results:
            all_detections.extend(result['detections'])
        
        if not all_detections:
            print("No detections found!")
            return {}
        
        # Extract confidences
        confidences = [det['confidence'] for det in all_detections]
        
        # Basic statistics
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        total_detections = len(all_detections)
        
        # High confidence detections (>0.5)
        high_conf_detections = [c for c in confidences if c > 0.5]
        high_conf_ratio = len(high_conf_detections) / len(confidences) if confidences else 0
        
        # Per-image statistics
        detections_per_image = [len(r['detections']) for r in self.results]
        avg_detections_per_image = np.mean(detections_per_image)
        
        metrics = {
            'total_images': len(self.results),
            'total_detections': total_detections,
            'avg_confidence': float(avg_confidence),
            'std_confidence': float(std_confidence),
            'high_confidence_ratio': float(high_conf_ratio),
            'avg_detections_per_image': float(avg_detections_per_image),
            'images_with_detections': len([r for r in self.results if r['detections']]),
            'images_without_detections': len([r for r in self.results if not r['detections']])
        }
        
        self.metrics = metrics
        return metrics
    
    def identify_best_and_worst_cases(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify best and worst detection cases
        
        Returns:
            Tuple of (best_cases, worst_cases)
        """
        print("\nIdentifying best and worst cases...")
        
        # Score each image based on detection quality
        scored_images = []
        for result in self.results:
            if not result['detections']:
                score = 0
                avg_conf = 0
            else:
                confidences = [det['confidence'] for det in result['detections']]
                avg_conf = np.mean(confidences)
                # Score based on number of detections and average confidence
                score = len(result['detections']) * avg_conf
            
            scored_images.append({
                'result': result,
                'score': score,
                'avg_confidence': avg_conf,
                'num_detections': len(result['detections'])
            })
        
        # Sort by score
        scored_images.sort(key=lambda x: x['score'], reverse=True)
        
        # Best cases (top 5)
        best_cases = scored_images[:5]
        
        # Worst cases (lowest scores, including no detections)
        worst_cases = scored_images[-10:]
        
        return best_cases, worst_cases
    
    def export_metrics_to_csv(self):
        """Export metrics to CSV file"""
        csv_path = self.output_dir / "metrics" / "validation_metrics.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in self.metrics.items():
                writer.writerow([key, value])
        
        print(f"\nMetrics exported to: {csv_path}")
    
    def export_detections_to_csv(self):
        """Export all detections to CSV"""
        csv_path = self.output_dir / "metrics" / "detections.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Detection_ID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence', 'Class'])
            
            for result in self.results:
                for idx, det in enumerate(result['detections'], 1):
                    x1, y1, x2, y2 = det['bbox']
                    writer.writerow([
                        result['image_name'],
                        idx,
                        x1, y1, x2, y2,
                        det['confidence'],
                        det['class_name']
                    ])
        
        print(f"Detections exported to: {csv_path}")
    
    def generate_report(self, best_cases: List[Dict], worst_cases: List[Dict]):
        """Generate comprehensive validation report"""
        report_path = self.output_dir / "validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Strawberry Detection Model - Validation Report\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Images Tested**: {self.metrics['total_images']}\n")
            f.write(f"- **Total Detections**: {self.metrics['total_detections']}\n")
            f.write(f"- **Average Confidence**: {self.metrics['avg_confidence']:.3f}\n")
            f.write(f"- **High Confidence Detections (>0.5)**: {self.metrics['high_confidence_ratio']:.1%}\n")
            f.write(f"- **Images with Detections**: {self.metrics['images_with_detections']}\n")
            f.write(f"- **Images without Detections**: {self.metrics['images_without_detections']}\n\n")
            
            # Best Cases
            f.write("## Best Detection Cases (Top 5)\n\n")
            for idx, case in enumerate(best_cases, 1):
                result = case['result']
                f.write(f"### {idx}. {result['image_name']}\n")
                f.write(f"- **Score**: {case['score']:.2f}\n")
                f.write(f"- **Detections**: {case['num_detections']}\n")
                f.write(f"- **Avg Confidence**: {case['avg_confidence']:.3f}\n")
                f.write(f"- **Visualization**: [View](visualizations/{result['image_name'].split('.')[0]}_detection.jpg)\n\n")
            
            # Worst Cases
            f.write("## Challenging Cases (Bottom 10)\n\n")
            for idx, case in enumerate(worst_cases, 1):
                result = case['result']
                f.write(f"### {idx}. {result['image_name']}\n")
                f.write(f"- **Score**: {case['score']:.2f}\n")
                f.write(f"- **Detections**: {case['num_detections']}\n")
                f.write(f"- **Avg Confidence**: {case['avg_confidence']:.3f}\n")
                if case['num_detections'] > 0:
                    f.write(f"- **Visualization**: [View](visualizations/{result['image_name'].split('.')[0]}_detection.jpg)\n")
                else:
                    f.write("- **Issue**: No detections found\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if self.metrics['images_without_detections'] > 20:
                f.write("- ⚠️ High number of images without detections - consider lowering confidence threshold\n")
            if self.metrics['avg_confidence'] < 0.6:
                f.write("- ⚠️ Low average confidence - model may need more training\n")
            else:
                f.write("- ✅ Good average confidence - model is performing well\n")
            
            if self.metrics['high_confidence_ratio'] > 0.7:
                f.write("- ✅ High ratio of high-confidence detections\n")
            else:
                f.write("- ⚠️ Low ratio of high-confidence detections - consider filtering predictions\n")
        
        print(f"\nValidation report saved to: {report_path}")

def main():
    """Main validation function"""
    # Setup paths
    base_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    model_path = base_dir / "model/weights/strawberry_yolov8n.pt"
    test_dir = base_dir / "model/dataset/test/images"
    output_dir = base_dir / "model/validation_results"
    
    print("=" * 60)
    print("STRAWBERRY DETECTION MODEL VALIDATION")
    print("=" * 60)
    
    # Initialize validator
    validator = StrawberryValidator(
        model_path=model_path,
        test_dir=test_dir,
        output_dir=output_dir
    )
    
    # Run inference
    results = validator.run_inference_on_test_set()
    
    # Calculate metrics
    metrics = validator.calculate_metrics()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Identify best and worst cases
    best_cases, worst_cases = validator.identify_best_and_worst_cases()
    
    # Export data
    validator.export_metrics_to_csv()
    validator.export_detections_to_csv()
    
    # Generate report
    validator.generate_report(best_cases, worst_cases)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()