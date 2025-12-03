#!/usr/bin/env python3
"""Validate all trained models and save results in each model's folder"""

import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
from datetime import datetime
import json

def validate_model(model_path, output_dir, test_images, model_name):
    """Validate a single model and save results"""
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"VALIDATING: {model_name}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create validation directory for this model
    val_dir = Path(output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'validation_date': datetime.now().isoformat(),
        'test_images': [],
        'summary': {
            'total_images': 0,
            'total_detections': 0,
            'images_with_detections': 0,
            'average_confidence': 0.0,
            'average_detections_per_image': 0.0
        }
    }
    
    total_detections = 0
    confidences = []
    
    for i, img_path in enumerate(test_images, 1):
        if Path(img_path).exists():
            print(f"🖼️  Testing image {i}: {Path(img_path).name}")
            
            try:
                # Run detection
                detections = model(img_path, conf=0.25, imgsz=416)
                
                # Count detections
                num_detections = len(detections[0].boxes)
                total_detections += num_detections
                
                # Extract confidences
                if num_detections > 0:
                    confs = [float(box.conf) for box in detections[0].boxes]
                    confidences.extend(confs)
                    avg_conf = sum(confs) / len(confs)
                    print(f"   ✅ Detected {num_detections} strawberries")
                    print(f"   📊 Average confidence: {avg_conf:.3f}")
                else:
                    avg_conf = 0.0
                    print(f"   ⚠️  No strawberries detected")
                
                # Save visualization
                result_img = detections[0].plot()
                output_path = val_dir / f"validation_{Path(img_path).name}"
                cv2.imwrite(str(output_path), result_img)
                print(f"   💾 Saved to: {output_path}")
                
                # Store image result
                image_result = {
                    'image_path': str(img_path),
                    'image_name': Path(img_path).name,
                    'detections': num_detections,
                    'average_confidence': avg_conf,
                    'confidences': [float(box.conf) for box in detections[0].boxes] if num_detections > 0 else []
                }
                results['test_images'].append(image_result)
                
            except Exception as e:
                print(f"   ❌ Error processing image: {e}")
                continue
        
        print()
    
    # Calculate summary statistics
    num_valid_images = len(results['test_images'])
    if num_valid_images > 0:
        results['summary']['total_images'] = num_valid_images
        results['summary']['total_detections'] = total_detections
        results['summary']['images_with_detections'] = len([img for img in results['test_images'] if img['detections'] > 0])
        results['summary']['average_detections_per_image'] = total_detections / num_valid_images
        
        if confidences:
            results['summary']['average_confidence'] = sum(confidences) / len(confidences)
    
    # Save results JSON
    results_file = val_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    summary_file = val_dir / "validation_summary.md"
    with open(summary_file, 'w') as f:
        f.write(f"# {model_name} - Validation Results\n\n")
        f.write(f"**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Images Tested**: {results['summary']['total_images']}\n")
        f.write(f"- **Total Detections**: {results['summary']['total_detections']}\n")
        f.write(f"- **Images with Detections**: {results['summary']['images_with_detections']}\n")
        f.write(f"- **Detection Rate**: {results['summary']['images_with_detections']/results['summary']['total_images']*100:.1f}%\n")
        f.write(f"- **Average Detections per Image**: {results['summary']['average_detections_per_image']:.2f}\n")
        f.write(f"- **Average Confidence**: {results['summary']['average_confidence']:.3f}\n\n")
        f.write(f"## Individual Image Results\n\n")
        
        for img in results['test_images']:
            f.write(f"### {img['image_name']}\n")
            f.write(f"- **Detections**: {img['detections']}\n")
            f.write(f"- **Average Confidence**: {img['average_confidence']:.3f}\n")
            f.write(f"- **Visualization**: validation_{img['image_name']}\n\n")
    
    # Print summary
    print(f"📊 VALIDATION SUMMARY - {model_name}")
    print(f"{'='*60}")
    print(f"✅ Images tested: {results['summary']['total_images']}")
    print(f"🍓 Total detections: {results['summary']['total_detections']}")
    print(f"📈 Detection rate: {results['summary']['images_with_detections']/results['summary']['total_images']*100:.1f}%")
    print(f"📊 Average confidence: {results['summary']['average_confidence']:.3f}")
    print(f"💾 Results saved to: {val_dir}")
    
    return results

def main():
    """Main validation function"""
    
    # Define models to validate
    models = [
        {
            'name': 'yolov8n_kaggle_2500images',
            'path': 'models/detection/yolov8n_kaggle_2500images_trained_20251203_130255/weights/best.pt',
            'output_dir': 'models/detection/yolov8n_kaggle_2500images_trained_20251203_130255'
        },
        {
            'name': 'yolov8s_improved_detection_v2',
            'path': 'models/detection/yolov8s_improved_detection_v2_20251202_153433/weights/best.pt',
            'output_dir': 'models/detection/yolov8s_improved_detection_v2_20251202_153433'
        },
        {
            'name': 'yolov8s_enhanced',
            'path': 'models/detection/yolov8s_enhanced/strawberry_yolov8s_enhanced.pt',
            'output_dir': 'models/detection/yolov8s_enhanced'
        },
        {
            'name': 'yolov8n_baseline',
            'path': 'models/detection/yolov8n/strawberry_yolov8n.pt',
            'output_dir': 'models/detection/yolov8n'
        },
        {
            'name': 'baseline',
            'path': 'models/detection/baseline/First_run_Baseline.pt',
            'output_dir': 'models/detection/baseline'
        }
    ]
    
    # Test images (same for all models)
    test_images = [
        '/home/user/Downloads/train/RottenStrawberry/RottenStrawberry (25).jpg',
        '/home/user/Downloads/train/RottenStrawberry/RottenStrawberry (84).jpg',
        '/home/user/Downloads/train/RottenStrawberry/RottenStrawberry (97).jpg',
        '/home/user/Downloads/train/RottenStrawberry/RottenStrawberry (116).jpg',
        '/home/user/Downloads/train/RottenStrawberry/RottenStrawberry (141).jpg'
    ]
    
    print("🍓 COMPREHENSIVE MODEL VALIDATION")
    print(f"Testing {len(models)} models on {len(test_images)} strawberry images")
    print("="*60)
    
    all_results = []
    
    # Validate each model
    for model in models:
        result = validate_model(
            model_path=model['path'],
            output_dir=model['output_dir'], 
            test_images=test_images,
            model_name=model['name']
        )
        if result:
            all_results.append(result)
        print("\n" + "="*80 + "\n")
    
    # Create comparison summary
    print("📋 COMPREHENSIVE VALIDATION SUMMARY")
    print("="*80)
    
    for result in all_results:
        summary = result['summary']
        print(f"\n🎯 **{result['model_name']}**")
        print(f"   📊 Detection Rate: {summary['images_with_detections']/summary['total_images']*100:.1f}%")
        print(f"   🍓 Avg Detections: {summary['average_detections_per_image']:.2f}")
        print(f"   📈 Avg Confidence: {summary['average_confidence']:.3f}")
        print(f"   💾 Results: validation/ folder")
    
    print(f"\n✅ All {len(all_results)} models validated successfully!")
    print(f"📁 Detailed results saved in each model's validation folder")

if __name__ == '__main__':
    main()