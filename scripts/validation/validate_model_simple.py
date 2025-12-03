#!/usr/bin/env python3
"""Simple validation script for trained strawberry detection model"""

import sys
from pathlib import Path
import cv2

def validate_model():
    """Validate the trained model on sample images"""
    
    # Model path
    model_path = Path("models/detection/yolov8n_kaggle_2500images_trained_20251203_130255/weights/best.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed")
        return False
    
    print(f"\n{'='*60}")
    print("VALIDATING TRAINED MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Test on a few sample images from the dataset
    test_images = [
        "model/test/images/1000_png.rf.ddadf0610c640b7eae60447b6a3a6b4f.jpg",
        "model/test/images/1003_png.rf.8ce57406da0b12e9d1c26897c7ceaa7a.jpg",
        "model/test/images/1005_png.rf.da9b0f8ff644c00c8c2cc2e5e1cb99f9.jpg"
    ]
    
    # Filter existing images
    existing_images = []
    for img_path in test_images:
        if Path(img_path).exists():
            existing_images.append(Path(img_path))
        else:
            # Try alternative path
            alt_path = Path("model/dataset_strawberry_kaggle") / "test" / "images" / Path(img_path).name
            if alt_path.exists():
                existing_images.append(alt_path)
    
    if not existing_images:
        print("❌ No test images found")
        return False
    
    print(f"Found {len(existing_images)} test images\n")
    
    # Run inference
    total_detections = 0
    confidences = []
    
    for i, img_path in enumerate(existing_images, 1):
        print(f"🖼️  Testing image {i}/{len(existing_images)}: {img_path.name}")
        
        # Run detection
        results = model(img_path, conf=0.25, imgsz=416)
        
        # Count detections
        detections = results[0].boxes
        num_detections = len(detections)
        total_detections += num_detections
        
        # Extract confidences
        if num_detections > 0:
            confs = [float(box.conf) for box in detections]
            confidences.extend(confs)
            avg_conf = sum(confs) / len(confs)
            print(f"   ✅ Detected {num_detections} strawberries")
            print(f"   📊 Average confidence: {avg_conf:.3f}")
        else:
            print(f"   ⚠️  No strawberries detected")
        
        # Save visualization
        result_img = results[0].plot()
        output_path = Path("model/validation_results") / f"validation_{img_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_img)
        print(f"   💾 Saved result to: {output_path}\n")
    
    # Summary
    print(f"{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Model loaded successfully: {model_path.name}")
    print(f"🖼️  Images tested: {len(existing_images)}")
    print(f"🍓 Total detections: {total_detections}")
    
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"📊 Average confidence: {avg_conf:.3f}")
        print(f"📈 Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    print(f"\n✅ Model validation completed successfully!")
    print(f"📁 Results saved to: model/validation_results/")
    
    return True

if __name__ == '__main__':
    success = validate_model()
    sys.exit(0 if success else 1)