#!/usr/bin/env python3
"""
Validate the Kaggle-trained strawberry detection model
- Run inference on test images
- Measure performance metrics
- Generate visualizations
"""

import sys
from pathlib import Path
import argparse
import time
import json
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def validate_kaggle_model(model_path=None, test_images=None, save_results=True):
    """
    Validate the Kaggle-trained strawberry detection model
    
    Args:
        model_path: Path to the trained model (defaults to kaggle_strawberry_yolov8n)
        test_images: Path to test images directory
        save_results: Whether to save validation results
    """
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
        import torch
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install with: pip install ultralytics pandas numpy tqdm")
        sys.exit(1)
    
    # Setup paths
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    
    if model_path is None:
        # Look for the latest kaggle model in models/detection/
        detection_dir = base_path / "models" / "detection"
        kaggle_models = sorted(detection_dir.glob("kaggle_strawberry_yolov8n_*"))
        if kaggle_models:
            latest_model = kaggle_models[-1]
            model_path = latest_model / "weights" / "best.pt"
        else:
            # Fallback to old location
            model_path = base_path / "model" / "results" / "kaggle_strawberry_yolov8n" / "weights" / "best.pt"
    
    if test_images is None:
        test_images = base_path / "model" / "dataset_strawberry_kaggle" / "test" / "images"
    
    # Save validation results inside the model's folder
    model_results_dir = model_path.parent.parent
    results_dir = model_results_dir / "validation" / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Validating model: {model_path.name}")
    print(f"📁 Test images: {test_images}")
    print(f"📊 Results will be saved to: {results_dir}")
    print("="*60)
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Check if test images exist
    if not test_images.exists():
        print(f"❌ Test images not found: {test_images}")
        return None
    
    # Load model
    print("🤖 Loading model...")
    model = YOLO(str(model_path))
    
    # Get all test images
    image_files = list(test_images.glob("*.jpg")) + list(test_images.glob("*.png")) + list(test_images.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No test images found in {test_images}")
        return None
    
    print(f"📸 Found {len(image_files)} test images")
    
    # Run inference and collect metrics
    print("\n🏃 Running inference...")
    inference_times = []
    detections = []
    all_confidences = []
    all_precisions = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Measure inference time
        start_time = time.time()
        results = model(img_path, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        # Extract detection data
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            # Get confidences
            confidences = boxes.conf.cpu().numpy()
            all_confidences.extend(confidences)
            
            # Count detections
            num_detections = len(boxes)
            detections.append(num_detections)
            
            # Save visualization
            save_path = results_dir / f"detection_{img_path.name}"
            result.save(str(save_path))
        else:
            detections.append(0)
    
    # Calculate metrics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    fps = 1000 / avg_inference_time
    
    avg_detections = np.mean(detections)
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    # Load training results for comparison
    training_results_path = model_path.parent.parent / "results.csv"
    training_metrics = {}
    
    if training_results_path.exists():
        df = pd.read_csv(training_results_path)
        last_epoch = df.iloc[-1]
        training_metrics = {
            "training_mAP50": float(last_epoch.get("metrics/mAP50(B)", 0)),
            "training_precision": float(last_epoch.get("metrics/precision(B)", 0)),
            "training_recall": float(last_epoch.get("metrics/recall(B)", 0)),
            "training_time": int(len(df))  # epochs
        }
    
    # Generate summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"🎯 Model: {model_path.name}")
    print(f"📸 Test Images: {len(image_files)}")
    print(f"⏱️  Inference Speed: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms/image")
    print(f"⚡ FPS: {fps:.1f}")
    print(f"🔍 Avg Detections/Image: {avg_detections:.2f}")
    print(f"💯 Avg Confidence: {avg_confidence:.3f}")
    
    if training_metrics:
        print(f"\n📈 Training vs Validation:")
        print(f"   mAP@50: {training_metrics['training_mAP50']:.3f} (train) vs Validation on {len(image_files)} test images")
        print(f"   Precision: {training_metrics['training_precision']:.3f} (train)")
        print(f"   Recall: {training_metrics['training_recall']:.3f} (train)")
    
    # Save detailed results
    if save_results:
        # Convert numpy types to Python native types for JSON serialization
        results_data = {
            "validation_date": datetime.now().isoformat(),
            "model": str(model_path),
            "test_images": str(test_images),
            "num_test_images": len(image_files),
            "inference_time_ms": {
                "mean": float(avg_inference_time),
                "std": float(std_inference_time),
                "min": float(np.min(inference_times)),
                "max": float(np.max(inference_times))
            },
            "fps": float(fps),
            "avg_detections_per_image": float(avg_detections),
            "avg_confidence": float(avg_confidence),
            "training_metrics": training_metrics,
            "detection_counts": [int(x) for x in detections],
            "inference_times": [float(x) for x in inference_times],
            "confidences": [float(x) for x in all_confidences]
        }
        
        # Save JSON results
        json_path = results_dir / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV of per-image results
        csv_data = []
        for i, img_path in enumerate(image_files):
            csv_data.append({
                "image": img_path.name,
                "inference_time_ms": inference_times[i],
                "detections": detections[i]
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = results_dir / "per_image_results.csv"
        csv_df.to_csv(csv_path, index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")
        print(f"   🖼️  Visualizations: {results_dir}")
    
    print("\n" + "="*60)
    print("✅ Validation complete!")
    
    return {
        "fps": fps,
        "avg_inference_time": avg_inference_time,
        "avg_detections": avg_detections,
        "results_dir": results_dir
    }

def main():
    parser = argparse.ArgumentParser(description='Validate Kaggle-trained strawberry detection model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: kaggle_strawberry_yolov8n)')
    parser.add_argument('--test-images', type=str, default=None,
                        help='Path to test images (default: dataset_strawberry_kaggle/test/images)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_kaggle_model(
        model_path=args.model,
        test_images=args.test_images,
        save_results=not args.no_save
    )
    
    if results:
        print(f"\n🎉 Validation successful!")
        print(f"⚡ Performance: {results['fps']:.1f} FPS ({results['avg_inference_time']:.2f} ms/image)")
        print(f"🔍 Avg detections: {results['avg_detections']:.2f} per image")
    else:
        print("\n❌ Validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()