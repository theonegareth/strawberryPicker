#!/usr/bin/env python3
"""
Validate all detection models in model/detection/ folder
- Runs inference on test set for each model
- Generates comparison report
- Saves results in each model's validation folder
"""

import sys
import time
from pathlib import Path
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def validate_all_models(test_images=None, save_results=True, max_models=None):
    """
    Validate all models in model/detection/ folder

    Args:
        test_images: Path to test images directory
        save_results: Whether to save validation results
        max_models: Maximum number of models to validate (for testing)
    """
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install with: pip install ultralytics pandas numpy tqdm")
        sys.exit(1)
    
    # Setup paths
    base_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker")
    
    if test_images is None:
        test_images = base_path / "model" / "dataset_strawberry_kaggle" / "test" / "images"
    
    detection_dir = base_path / "model" / "detection"
    
    # Find all model directories with best.pt
    print(f"🔍 Scanning {detection_dir} for models...")
    model_dirs = []
    for model_dir in detection_dir.iterdir():
        if model_dir.is_dir():
            best_pt = model_dir / "weights" / "best.pt"
            if best_pt.exists():
                model_dirs.append({
                    "dir": model_dir,
                    "name": model_dir.name,
                    "model_path": best_pt
                })
    
    if not model_dirs:
        print("❌ No models found in model/detection/")
        return None
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=lambda x: x["dir"].stat().st_mtime, reverse=True)
    
    print(f"📊 Found {len(model_dirs)} models")
    print("="*60)
    
    # Limit models if specified
    if max_models:
        model_dirs = model_dirs[:max_models]
        print(f"🔬 Validating first {max_models} models")
    
    # Validate each model
    all_results = []
    
    for i, model_info in enumerate(model_dirs, 1):
        print(f"\n[{i}/{len(model_dirs)}] Validating: {model_info['name']}")
        print("-" * 60)
        
        try:
            # Load model
            model = YOLO(str(model_info['model_path']))
            
            # Get all test images
            image_files = list(test_images.glob("*.jpg")) + list(test_images.glob("*.png")) + list(test_images.glob("*.jpeg"))
            
            if not image_files:
                print(f"⚠️  No test images found")
                continue
            
            # Run inference and collect metrics
            inference_times = []
            detections = []
            all_confidences = []
            
            for img_path in tqdm(image_files, desc=f"Processing {model_info['name']}", leave=False):
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
                else:
                    detections.append(0)
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            fps = 1000 / avg_inference_time
            
            avg_detections = np.mean(detections)
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            
            # Load training results if available
            training_results_path = model_info['model_path'].parent.parent / "results.csv"
            training_metrics = {}
            
            if training_results_path.exists():
                df = pd.read_csv(training_results_path)
                if not df.empty:
                    last_epoch = df.iloc[-1]
                    training_metrics = {
                        "training_mAP50": float(last_epoch.get("metrics/mAP50(B)", 0)),
                        "training_precision": float(last_epoch.get("metrics/precision(B)", 0)),
                        "training_recall": float(last_epoch.get("metrics/recall(B)", 0)),
                        "training_epochs": int(len(df))
                    }
            
            # Save results for this model
            model_results = {
                "model_name": model_info['name'],
                "model_path": str(model_info['model_path']),
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
                "validation_date": datetime.now().isoformat()
            }
            
            all_results.append(model_results)
            
            # Save detailed results in model folder
            if save_results:
                validation_dir = model_info['model_path'].parent.parent / "validation" / f"batch_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                validation_dir.mkdir(parents=True, exist_ok=True)
                
                # Save JSON
                json_path = validation_dir / "validation_results.json"
                with open(json_path, 'w') as f:
                    json.dump(model_results, f, indent=2)
                
                # Save per-image CSV
                csv_data = []
                for i, img_path in enumerate(image_files):
                    csv_data.append({
                        "image": img_path.name,
                        "inference_time_ms": float(inference_times[i]),
                        "detections": int(detections[i])
                    })
                
                csv_df = pd.DataFrame(csv_data)
                csv_path = validation_dir / "per_image_results.csv"
                csv_df.to_csv(csv_path, index=False)
                
                print(f"💾 Results saved to: {validation_dir}")
            
            # Print summary for this model
            print(f"⚡ FPS: {fps:.1f} | mAP@50: {training_metrics.get('training_mAP50', 0):.3f} | Avg Det: {avg_detections:.2f}")
            
        except Exception as e:
            print(f"❌ Error validating {model_info['name']}: {e}")
            continue
    
    # Generate comparison report
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON REPORT")
    print("="*80)
    
    if all_results:
        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                "Model": result["model_name"],
                "FPS": f"{result['fps']:.1f}",
                "mAP@50": f"{result['training_metrics'].get('training_mAP50', 0):.3f}",
                "Precision": f"{result['training_metrics'].get('training_precision', 0):.3f}",
                "Recall": f"{result['training_metrics'].get('training_recall', 0):.3f}",
                "Avg Detections": f"{result['avg_detections_per_image']:.2f}",
                "Avg Confidence": f"{result['avg_confidence']:.3f}",
                "Epochs": result['training_metrics'].get('training_epochs', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by mAP@50 (descending)
        comparison_df['mAP@50_num'] = comparison_df['mAP@50'].astype(float)
        comparison_df = comparison_df.sort_values('mAP@50_num', ascending=False).drop('mAP@50_num', axis=1)
        
        print(comparison_df.to_string(index=False))
        
        # Save comparison report
        if save_results:
            report_path = base_path / "model" / "detection" / "validation_comparison_report.csv"
            comparison_df.to_csv(report_path, index=False)
            print(f"\n💾 Comparison report saved to: {report_path}")
        
        # Find best model
        best_model = max(all_results, key=lambda x: x['training_metrics'].get('training_mAP50', 0))
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   mAP@50: {best_model['training_metrics'].get('training_mAP50', 0):.3f}")
        print(f"   FPS: {best_model['fps']:.1f}")
        print(f"   Path: {best_model['model_path']}")
    
    print("\n" + "="*80)
    print("✅ All models validated successfully!")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Validate all detection models')
    parser.add_argument('--test-images', type=str, default=None,
                        help='Path to test images (default: dataset_strawberry_kaggle/test/images)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--max-models', type=int, default=None,
                        help='Maximum number of models to validate (for testing)')
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_all_models(
        test_images=args.test_images,
        save_results=not args.no_save,
        max_models=args.max_models
    )
    
    if results:
        print(f"\n🎉 Successfully validated {len(results)} models")
    else:
        print("\n❌ Validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()