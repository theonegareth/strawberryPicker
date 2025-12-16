#!/usr/bin/env python3
"""
Benchmark YOLO models for performance on Raspberry Pi 4B (or current machine).
Measures inference time, FPS, and memory usage for different model formats.
"""

import argparse
import time
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO
import psutil
import platform

def get_system_info():
    """Get system information for benchmarking context."""
    info = {
        'system': platform.system(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=False),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
    }
    return info

def load_test_images(dataset_path, max_images=50):
    """Load test images from dataset for benchmarking."""
    test_images = []
    
    # Try multiple possible locations
    possible_paths = [
        Path(dataset_path) / "test" / "images",
        Path(dataset_path) / "valid" / "images",
        Path(dataset_path) / "val" / "images",
        Path(dataset_path) / "train" / "images",
    ]
    
    for path in possible_paths:
        if path.exists():
            image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if image_files:
                test_images = [str(p) for p in image_files[:max_images]]
                print(f"Found {len(test_images)} images in {path}")
                break
    
    if not test_images:
        # Create dummy images if no dataset found
        print("No test images found. Creating dummy images for benchmarking.")
        test_images = []
        for i in range(10):
            # Create a dummy image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_path = f"/tmp/dummy_{i}.jpg"
            cv2.imwrite(dummy_path, dummy_img)
            test_images.append(dummy_path)
    
    return test_images

def benchmark_model(model_path, test_images, img_size=640, warmup=10, runs=100):
    """
    Benchmark a single model.
    
    Args:
        model_path: Path to model file (.pt, .onnx, .tflite)
        test_images: List of image paths for testing
        img_size: Input image size
        warmup: Number of warmup runs
        runs: Number of benchmark runs
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*60}")
    
    results = {
        'model': os.path.basename(model_path),
        'format': Path(model_path).suffix[1:],
        'size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0,
        'inference_times': [],
        'memory_usage_mb': [],
        'success': False
    }
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return results
    
    try:
        # Load model
        print(f"  Loading model...")
        start_load = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_load
        results['load_time'] = load_time
        
        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for i in range(warmup):
            if i >= len(test_images):
                img_path = test_images[0]
            else:
                img_path = test_images[i]
            img = cv2.imread(img_path)
            if img is None:
                # Create dummy image
                img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            _ = model(img, verbose=False)
        
        # Benchmark runs
        print(f"  Running benchmark ({runs} runs)...")
        for i in range(runs):
            # Cycle through test images
            img_idx = i % len(test_images)
            img_path = test_images[img_idx]
            img = cv2.imread(img_path)
            if img is None:
                img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Inference
            start_time = time.perf_counter()
            results_inference = model(img, verbose=False)
            inference_time = time.perf_counter() - start_time
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            results['inference_times'].append(inference_time)
            results['memory_usage_mb'].append(mem_used)
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"    Completed {i+1}/{runs} runs...")
        
        # Calculate statistics
        if results['inference_times']:
            times = np.array(results['inference_times'])
            results['avg_inference_ms'] = np.mean(times) * 1000
            results['std_inference_ms'] = np.std(times) * 1000
            results['min_inference_ms'] = np.min(times) * 1000
            results['max_inference_ms'] = np.max(times) * 1000
            results['fps'] = 1.0 / np.mean(times)
            results['avg_memory_mb'] = np.mean(results['memory_usage_mb'])
            results['success'] = True
            
            print(f"  ✅ Benchmark completed:")
            print(f"     Model size: {results['size_mb']:.2f} MB")
            print(f"     Avg inference: {results['avg_inference_ms']:.2f} ms")
            print(f"     FPS: {results['fps']:.2f}")
            print(f"     Memory usage: {results['avg_memory_mb']:.2f} MB")
        else:
            print(f"  ❌ No inference times recorded")
            
    except Exception as e:
        print(f"  ❌ Error benchmarking {model_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def benchmark_all_models(models_to_test, test_images, img_size=640):
    """Benchmark multiple models and return results."""
    all_results = []
    
    for model_info in models_to_test:
        model_path = model_info['path']
        if not os.path.exists(model_path):
            print(f"Skipping {model_path} - not found")
            continue
        
        results = benchmark_model(
            model_path=model_path,
            test_images=test_images,
            img_size=img_size,
            warmup=10,
            runs=50  # Reduced for faster benchmarking
        )
        
        results.update({
            'name': model_info['name'],
            'description': model_info.get('description', '')
        })
        all_results.append(results)
    
    return all_results

def print_results_table(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(f"{'Model':<30} {'Format':<8} {'Size (MB)':<10} {'Inference (ms)':<15} {'FPS':<10} {'Memory (MB)':<12} {'Status':<10}")
    print("-"*100)
    
    for r in results:
        if r['success']:
            print(f"{r['name'][:28]:<30} {r['format']:<8} {r['size_mb']:>9.2f} "
                  f"{r['avg_inference_ms']:>14.2f} {r['fps']:>9.2f} {r['avg_memory_mb']:>11.2f} {'✅':<10}")
        else:
            print(f"{r['name'][:28]:<30} {r['format']:<8} {r['size_mb']:>9.2f} "
                  f"{'N/A':>14} {'N/A':>9} {'N/A':>11} {'❌':<10}")
    
    print("="*100)
    
    # Find best model by FPS
    successful = [r for r in results if r['success']]
    if successful:
        best_by_fps = max(successful, key=lambda x: x['fps'])
        best_by_size = min(successful, key=lambda x: x['size_mb'])
        best_by_memory = min(successful, key=lambda x: x['avg_memory_mb'])
        
        print(f"\n🏆 Best by FPS: {best_by_fps['name']} ({best_by_fps['fps']:.2f} FPS)")
        print(f"🏆 Best by size: {best_by_size['name']} ({best_by_size['size_mb']:.2f} MB)")
        print(f"🏆 Best by memory: {best_by_memory['name']} ({best_by_memory['avg_memory_mb']:.2f} MB)")

def save_results_to_csv(results, output_path="benchmark_results.csv"):
    """Save benchmark results to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['name', 'format', 'size_mb', 'avg_inference_ms', 
                     'std_inference_ms', 'min_inference_ms', 'max_inference_ms',
                     'fps', 'avg_memory_mb', 'load_time', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            writer.writerow({
                'name': r['name'],
                'format': r['format'],
                'size_mb': r.get('size_mb', 0),
                'avg_inference_ms': r.get('avg_inference_ms', 0),
                'std_inference_ms': r.get('std_inference_ms', 0),
                'min_inference_ms': r.get('min_inference_ms', 0),
                'max_inference_ms': r.get('max_inference_ms', 0),
                'fps': r.get('fps', 0),
                'avg_memory_mb': r.get('avg_memory_mb', 0),
                'load_time': r.get('load_time', 0),
                'success': r['success']
            })
    
    print(f"\n📊 Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO models for performance')
    parser.add_argument('--dataset', type=str, default='model/dataset_strawberry_detect_v3',
                        help='Path to dataset for test images')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size for inference')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get system info
    system_info = get_system_info()
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title():<20}: {value}")
    
    # Define models to test
    models_to_test = [
        # Base YOLO models
        {'name': 'YOLOv8n', 'path': 'yolov8n.pt', 'description': 'Ultralytics YOLOv8n'},
        {'name': 'YOLOv8s', 'path': 'yolov8s.pt', 'description': 'Ultralytics YOLOv8s'},
        {'name': 'YOLOv8m', 'path': 'yolov8m.pt', 'description': 'Ultralytics YOLOv8m'},
        
        # Custom trained models
        {'name': 'Strawberry YOLOv11n', 'path': 'model/weights/strawberry_yolov11n.pt', 'description': 'Custom trained on strawberry dataset'},
        {'name': 'Strawberry YOLOv11n ONNX', 'path': 'model/weights/strawberry_yolov11n.onnx', 'description': 'ONNX export'},
        
        # Ripeness detection models
        {'name': 'Ripeness YOLOv11n', 'path': 'model/weights/ripeness_detection_yolov11n.pt', 'description': 'Ripeness detection model'},
        {'name': 'Ripeness YOLOv11n ONNX', 'path': 'model/weights/ripeness_detection_yolov11n.onnx', 'description': 'ONNX export'},
    ]
    
    # Check which models exist
    existing_models = []
    for model in models_to_test:
        if os.path.exists(model['path']):
            existing_models.append(model)
        else:
            print(f"⚠️  Model not found: {model['path']}")
    
    if not existing_models:
        print("❌ No models found for benchmarking.")
        print("Please train a model first or download pretrained weights.")
        sys.exit(1)
    
    # Load test images
    print(f"\n📷 Loading test images from {args.dataset}...")
    test_images = load_test_images(args.dataset, max_images=50)
    print(f"   Loaded {len(test_images)} test images")
    
    # Run benchmarks
    print(f"\n🚀 Starting benchmarks...")
    results = benchmark_all_models(existing_models, test_images, img_size=args.img_size)
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results_to_csv(results, args.output)
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS FOR RASPBERRY PI 4B:")
    print(f"   1. For fastest inference: Choose model with highest FPS")
    print(f"   2. For memory-constrained environments: Choose smallest model")
    print(f"   3. For best accuracy/speed tradeoff: Consider YOLOv8s")
    print(f"   4. For edge deployment: Convert to TFLite INT8 for ~2-3x speedup")
    
    # Check if we're on Raspberry Pi
    if 'arm' in platform.machine().lower() or 'raspberry' in platform.system().lower():
        print(f"\n🎯 Running on Raspberry Pi - results are accurate for deployment.")
    else:
        print(f"\n⚠️  Not running on Raspberry Pi - results are for reference only.")
        print(f"   Actual Raspberry Pi performance may be 2-5x slower.")

if __name__ == '__main__':
    main()