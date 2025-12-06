#!/usr/bin/env python3
"""
Comprehensive Inference Speed Benchmark Script
Compares PyTorch vs ONNX vs FP16 quantized models for strawberry detection and ripeness classification
"""

import time
import torch
import onnxruntime as ort
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import statistics

# Add training modules to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

class StrawberryDetectionBenchmark:
    """Benchmark class for strawberry detection models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def load_pytorch_model(self, model_path: str):
        """Load PyTorch YOLO model"""
        from ultralytics import YOLO
        print(f"Loading PyTorch YOLO model: {model_path}")
        self.pytorch_model = YOLO(model_path)
        print(f"✅ PyTorch model loaded on {self.device}")
    
    def load_onnx_model(self, model_path: str):
        """Load ONNX model"""
        print(f"Loading ONNX model: {model_path}")
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        print(f"✅ ONNX model loaded with providers: {self.onnx_session.get_providers()}")
    
    def load_fp16_model(self, model_path: str):
        """Load FP16 quantized ONNX model"""
        print(f"Loading FP16 quantized model: {model_path}")
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.fp16_session = ort.InferenceSession(model_path, providers=providers)
        print(f"✅ FP16 model loaded with providers: {self.fp16_session.get_providers()}")
    
    def preprocess_image(self, image_path: str, size=(640, 640)) -> np.ndarray:
        """Preprocess image for inference"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize and normalize
        image_resized = cv2.resize(image, size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        image_input = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_input, axis=0)
        
        return image_batch
    
    def benchmark_pytorch_inference(self, image_path: str, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark PyTorch model inference"""
        print(f"\n🔍 Benchmarking PyTorch inference ({num_runs} runs)...")
        
        # Warm up
        for _ in range(10):
            _ = self.pytorch_model(image_path, verbose=False)
        
        # Actual benchmark
        inference_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            results = self.pytorch_model(image_path, verbose=False)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        return self._calculate_stats(inference_times, "PyTorch")
    
    def benchmark_onnx_inference(self, image_path: str, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX model inference"""
        print(f"\n🔍 Benchmarking ONNX inference ({num_runs} runs)...")
        
        # Preprocess image
        image_input = self.preprocess_image(image_path)
        
        # Get input/output names
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        # Warm up
        for _ in range(10):
            _ = self.onnx_session.run([output_name], {input_name: image_input})
        
        # Actual benchmark
        inference_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            outputs = self.onnx_session.run([output_name], {input_name: image_input})
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        return self._calculate_stats(inference_times, "ONNX")
    
    def benchmark_fp16_inference(self, image_path: str, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark FP16 quantized model inference"""
        print(f"\n🔍 Benchmarking FP16 inference ({num_runs} runs)...")
        
        # Preprocess image
        image_input = self.preprocess_image(image_path)
        
        # Get input/output names
        input_name = self.fp16_session.get_inputs()[0].name
        output_name = self.fp16_session.get_outputs()[0].name
        
        # Warm up
        for _ in range(10):
            _ = self.fp16_session.run([output_name], {input_name: image_input})
        
        # Actual benchmark
        inference_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            outputs = self.fp16_session.run([output_name], {input_name: image_input})
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        return self._calculate_stats(inference_times, "FP16")
    
    def _calculate_stats(self, times: List[float], model_name: str) -> Dict[str, float]:
        """Calculate statistics from inference times"""
        stats = {
            f'{model_name}_mean_ms': statistics.mean(times),
            f'{model_name}_median_ms': statistics.median(times),
            f'{model_name}_std_ms': statistics.stdev(times) if len(times) > 1 else 0,
            f'{model_name}_min_ms': min(times),
            f'{model_name}_max_ms': max(times),
            f'{model_name}_fps': 1000.0 / statistics.mean(times)
        }
        
        print(f"  📊 {model_name} Results:")
        print(f"    Mean: {stats[f'{model_name}_mean_ms']:.2f} ms")
        print(f"    Median: {stats[f'{model_name}_median_ms']:.2f} ms")
        print(f"    Std Dev: {stats[f'{model_name}_std_ms']:.2f} ms")
        print(f"    Min: {stats[f'{model_name}_min_ms']:.2f} ms")
        print(f"    Max: {stats[f'{model_name}_max_ms']:.2f} ms")
        print(f"    FPS: {stats[f'{model_name}_fps']:.2f}")
        
        return stats
    
    def run_comprehensive_benchmark(self, image_path: str, num_runs: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing all models"""
        print(f"\n🚀 Running Comprehensive Inference Benchmark")
        print(f"📷 Test image: {image_path}")
        print(f"🔄 Number of runs per model: {num_runs}")
        print(f"💻 Device: {self.device}")
        
        all_results = {}
        
        # Benchmark PyTorch model
        if hasattr(self, 'pytorch_model'):
            pytorch_results = self.benchmark_pytorch_inference(image_path, num_runs)
            all_results.update(pytorch_results)
        
        # Benchmark ONNX model
        if hasattr(self, 'onnx_session'):
            onnx_results = self.benchmark_onnx_inference(image_path, num_runs)
            all_results.update(onnx_results)
        
        # Benchmark FP16 model
        if hasattr(self, 'fp16_session'):
            fp16_results = self.benchmark_fp16_inference(image_path, num_runs)
            all_results.update(fp16_results)
        
        # Calculate speedup ratios
        all_results = self._calculate_speedup_ratios(all_results)
        
        return all_results
    
    def _calculate_speedup_ratios(self, results: Dict[str, float]) -> Dict[str, float]:
        """Calculate speedup ratios between models"""
        if 'PyTorch_mean_ms' in results and 'ONNX_mean_ms' in results:
            speedup_onnx = results['PyTorch_mean_ms'] / results['ONNX_mean_ms']
            results['ONNX_speedup'] = speedup_onnx
            print(f"\n🚀 ONNX speedup over PyTorch: {speedup_onnx:.2f}x")
        
        if 'ONNX_mean_ms' in results and 'FP16_mean_ms' in results:
            speedup_fp16 = results['ONNX_mean_ms'] / results['FP16_mean_ms']
            results['FP16_speedup_over_ONNX'] = speedup_fp16
            print(f"🚀 FP16 speedup over ONNX: {speedup_fp16:.2f}x")
        
        if 'PyTorch_mean_ms' in results and 'FP16_mean_ms' in results:
            speedup_total = results['PyTorch_mean_ms'] / results['FP16_mean_ms']
            results['Total_FP16_speedup'] = speedup_total
            print(f"🚀 Total FP16 speedup over PyTorch: {speedup_total:.2f}x")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    def create_performance_plot(self, results: Dict[str, Any], output_path: str):
        """Create performance comparison plot"""
        models = []
        mean_times = []
        fps_values = []
        
        if 'PyTorch_mean_ms' in results:
            models.append('PyTorch')
            mean_times.append(results['PyTorch_mean_ms'])
            fps_values.append(results['PyTorch_fps'])
        
        if 'ONNX_mean_ms' in results:
            models.append('ONNX')
            mean_times.append(results['ONNX_mean_ms'])
            fps_values.append(results['ONNX_fps'])
        
        if 'FP16_mean_ms' in results:
            models.append('FP16')
            mean_times.append(results['FP16_mean_ms'])
            fps_values.append(results['FP16_fps'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Inference time plot
        bars1 = ax1.bar(models, mean_times, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Inference Time Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars1, mean_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}ms', ha='center', va='bottom')
        
        # FPS plot
        bars2 = ax2.bar(models, fps_values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax2.set_ylabel('Frames Per Second (FPS)')
        ax2.set_title('Throughput Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, fps in zip(bars2, fps_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{fps:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Performance plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark strawberry detection inference speed')
    parser.add_argument('--pytorch-model', type=str,
                       help='Path to PyTorch YOLO model (.pt file)')
    parser.add_argument('--onnx-model', type=str,
                       help='Path to ONNX model (.onnx file)')
    parser.add_argument('--fp16-model', type=str,
                       help='Path to FP16 quantized ONNX model (.onnx file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of inference runs per model')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = StrawberryDetectionBenchmark(device=args.device)
    
    # Load models
    if args.pytorch_model:
        benchmark.load_pytorch_model(args.pytorch_model)
    
    if args.onnx_model:
        benchmark.load_onnx_model(args.onnx_model)
    
    if args.fp16_model:
        benchmark.load_fp16_model(args.fp16_model)
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(args.image, args.runs)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_results_{timestamp}.json"
    plot_file = output_dir / f"performance_comparison_{timestamp}.png"
    
    benchmark.save_results(results, str(results_file))
    benchmark.create_performance_plot(results, str(plot_file))
    
    # Print summary
    print(f"\n🎯 Benchmark Summary:")
    print(f"📊 Total models tested: {len([k for k in results.keys() if '_mean_ms' in k])}")
    print(f"🚀 Best performing model: {min([(k, v) for k, v in results.items() if '_mean_ms' in k], key=lambda x: x[1])[0].replace('_mean_ms', '')}")
    
    if 'Total_FP16_speedup' in results:
        print(f"💨 Total speedup with FP16: {results['Total_FP16_speedup']:.2f}x")

if __name__ == "__main__":
    main()