#!/usr/bin/env python3
"""
Optimized ONNX Inference for Raspberry Pi
High-performance inference with ONNX Runtime optimizations
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import Tuple, List, Optional

class OptimizedONNXInference:
    """
    Optimized ONNX inference engine for Raspberry Pi
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize optimized ONNX inference engine

        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.session = self._create_optimized_session()
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Extract input dimensions
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"✅ Optimized ONNX model loaded: {model_path}")
        print(f"📐 Input shape: {self.input_shape}")
        print(f"🎯 Confidence threshold: {conf_threshold}")

    def _create_optimized_session(self) -> ort.InferenceSession:
        """
        Create ONNX session with Raspberry Pi optimizations
        """
        # Set environment variables for optimization
        os.environ["OMP_NUM_THREADS"] = "4"  # Raspberry Pi 4 has 4 cores
        os.environ["OMP_THREAD_LIMIT"] = "4"
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        os.environ["MKL_NUM_THREADS"] = "4"

        # Session options for maximum performance
        session_options = ort.SessionOptions()

        # Enable all graph optimizations
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use sequential execution for consistency
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Optimize thread usage for Raspberry Pi
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True

        # CPU execution provider (Raspberry Pi doesn't have CUDA)
        providers = ['CPUExecutionProvider']

        try:
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            return session
        except Exception as e:
            print(f"❌ Failed to create optimized session: {e}")
            # Fallback to basic session
            return ort.InferenceSession(self.model_path, providers=providers)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized preprocessing for Raspberry Pi

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize with optimization
        image = cv2.resize(image, (self.input_width, self.input_height),
                          interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0

        # Transpose to CHW format (ONNX expects this)
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, outputs: np.ndarray) -> List[dict]:
        """
        Post-process YOLOv8 outputs

        Args:
            outputs: Raw model outputs

        Returns:
            List of detections
        """
        detections = []

        # YOLOv8 output shape: [1, 5, 8400] for 640x640
        # Where 5 = [x, y, w, h, conf] and 8400 = 80x80 + 40x40 + 20x20

        # Reshape outputs
        outputs = outputs[0]  # Remove batch dimension

        # Filter by confidence
        conf_mask = outputs[4] > self.conf_threshold
        filtered_outputs = outputs[:, conf_mask]

        if filtered_outputs.shape[1] == 0:
            return detections

        # Extract boxes and scores
        boxes = filtered_outputs[:4].T  # [x, y, w, h]
        scores = filtered_outputs[4]    # confidence scores

        # Convert from center format to corner format
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Clip to image boundaries
        x1 = np.clip(x1, 0, self.input_width)
        y1 = np.clip(y1, 0, self.input_height)
        x2 = np.clip(x2, 0, self.input_width)
        y2 = np.clip(y2, 0, self.input_height)

        # Create detection dictionaries
        for i in range(len(scores)):
            detection = {
                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                'confidence': float(scores[i]),
                'class': 0,  # Strawberry class
                'class_name': 'strawberry'
            }
            detections.append(detection)

        return detections

    def predict(self, image: np.ndarray) -> Tuple[List[dict], float]:
        """
        Run optimized inference

        Args:
            image: Input image

        Returns:
            Tuple of (detections, inference_time)
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference with timing
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.perf_counter() - start_time

        # Post-process
        detections = self.postprocess(outputs)

        return detections, inference_time

    def predict_batch(self, images: List[np.ndarray]) -> Tuple[List[List[dict]], float]:
        """
        Run batch inference for multiple images

        Args:
            images: List of input images

        Returns:
            Tuple of (list_of_detections, total_inference_time)
        """
        if not images:
            return [], 0.0

        # Preprocess all images
        input_tensors = [self.preprocess(img) for img in images]
        batch_tensor = np.concatenate(input_tensors, axis=0)

        # Run batch inference
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        inference_time = time.perf_counter() - start_time

        # Post-process each image in batch
        all_detections = []
        for i in range(len(images)):
            single_output = outputs[0][i:i+1]  # Extract single image output
            detections = self.postprocess([single_output])
            all_detections.append(detections)

        return all_detections, inference_time

def benchmark_model(model_path: str, test_image_path: str, runs: int = 10) -> dict:
    """
    Benchmark model performance

    Args:
        model_path: Path to ONNX model
        test_image_path: Path to test image
        runs: Number of benchmark runs

    Returns:
        Benchmark results dictionary
    """
    # Load model
    model = OptimizedONNXInference(model_path)

    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise ValueError(f"Could not load test image: {test_image_path}")

    # Warmup run
    _ = model.predict(test_image)

    # Benchmark runs
    times = []
    for _ in range(runs):
        _, inference_time = model.predict(test_image)
        times.append(inference_time * 1000)  # Convert to milliseconds

    # Calculate statistics
    times_array = np.array(times)
    results = {
        'mean_ms': float(np.mean(times_array)),
        'median_ms': float(np.median(times_array)),
        'std_ms': float(np.std(times_array)),
        'min_ms': float(np.min(times_array)),
        'max_ms': float(np.max(times_array)),
        'fps': float(1000 / np.mean(times_array)),
        'runs': runs
    }

    return results

if __name__ == "__main__":
    # Example usage
    model_path = "model/detection/yolov8n/best_416.onnx"
    test_image = "test_detection_result.jpg"

    if os.path.exists(model_path) and os.path.exists(test_image):
        print("🚀 Testing Optimized ONNX Inference")
        print("=" * 50)

        # Load model
        model = OptimizedONNXInference(model_path)

        # Load and predict
        image = cv2.imread(test_image)
        detections, inference_time = model.predict(image)

        print(".2f"        print(f"📊 Detections found: {len(detections)}")

        # Benchmark
        print("\n📈 Running benchmark (10 runs)...")
        results = benchmark_model(model_path, test_image, runs=10)

        print("📊 Benchmark Results:"        print(".2f"        print(".2f"        print(".2f"        print(".2f"        print(".2f"        print(".1f"
        print("\n✅ Optimized inference test complete!")
    else:
        print("❌ Model or test image not found")
        print(f"Model: {model_path} - {'✅' if os.path.exists(model_path) else '❌'}")
        print(f"Image: {test_image} - {'✅' if os.path.exists(test_image) else '❌'}")