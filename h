# Model Configuration for Strawberry YOLOv8s Detector

# Model Architecture
model_name: "yolov8s"
task: "object_detection"
framework: "ultralytics"

# Training Configuration
training:
  epochs: 150
  batch_size: 4
  image_size: 640
  learning_rate: 0.01
  patience: 100

# Dataset Configuration
dataset:
  name: "strawberry_detection"
  classes: ["strawberry"]
  num_classes: 1
  train_split: 0.8
  val_split: 0.2

# Model Performance
performance:
  model_size_mb: 22
  inference_speed: "real-time"
  accuracy: "high"
  precision: "optimized for agricultural use"

# Hardware Requirements
requirements:
  gpu_recommended: true
  min_memory_gb: 4
  python_version: ">=3.8"

# Usage
usage:
  real_time_detection: true
  batch_processing: true
  integration_ready: true
  robotic_compatible: true