# Training History Summary

## Overview
Total training runs: 1

### 2025-11-25

| Run ID | Dataset | Model | Batch | Input Size | LR/Epochs | Optimizer | Epochs | Precision | Recall | mAP@50 | Training Time | GPU |
|--------|---------|-------|-------|------------|-----------|-----------|--------|-----------|--------|--------|---------------|-----|
| run_2025 | straw-detect.v1-straw-detect.yolov8 | YOLOv8-n | 8 | 416x416 | 0.002/50 | AdamW | 50/50 | 0.916 | 0.855 | 0.937 | 4.3 min | NVIDIA GeForce RTX 3050 Ti Laptop GPU |

## Detailed Run Information

### Run: run_20251125_150400_manual_baseline
- **Date**: 2025-11-25 15:04:00
- **Experiment**: Baseline_YOLOv8n
- **Status**: completed
- **Dataset**: straw-detect.v1-straw-detect.yolov8 (392 images, 1 classes)
- **Model**: YOLOv8-n (Pretrained: True)
- **Hyperparameters**: Batch=8, Image Size=416, LR=0.002, Optimizer=AdamW
- **Performance**: Precision=0.916, Recall=0.855, mAP@50=0.937, mAP@50-95=0.581
- **Training**: 50/50 epochs, 4.3 minutes
- **System**: NVIDIA GeForce RTX 3050 Ti Laptop GPU, Peak GPU Memory: 1.44 GB
- **Paths**: Model=model/weights/strawberry_yolov8n.pt, Results=model/results/strawberry_detection

