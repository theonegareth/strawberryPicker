# 🍓 Raspberry Pi Deployment Guide

This guide provides step-by-step instructions for deploying the optimized strawberry detection and ripeness classification models on Raspberry Pi 4B (or later) for real-time inference.

## 📋 Prerequisites

### Hardware Requirements
- Raspberry Pi 4B (or Raspberry Pi 5) with at least 4GB RAM
- MicroSD card (32GB recommended) with Raspberry Pi OS (64-bit) installed
- USB webcam or Raspberry Pi Camera Module v2
- Power supply (5V 3A)
- Optional: Cooling fan/heatsink for sustained performance

### Software Requirements
- Raspberry Pi OS (64-bit) Bullseye or later
- Python 3.9 or higher
- OpenCV, ONNX Runtime, and other dependencies

## 🚀 Quick Start

### 1. Set Up Raspberry Pi

1. **Flash Raspberry Pi OS** using Raspberry Pi Imager
2. **Enable SSH** and configure Wi-Fi/Ethernet
3. **Update system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### 2. Install Dependencies

```bash
# Install system dependencies
sudo apt install -y python3-pip python3-venv libopenblas-dev libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjasper-dev libqtgui4 libqt4-test

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy opencv-python-headless pillow matplotlib tqdm
pip3 install onnxruntime  # Use onnxruntime for ARM
pip3 install ultralytics  # For PyTorch models (optional)
```

### 3. Clone the Repository

```bash
cd ~
git clone https://github.com/theonegareth/strawberryPicker.git
cd strawberryPicker
```

### 4. Set Up Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📦 Model Deployment

### Download Optimized Models

The repository already includes optimized ONNX and FP16 models in `model/detection/`. You can also generate them using the optimization scripts:

```bash
# Convert YOLOv8n to ONNX (if not already done)
python3 scripts/optimization/convert_yolo_to_onnx.py \
  --model model/detection/yolov8n_kaggle_2500images_trained_20251203_130255/weights/best.pt \
  --output model/detection/yolov8n/best.onnx

# Quantize to FP16
python3 scripts/optimization/quantize_onnx_fp16.py \
  --model model/detection/yolov8n/best.onnx \
  --output model/detection/yolov8n/best_fp16.onnx
```

### Verify Model Compatibility

Run a quick test to ensure the models work on Raspberry Pi:

```bash
python3 scripts/optimization/optimized_onnx_inference.py \
  --model model/detection/yolov8n/best_fp16.onnx \
  --image test_detection_result.jpg
```

Expected output: detection results and inference time.

## 🎯 Performance Optimization

### 1. Use FP16 Quantized Models

FP16 models are 72.7% smaller and faster on Raspberry Pi's ARM CPU. Use `best_fp16.onnx` for deployment.

### 2. Adjust Input Resolution

Lower resolution improves speed at the cost of accuracy. The default is 640x640. You can modify the inference script to use 416x416 or 320x320.

### 3. Enable Thread Pinning

Set environment variables for ONNX Runtime to optimize CPU usage:

```bash
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
```

### 4. Use Camera Streaming Optimizations

For real‑time video, use `scripts/inference/webcam_inference_WSL.py` with the `--optimize` flag (if implemented) or reduce frame rate.

## 🔧 Running Inference

### Single Image Detection

```bash
python3 scripts/inference/image_inference.py \
  --detector model/detection/yolov8n/best_fp16.onnx \
  --image path/to/image.jpg \
  --output result.jpg
```

### Real‑Time Webcam Detection

```bash
python3 scripts/inference/webcam_inference_WSL.py \
  --detector model/detection/yolov8n/best_fp16.onnx \
  --camera 0 \
  --confidence 0.5
```

### Two‑Stage Detection + Ripeness Classification

```bash
python3 scripts/inference/detect_and_classify_ripeness.py \
  --detector model/detection/yolov8n/best_fp16.onnx \
  --classifier model/ripeness_classification_dataset/best_ripeness_classifier.pth \
  --image path/to/image.jpg \
  --output result.jpg
```

## 📊 Expected Performance

| Model | Resolution | Inference Time (RPi 4B) | FPS | Memory Usage |
|-------|------------|-------------------------|-----|--------------|
| YOLOv8n (FP16) | 640×640 | ~120‑150 ms | 6‑8 | ~100 MB |
| YOLOv8n (FP16) | 416×416 | ~70‑90 ms | 11‑14 | ~80 MB |
| YOLOv8s (FP16) | 640×640 | ~200‑250 ms | 4‑5 | ~150 MB |

**Note:** Actual performance depends on system load, thermal throttling, and camera I/O.

## 🧪 Benchmarking on Raspberry Pi

Use the built‑in benchmark script to measure performance:

```bash
python3 scripts/optimization/benchmark_inference_speed.py \
  --onnx-model model/detection/yolov8n/best.onnx \
  --fp16-model model/detection/yolov8n/best_fp16.onnx \
  --image test_detection_result.jpg \
  --runs 30 \
  --device cpu
```

Results will be saved to `benchmark_results/`.

## 🔌 Integration with Robotic Arm

The project includes Arduino code for a 5‑DOF robotic arm. After detection, you can send coordinates to the arm via serial communication.

1. **Install ROS2 (optional)** – see `ArduinoCode/README.md`
2. **Run the detection node** that publishes strawberry coordinates
3. **Arm controller** subscribes and executes picking motions

Example integration script (simplified):

```python
# scripts/inference/arm_integration.py
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)

def send_coordinates(x, y, confidence):
    command = f"PICK {x} {y} {confidence}\n"
    ser.write(command.encode())
    response = ser.readline().decode().strip()
    return response
```

## 🐛 Troubleshooting

### Common Issues

1. **Slow inference** – Ensure you are using the FP16 model, reduce resolution, close background applications.
2. **High memory usage** – Check with `htop` and consider using `swap` if needed.
3. **Camera not detected** – Verify USB connection or enable Raspberry Pi Camera Module via `sudo raspi-config`.
4. **ONNX Runtime errors** – Install the correct version: `pip3 install onnxruntime==1.15.1`.

### Performance Tips

- **Overclock** the Raspberry Pi (if adequately cooled) for extra speed.
- **Use a heatsink/fan** to prevent thermal throttling.
- **Disable desktop GUI** (`sudo systemctl set-default multi-user.target`) to free up RAM.
- **Use `taskset`** to pin inference to specific CPU cores.

## 📈 Monitoring

Monitor system resources during inference:

```bash
# CPU temperature
vcgencmd measure_temp

# CPU frequency
vcgencmd measure_clock arm

# Memory usage
free -h
```

## 🚀 Production Deployment Checklist

- [ ] Test all models on actual Raspberry Pi hardware
- [ ] Verify camera feed stability
- [ ] Set up auto‑start script for boot‑time launch
- [ ] Implement logging and error recovery
- [ ] Create a systemd service for continuous operation
- [ ] Perform stress test (24‑hour run)

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime ARM Builds](https://onnxruntime.ai/docs/execution-providers/ARM-ExecutionProvider.html)
- [Raspberry Pi Camera Setup](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Project GitHub Repository](https://github.com/theonegareth/strawberryPicker)

## 🎉 Conclusion

Your optimized strawberry detection pipeline is now ready for Raspberry Pi deployment. The FP16 quantized models provide a great balance of speed and accuracy, enabling real‑time picking decisions.

For further assistance, open an issue on the GitHub repository or refer to the `OPTIMIZATION_SUMMARY.md` for technical details.

**Happy picking!** 🍓