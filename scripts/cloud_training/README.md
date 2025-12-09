# 🚀 Cloud Training for YOLOv8 Models

This directory contains resources for training YOLOv8 models on cloud GPU providers (RunPod, Google Colab, AWS, etc.) to accelerate training of larger models like YOLOv8m and YOLOv8l.

## 📋 Contents

- `runpod_yolov8_training.ipynb` - Jupyter notebook for RunPod cloud training
- `runpod_yolov8_training.py` - Python script alternative (standalone)
- `create_runpod_notebook.py` - Script to regenerate the notebook
- `README.md` - This file

## 🎯 Training Goals

Train two larger YOLOv8 models for improved accuracy:

1. **YOLOv8m (Medium)** - Better accuracy than YOLOv8s, suitable for production
2. **YOLOv8l (Large)** - Maximum accuracy for validation/research

## ⚙️ Hardware Requirements

| Provider | GPU | VRAM | Cost/Hour | Estimated Training Time | Total Cost |
|----------|-----|------|-----------|------------------------|------------|
| **RunPod** | RTX 3080 | 10GB | **$0.17** | 3-4 hours | **$0.51-$0.68** |
| RunPod | RTX 3090 | 24GB | $0.22 | 2.5-3.5 hours | $0.55-$0.77 |
| RunPod | RTX A5000 | 24GB | $0.16 | 3-4 hours | $0.48-$0.64 |
| Google Colab | T4/P100 | 16GB | Free (limited) | 4-6 hours | Free |
| AWS | g4dn.xlarge | 16GB | $0.526 | 3-4 hours | $1.58-$2.10 |

**Recommendation**: Use **RunPod RTX 3080** for best price/performance ratio.

## 📊 Expected Performance

Based on YOLOv8s training results (98.5% mAP@50):

| Model | Target mAP@50 | Training Time | Model Size | Inference Speed (RTX 3080) |
|-------|---------------|---------------|------------|----------------------------|
| **YOLOv8m** | 99.0-99.2% | 3-4 hours | ~49MB | ~25-30 FPS |
| **YOLOv8l** | 99.2-99.4% | 4-5 hours | ~87MB | ~15-20 FPS |

## 🚀 Quick Start

### Option 1: RunPod (Recommended)

1. **Sign up** at [runpod.io](https://runpod.io) and add credits ($5-10)
2. **Deploy** a GPU instance:
   - Template: `RunPod PyTorch` or `RunPod Jupyter`
   - GPU: RTX 3080 (recommended) or RTX 3090
   - Storage: At least 50GB
3. **Upload files**:
   ```bash
   # On your local machine
   scp -P <port> runpod_yolov8_training.ipynb user@runpod.io:/workspace/
   scp -P <port> -r model/dataset_strawberry_kaggle user@runpod.io:/workspace/
   ```
4. **Open Jupyter** in RunPod web interface
5. **Run the notebook** cells sequentially

### Option 2: Google Colab

1. Upload `runpod_yolov8_training.ipynb` to Google Drive
2. Open with Google Colab
3. Mount Google Drive and adjust paths
4. Run cells (free GPU but limited runtime)

### Option 3: Local with GPU

If you have a local GPU with sufficient VRAM (≥8GB):
```bash
python scripts/cloud_training/runpod_yolov8_training.py
```

## 📁 Dataset Preparation

Ensure your dataset is properly organized:
```
model/dataset_strawberry_kaggle/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

## ⚡ Training Commands

### YOLOv8m (Medium Model)
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(
    data='model/dataset_strawberry_kaggle/data.yaml',
    epochs=120,
    imgsz=640,
    batch=24,  # Adjust based on GPU memory
    device=0,
    project='model/detection/cloud_training',
    name='yolov8m_runpod',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.01,
    amp=True,  # Mixed precision
    plots=True,
    save_period=10
)
```

### YOLOv8l (Large Model)
```python
model = YOLO('yolov8l.pt')
results = model.train(
    data='model/dataset_strawberry_kaggle/data.yaml',
    epochs=150,
    imgsz=640,
    batch=12,  # Smaller batch for larger model
    device=0,
    project='model/detection/cloud_training',
    name='yolov8l_runpod',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.008,  # Slightly lower learning rate
    amp=True,
    plots=True,
    save_period=10
)
```

## 📈 Monitoring Training

The notebook includes real-time monitoring functions:

```python
# Monitor training progress
progress_df = monitor_training_progress("m")  # For YOLOv8m
```

Key metrics to watch:
- **Training Loss**: Should decrease steadily
- **Validation mAP@50**: Should increase toward 0.99+
- **GPU Utilization**: Should be >90% during training
- **Memory Usage**: Should be stable

## 💾 Downloading Results

After training completes, package models for download:

```python
zip_file = package_trained_models()
print(f"Download: {zip_file}")
```

The package includes:
- `best.pt` and `last.pt` for each model
- Training results CSV
- Performance plots (results.png, confusion_matrix.png, etc.)

## 🔄 Integration with Local Pipeline

After downloading trained models:

1. **Move models** to your local detection folder:
   ```bash
   unzip yolov8_cloud_trained_*.zip -d model/detection/
   ```

2. **Update model registry**:
   ```bash
   python scripts/model_consolidation.py --add-model model/detection/cloud_training/yolov8m_runpod/weights/best.pt --name yolov8m_cloud
   ```

3. **Test inference**:
   ```bash
   python scripts/inference/detect_and_classify.py --model model/detection/cloud_training/yolov8m_runpod/weights/best.pt
   ```

## ⚠️ Cost Management Tips

1. **Monitor usage**: RunPod shows real-time cost in dashboard
2. **Auto-stop**: Configure auto-stop after training completes
3. **Download promptly**: Download models before stopping instance
4. **Clean up**: Delete unnecessary files to avoid storage charges
5. **Use spot instances**: Cheaper but may be interrupted

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size (`batch=16` or `batch=8`)
2. **Dataset not found**: Check paths in `data.yaml`
3. **Training too slow**: Ensure GPU is being used (`device=0`)
4. **Validation errors**: Check dataset format and labels

### Performance Optimization

- **Mixed Precision**: Enabled by default (`amp=True`)
- **Data Loading**: Use `workers=8` for parallel loading
- **Image Caching**: Enable with `cache=True` if you have enough RAM
- **Early Stopping**: Monitor validation metrics to stop early if plateaued

## 📞 Support

- **RunPod Documentation**: https://docs.runpod.io
- **Ultralytics YOLOv8 Docs**: https://docs.ultralytics.com
- **Issues**: Check GitHub repository issues

## 📝 License

This training setup is part of the Strawberry Picker project. See main repository for license details.

---

**Happy Training! 🍓🚀**