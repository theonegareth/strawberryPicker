# 🍓 Strawberry Model Training Workflow

This guide explains how to train new models using the organized training system.

## 📋 Prerequisites

1. **Dataset ready** in YOLO format
2. **Environment set up** with required packages
3. **GPU available** (recommended for faster training)

## 🚀 Complete Training Workflow

### **Step 1: Create New Model Structure**

Before training, create an organized folder for your new model:

```bash
cd /home/user/machine-learning/GitHubRepos/strawberryPicker

# Create new model structure
python3 train_and_organize.py \
  --type detection \
  --architecture yolov8s \
  --description "experiment_name_here"
```

**Example:**
```bash
python3 train_and_organize.py \
  --type detection \
  --architecture yolov8s \
  --description "higher_confidence_threshold"
```

**Output:**
```
🚀 Creating new detection model...
📛 Model name: yolov8s_higher_confidence_threshold_20251202_153433
✅ Created model structure: models/detection/yolov8s_higher_confidence_threshold_20251202_153433
```

### **Step 2: Train the Model**

Use the YOLO training script with your new model directory:

```bash
# Basic training command
python3 model/training/train_yolov8.py \
  --data model/data.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --weights yolov8s.pt \
  --project models/detection/yolov8s_higher_confidence_threshold_20251202_153433 \
  --name training_run
```

**Training Parameters Explained:**
- `--data`: Path to your dataset configuration
- `--epochs`: Number of training iterations (100-300 recommended)
- `--batch`: Batch size (adjust based on GPU memory)
- `--imgsz`: Image size for training
- `--weights`: Pre-trained weights to start from
- `--project`: Output directory (use your new model folder)
- `--name`: Name of this specific training run

**Example with custom parameters:**
```bash
python3 model/training/train_yolov8.py \
  --data model/data.yaml \
  --epochs 150 \
  --batch 8 \
  --imgsz 640 \
  --weights yolov8s.pt \
  --project models/detection/yolov8s_higher_confidence_threshold_20251202_153433 \
  --name enhanced_training \
  --patience 50 \
  --optimizer AdamW \
  --lr0 0.001
```

### **Step 3: Monitor Training**

Training will create several output files:
- `training_run/weights/best.pt` - Best model weights
- `training_run/weights/last.pt` - Last checkpoint
- `training_run/results.csv` - Training metrics
- `training_run/results.png` - Training curves

**Check training progress:**
```bash
# View training curves
ls models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/results.png

# Check metrics
cat models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/results.csv
```

### **Step 4: Organize Trained Model**

After training completes, copy the best weights to the organized structure:

```bash
python3 train_and_organize.py \
  --type detection \
  --architecture yolov8s \
  --description "higher_confidence_threshold" \
  --weights models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/weights/best.pt
```

**This will:**
- Copy `best.pt` to `weights/model.pt` and `weights/best.pt`
- Create `training_config.json` with metadata
- Generate a `README.md` with model information

### **Step 5: Validate the Model**

Run validation on test images:

```bash
python3 validate_models.py \
  --detector models/detection/yolov8s_higher_confidence_threshold_20251202_153433/weights/best.pt \
  --test-dir model/test/images \
  --output-dir models/detection/yolov8s_higher_confidence_threshold_20251202_153433/validation/detection_results \
  --num-samples 25 \
  --grid-size 5
```

**Check validation results:**
```bash
# View validation grid
ls models/detection/yolov8s_higher_confidence_threshold_20251202_153433/validation/detection_results/validation_grid.jpg

# Check detection metrics
cat models/detection/yolov8s_higher_confidence_threshold_20251202_153433/validation/detection_results/validation_report.json
```

### **Step 6: Test with Individual Images**

Test the model on specific images:

```bash
python3 image_inference.py \
  --detector models/detection/yolov8s_higher_confidence_threshold_20251202_153433/weights/best.pt \
  --image path/to/your/image.jpg \
  --output test_result.jpg
```

### **Step 7: Push to GitHub**

Add and commit your new model:

```bash
# Add the new model
git add models/detection/yolov8s_higher_confidence_threshold_20251202_153433/

# Commit with descriptive message
git commit -m "Add new detection model: yolov8s_higher_confidence_threshold

- Trained for 150 epochs with batch size 8
- Achieved X% mAP on validation set
- Detection rate: Y strawberries per image average"

# Push to GitHub
git push origin main
```

## 🔧 Advanced Training Options

### **Resume Training from Checkpoint**

If training was interrupted:

```bash
python3 model/training/train_yolov8.py \
  --data model/data.yaml \
  --resume models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/weights/last.pt
```

### **Hyperparameter Tuning**

Create a custom hyperparameter file:

```bash
# Create hyperparameter config
cat > models/detection/yolov8s_higher_confidence_threshold_20251202_153433/config/hyp.yaml << EOF
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
EOF

# Train with custom hyperparameters
python3 model/training/train_yolov8.py \
  --data model/data.yaml \
  --hyp models/detection/yolov8s_higher_confidence_threshold_20251202_153433/config/hyp.yaml \
  --epochs 150 \
  --batch 8 \
  --project models/detection/yolov8s_higher_confidence_threshold_20251202_153433 \
  --name custom_hyp_training
```

### **Multi-GPU Training**

If you have multiple GPUs:

```bash
python3 -m torch.distributed.run --nproc_per_node 2 model/training/train_yolov8.py \
  --data model/data.yaml \
  --epochs 150 \
  --batch 16 \
  --project models/detection/yolov8s_higher_confidence_threshold_20251202_153433 \
  --name multi_gpu_training
```

## 📊 Monitoring Training Progress

### **View Real-time Metrics**

Training automatically saves metrics that you can monitor:

```bash
# Watch training log
tail -f models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/results.csv

# Plot training curves
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('models/detection/yolov8s_higher_confidence_threshold_20251202_153433/training_run/results.csv')
df[['train/box_loss', 'val/box_loss']].plot()
plt.savefig('training_loss.png')
"
```

### **Key Metrics to Watch:**

- **mAP@0.5**: Mean Average Precision at IoU 0.5 (higher is better, aim for >0.7)
- **Precision**: Correct detections / total detections
- **Recall**: Correct detections / total ground truth
- **box_loss**: Bounding box regression loss (should decrease)
- **cls_loss**: Classification loss (should decrease)

## 🎯 Best Practices

1. **Start Small**: Begin with 50-100 epochs to test your setup
2. **Monitor Overfitting**: If validation loss increases while training loss decreases, reduce epochs or use more data augmentation
3. **Save Checkpoints**: Training saves checkpoints every epoch automatically
4. **Document Everything**: Update the model's README.md with training details
5. **Validate Early**: Run validation after 50 epochs to check if training is working
6. **Use Version Control**: Commit each new model to GitHub with descriptive messages

## 🆘 Troubleshooting

**If training fails to start:**
```bash
# Check GPU availability
nvidia-smi

# Check dataset paths
cat model/data.yaml

# Verify images exist
ls model/train/images/ | head -5
```

**If detection is poor after training:**
- Increase training epochs
- Add more training data
- Adjust confidence threshold in validation
- Try different model architecture (yolov8m instead of yolov8s)

## 📚 Quick Reference Commands

```bash
# Create new model
python3 train_and_organize.py --type detection --architecture yolov8s --description "my_experiment"

# List all models
python3 train_and_organize.py --list

# Get latest model
python3 train_and_organize.py --type detection --latest

# Validate model
python3 validate_models.py --detector path/to/model.pt --test-dir model/test/images --output-dir validation_results/

# Test on image
python3 image_inference.py --detector path/to/model.pt --image test.jpg --output result.jpg
```

This workflow ensures every training run is organized, reproducible, and properly versioned!