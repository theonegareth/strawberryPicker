#!/usr/bin/env python3
"""
Training Progress Monitor
Real-time monitoring of YOLOv8 multi-model training progress
"""

import os
import time
import pandas as pd
from pathlib import Path

def monitor_training():
    """Monitor training progress in real-time"""
    
    # Training directory
    training_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/detection/multi_model_training")
    
    # Find the latest training directory
    training_dirs = [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith('yolov8s_')]
    if not training_dirs:
        print("❌ No training directories found!")
        return
    
    # Get the most recent training directory
    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / "train" / "results.csv"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"🎯 Monitoring training in: {latest_dir.name}")
    print("=" * 80)
    
    try:
        while True:
            # Read the latest results
            if os.path.exists(results_file):
                df = pd.read_csv(results_file)
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    
                    # Clear screen and show progress
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                    print(f"🚀 YOLOv8s Training Progress Monitor")
                    print(f"📁 Training Directory: {latest_dir.name}")
                    print("=" * 80)
                    
                    # Progress bar
                    epoch = int(latest['epoch'])
                    total_epochs = 100
                    progress = (epoch / total_epochs) * 100
                    
                    bar_length = 50
                    filled_length = int(bar_length * progress / 100)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    print(f"📊 Progress: |{bar}| {progress:.1f}% ({epoch}/{total_epochs})")
                    print()
                    
                    # Key metrics
                    print("📈 Key Performance Metrics:")
                    print(f"   🎯 mAP@50:     {latest['metrics/mAP50(B)']:.3f} ({latest['metrics/mAP50(B)']*100:.1f}%)")
                    print(f"   🎯 mAP50-95:   {latest['metrics/mAP50-95(B)']:.3f} ({latest['metrics/mAP50-95(B)']*100:.1f}%)")
                    print(f"   📐 Precision:   {latest['metrics/precision(B)']:.3f}")
                    print(f"   📐 Recall:      {latest['metrics/recall(B)']:.3f}")
                    print()
                    
                    # Training losses
                    print("🔥 Training Losses:")
                    print(f"   📦 Box Loss:    {latest['train/box_loss']:.3f}")
                    print(f"   🏷️  Cls Loss:    {latest['train/cls_loss']:.3f}")
                    print(f"   📊 DFL Loss:    {latest['train/dfl_loss']:.3f}")
                    print()
                    
                    # Validation losses
                    print("✅ Validation Losses:")
                    print(f"   📦 Box Loss:    {latest['val/box_loss']:.3f}")
                    print(f"   🏷️  Cls Loss:    {latest['val/cls_loss']:.3f}")
                    print(f"   📊 DFL Loss:    {latest['val/dfl_loss']:.3f}")
                    print()
                    
                    # Training time
                    total_time = latest['time']
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    
                    print(f"⏱️  Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    
                    # Estimated time remaining
                    if epoch > 0:
                        avg_time_per_epoch = total_time / epoch
                        remaining_epochs = total_epochs - epoch
                        eta_seconds = remaining_epochs * avg_time_per_epoch
                        eta_hours = int(eta_seconds // 3600)
                        eta_minutes = int((eta_seconds % 3600) // 60)
                        
                        print(f"🕐 ETA: {eta_hours:02d}:{eta_minutes:02d}:{int(eta_seconds % 60):02d}")
                    
                    print()
                    print("📍 Log Files Location:")
                    print(f"   📄 CSV Log: {results_file}")
                    print(f"   📊 Training Curves: {latest_dir}/train/results.png")
                    print(f"   ⚙️  Config: {latest_dir}/train/args.yaml")
                    print()
                    print("🔄 Auto-refreshing every 5 seconds... (Ctrl+C to exit)")
                    
                else:
                    print("⏳ Waiting for training data...")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Training monitoring stopped.")
    except Exception as e:
        print(f"❌ Error monitoring training: {e}")

if __name__ == "__main__":
    monitor_training()