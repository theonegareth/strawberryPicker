#!/usr/bin/env python3
"""
Debug script to diagnose the training callback issue
"""

import torch
from ultralytics import YOLO

def debug_callback(trainer):
    """Debug version of the callback to see what's available"""
    print(f"\n=== DEBUG: Epoch {trainer.epoch} ===")
    
    # Check what attributes trainer has
    print(f"Trainer attributes: {[attr for attr in dir(trainer) if not attr.startswith('_')][:20]}...")
    
    # Check lr
    if hasattr(trainer, 'lr'):
        print(f"trainer.lr = {trainer.lr}")
        print(f"type(trainer.lr) = {type(trainer.lr)}")
        if isinstance(trainer.lr, (list, tuple)) and len(trainer.lr) > 0:
            print(f"trainer.lr[0] = {trainer.lr[0]}")
        elif isinstance(trainer.lr, (float, int)):
            print(f"trainer.lr as scalar = {trainer.lr}")
        elif trainer.lr is None:
            print("trainer.lr is None")
        else:
            print(f"trainer.lr is something else: {type(trainer.lr)}")
    else:
        print("trainer.lr does not exist")
    
    # Check metrics
    print(f"\ntrainer.metrics = {trainer.metrics}")
    print(f"type(trainer.metrics) = {type(trainer.metrics)}")
    
    # Check specific metrics
    for key in ['train/box_loss', 'train/cls_loss', 'metrics/precision', 'metrics/mAP50']:
        value = trainer.metrics.get(key, 'NOT_FOUND')
        print(f"  {key}: {value}")

def test_training():
    """Test training with debug callback"""
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Add debug callback
    def debug_on_train_epoch_end(trainer):
        debug_callback(trainer)
    
    model.add_callback('on_train_epoch_end', debug_on_train_epoch_end)
    
    # Try a very short training to see what happens
    try:
        results = model.train(
            data='model/dataset/data.yaml',
            epochs=1,  # Just 1 epoch for debugging
            imgsz=416,
            batch=8,
            device='0',
            project='model/results',
            name='debug_run',
            exist_ok=True,
            verbose=True
        )
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_training()