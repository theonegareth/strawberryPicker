#!/usr/bin/env python3
"""
Train lightweight CNN for strawberry ripeness classification
Optimized for Raspberry Pi 4B deployment
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

class StrawberryRipenessDataset(Dataset):
    """Dataset for strawberry ripeness classification"""
    
    def __init__(self, data_dir, transform=None, split="train"):
        """
        Args:
            data_dir: Root directory of the dataset
            transform: Transformations to apply
            split: 'train', 'valid', or 'test'
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ['unripe', 'ripe', 'overripe']
        
        self.images = []
        self.labels = []
        
        # Load all images
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"📂 Loaded {len(self.images)} images for {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes=3, model_type='mobilenet_v2'):
    """Create lightweight model for Raspberry Pi"""
    
    if model_type == 'mobilenet_v2':
        # Use MobileNetV2 - excellent for edge devices
        model = models.mobilenet_v2(pretrained=True)
        # Modify classifier for our 3 classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'mobilenet_v3':
        # MobileNetV3 - even more efficient
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        # EfficientNet-B0 - good balance of accuracy/speed
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

def train_model(
    data_dir: str,
    model_type: str = 'mobilenet_v2',
    batch_size: int = 32,
    epochs: int = 30,
    learning_rate: float = 0.001,
    device: str = 'auto'
):
    """
    Train ripeness classification model
    
    Args:
        data_dir: Path to dataset directory
        model_type: 'mobilenet_v2', 'mobilenet_v3', or 'efficientnet_b0'
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: 'auto', 'cpu', or 'cuda'
    """
    
    print(f"🍓 Training Strawberry Ripeness Classifier")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}")
    
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = StrawberryRipenessDataset(data_dir, train_transform, "train")
    val_dataset = StrawberryRipenessDataset(data_dir, val_transform, "valid")
    test_dataset = StrawberryRipenessDataset(data_dir, val_transform, "test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model(num_classes=3, model_type=model_type)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    training_history = []
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/len(pbar):.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = Path(data_dir).parent / "best_ripeness_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': model_type
            }, best_model_path)
        
        # Log history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%")
    
    # Test evaluation
    print(f"\n🧪 Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * 3
    class_total = [0] * 3
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\n📊 Per-class accuracy:")
    classes = ['unripe', 'ripe', 'overripe']
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"   {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Save training history
    history_file = Path(data_dir).parent / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'per_class_accuracy': {
                classes[i]: 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                for i in range(len(classes))
            }
        }, f, indent=2)
    
    print(f"\n💾 Model saved to: {best_model_path}")
    print(f"💾 Training history saved to: {history_file}")
    
    return {
        'model_path': str(best_model_path),
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'per_class_accuracy': {
            classes[i]: 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(len(classes))
        }
    }

def main():
    """Main training function"""
    
    data_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/strawberry_ripeness_classification"
    
    # Train the model
    results = train_model(
        data_dir=data_dir,
        model_type='mobilenet_v2',  # Best for Raspberry Pi
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    )
    
    print(f"\n✅ Training complete! Model ready for deployment.")

if __name__ == "__main__":
    main()