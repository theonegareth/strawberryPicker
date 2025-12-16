#!/usr/bin/env python3
"""
Strawberry Ripeness Classification Training Script
Trains a 3-class classifier (unripe/ripe/overripe) using transfer learning
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RipenessDataset(Dataset):
    """Custom dataset for strawberry ripeness classification"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Get class names and counts (exclude 'to_label' directory)
        self.classes = sorted([d.name for d in self.data_dir.iterdir()
                             if d.is_dir() and d.name != 'to_label'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"{split} dataset: {len(self.samples)} samples")
        print(f"Classes: {self.classes}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size=224):
    """Get data transforms for training and validation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes=3, backbone='resnet18', pretrained=True):
    """Create model with transfer learning"""
    
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """Train the model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
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
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model/ripeness_classifier_best.pth')
            print(f'New best model saved! Val Acc: {best_val_acc:.2f}%')
        
        scheduler.step(val_acc)
    
    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and generate reports"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model/ripeness_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return report, cm

def plot_training_history(train_losses, val_accuracies, save_path):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train strawberry ripeness classifier')
    parser.add_argument('--data-dir', default='model/ripeness_manual_dataset',
                       help='Directory containing labeled images')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--backbone', default='resnet18', 
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output-dir', default='model/ripeness_classifier',
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(args.img_size)
    
    # Create datasets
    train_dataset = RipenessDataset(args.data_dir, transform=train_transform, split='train')
    val_dataset = RipenessDataset(args.data_dir, transform=val_transform, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes=num_classes, backbone=args.backbone, pretrained=True)
    model = model.to(device)
    
    print(f"Model created with {num_classes} classes: {train_dataset.classes}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, device, 
        num_epochs=args.epochs, lr=args.lr
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('model/ripeness_classifier_best.pth'))
    
    # Evaluate model
    print("Evaluating model...")
    report, cm = evaluate_model(model, val_loader, device, train_dataset.classes)
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies, 
                         f'{args.output_dir}/training_history.png')
    
    # Save results
    results = {
        'model_architecture': args.backbone,
        'num_classes': num_classes,
        'class_names': train_dataset.classes,
        'best_val_accuracy': best_val_acc,
        'training_config': {
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'val_split': args.val_split
        },
        'dataset_info': {
            'total_samples': len(train_dataset),
            'class_distribution': {cls: len(list(Path(args.data_dir, cls).glob('*.jpg'))) 
                                 for cls in train_dataset.classes}
        }
    }
    
    with open(f'{args.output_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report
    with open(f'{args.output_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {args.output_dir}")
    print(f"Model saved to: model/ripeness_classifier_best.pth")

if __name__ == '__main__':
    main()