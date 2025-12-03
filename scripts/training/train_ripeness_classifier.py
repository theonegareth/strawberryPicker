#!/usr/bin/env python3
"""
Train Ripeness Classifier using the converted dataset
Lightweight CNN for strawberry ripeness classification
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
import matplotlib.pyplot as plt
import numpy as np

class StrawberryRipenessDataset(Dataset):
    """Dataset for strawberry ripeness classification"""
    def __init__(self, data_dir, transform=None, split="train"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ["unripe", "ripe"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LightweightRipenessClassifier(nn.Module):
    """Lightweight CNN for ripeness classification"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use MobileNetV2 as backbone (lightweight and fast)
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Modify for 2 classes (unripe, ripe)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def train_model(data_dir, epochs=30, batch_size=32, learning_rate=0.001):
    """Train the ripeness classifier"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = StrawberryRipenessDataset(data_dir, train_transform, "train")
    val_dataset = StrawberryRipenessDataset(data_dir, val_transform, "valid")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    
    # Model
    model = LightweightRipenessClassifier(num_classes=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   patience=3, factor=0.5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    
    # Results directory
    results_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
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
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{val_loss/len(pbar):.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), results_dir / "best_ripeness_classifier.pth")
            print(f"  ✓ Best model saved! ({val_acc:.2f}% val acc)")
        
        # Update scheduler
        scheduler.step(val_loss)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model and history
    torch.save(model.state_dict(), results_dir / "final_ripeness_classifier.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy')
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    print(f"\n📊 Training curves saved to: {results_dir / 'training_curves.png'}")
    
    return model, history, best_val_acc

def main():
    """Main training function"""
    data_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_classification_converted"
    
    if not Path(data_dir).exists():
        print(f"❌ Dataset not found: {data_dir}")
        print("Please run the conversion script first:")
        print("python3 training/convert_ripeness_detection_to_classification.py")
        return
    
    print("🍓 Training Strawberry Ripeness Classifier")
    print("="*60)
    print(f"Dataset: {data_dir}")
    
    # Train the model
    model, history, best_acc = train_model(data_dir, epochs=30, batch_size=32)
    
    print("\n✅ Training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Save training summary
    summary_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification/training_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("# Ripeness Classifier Training Summary\n\n")
        f.write(f"- **Best Validation Accuracy**: {best_acc:.2f}%\n")
        f.write(f"- **Final Training Accuracy**: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"- **Final Validation Accuracy**: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"- **Training Images**: 1,436 (564 unripe + 872 ripe)\n")
        f.write(f"- **Model**: MobileNetV2 (lightweight, fast)\n")
        f.write(f"- **Training Time**: ~10-15 minutes on GPU\n\n")
        f.write("## Class Distribution\n\n")
        f.write("- **Unripe**: 564 training, 163 validation, 79 test\n")
        f.write("- **Ripe**: 872 training, 259 validation, 148 test\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Test the classifier on sample images\n")
        f.write("2. Integrate with the strawberry detector\n")
        f.write("3. Test the two-stage pipeline\n")
        f.write("4. Export to TFLite for Raspberry Pi\n")
    
    print(f"\n📄 Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()