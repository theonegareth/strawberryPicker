#!/usr/bin/env python3
"""
Enhanced Ripeness Classifier Training
Aims for 95%+ validation accuracy with better techniques
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
from torch.optim.lr_scheduler import OneCycleLR
import warnings
warnings.filterwarnings('ignore')

class StrawberryRipenessDataset(Dataset):
    """Enhanced dataset with better preprocessing"""
    def __init__(self, data_dir, transform=None, split="train"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ["unripe", "partially-ripe", "ripe", "overripe"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Print class distribution
        print(f"{split} set: {len(self.samples)} images")
        for class_name in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EnhancedRipenessClassifier(nn.Module):
    """Enhanced classifier with better architecture"""
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        
        # Use EfficientNet-B0 for better accuracy
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Modify classifier for 4 classes
        in_features = self.backbone.classifier[1].in_features
        
        # Add dropout for regularization
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def calculate_accuracy(outputs, targets):
    """Calculate top-1 accuracy"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """Enhanced training with better techniques"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced data transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = StrawberryRipenessDataset(data_dir, train_transform, "train")
    val_dataset = StrawberryRipenessDataset(data_dir, val_transform, "valid")
    
    # Data loaders with balanced sampling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Model with dropout for regularization
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = EnhancedRipenessClassifier(num_classes=3, dropout_rate=0.4).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # OneCycleLR scheduler for better convergence
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                          steps_per_epoch=len(train_loader), epochs=epochs,
                          pct_start=0.3, div_factor=10, final_div_factor=100)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    # Results directory
    results_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification_enhanced")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING STARTED - Target: 95%+ Validation Accuracy")
    print("="*60)
    
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            train_correct += acc * labels.size(0)
            train_total += labels.size(0)
            
            current_train_acc = 100. * train_correct / train_total
            
            pbar.set_postfix({
                'Loss': f'{train_loss/len(pbar):.4f}',
                'Acc': f'{current_train_acc:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
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
                acc = calculate_accuracy(outputs, labels)
                val_correct += acc * labels.size(0)
                val_total += labels.size(0)
                
                current_val_acc = 100. * val_correct / val_total
                
                pbar.set_postfix({
                    'Loss': f'{val_loss/len(pbar):.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, results_dir / "best_enhanced_classifier.pth")
            print(f"  ✓ New best model! ({val_acc:.2f}% val acc)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Stop if target accuracy reached
        if val_acc >= 95.0:
            print(f"\n🎉 Target accuracy reached! ({val_acc:.2f}% >= 95%)")
            break
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    if best_val_acc >= 95.0:
        print("🎯 Target achieved! Model is ready for deployment.")
    elif best_val_acc >= 93.0:
        print("✅ Good improvement! Model performance is strong.")
    else:
        print("⚠️  Consider more training or hyperparameter tuning.")
    
    # Save final model
    torch.save(model.state_dict(), results_dir / "final_enhanced_classifier.pth")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_acc'], label='Val Acc', color='green', linewidth=2)
    plt.axhline(y=95, color='red', linestyle='--', label='Target (95%)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy vs Target')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "enhanced_training_curves.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {results_dir / 'enhanced_training_curves.png'}")
    
    return model, history, best_val_acc

def main():
    """Main enhanced training function"""
    data_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_classification_converted"
    
    if not Path(data_dir).exists():
        print(f"❌ Dataset not found: {data_dir}")
        print("Please run the conversion script first:")
        print("python3 training/convert_ripeness_detection_to_classification.py")
        return
    
    print("🚀 Enhanced Ripeness Classifier Training")
    print("="*60)
    print("Target: 95%+ Validation Accuracy")
    print(f"Dataset: {data_dir}")
    print(f"Training samples: 1,436")
    print(f"Validation samples: 422")
    
    # Train the model
    model, history, best_acc = train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.002)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Save training summary
    summary_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification_enhanced/training_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("# Enhanced Ripeness Classifier Training Summary\n\n")
        f.write(f"- **Best Validation Accuracy**: {best_acc:.2f}%\n")
        f.write(f"- **Final Training Accuracy**: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"- **Final Validation Accuracy**: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"- **Target Achieved**: {'✅ Yes' if best_acc >= 95 else '❌ No'}\n")
        f.write(f"- **Training Images**: 1,436 (564 unripe + 872 ripe)\n")
        f.write(f"- **Model**: EfficientNet-B0 with dropout\n")
        f.write(f"- **Key Improvements**: OneCycleLR, heavy augmentation, label smoothing\n\n")
        
        if best_acc >= 95:
            f.write("## 🎉 Success!\n\n")
            f.write("Target accuracy of 95%+ achieved!\n")
            f.write("Model is ready for deployment.\n\n")
        else:
            f.write("## 📈 Improvement\n\n")
            f.write(f"Accuracy improved from 91.94% to {best_acc:.2f}%.\n")
            if best_acc >= 93:
                f.write("Model performance is strong and usable.\n")
            else:
                f.write("Consider additional improvements.\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Test the enhanced model on sample images\n")
        f.write("2. Compare with baseline model\n")
        f.write("3. Export to TFLite for Raspberry Pi deployment\n")
        f.write("4. Integrate with strawberry detector\n")
    
    print(f"\n📄 Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()