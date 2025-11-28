#!/usr/bin/env python3
"""
Train a lightweight ripeness classifier
Detects: unripe, ripe, overripe
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from collections import Counter

class RipenessDataset(Dataset):
    """Dataset for ripeness classification"""
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ['unripe', 'ripe', 'overripe']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images for {split}")
        print(f"Class distribution: {self.get_class_distribution()}")
    
    def get_class_distribution(self):
        """Get class distribution"""
        if not self.samples:
            return {}
        labels = [sample[1] for sample in self.samples]
        counter = Counter(labels)
        return {self.classes[idx]: count for idx, count in counter.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class RipenessClassifier(nn.Module):
    """Lightweight ripeness classifier"""
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Use MobileNetV3 for efficiency
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
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
        
        pbar.set_postfix({
            'Loss': f'{running_loss/total:.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(dataloader), 100.*correct/total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * 3
    class_total = [0] * 3
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            
            pbar.set_postfix({
                'Loss': f'{running_loss/total:.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    # Per-class accuracy
    class_acc = [100.*c/t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    return running_loss/len(dataloader), 100.*correct/total, class_acc

def main():
    # Configuration
    DATA_DIR = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/ripeness_manual_dataset"
    MODEL_SAVE_PATH = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/weights/ripeness_classifier.pt"
    RESULTS_DIR = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification"
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 128
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if dataset exists
    if not Path(DATA_DIR).exists():
        print(f"Error: Dataset not found at {DATA_DIR}")
        print("Please run: python3 create_manual_ripeness_dataset.py")
        print("Then manually label the crops before training.")
        return
    
    # Check if labeled data exists
    train_dir = Path(DATA_DIR) / 'train'
    if not train_dir.exists():
        print(f"Error: No training data found at {train_dir}")
        print("Please manually label the crops first!")
        print("See the LABELING_GUIDE.md in the dataset directory.")
        return
    
    # Check class distribution
    classes = ['unripe', 'ripe', 'overripe']
    print("\nChecking dataset structure...")
    for split in ['train', 'valid', 'test']:
        split_dir = Path(DATA_DIR) / split
        if split_dir.exists():
            print(f"\n{split}:")
            for cls in classes:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    count = len(list(cls_dir.glob("*.jpg")))
                    print(f"  {cls}: {count} images")
                else:
                    print(f"  {cls}: 0 images (directory missing)")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RipenessDataset(DATA_DIR, transform=train_transform, split='train')
    val_dataset = RipenessDataset(DATA_DIR, transform=val_transform, split='valid')
    test_dataset = RipenessDataset(DATA_DIR, transform=val_transform, split='test')
    
    if len(train_dataset) == 0:
        print("Error: No training images found!")
        print("Please manually label the crops first.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print("\nCreating model...")
    model = RipenessClassifier(num_classes=3, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    best_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_class_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Per-class Val Acc: {val_class_acc}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': classes
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    # Test on test set
    print("\n" + "="*60)
    print("TESTING ON TEST SET")
    print("="*60)
    test_loss, test_acc, test_class_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Per-class Test Acc: {test_class_acc}")
    
    # Save results
    results = {
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'test_class_acc': test_class_acc,
        'class_names': classes,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'image_size': IMAGE_SIZE
        }
    }
    
    with open(f"{RESULTS_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()