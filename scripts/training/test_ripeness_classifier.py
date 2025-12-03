#!/usr/bin/env python3
"""
Test the ripeness classifier on sample images
Shows predictions with confidence scores and visualizations
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

class LightweightRipenessClassifier(nn.Module):
    """Lightweight CNN for ripeness classification"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    """Load the trained model"""
    model = LightweightRipenessClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device):
    """Predict ripeness for a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ["unripe", "ripe"]
    return class_names[predicted.item()], confidence.item(), image

def test_on_samples(data_dir, model_path, num_samples=20):
    """Test the classifier on random samples from test set"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get test images
    test_dir = Path(data_dir) / "test"
    all_images = []
    
    for class_name in ["unripe", "ripe"]:
        class_dir = test_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))
            all_images.extend([(img, class_name) for img in images])
    
    if not all_images:
        print("❌ No test images found!")
        return
    
    # Randomly sample images
    if len(all_images) > num_samples:
        samples = random.sample(all_images, num_samples)
    else:
        samples = all_images
    
    print(f"\nTesting on {len(samples)} random samples...")
    
    # Create results directory
    results_dir = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results for metrics
    results = []
    
    # Create figure for visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (img_path, true_class) in enumerate(tqdm(samples, desc="Testing")):
        if idx >= 20:  # Limit to 20 images for visualization
            break
        
        # Make prediction
        pred_class, confidence, image = predict_image(model, img_path, transform, device)
        
        # Store result
        correct = pred_class == true_class
        results.append({
            "image": img_path.name,
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": confidence,
            "correct": correct
        })
        
        # Display result
        if idx < 20:
            ax = axes[idx]
            
            # Convert PIL image to numpy for display
            img_display = np.array(image)
            
            ax.imshow(img_display)
            ax.axis('off')
            
            # Set title with color coding
            title_color = 'green' if correct else 'red'
            ax.set_title(f"True: {true_class}\nPred: {pred_class} ({confidence:.2f})", 
                        color=title_color, fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(samples), 20):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / "test_predictions.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Prediction visualization saved to: {results_dir / 'test_predictions.png'}")
    
    # Calculate metrics
    correct_predictions = sum(1 for r in results if r["correct"])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions * 100
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total samples tested: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    print(f"\nPer-class performance:")
    for class_name in ["unripe", "ripe"]:
        class_results = [r for r in results if r["true_class"] == class_name]
        if class_results:
            class_correct = sum(1 for r in class_results if r["correct"])
            class_acc = class_correct / len(class_results) * 100
            print(f"  {class_name}: {class_correct}/{len(class_results)} ({class_acc:.2f}%)")
    
    # Confidence analysis
    confidences = [r["confidence"] for r in results]
    avg_confidence = np.mean(confidences)
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Show some example predictions
    print(f"\n{'='*60}")
    print("EXAMPLE PREDICTIONS")
    print(f"{'='*60}")
    
    # Show 5 correct and 5 incorrect predictions
    correct_samples = [r for r in results if r["correct"]][:5]
    incorrect_samples = [r for r in results if not r["correct"]][:5]
    
    if correct_samples:
        print("\n✅ Correct predictions:")
        for sample in correct_samples:
            print(f"  {sample['image']}: {sample['pred_class']} ({sample['confidence']:.3f})")
    
    if incorrect_samples:
        print("\n❌ Incorrect predictions:")
        for sample in incorrect_samples:
            print(f"  {sample['image']}: True={sample['true_class']}, Pred={sample['pred_class']} ({sample['confidence']:.3f})")
    
    # Save detailed results
    results_path = results_dir / "detailed_test_results.txt"
    with open(results_path, 'w') as f:
        f.write("RIPENESS CLASSIFIER TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Samples: {total_predictions}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n\n")
        
        f.write("PER-CLASS RESULTS:\n")
        for class_name in ["unripe", "ripe"]:
            class_results = [r for r in results if r["true_class"] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r["correct"])
                class_acc = class_correct / len(class_results) * 100
                f.write(f"  {class_name}: {class_correct}/{len(class_results)} ({class_acc:.2f}%)\n")
        
        f.write(f"\nAVERAGE CONFIDENCE: {avg_confidence:.3f}\n\n")
        
        f.write("ALL PREDICTIONS:\n")
        for result in results:
            status = "✓" if result["correct"] else "✗"
            f.write(f"{status} {result['image']}: True={result['true_class']}, "
                   f"Pred={result['pred_class']}, Conf={result['confidence']:.3f}\n")
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    return results, accuracy

def main():
    """Main testing function"""
    data_dir = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/datasets/ripeness_classification_converted"
    model_path = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/results/ripeness_classification/best_ripeness_classifier.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("python3 training/train_ripeness_classifier.py")
        return
    
    print("🍓 Testing Ripeness Classifier")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test data: {data_dir}")
    
    # Test on samples
    results, accuracy = test_on_samples(data_dir, model_path, num_samples=50)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE!")
    print(f"{'='*60}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 90:
        print("🎉 Excellent! Model is ready for deployment.")
    elif accuracy >= 85:
        print("✅ Good! Model should work well.")
    else:
        print("⚠️  Model may need more training or data.")

if __name__ == "__main__":
    main()