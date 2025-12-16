#!/usr/bin/env python3
"""
Model Validation Script for Strawberry Ripeness Classification
Tests the trained model on sample images to verify functionality
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append('.')

from train_ripeness_classifier import create_model, get_transforms

def load_model(model_path):
    """Load the trained classification model"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model architecture
    model = create_model(num_classes=3, backbone='resnet18', pretrained=False)
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, device

def get_test_images():
    """Get sample test images from the dataset"""
    test_dirs = [
        'model/ripeness_manual_dataset/unripe',
        'model/ripeness_manual_dataset/ripe', 
        'model/ripeness_manual_dataset/overripe'
    ]
    
    test_images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = list(Path(test_dir).glob('*.jpg'))[:3]  # Get first 3 images from each class
            for img_path in images:
                test_images.append({
                    'path': str(img_path),
                    'true_label': os.path.basename(test_dir),
                    'class_name': os.path.basename(test_dir)
                })
    
    return test_images

def predict_image(model, device, image_path, transform):
    """Predict ripeness for a single image"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image_pil = Image.fromarray(image)
        
        # Apply transforms
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Get class names
        class_names = ['overripe', 'ripe', 'unripe']
        predicted_class = class_names[predicted_class_idx]
        
        # Get all probabilities
        probs_dict = {
            class_names[i]: float(probabilities[0][i].item()) 
            for i in range(len(class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs_dict
        }, None
        
    except Exception as e:
        return None, str(e)

def validate_model():
    """Main validation function"""
    print("=== Strawberry Ripeness Classification Model Validation ===")
    print(f"Validation time: {datetime.now().isoformat()}")
    print()
    
    # Load model
    model_path = 'model/ripeness_classifier_best.pth'
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get transforms
    _, transform = get_transforms(img_size=224)
    
    # Get test images
    test_images = get_test_images()
    if not test_images:
        print("❌ No test images found")
        return False
    
    print(f"Found {len(test_images)} test images")
    print()
    
    # Test predictions
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print("Testing predictions...")
    print("-" * 80)
    
    for i, test_img in enumerate(test_images):
        image_path = test_img['path']
        true_label = test_img['true_label']
        
        # Make prediction
        prediction, error = predict_image(model, device, image_path, transform)
        
        if error:
            print(f"❌ Image {i+1}: Error - {error}")
            continue
        
        predicted_class = prediction['predicted_class']
        confidence = prediction['confidence']
        
        # Check if prediction is correct
        is_correct = predicted_class == true_label
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"{status} Image {i+1}: {os.path.basename(image_path)}")
        print(f"   True: {true_label} | Predicted: {predicted_class} ({confidence:.3f})")
        print(f"   Probabilities: overripe={prediction['probabilities']['overripe']:.3f}, "
              f"ripe={prediction['probabilities']['ripe']:.3f}, "
              f"unripe={prediction['probabilities']['unripe']:.3f}")
        print()
        
        # Store result
        results.append({
            'image_path': image_path,
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prediction['probabilities'],
            'correct': is_correct
        })
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total images tested: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    
    # Class-wise analysis
    class_stats = {}
    for result in results:
        true_class = result['true_label']
        if true_class not in class_stats:
            class_stats[true_class] = {'correct': 0, 'total': 0}
        class_stats[true_class]['total'] += 1
        if result['correct']:
            class_stats[true_class]['correct'] += 1
    
    print("Class-wise Performance:")
    for class_name, stats in class_stats.items():
        class_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {class_name}: {stats['correct']}/{stats['total']} ({class_accuracy:.1f}%)")
    print()
    
    # Save detailed results
    validation_results = {
        'validation_time': datetime.now().isoformat(),
        'model_path': model_path,
        'device': str(device),
        'total_images': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy_percent': accuracy,
        'class_stats': class_stats,
        'detailed_results': results
    }
    
    results_path = 'model_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_path}")
    
    # Validation verdict
    if accuracy >= 90:
        print("🎉 VALIDATION PASSED: Model performs excellently!")
        return True
    elif accuracy >= 80:
        print("⚠️  VALIDATION WARNING: Model performs moderately well")
        return True
    else:
        print("❌ VALIDATION FAILED: Model performance is poor")
        return False

if __name__ == '__main__':
    success = validate_model()
    sys.exit(0 if success else 1)
