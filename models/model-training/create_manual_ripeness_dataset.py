#!/usr/bin/env python3
"""
Create manual ripeness dataset using your detector
1. Detect strawberries in images
2. Extract crops
3. Save to folders for manual labeling
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

def create_ripeness_dataset_from_detector(detector_model, images_dir, output_dir, confidence=0.3):
    """
    Use detector to find strawberries, extract crops for manual labeling
    
    Args:
        detector_model: Path to YOLO detection model
        images_dir: Directory with images to process
        output_dir: Output directory for crops
        confidence: Detection confidence threshold
    """
    # Load detector
    print(f"Loading detector: {detector_model}")
    model = YOLO(detector_model)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Create output directories for manual labeling
    for ripeness in ['unripe', 'ripe', 'overripe']:
        (output_dir / ripeness).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing images from: {images_dir}")
    print(f"Output to: {output_dir}")
    print(f"Confidence threshold: {confidence}")
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.rglob(ext))
    
    print(f"Found {len(image_files)} images")
    
    total_crops = 0
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Run detection
            results = model(img, conf=confidence, verbose=False)
            
            # Extract detections
            for idx, result in enumerate(results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box_idx, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)
                    
                    # Extract crop
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Resize to standard size
                    crop_resized = cv2.resize(crop, (128, 128))
                    
                    # Save crop (will be manually sorted later)
                    # Use a temporary folder first
                    temp_dir = output_dir / "to_label"
                    temp_dir.mkdir(exist_ok=True)
                    
                    crop_file = temp_dir / f"{img_file.stem}_box{box_idx}_conf{conf:.2f}.jpg"
                    cv2.imwrite(str(crop_file), crop_resized)
                    
                    total_crops += 1
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"\n✓ Extracted {total_crops} strawberry crops")
    print(f"✓ Saved to: {output_dir}")
    
    # Print instructions
    print("\n" + "="*60)
    print("MANUAL LABELING INSTRUCTIONS")
    print("="*60)
    print(f"1. Go to: {output_dir}")
    print(f"2. Open the 'to_label' folder")
    print(f"3. For each strawberry image, decide if it's:")
    print(f"   - unripe (green, small)")
    print(f"   - ripe (red, full size)")
    print(f"   - overripe (dark red, soft)")
    print(f"4. Move each image to the appropriate folder")
    print(f"5. Try to get at least 50-100 images per class")
    print(f"6. Delete the 'to_label' folder when done")
    print("="*60)
    
    return output_dir

def create_labeling_guide(output_dir):
    """Create a visual guide for labeling"""
    guide_path = output_dir / "LABELING_GUIDE.md"
    
    guide_content = """# Strawberry Ripeness Labeling Guide

## Classes

### 1. Unripe 🟢
- **Color**: Green, white, or pale pink
- **Size**: Small, not fully grown
- **Texture**: Firm, hard
- **Examples**: Young strawberries still developing

### 2. Ripe 🔴
- **Color**: Bright red, uniform color
- **Size**: Full size, plump
- **Texture**: Firm but not hard
- **Examples**: Ready to pick and eat

### 3. Overripe 🟤
- **Color**: Dark red, maroon, or starting to rot
- **Size**: May be soft or shriveled
- **Texture**: Soft, mushy, or wrinkled
- **Examples**: Past prime, too soft

## Labeling Tips

1. **When in doubt**: If you're not sure, it's probably unripe or ripe
2. **Focus on color**: Red = ripe, Green/White = unripe, Dark = overripe
3. **Check texture**: Firm = unripe/ripe, Soft = overripe
4. **Be consistent**: Try to use the same criteria for all images

## After Labeling

Once you've sorted all images into the three folders:
1. Delete the 'to_label' folder
2. Count images in each folder (should be roughly balanced)
3. Run: python3 train_ripeness_classifier.py
"""
    
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print(f"✓ Created labeling guide: {guide_path}")

if __name__ == "__main__":
    # Configuration
    DETECTOR_MODEL = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/weights/strawberry_yolov8s_enhanced.pt"
    
    # Use images from your existing dataset
    IMAGES_DIR = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/dataset/train/images"
    
    # Output directory
    OUTPUT_DIR = "/home/user/machine-learning/GitHubRepos/strawberryPicker/model/ripeness_manual_dataset"
    
    print("Creating Manual Ripeness Dataset")
    print("="*60)
    print(f"Detector: {DETECTOR_MODEL}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)
    
    # Create dataset
    output_path = create_ripeness_dataset_from_detector(
        detector_model=DETECTOR_MODEL,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        confidence=0.3
    )
    
    # Create labeling guide
    create_labeling_guide(output_path)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Manually label the extracted crops")
    print("2. Move images from 'to_label' to appropriate ripeness folders")
    print("3. Aim for 50-100 images per class")
    print("4. Run: python3 train_ripeness_classifier.py")
    print("="*60)