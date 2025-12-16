# Strawberry Ripeness Labeling Guide

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
