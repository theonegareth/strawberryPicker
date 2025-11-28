# Strawberry Ripeness Annotation Guide

## Ripeness Classes

### 1. **Unripe** (Class 0)
- **Color**: Predominantly green, white, or pale pink
- **Texture**: Hard, firm to the touch
- **Size**: Smaller, not fully developed
- **Examples**: 
  - Completely green strawberries
  - White/pale strawberries with no red
  - Very light pink strawberries

### 2. **Ripe** (Class 1)
- **Color**: Bright, uniform red
- **Texture**: Firm but slightly soft
- **Size**: Full size, plump
- **Examples**:
  - Bright red strawberries
  - Uniform color without white/green patches
  - Ready to eat/pick

### 3. **Overripe** (Class 2)
- **Color**: Dark red, burgundy, or with dark spots
- **Texture**: Soft, mushy, or wrinkled
- **Appearance**: May have bruises, mold, or be shriveled
- **Examples**:
  - Very dark red/almost purple
  - Wrinkled surface
  - Visible bruises or soft spots

## Annotation Rules

1. **Primary Criterion**: Color is the main indicator
2. **Borderline Cases**: When in doubt, classify as "ripe"
3. **Partial Visibility**: If you can see >50% of the strawberry, classify based on visible portion
4. **Occlusion**: If heavily occluded (>50% hidden), skip or make best guess
5. **Multiple Strawberries**: Each strawberry gets its own bounding box with ripeness label

## Labeling Format

YOLO format: `class_id center_x center_y width height`

- **Class 0**: Unripe
- **Class 1**: Ripe  
- **Class 2**: Overripe

Example:
```
1 0.45 0.32 0.08 0.12  # Ripe strawberry
0 0.67 0.58 0.06 0.09  # Unripe strawberry
2 0.23 0.71 0.07 0.10  # Overripe strawberry
```

## Quality Check

- Verify all strawberries are labeled
- Check for consistent ripeness classification
- Ensure no strawberries are missed
- Confirm bounding boxes are tight (not too loose)