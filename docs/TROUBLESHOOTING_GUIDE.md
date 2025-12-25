# Enhanced Strawberry Locator Troubleshooting Guide

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

1. **✅ Cameras working?** Test with basic OpenCV capture
2. **✅ Model loading?** Verify YOLO model path and format
3. **✅ Calibration correct?** Check camera calibration parameters
4. **✅ Configuration valid?** Validate YAML configuration
5. **✅ Dependencies installed?** Verify all required packages

## Common Issues and Solutions

### 🔴 Camera Issues

#### Problem: "Camera capture failed" or returning None
**Symptoms:**
- Error message: "Camera capture failed"
- `capture_frames()` returns (None, None)
- No images displayed in visualization

**Solutions:**
```python
# Test cameras independently
import cv2

# Test left camera
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret, frame = cap_left.read()
print(f"Left camera: {ret}, Frame shape: {frame.shape if ret else 'None'}")

# Test right camera  
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
ret, frame = cap_right.read()
print(f"Right camera: {ret}, Frame shape: {frame.shape if ret else 'None'}")

cap_left.release()
cap_right.release()
```

**Configuration Fix:**
```yaml
cameras:
  left_camera_id: 1          # Try different IDs (0, 1, 2, 3)
  right_camera_id: 2
  capture_backend: "CAP_DSHOW"  # Try different backends
```

#### Problem: Frame synchronization issues
**Symptoms:**
- Different timestamps on left/right frames
- Misaligned detections between cameras
- Poor depth accuracy

**Solutions:**
```yaml
cameras:
  warmup_frames: 5           # Increase warmup frames
  sync_tolerance_ms: 50      # Tighter synchronization
```

---

### 🔴 Detection Issues

#### Problem: "No strawberries detected"
**Symptoms:**
- Warning: "No strawberries detected"
- Empty results list
- Visualization shows no bounding boxes

**Diagnostic Steps:**
```python
# Test model independently
from ultralytics import YOLO
import cv2

model = YOLO("path/to/your/model.pt")
frame = cv2.imread("test_image.jpg")
results = model(frame)

print(f"Detections: {len(results[0].boxes)}")
for box in results[0].boxes:
    print(f"Confidence: {box.conf[0]:.3f}")
```

**Solutions:**
1. **Lower confidence threshold:**
```yaml
depth_detection:
  min_confidence_threshold: 0.4    # Lower from 0.6
```

2. **Check model path:**
```python
# Verify model exists and loads
import os
print(f"Model exists: {os.path.exists('path/to/model.pt')}")
model = YOLO('path/to/model.pt')
print(f"Model classes: {model.names}")
```

3. **Test with different images:**
```python
# Use test_enhanced_locator.py for synthetic testing
python test_enhanced_locator.py
```

#### Problem: Poor detection quality
**Symptoms:**
- Low confidence scores (< 0.5)
- Inconsistent detections between cameras
- High false positive rate

**Solutions:**
```yaml
# Improve quality filtering
robustness:
  enable_quality_filtering: true
  enable_calibration_monitoring: true
  
depth_detection:
  min_confidence_threshold: 0.7    # Increase threshold
  method_thresholds:
    bbox_corners: 0.8              # Higher corner method threshold
    bbox_perimeter: 0.7            # Higher perimeter method threshold
```

---

### 🔴 Depth Calculation Issues

#### Problem: "Depth calculation failed"
**Symptoms:**
- Error: "Depth calculation failed"
- depth_cm: None in results
- confidence: 0.0

**Diagnostic Steps:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with simple center method only
locator.config['depth_detection']['enable_bbox_corners'] = False
locator.config['depth_detection']['enable_bbox_perimeter'] = False
```

**Solutions:**

1. **Check camera calibration:**
```python
# Verify calibration matrices
print(f"Left K matrix: {locator.K_A}")
print(f"Right K matrix: {locator.K_B}")
print(f"Baseline: {locator.baseline_cm}cm")
```

2. **Test fallback methods:**
```yaml
depth_detection:
  enable_bbox_corners: true        # Enable all methods
  enable_bbox_perimeter: true
  enable_fallback_strategies: true  # Ensure fallbacks enabled
```

3. **Check stereo alignment:**
```python
# Test with known good calibration
locator.config['cameras']['baseline_cm'] = 23.0  # Your known baseline
```

#### Problem: Inaccurate depth values
**Symptoms:**
- Depth values clearly wrong (negative, extremely large)
- Inconsistent depth between similar strawberries
- Poor correlation with actual distance

**Solutions:**

1. **Improve outlier removal:**
```yaml
robustness:
  enable_outlier_removal: true
  outlier_threshold: 1.5           # Stricter outlier removal (was 2.0)
```

2. **Increase sampling points:**
```yaml
depth_detection:
  num_perimeter_points: 12         # More points for better statistics
```

3. **Check for calibration drift:**
```python
# Monitor calibration quality
if locator.config['robustness']['enable_calibration_monitoring']:
    print("Calibration monitoring enabled")
```

---

### 🔴 Performance Issues

#### Problem: Slow processing speed
**Symptoms:**
- Low FPS (< 10)
- High CPU usage
- Frame dropping

**Solutions:**

1. **Reduce processing load:**
```yaml
# Speed optimization
depth_detection:
  num_perimeter_points: 4          # Reduce from 8 to 4
  enable_bbox_perimeter: false     # Disable perimeter sampling
  
performance:
  enable_caching: true             # Enable result caching
  processing_fps: 15               # Lower target FPS
  
visualization:
  enable_3d_visualization: false   # Disable 3D visualization
  show_perimeter_points: false     # Don't show processing points
```

2. **Enable threading (experimental):**
```yaml
performance:
  enable_threading: true           # Enable multi-threading
  thread_pool_size: 4              # Adjust based on CPU cores
```

#### Problem: High memory usage
**Symptoms:**
- System runs out of memory
- Crashes during long operation
- Memory leaks detected

**Solutions:**
```yaml
performance:
  max_memory_usage_mb: 256         # Limit memory usage
  enable_garbage_collection: true  # Aggressive garbage collection
  cache_size: 50                   # Reduce cache size
  
visualization:
  enable_3d_visualization: false   # Memory-intensive feature
```

---

### 🔴 Configuration Issues

#### Problem: Configuration file not found
**Symptoms:**
- Warning: "Config file not found, using defaults"
- Settings not being applied

**Solutions:**
```python
# Specify full path to config file
locator = StrawberryLocator("/full/path/to/locator_config.yaml")

# Or place config in working directory
import os
print(f"Current directory: {os.getcwd()}")
print(f"Config exists: {os.path.exists('locator_config.yaml')}")
```

#### Problem: Invalid configuration values
**Symptoms:**
- Error messages about invalid values
- System falls back to defaults
- Unexpected behavior

**Validation Script:**
```python
# Validate configuration
def validate_config():
    config_ranges = {
        'min_confidence_threshold': (0.0, 1.0),
        'num_perimeter_points': (4, 16),
        'baseline_cm': (10.0, 50.0),
        'outlier_threshold': (1.0, 3.0)
    }
    
    for param, (min_val, max_val) in config_ranges.items():
        value = locator.config['depth_detection'].get(param, min_val)
        if not (min_val <= value <= max_val):
            print(f"Invalid {param}: {value} (should be {min_val}-{max_val})")

validate_config()
```

---

### 🔴 Arduino Integration Issues

#### Problem: Arduino not responding to enhanced coordinates
**Symptoms:**
- Arduino receives coordinates but doesn't move
- Movement commands ignored
- Serial communication errors

**Solutions:**

1. **Check coordinate format:**
```python
# Verify coordinate format matches Arduino expectations
for result in results:
    x, y, z = result['position_3d']
    print(f"Sending: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
    # Should match Arduino input format: "x,y,z"
```

2. **Test Arduino communication:**
```python
# Test basic Arduino communication
import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)
ser.write(b"10.0,5.0,15.0\n")  # Test coordinates
response = ser.readline()
print(f"Arduino response: {response}")
```

3. **Check confidence filtering:**
```python
# Ensure you're sending high-confidence results
for result in results:
    if result['confidence'] > 0.7:  # Adjust threshold as needed
        x, y, z = result['position_3d']
        send_to_arduino(x, y, z)
    else:
        print(f"Skipping low confidence: {result['confidence']}")
```

---

### 🔴 Visualization Issues

#### Problem: Visualization not displaying
**Symptoms:**
- No visualization window
- Empty visualization image
- Crashes during visualization

**Solutions:**

1. **Check OpenCV installation:**
```python
import cv2
print(f"OpenCV version: {cv2.__version__}")

# Test basic display
test_img = cv2.imread("test.jpg")
cv2.imshow("Test", test_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

2. **Disable problematic features:**
```yaml
visualization:
  enable_3d_visualization: false   # Disable if causing issues
  show_perimeter_points: false     # Simplify visualization
  opacity: 1.0                     # Full opacity
```

---

## Advanced Debugging

### Enable Verbose Logging
```python
# Maximum debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Set all loggers to DEBUG
for logger_name in ['StrawberryLocator', 'ultralytics']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

# Run with detailed logging
locator = StrawberryLocator()
locator.demo_immediate_enhancements()
```

### Performance Profiling
```python
# Profile processing time
import time

start_time = time.time()
results = locator.process_frame_pair(left_frame, right_frame, model)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.3f}s")
print(f"Results: {len(results)}")
```

### Memory Usage Monitoring
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Before and after processing
before_memory = process.memory_info().rss
results = locator.process_frame_pair(left_frame, right_frame, model)
after_memory = process.memory_info().rss
print(f"Memory delta: {(after_memory - before_memory) / 1024 / 1024:.1f} MB")
```

## Recovery Procedures

### System Reset Procedure
```python
def reset_system():
    """Complete system reset"""
    # 1. Release camera resources
    if locator.cap_left:
        locator.cap_left.release()
    if locator.cap_right:
        locator.cap_right.release()
    
    # 2. Clear caches
    if hasattr(locator, 'cache'):
        locator.cache.clear()
    
    # 3. Reset configuration
    locator = StrawberryLocator()  # Fresh instance
    
    # 4. Test basic functionality
    test_results = locator.test_enhanced_vs_original()
    return test_results

# Execute reset
reset_system()
```

### Emergency Fallback
```python
def emergency_fallback():
    """Fallback to basic center-point detection"""
    # Disable all enhanced features
    locator.config['depth_detection']['enable_bbox_corners'] = False
    locator.config['depth_detection']['enable_bbox_perimeter'] = False
    locator.config['depth_detection']['enable_fallback_strategies'] = False
    
    # Use only center point method
    print("Emergency fallback: Using center point only")
    return locator
```

## Getting Help

### Information to Include in Bug Reports
When reporting issues, include:

1. **System Information:**
   - Operating system and version
   - Python version
   - OpenCV version
   - Camera models and IDs

2. **Configuration:**
   - Full configuration file (or relevant sections)
   - Any custom modifications

3. **Error Details:**
   - Complete error messages
   - Stack traces if available
   - Log files (strawberry_locator.log)

4. **Test Results:**
   - Output from `test_enhanced_locator.py`
   - Results from diagnostic scripts above

### Community Resources
- **GitHub Issues:** Report bugs and request features
- **Documentation:** Check updated guides and examples
- **Test Scripts:** Use built-in testing tools

## Quick Reference

### Most Common Fixes
1. **Camera not found:** Try different camera IDs (0, 1, 2, 3)
2. **No detections:** Lower confidence threshold or check model path
3. **Depth failures:** Enable fallback strategies and check calibration
4. **Slow performance:** Reduce perimeter points or disable threading
5. **Poor accuracy:** Enable outlier removal and increase sampling

### Emergency Commands
```bash
# Test basic functionality
python test_enhanced_locator.py

# Run with debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from strawberrylocator import StrawberryLocator
locator = StrawberryLocator()
locator.demo_immediate_enhancements()
"
```

Remember: **Start simple, then add complexity.** Begin with default settings and basic functionality, then gradually enable advanced features as needed.