# Enhanced Strawberry Locator Configuration Guide

## Overview

The enhanced strawberry locator uses a comprehensive YAML configuration system that allows fine-tuning of all aspects of the depth detection pipeline. This guide covers all configuration options and their effects on system performance.

## Configuration File Structure

The configuration file follows a hierarchical structure with logical groupings:

```yaml
strawberry_locator:
  cameras:          # Camera hardware settings
  depth_detection:  # Depth calculation parameters  
  processing:       # Processing and performance settings
  logging:          # Logging configuration
  performance:      # Performance optimization
  robustness:       # Reliability and error handling
  visualization:    # Display and debugging options
```

## Camera Configuration

### Basic Camera Settings
```yaml
cameras:
  baseline_cm: 23.0          # Stereo baseline distance (your current setup)
  left_camera_id: 1          # Left camera device ID
  right_camera_id: 2         # Right camera device ID
  resolution: [640, 408]     # Frame resolution [width, height]
```

**Key Parameters:**
- `baseline_cm`: Your current 23cm baseline setup (do not change)
- `camera_id`: Device IDs for your stereo cameras (adjust based on system)
- `resolution`: Frame resolution affects processing speed and accuracy

### Advanced Camera Settings
```yaml
cameras:
  warmup_frames: 3           # Number of frames to discard on startup
  capture_backend: "CAP_DSHOW"  # OpenCV capture backend
  sync_tolerance_ms: 100     # Maximum time difference between frames
```

## Depth Detection Configuration

### Confidence and Quality Settings
```yaml
depth_detection:
  min_confidence_threshold: 0.6    # Minimum confidence to accept depth (0-1)
  enable_bbox_corners: true        # Enable 4-corner triangulation
  enable_bbox_perimeter: true      # Enable perimeter sampling
  num_perimeter_points: 8          # Number of perimeter points (4-16)
  enable_fallback_strategies: true # Enable automatic fallback methods
```

**Parameter Effects:**

| Parameter | Low Value | High Value | Recommended |
|-----------|-----------|------------|-------------|
| `min_confidence_threshold` | More detections, lower quality | Fewer detections, higher quality | 0.6-0.8 |
| `num_perimeter_points` | Faster processing, less data | Slower processing, more data | 8-12 |

### Method Priority and Thresholds
```yaml
depth_detection:
  method_thresholds:
    bbox_corners: 0.7      # Minimum confidence for corner method
    bbox_perimeter: 0.6    # Minimum confidence for perimeter method
    center_fallback: 0.4   # Minimum confidence for center method
```

## Processing Configuration

### Frame Processing Limits
```yaml
processing:
  max_strawberries_per_frame: 5    # Maximum strawberries to process per frame
  enable_error_recovery: true      # Enable automatic error recovery
  max_retries: 3                   # Maximum retry attempts for failed operations
```

### Performance Tuning
```yaml
processing:
  frame_buffer_size: 10            # Number of frames to buffer
  enable_frame_dropping: true      # Drop frames if processing falls behind
  target_latency_ms: 100           # Target processing latency
```

## Logging Configuration

### Log Levels and Output
```yaml
logging:
  level: INFO                      # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: strawberry_locator.log     # Log file name
  max_file_size_mb: 10             # Maximum log file size
  backup_count: 5                  # Number of backup log files
```

### Log Format Options
```yaml
logging:
  console_format: "[%(levelname)s] %(message)s"
  file_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  include_timestamp: true
  include_function_name: true
```

## Performance Configuration

### Multi-threading Options
```yaml
performance:
  enable_threading: false          # Enable multi-threading (experimental)
  thread_pool_size: 4              # Number of worker threads
  enable_caching: true             # Enable result caching
  processing_fps: 30               # Target processing FPS
```

### Memory Management
```yaml
performance:
  max_memory_usage_mb: 512         # Maximum memory usage
  enable_garbage_collection: true  # Enable aggressive garbage collection
  cache_size: 100                  # Number of results to cache
```

## Robustness Configuration

### Calibration Monitoring
```yaml
robustness:
  enable_calibration_monitoring: true   # Monitor calibration quality
  enable_quality_filtering: true        # Filter low-quality detections
  enable_outlier_removal: true          # Remove depth outliers
  outlier_threshold: 2.0               # MAD threshold (standard deviations)
```

### Error Recovery
```yaml
robustness:
  auto_reset_on_failure: true      # Automatically reset on critical failure
  failure_threshold: 10            # Number of failures before reset
  recovery_delay_seconds: 5        # Delay before recovery attempt
```

## Visualization Configuration

### Display Options
```yaml
visualization:
  enable_depth_overlay: true       # Show depth information on visualization
  enable_confidence_overlay: true  # Show confidence scores
  enable_method_overlay: true      # Show calculation method used
  box_color: [0, 255, 0]          # Bounding box color (BGR)
  center_color: [0, 0, 255]       # Center point color (BGR)
  text_color: [255, 255, 0]       # Text color (BGR)
```

### Advanced Visualization
```yaml
visualization:
  enable_3d_visualization: false   # Enable 3D point cloud display
  show_perimeter_points: false     # Show perimeter sampling points
  show_corner_points: false        # Show corner triangulation points
  opacity: 0.7                     # Overlay opacity (0-1)
```

## Environment-Specific Configurations

### Greenhouse Environment
```yaml
# Optimized for greenhouse conditions
strawberry_locator:
  cameras:
    baseline_cm: 23.0
    resolution: [640, 408]
    
  depth_detection:
    min_confidence_threshold: 0.7    # Higher threshold for stable conditions
    num_perimeter_points: 12         # More points for accuracy
    
  robustness:
    enable_calibration_monitoring: true
    outlier_threshold: 1.5           # Stricter outlier removal
```

### Laboratory Environment
```yaml
# Optimized for controlled lab conditions
strawberry_locator:
  depth_detection:
    min_confidence_threshold: 0.5    # Lower threshold for testing
    num_perimeter_points: 8          # Standard point count
    
  visualization:
    enable_3d_visualization: true    # Enable detailed visualization
    show_perimeter_points: true
    show_corner_points: true
```

### Production Environment
```yaml
# Optimized for production deployment
strawberry_locator:
  processing:
    max_strawberries_per_frame: 3    # Focus on nearest strawberries
    enable_error_recovery: true
    
  performance:
    enable_caching: true
    processing_fps: 30
    
  robustness:
    enable_quality_filtering: true
    auto_reset_on_failure: true
```

## Configuration Validation

### Built-in Validation
The system automatically validates configuration on startup:

```python
locator = StrawberryLocator()
# Configuration is validated automatically
# Warnings issued for invalid values
# Fallback to defaults for critical errors
```

### Manual Validation
```python
# Test configuration without cameras
locator = StrawberryLocator()
config_valid = locator.validate_config()
print(f"Configuration valid: {config_valid}")
```

## Performance Impact Guide

### Speed Optimizations
```yaml
# Fastest configuration
performance:
  enable_threading: true
  enable_caching: true
  
depth_detection:
  num_perimeter_points: 4    # Minimum points
  enable_bbox_perimeter: false  # Disable perimeter sampling
  
visualization:
  enable_3d_visualization: false
  show_perimeter_points: false
```

### Accuracy Optimizations
```yaml
# Most accurate configuration
depth_detection:
  min_confidence_threshold: 0.8    # High confidence requirement
  num_perimeter_points: 16         # Maximum points
  enable_bbox_corners: true
  enable_bbox_perimeter: true
  
robustness:
  enable_outlier_removal: true
  outlier_threshold: 1.0           # Strict outlier removal
  enable_quality_filtering: true
```

## Troubleshooting Configuration Issues

### Common Problems

1. **Low detection rate**: Decrease `min_confidence_threshold`
2. **Slow processing**: Reduce `num_perimeter_points` or disable perimeter sampling
3. **High memory usage**: Disable caching and reduce thread pool size
4. **Poor depth accuracy**: Enable outlier removal and increase method thresholds

### Debug Configuration
```yaml
logging:
  level: DEBUG                     # Enable detailed logging
  
visualization:
  show_perimeter_points: true      # Show all processing steps
  show_corner_points: true
  enable_method_overlay: true
```

## Migration from Default Settings

### Step 1: Start with Defaults
```yaml
# Use default configuration initially
strawberry_locator: {}  # Empty config uses all defaults
```

### Step 2: Adjust Key Parameters
```yaml
# Tune these based on your environment
depth_detection:
  min_confidence_threshold: 0.6    # Adjust based on detection quality
  
cameras:
  left_camera_id: 1                # Set your camera IDs
  right_camera_id: 2
```

### Step 3: Optimize for Your Use Case
```yaml
# Add optimizations based on testing
performance:
  enable_caching: true             # If processing same scenes
  
robustness:
  enable_outlier_removal: true     # If depth accuracy is poor
```

## See Also

- [API Reference](API_REFERENCE_ENHANCED_LOCATOR.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Performance Comparison](PERFORMANCE_COMPARISON.md)