# Strawberry Picker Control ROS2 Package

This package provides ROS2 nodes for controlling a robotic arm to pick strawberries based on YOLOv8 detections.

## Overview

The package includes:

- `arm_controller_node`: Subscribes to `/detection` topic, converts pixel coordinates to robot workspace coordinates, and sends inverse kinematics commands to an Arduino via serial.
- `test_arm_controller_node`: Publishes mock detection messages for testing without a camera.
- Launch files for easy startup.

## Dependencies

- ROS2 Humble (or other distributions)
- `rclpy`
- `std_msgs`
- `sensor_msgs`
- `cv_bridge` (for detection publisher, but not required for this package)
- `ultralytics` (for YOLOv8 detection, but not required for this package)
- `pyserial`

Install dependencies:

```bash
pip install pyserial
```

## Installation

1. Copy this package into your ROS2 workspace `src/` directory.

2. Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select strawberry_picker_control
source install/setup.bash
```

## Usage

### 1. Running with Mock Detection (No Camera)

Launch the arm controller with a mock detection publisher:

```bash
ros2 launch strawberry_picker_control picker.launch.py
```

This will start:
- `arm_controller` node (listens to `/detection`)
- `mock_detection_publisher` node (publishes dummy detections)

### 2. Running with Real YOLOv8 Detection

Ensure the `yolov8_cam` package is built and running. Then launch:

```bash
ros2 launch strawberry_picker_control picker_with_detection.launch.py
```

This will start:
- `detection_publisher` from `yolov8_cam` (requires camera)
- `arm_controller` node

### 3. Running Nodes Individually

Start the arm controller:

```bash
ros2 run strawberry_picker_control arm_controller
```

Start the mock detection publisher:

```bash
ros2 run strawberry_picker_control test_arm_controller
```

## Configuration

Parameters can be adjusted via the launch file or at runtime:

- `serial_port`: Serial device path (default `/dev/ttyUSB0`)
- `baud_rate`: Baud rate (default `9600`)
- `confidence_threshold`: Minimum detection confidence (default `0.5`)
- `image_width`, `image_height`: Camera resolution (default `640`)
- `workspace_x_min`, `workspace_x_max`, `workspace_y_min`, `workspace_y_max`: Robot workspace boundaries in cm.
- `pick_height`: Z coordinate for picking (cm).

Example of overriding parameters:

```bash
ros2 run strawberry_picker_control arm_controller --ros-args -p serial_port:=/dev/ttyACM0 -p workspace_x_min:=-15.0
```

## Coordinate Transformation

The node maps pixel coordinates (0–640) to robot coordinates using a simple linear mapping:

```
robot_x = workspace_x_min + (pixel_x / image_width) * (workspace_x_max - workspace_x_min)
robot_y = workspace_y_max - (pixel_y / image_height) * (workspace_y_max - workspace_y_min)
```

The Y axis is inverted because pixel Y increases downward.

## Serial Protocol

The node sends commands to the Arduino in the format:

```
I x y z
```

Where `x`, `y`, `z` are floating‑point numbers in centimeters. The Arduino is expected to run an inverse‑kinematics solver and move the arm accordingly.

## Testing

To test without an Arduino, you can simulate the serial port using a loopback (e.g., `socat`). Alternatively, you can modify the node to log commands instead of sending them.

## Troubleshooting

- **No serial connection**: Check the `serial_port` parameter and ensure the Arduino is connected and the user has permissions.
- **No detections**: Verify that the `/detection` topic is being published (use `ros2 topic echo /detection`).
- **Incorrect coordinates**: Calibrate the workspace parameters to match your camera‑robot setup.

## License

MIT