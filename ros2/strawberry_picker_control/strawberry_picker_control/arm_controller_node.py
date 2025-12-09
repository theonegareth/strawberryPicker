#!/usr/bin/env python3
"""
Arm Controller Node for Strawberry Picker

Subscribes to /detection topic (JSON string of detections)
Selects the most confident detection, converts pixel coordinates to robot coordinates,
and sends inverse kinematics commands to Arduino via serial.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
import json
import serial
import time
import math

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # Declare parameters with default values
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 640)
        self.declare_parameter('workspace_x_min', -10.0)
        self.declare_parameter('workspace_x_max', 10.0)
        self.declare_parameter('workspace_y_min', 0.0)
        self.declare_parameter('workspace_y_max', 20.0)
        self.declare_parameter('pick_height', 5.0)

        # Get parameter values
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.workspace_x_min = self.get_parameter('workspace_x_min').value
        self.workspace_x_max = self.get_parameter('workspace_x_max').value
        self.workspace_y_min = self.get_parameter('workspace_y_min').value
        self.workspace_y_max = self.get_parameter('workspace_y_max').value
        self.pick_height = self.get_parameter('pick_height').value

        self.get_logger().info(f"Serial port: {self.serial_port}")
        self.get_logger().info(f"Workspace X: [{self.workspace_x_min}, {self.workspace_x_max}]")
        self.get_logger().info(f"Workspace Y: [{self.workspace_y_min}, {self.workspace_y_max}]")

        self.subscription = self.create_subscription(
            String,
            'detection',
            self.detection_callback,
            10)
        self.get_logger().info("Arm Controller node started. Waiting for detections...")

        # Serial communication with Arduino
        self.ser = None
        self.connect_serial()

    def connect_serial(self):
        """Establish serial connection to Arduino."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  # wait for Arduino to reset
            self.get_logger().info(f"Connected to Arduino on {self.serial_port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to Arduino: {e}")
            self.ser = None

    def pixel_to_robot(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates (0-640) to robot coordinates (cm).
        Simple linear mapping.
        """
        # Normalize pixel coordinates to 0-1
        norm_x = pixel_x / self.image_width
        norm_y = pixel_y / self.image_height

        # Map to robot workspace (invert Y axis because pixel Y increases downward)
        robot_x = self.workspace_x_min + norm_x * (self.workspace_x_max - self.workspace_x_min)
        robot_y = self.workspace_y_max - norm_y * (self.workspace_y_max - self.workspace_y_min)

        return robot_x, robot_y

    def send_ik_command(self, x, y, z):
        """
        Send inverse kinematics command to Arduino.
        Format: "I x y z"
        """
        if self.ser is None:
            self.get_logger().warn("Serial not connected, skipping command.")
            return False
        cmd = f"I {x:.2f} {y:.2f} {z:.2f}\n"
        self.get_logger().info(f"Sending IK command: {cmd.strip()}")
        try:
            self.ser.write(cmd.encode())
            # Wait for response (optional)
            response = self.ser.readline().decode().strip()
            if response:
                self.get_logger().info(f"Arduino response: {response}")
            return True
        except Exception as e:
            self.get_logger().error(f"Serial write error: {e}")
            return False

    def detection_callback(self, msg):
        """Process detection messages."""
        try:
            detections = json.loads(msg.data)
            if not detections:
                self.get_logger().debug("No detections")
                return

            # Filter by confidence and select the detection with highest confidence
            valid_detections = [d for d in detections if d.get('confidence', 0) > self.confidence_threshold]
            if not valid_detections:
                self.get_logger().debug("No detections above confidence threshold")
                return

            # Choose detection with highest confidence
            best = max(valid_detections, key=lambda d: d['confidence'])
            bbox = best['bbox']  # [x1, y1, x2, y2]
            class_name = best.get('class', 'unknown')
            confidence = best['confidence']

            # Compute center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            self.get_logger().info(
                f"Selected detection: {class_name} "
                f"at ({center_x:.1f}, {center_y:.1f}) "
                f"conf {confidence:.2f}"
            )

            # Convert to robot coordinates
            robot_x, robot_y = self.pixel_to_robot(center_x, center_y)
            robot_z = self.pick_height  # fixed picking height

            self.get_logger().info(
                f"Robot target: ({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f}) cm"
            )

            # Send command to Arduino
            success = self.send_ik_command(robot_x, robot_y, robot_z)
            if success:
                self.get_logger().info("Command sent successfully.")
            else:
                self.get_logger().error("Failed to send command.")

        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse detection message")
        except KeyError as e:
            self.get_logger().error(f"Missing key in detection data: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in detection_callback: {e}")

    def destroy_node(self):
        if self.ser is not None:
            self.ser.close()
            self.get_logger().info("Serial connection closed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()