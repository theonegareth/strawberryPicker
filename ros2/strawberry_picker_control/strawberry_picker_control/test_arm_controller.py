#!/usr/bin/env python3
"""
Test node that publishes mock detection messages for arm controller testing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class MockDetectionPublisher(Node):
    def __init__(self):
        super().__init__('mock_detection_publisher')
        self.publisher = self.create_publisher(String, 'detection', 10)
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Mock detection publisher started. Publishing every 2 seconds.')

    def timer_callback(self):
        # Simulate a detection of a strawberry at center of image
        detection = [
            {
                'class': 'strawberry',
                'confidence': 0.95,
                'bbox': [300.0, 200.0, 340.0, 240.0]  # x1, y1, x2, y2
            }
        ]
        msg = String()
        msg.data = json.dumps(detection)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published mock detection: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = MockDetectionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()