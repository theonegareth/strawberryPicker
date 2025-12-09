#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class DetectionListener(Node):
    def __init__(self):
        super().__init__('detection_listener')
        self.subscription = self.create_subscription(
            String,
            'detection',
            self.listener_callback,
            10)
        self.get_logger().info("🔍 Listening to /detection topic...")

    def listener_callback(self, msg):
        try:
            detections = json.loads(msg.data)
            if detections:
                self.get_logger().info("=" * 50)
                self.get_logger().info("📦 DETECTIONS FOUND:")
                for i, detection in enumerate(detections, 1):
                    bbox = detection['bbox']  # [x1, y1, x2, y2]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    self.get_logger().info(
                        f"  {i}. Class: {detection['class']}, "
                        f"Center X: {center_x:.1f}, "
                        f"Center Y: {center_y:.1f}, "
                        f"Confidence: {detection['confidence']:.4f}, "
                        f"Bbox: {bbox}"
                    )
                self.get_logger().info("=" * 50)
            else:
                self.get_logger().info("❌ No detections in this frame")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse detection message")
        except KeyError as e:
            self.get_logger().error(f"Missing key in detection data: {e}")
            self.get_logger().info(f"Raw detection data: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()