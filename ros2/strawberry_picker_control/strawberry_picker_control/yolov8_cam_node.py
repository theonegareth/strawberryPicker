#!/usr/bin/env python3
import sys
import os

# Build conda environment path dynamically based on user's home directory
home_dir = os.path.expanduser('~')
conda_env_python = os.path.join(home_dir, 'miniconda3/envs/yolov8_env/bin/python')
current_python = sys.executable

if current_python != conda_env_python and os.path.exists(conda_env_python):
    # Re-run this script with the conda environment Python
    os.execv(conda_env_python, [conda_env_python] + sys.argv)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class YoloV8CamNode(Node):
    def __init__(self):
        super().__init__('yolov8_cam_node')
        self.publisher_ = self.create_publisher(Image, 'yolov8_detected_image', 10)
        self.bridge = CvBridge()

        # Load YOLOv8 model
        self.model = YOLO('best.pt')  # or replace with your custom model path
        self.get_logger().info("Loaded YOLOv8 model: yolov8n.pt")

        # Open the camera
        self.cap = cv2.VideoCapture("http://10.24.30.80:5000/video_feed")
        if not self.cap.isOpened():
            self.get_logger().error("❌ Failed to open camera")
        else:
            self.get_logger().info("✅ Camera opened successfully")

        # Timer to process frames every 0.1 second (~10 FPS)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("🚀 YOLOv8 Camera Node started")

        # Set whether to show OpenCV window
        self.show_window = True

        # FPS calculation
        self.prev_frame_time = 0
        self.fps_update_interval = 2.0  # Update FPS every 2 seconds
        self.last_fps_print = 0.0
        import time
        self.start_time = time.time()

    def timer_callback(self):
        import time
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to read from camera")
            return

        # Run YOLOv8 inference
        results = self.model(frame)
        annotated = results[0].plot()

        # Calculate FPS
        current_time = time.time()
        time_elapsed = current_time - self.prev_frame_time
        if time_elapsed > 0:
            fps = 1.0 / time_elapsed
            
            # Print FPS every 2 seconds
            if current_time - self.last_fps_print >= self.fps_update_interval:
                self.get_logger().info(f"📊 FPS: {fps:.1f}")
                self.last_fps_print = current_time

            # Add FPS text to frame
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.prev_frame_time = current_time

        # Publish annotated frame to ROS2 topic
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.publisher_.publish(msg)

        # Show both original and annotated frames
        if self.show_window:
            # Show original frame
            cv2.imshow("Raw Camera Feed", frame)
            # Show annotated frame
            cv2.imshow("YOLOv8 Detection", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("🛑 Quitting display window")
                rclpy.shutdown()

    def destroy_node(self):
        self.get_logger().info("Releasing camera...")
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloV8CamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
