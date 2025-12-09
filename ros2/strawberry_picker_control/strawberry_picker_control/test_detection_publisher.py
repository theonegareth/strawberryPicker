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
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import json
import time


class TestYoloDetectionPublisher(Node):
    def __init__(self):
        super().__init__('test_yolo_detection_publisher')
        
        # Publishers
        self.image_publisher_ = self.create_publisher(Image, 'yolov8_detected_image', 10)
        self.detection_publisher_ = self.create_publisher(String, 'detection', 10)
        
        self.bridge = CvBridge()

        # Load YOLOv8 model - using the best strawberry detection model
        model_path = '/home/user/machine-learning/GitHubRepos/strawberryPicker/model/detection/kaggle_strawberry_yolov8n_20251204_115538/weights/best.pt'
        self.get_logger().info(f"Loading YOLOv8 model from: {model_path}")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = YOLO(model_path)
        self.get_logger().info("✅ YOLOv8 model loaded successfully")
        self.get_logger().info(f"Model classes: {self.model.names}")

        # Try to open camera - with fallback options
        camera_urls = [
            "http://10.24.30.80:5000/video_feed",  # Primary webcam URL
            "http://127.0.0.1:5000/video_feed",    # Localhost fallback
            0                                       # Local webcam (index 0)
        ]
        
        self.cap = None
        for url in camera_urls:
            try:
                if isinstance(url, int):
                    self.get_logger().info(f"Trying local camera index {url}...")
                    self.cap = cv2.VideoCapture(url)
                else:
                    self.get_logger().info(f"Trying camera URL: {url}")
                    self.cap = cv2.VideoCapture(url)
                
                if self.cap.isOpened():
                    self.get_logger().info(f"✅ Camera opened successfully: {url}")
                    break
                else:
                    self.get_logger().warn(f"⚠️ Failed to open camera: {url}")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Error opening camera {url}: {e}")
        
        if not self.cap or not self.cap.isOpened():
            self.get_logger().error("❌ Could not open any camera source")
            self.get_logger().info("💡 Starting in test mode with sample detection")
            self.test_mode = True
        else:
            self.test_mode = False

        # Timer to process frames
        timer_period = 0.1  # 10 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("🚀 Test YOLO Detection Publisher started")
        self.get_logger().info("📡 Publishing to topics: /yolov8_detected_image, /detection")

        # FPS calculation
        self.prev_frame_time = 0
        self.frame_count = 0

    def timer_callback(self):
        import time
        
        if self.test_mode:
            # Test mode: Create a dummy frame and simulate detection
            # Create a blank image
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "TEST MODE - No Camera", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Simulate detection results
            detections = [
                {"x": 0.4523, "y": 0.3847, "class": "RipeStrawberry", "confidence": 0.9234},
                {"x": 0.7231, "y": 0.5612, "class": "UnripeStrawberry", "confidence": 0.8745}
            ]
            
            # Draw simulated boxes
            for det in detections:
                x = int(det["x"] * frame.shape[1])
                y = int(det["y"] * frame.shape[0])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"{det['class']}: {det['confidence']:.2f}", 
                           (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated = frame
            
        else:
            # Normal mode: Read from camera and run inference
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("⚠️ Failed to read from camera")
                return

            # Run YOLOv8 inference
            results = self.model(frame)
            result = results[0]
            annotated = result.plot()

            # Extract detections
            detections = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates (normalized 0-1)
                    x_center = float(box.xywh[0][0]) / frame.shape[1]
                    y_center = float(box.xywh[0][1]) / frame.shape[0]
                    
                    # Get class name and confidence
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf)
                    
                    detection = {
                        "x": round(x_center, 4),
                        "y": round(y_center, 4),
                        "class": class_name,
                        "confidence": round(confidence, 4)
                    }
                    detections.append(detection)

        # Publish detection results
        detection_msg = String()
        detection_msg.data = json.dumps(detections)
        self.detection_publisher_.publish(detection_msg)
        
        if detections:
            self.get_logger().info(f"📡 Published {len(detections)} detections")
            for i, det in enumerate(detections, 1):
                self.get_logger().info(f"  {i}. {det['class']} at ({det['x']}, {det['y']}) conf={det['confidence']}")

        # Calculate and display FPS
        current_time = time.time()
        time_elapsed = current_time - self.prev_frame_time
        if time_elapsed > 0:
            fps = 1.0 / time_elapsed
            self.frame_count += 1
            
            if self.frame_count % 10 == 0:  # Log every 10 frames
                self.get_logger().info(f"📊 FPS: {fps:.1f}")

            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.prev_frame_time = current_time

        # Publish annotated frame
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.image_publisher_.publish(msg)

        # Show annotated frame
        cv2.imshow("YOLOv8 Detection", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("🛑 Quitting...")
            rclpy.shutdown()

    def destroy_node(self):
        self.get_logger().info("Cleaning up...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TestYoloDetectionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()