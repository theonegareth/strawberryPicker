#!/usr/bin/env python3
"""
Real-time Strawberry Detection and Ripeness Classification using Webcam
Optimized for WSL (Windows Subsystem for Linux) environments
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import time
from pathlib import Path
import sys
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class StrawberryPickerWebcam:
    def __init__(self, detector_path, classifier_path=None, device='cpu'):
        """
        Initialize the strawberry picker system
        
        Args:
            detector_path: Path to YOLOv8 detection model
            classifier_path: Path to classification model (optional)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        print("🍓 Initializing Strawberry Picker AI System...")
        
        self.device = device
        self.has_classifier = classifier_path is not None
        
        if self.has_classifier:
            self.ripeness_classes = ['unripe', 'partially-ripe', 'ripe', 'overripe']
            # Color mapping for visualization
            self.colors = {
                'unripe': (0, 255, 0),        # Green
                'partially-ripe': (0, 255, 255),  # Yellow
                'ripe': (0, 0, 255),          # Red
                'overripe': (128, 0, 128)     # Purple
            }
        else:
            self.ripeness_classes = ['strawberry']
            self.colors = {
                'strawberry': (0, 255, 0)     # Green for all strawberries
            }
        
        # Load detection model
        print("Loading detection model...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO(detector_path)
            print("✅ Detection model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading detection model: {e}")
            print("💡 Tip: Make sure the model file exists at the specified path")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Model path: {detector_path}")
            # Try to find available models
            print("🔍 Searching for available model files...")
            import glob
            model_files = glob.glob("**/*.pt", recursive=True)
            print("Available model files:")
            for model_file in model_files:
                print(f"  - {model_file}")
            sys.exit(1)
        
        # Load classification model if provided
        if self.has_classifier:
            print("Loading classification model...")
            try:
                self.classifier = torch.load(classifier_path, map_location=device)
                self.classifier.eval()
                print("✅ Classification model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading classification model: {e}")
                sys.exit(1)
        else:
            self.classifier = None
            print("ℹ️  No classification model provided - running detection only")
        
        # Setup preprocessing for classification
        if self.has_classifier:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        print("✅ System initialized and ready!")
    
    def detect_and_classify(self, frame):
        """
        Detect strawberries and classify their ripeness in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            results: List of detection/classification results
            visualized_frame: Frame with visualizations
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect strawberries
        detection_results = self.detector(frame_rgb)
        
        results = []
        
        for result in detection_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf < 0.3:  # Lowered confidence threshold for better detection
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Crop strawberry
                crop = frame_rgb[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Classify ripeness if classifier available
                if self.has_classifier:
                    try:
                        crop_pil = Image.fromarray(crop)
                        input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            output = self.classifier(input_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                        
                        ripeness = self.ripeness_classes[predicted_class]
                        
                        results.append({
                            'bbox': (x1, y1, x2, y2),
                            'ripeness': ripeness,
                            'confidence': confidence,
                            'detection_confidence': float(conf)
                        })
                        
                    except Exception as e:
                        print(f"Warning: Error classifying crop: {e}")
                        continue
                else:
                    # Detection only mode
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'ripeness': 'strawberry',
                        'confidence': 1.0,
                        'detection_confidence': float(conf)
                    })
        
        return results
    
    def visualize(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: Detection/classification results
            
        Returns:
            visualized_frame: Frame with drawings
        """
        vis_frame = frame.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            ripeness = result['ripeness']
            conf = result['confidence']
            
            # Draw bounding box
            color = self.colors[ripeness]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{ripeness} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(vis_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add title
        title = "Strawberry Picker AI - Press 'q' to quit"
        cv2.putText(vis_frame, title, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def run_webcam(self, camera_index=0, width=640, height=480, ip_camera_url=None):
        """
        Run real-time inference on webcam or IP camera
        
        Args:
            camera_index: Camera index (0 for default webcam)
            width: Frame width
            height: Frame height
            ip_camera_url: IP camera URL (if using IP camera)
        """
        if ip_camera_url:
            print(f"\n📹 Starting IP camera: {ip_camera_url}")
            # Try to open IP camera
            cap = cv2.VideoCapture(ip_camera_url)
            
            if not cap.isOpened():
                print(f"❌ Error: Could not connect to IP camera at {ip_camera_url}")
                print("\nTroubleshooting tips:")
                print("1. Check that the IP camera is running and accessible")
                print("2. Verify the URL is correct")
                print("3. Make sure you're on the same network")
                return
        else:
            print(f"\n📹 Starting webcam (camera {camera_index})...")
            # Try to open webcam
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"❌ Error: Could not open camera {camera_index}")
                print("\nTroubleshooting tips for WSL:")
                print("1. Install v4l2loopback: sudo apt-get install v4l2loopback-dkms")
                print("2. Load module: sudo modprobe v4l2loopback")
                print("3. Use IP webcam app on phone as alternative")
                print("4. Or use pre-recorded video file")
                return
        
        print("Press 'q' to quit, 's' to save screenshot")
        print("Make sure strawberries are well-lit and clearly visible\n")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # FPS tracking
        self.fps = 0
        frame_count = 0
        start_time = time.time()
        
        # Screenshot counter
        screenshot_count = 0
        
        try:
            while frame_count < 100000:  # Process 50 frames then stop
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Detect and classify
                results = self.detect_and_classify(frame)
                
                # Visualize results
                vis_frame = self.visualize(frame, results)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    self.fps = frame_count / elapsed
                
                # Save frame to file (for WSL environments without display)
                screenshot_path = f"webcam_frame.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                
                # Log every 10 frames
                if frame_count % 10 == 0:
                    detection_count = len(results)
                    print(f"📊 Frame {frame_count:4d} - FPS: {self.fps:.1f} - Detections: {detection_count}")
                    print(f"📸 Saved: {screenshot_path}")
                
                # Add a small delay to avoid overwhelming the system
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Webcam session ended")
    
    def run_video_file(self, video_path):
        """
        Run inference on a video file
        
        Args:
            video_path: Path to video file
        """
        print(f"\n🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        output_path = f"output_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                results = self.detect_and_classify(frame)
                vis_frame = self.visualize(frame, results)
                
                # Write to output
                out.write(vis_frame)
                
                # Display progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                          f"Time: {elapsed:.1f}s")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"✅ Video processing complete. Output saved to: {output_path}")

def main():
    """Main function with argument parsing"""
    # Change to script directory to ensure relative paths work
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Real-time Strawberry Detection and Ripeness Classification'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        default='model/weights/strawberry_yolov8s_enhanced.pt',
        help='Path to YOLOv8 detection model'
    )
    
    parser.add_argument(
        '--classifier',
        type=str,
        default=None,
        help='Path to classification model (optional - for ripeness classification)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['webcam', 'video', 'ip_camera'],
        default='webcam',
        help='Mode: webcam, video file, or IP camera'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to video file (if mode=video) or IP camera URL (if mode=ip_camera)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Camera frame width'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Camera frame height'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  Running on CPU - this will be slower. Consider using GPU if available.")
    
    # Initialize system
    try:
        picker = StrawberryPickerWebcam(
            detector_path=args.detector,
            classifier_path=args.classifier,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)
    
    # Run inference
    if args.mode == 'webcam':
        picker.run_webcam(
            camera_index=args.camera,
            width=args.width,
            height=args.height
        )
    elif args.mode == 'ip_camera':
        if not args.input:
            print("❌ Error: --input required for IP camera mode")
            print("Example: --input http://192.168.1.100:5000/video")
            sys.exit(1)
        picker.run_webcam(ip_camera_url=args.input)
    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: --input required for video mode")
            sys.exit(1)
        picker.run_video_file(args.input)

if __name__ == "__main__":
    # Check for required libraries
    try:
        import torch
        import cv2
        from PIL import Image
        from torchvision import transforms
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Install with: pip install torch torchvision opencv-python pillow")
        sys.exit(1)
    
    main()