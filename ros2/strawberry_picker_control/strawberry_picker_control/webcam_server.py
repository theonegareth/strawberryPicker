#!/usr/bin/env python3
"""
Simple webcam server for ROS2 YOLO detection
Run this first to provide video feed to the detection publisher
"""

import cv2
from flask import Flask, Response
import threading

app = Flask(__name__)

class WebcamServer:
    def __init__(self, camera_index=0, host='0.0.0.0', port=5000):
        self.camera_index = camera_index
        self.host = host
        self.port = port
        self.cap = None
        self.is_running = False
        
    def start_camera(self):
        """Initialize the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera index {self.camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera opened successfully (index: {self.camera_index})")
        print(f"📹 Streaming at: http://{self.host}:{self.port}/video_feed")
        return True
    
    def generate_frames(self):
        """Generate video frames for streaming"""
        while self.is_running:
            success, frame = self.cap.read()
            if not success:
                print("⚠️ Failed to read frame from camera")
                break
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def start_server(self):
        """Start the Flask server"""
        if not self.start_camera():
            return False
        
        self.is_running = True
        
        # Define video feed endpoint
        @app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/')
        def index():
            return """
            <html>
            <head><title>Webcam Server</title></head>
            <body>
                <h1>Webcam Server is Running</h1>
                <p>Video feed available at: <a href="/video_feed">/video_feed</a></p>
                <p>ROS2 Detection Publisher will connect to this feed</p>
            </body>
            </html>
            """
        
        print("🚀 Starting webcam server...")
        print("💡 Keep this running in a separate terminal")
        print("🎯 Run the detection publisher in another terminal")
        
        # Run server in a thread so we can stop it cleanly
        server_thread = threading.Thread(
            target=lambda: app.run(host=self.host, port=self.port, debug=False, threaded=True)
        )
        server_thread.daemon = True
        server_thread.start()
        
        return True
    
    def stop(self):
        """Stop the server and release camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            print("✅ Camera released")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Webcam Server for ROS2 YOLO Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    
    args = parser.parse_args()
    
    server = WebcamServer(camera_index=args.camera, host=args.host, port=args.port)
    
    try:
        if server.start_server():
            print("\n" + "="*60)
            print("Webcam Server is running!")
            print(f"Camera: Index {args.camera}")
            print(f"Stream URL: http://{args.host}:{args.port}/video_feed")
            print("="*60)
            print("\nPress Ctrl+C to stop the server\n")
            
            # Keep main thread alive
            while True:
                import time
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down webcam server...")
    finally:
        server.stop()


if __name__ == '__main__':
    main()