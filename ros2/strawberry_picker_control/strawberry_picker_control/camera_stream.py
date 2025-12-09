# save as camera_stream.py on Windows and run with `python camera_stream.py`
import cv2
from flask import Flask, Response, render_template_string
import io
from PIL import Image

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Windows webcam index

def gen():
    """Video streaming generator function"""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            continue
        
        # Convert frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Home page with video feed"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Stream</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
            h1 { color: #333; }
            img { border: 2px solid #333; max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Camera Live Stream</h1>
        <img src="{{ url_for('video_feed') }}" alt="Video Feed">
        <p>Streaming from webcam...</p>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def favicon():
    """Return empty favicon to prevent 404 errors"""
    return '', 204

if __name__ == '__main__':
    print("Starting camera server...")
    print("Access the stream at: http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    app.run(host='0.0.0.0', port=5000, debug=False)