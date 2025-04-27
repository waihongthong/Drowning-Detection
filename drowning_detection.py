import cv2
import torch
import requests
import socket
import pickle
import struct
import threading
import time
from flask import Flask, Response

# Initialize Flask app for streaming
app = Flask(__name__)

# Global variable to store the latest frame and detection status
latest_frame = None
drowning_detected = False
lock = threading.Lock()

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("Model loaded!")

# Detection function to run in background
def detection_loop():
    global latest_frame, drowning_detected
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Raspberry Pi camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            time.sleep(0.1)
            continue

        # Store the latest frame for streaming
        with lock:
            latest_frame = frame.copy()
        
        # Run inference
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        # Check for drowning detection
        drowning_detected_current = False
        for _, row in detections.iterrows():
            if row['name'] == 'drowning':
                drowning_detected_current = True
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "DROWNING", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Update drowning status and frame with overlays
        with lock:
            drowning_detected = drowning_detected_current
            latest_frame = frame.copy()
        
        # If drowning is detected, send alert
        if drowning_detected_current:
            print("🚨 Drowning detected!")
            try:
                # Send notification via FCM here (replace with your key + device token)
                requests.post("https://fcm.googleapis.com/fcm/send", json={
                    "to": "<DEVICE_TOKEN>",
                    "notification": {"title": "Drowning Alert", "body": "Possible drowning detected!"},
                }, headers={
                    "Authorization": "key=<YOUR_SERVER_KEY>",
                    "Content-Type": "application/json"
                })
            except Exception as e:
                print(f"Error sending notification: {e}")
        
        # Small delay
        time.sleep(0.03)

# Function to generate frames for streaming
def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is not None:
                # Encode frame
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                frame_bytes = jpeg.tobytes()
                
                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# API endpoint to check if drowning is detected
@app.route('/api/status')
def get_status():
    global drowning_detected
    with lock:
        status = drowning_detected
    return {"drowning_detected": status}

# Video streaming route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Main endpoint
@app.route('/')
def index():
    return "Raspberry Pi Drowning Detection Server"

if __name__ == '__main__':
    # Start detection in a background thread
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()
    
    # Run Flask app
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)