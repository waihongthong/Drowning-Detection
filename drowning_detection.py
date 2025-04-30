import cv2
import torch
import requests
import socket
import pickle
import struct
import threading
import time
from flask import Flask, Response
import RPi.GPIO as GPIO  # Added for Raspberry Pi GPIO control

# Initialize Flask app for streaming
app = Flask(__name__)

# Global variable to store the latest frame and detection status
latest_frame = None
drowning_detected = False
lock = threading.Lock()

# GPIO Setup
LED_PIN = 17    # GPIO pin for LED
BUZZER_PIN = 18  # GPIO pin for buzzer
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Ensure LED is off initially
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off initially

# Create PWM object for buzzer (frequency 2000 Hz)
buzzer_pwm = GPIO.PWM(BUZZER_PIN, 2000)

# Function to trigger alarm for 10 seconds
def trigger_alarm(duration=10):
    print("🚨 Triggering alarm for 10 seconds!")
    
    # Start the buzzer with 50% duty cycle
    buzzer_pwm.start(50)
    
    # Turn on LED
    GPIO.output(LED_PIN, GPIO.HIGH)
    
    # Keep alarm on for specified duration
    time.sleep(duration)
    
    # Turn off alarm
    buzzer_pwm.stop()
    GPIO.output(LED_PIN, GPIO.LOW)
    
    print("Alarm stopped")

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("Model loaded!")

# Detection function to run in background
def detection_loop():
    global latest_frame, drowning_detected
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Raspberry Pi camera
    
    # Variable to track if alarm is currently active
    alarm_active = False
    alarm_thread = None
    
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
        
        # If drowning is detected, send alert and trigger alarm
        if drowning_detected_current and not alarm_active:
            print("🚨 Drowning detected!")
            
            # Start the alarm in a separate thread to avoid blocking detection
            alarm_active = True
            alarm_thread = threading.Thread(target=trigger_alarm)
            alarm_thread.daemon = True
            alarm_thread.start()
            
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
                
            # Schedule the alarm to be deactivated after the thread completes
            def reset_alarm_status():
                nonlocal alarm_active
                alarm_thread.join()
                alarm_active = False
                
            reset_thread = threading.Thread(target=reset_alarm_status)
            reset_thread.daemon = True
            reset_thread.start()
        
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
    try:
        # Start detection in a background thread
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Run Flask app
        print("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # Clean up GPIO on program exit
        GPIO.cleanup()