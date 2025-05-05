import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import json
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Check if we're on Raspberry Pi and import picamera2 if available
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
    print("Picamera2 module found, will use it for camera access")
except ImportError:
    HAS_PICAMERA2 = False
    print("Picamera2 module not found, will use OpenCV for camera access")

# Load YOLOv8 model
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"Warning: Model file {model_path} not found. Please check the path.")
    exit(1)

try:
    model = YOLO(model_path)
    print(f"Successfully loaded YOLO model from {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Firebase setup
PROJECT_ID = "drowning-notification"
DEVICE_TOKEN = "fidLtsmJTvqYV7a7oYz1Cp:APA91bE_qVjw7hLQt_WjTYkrwNlYWhq6_V4HgvJwrzd7huMzLaR_YK5ejOVtvjGIkkRdoUl4RJnYDNHX0H595yNM449cIPwI1UpIxkwFKuoYyAp3_Eq8498"
SERVICE_ACCOUNT_FILE = "drowning-notification-firebase-adminsdk-fbsvc-55feeeaf68.json"

# Get OAuth2 access token from service account
def get_access_token():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"]
        )
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

# Send notification via HTTP v1
def send_alert():
    try:
        access_token = get_access_token()
        if not access_token:
            print("Failed to get access token, can't send alert")
            return False
            
        url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; UTF-8",
        }
        message = {
            "message": {
                "token": DEVICE_TOKEN,
                "notification": {
                    "title": "🚨 Drowning Alert",
                    "body": "Drowning detected by Raspberry Pi!"
                }
            }
        }
        response = requests.post(url, headers=headers, json=message)
        print("Notification sent:", response.status_code, response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending alert: {e}")
        return False

# Initialize camera using picamera2 if available, otherwise fall back to OpenCV
def init_camera():
    if HAS_PICAMERA2:
        try:
            print("Initializing camera with picamera2...")
            picam2 = Picamera2()
            # Configure camera
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(2)  # Give camera time to warm up
            print("picamera2 initialized successfully")
            return picam2, True  # True indicates it's picamera2
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
    
    # Fall back to OpenCV
    print("Trying OpenCV camera methods...")
    camera_methods = [
        # Method 1: Standard OpenCV camera
        lambda: cv2.VideoCapture(0),
        
        # Method 2: Direct V4L2 access
        lambda: cv2.VideoCapture('/dev/video0'),
        
        # Method 3: V4L2 with specific resolution
        lambda: cv2.VideoCapture('v4l2:///dev/video0')
    ]
    
    for i, method in enumerate(camera_methods):
        print(f"Trying OpenCV camera method {i+1}...")
        cap = method()
        if cap.isOpened():
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Test read
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully connected to camera using OpenCV method {i+1}")
                return cap, False  # False indicates it's OpenCV
            else:
                print(f"Method {i+1} opened camera but couldn't read frame")
                cap.release()
        else:
            print(f"Method {i+1} failed to open camera")
    
    print("All camera methods failed!")
    return None, False

# Initialize camera
camera, is_picamera = init_camera()
if camera is None:
    print("Failed to initialize any camera. Exiting.")
    exit(1)

last_alert_time = 0
alert_cooldown = 10  # seconds
frame_count = 0
error_count = 0
MAX_ERRORS = 5

print("Starting detection... Press 'q' to exit.")

try:
    while True:
        try:
            # Get frame from appropriate camera
            if is_picamera:
                frame = camera.capture_array()
                ret = True
            else:
                ret, frame = camera.read()
            
            if not ret:
                error_count += 1
                print(f"Failed to grab frame (error {error_count}/{MAX_ERRORS})")
                
                if error_count >= MAX_ERRORS:
                    print("Too many errors, trying to reinitialize camera...")
                    if not is_picamera:
                        camera.release()
                    camera, is_picamera = init_camera()
                    if camera is None:
                        print("Camera reinitialization failed. Exiting.")
                        break
                    error_count = 0
                time.sleep(0.5)
                continue
            
            # Reset error count when we successfully get a frame
            error_count = 0
            frame_count += 1
            
            # Only process every 3rd frame to reduce CPU load
            if frame_count % 3 == 0:
                # Ensure frame is in correct format for YOLO
                if is_picamera:
                    # picamera2 returns RGB, convert to BGR for OpenCV
                    frame_for_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_for_display = frame.copy()
                
                results = model(frame, imgsz=640)[0]
                
                # Process detections
                for r in results.boxes:
                    cls_id = int(r.cls[0])
                    conf = float(r.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_for_display, f"{class_name} {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if class_name.lower() == "drowning" and conf > 0.5:
                        print(f"Drowning detected (conf={conf:.2f})")
                        if time.time() - last_alert_time > alert_cooldown:
                            send_alert()
                            last_alert_time = time.time()
            
                # Add status text
                cv2.putText(frame_for_display, "Status: Running", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Drowning Detection System", frame_for_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit")
                break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)

except Exception as e:
    print(f"Unhandled exception: {e}")
finally:
    # Clean up
    if camera is not None:
        if not is_picamera:
            camera.release()
        else:
            camera.stop()
    cv2.destroyAllWindows()
    print("Program terminated")
