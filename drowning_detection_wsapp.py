import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import json
import os
import subprocess
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

# Try to configure libcamera for V4L2 compatibility
def setup_libcamera_compatibility():
    try:
        # Check if the bcm2835-v4l2 module is loaded
        modprobe_result = subprocess.run(["lsmod"], 
                                         capture_output=True, 
                                         text=True)
        
        if "bcm2835_v4l2" not in modprobe_result.stdout:
            print("Loading bcm2835-v4l2 module for libcamera compatibility...")
            subprocess.run(["sudo", "modprobe", "bcm2835-v4l2"], 
                           capture_output=True)
    except Exception as e:
        print(f"Note: Could not set up libcamera compatibility: {e}")

# Initialize camera using picamera2 if available, otherwise fall back to OpenCV
def init_camera():
    # Try to set up libcamera compatibility
    setup_libcamera_compatibility()
    
    # First, try picamera2 (preferred for Raspberry Pi)
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
            
            # Test capture
            test_frame = picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print("picamera2 initialized successfully")
                return picam2, True  # True indicates it's picamera2
            else:
                print("picamera2 started but couldn't capture valid frame")
                picam2.stop()
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
    
    # Fall back to OpenCV
    print("Trying OpenCV camera methods with libcamera compatibility...")
    
    # Try different camera options
    camera_options = [
        {"device": 0, "api": cv2.CAP_V4L2},
        {"device": 0, "api": cv2.CAP_ANY},
        {"device": "libcamera:///dev/video0", "api": cv2.CAP_V4L2},
        {"device": "/dev/video0", "api": cv2.CAP_V4L2},
        {"device": "device=/dev/video0,format=YUY2", "api": cv2.CAP_V4L2}
    ]
    
    for opt in camera_options:
        try:
            print(f"Trying camera: {opt['device']} with API: {opt['api']}")
            cap = cv2.VideoCapture(opt['device'], opt['api'])
            
            if cap.isOpened():
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Test read with multiple attempts
                for attempt in range(5):  # More attempts
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Successfully connected to camera using {opt['device']}")
                        print(f"Frame shape: {test_frame.shape}")
                        return cap, False  # False indicates it's OpenCV
                    time.sleep(0.5)
                
                print(f"Device {opt['device']} opened but couldn't read valid frames")
                cap.release()
            else:
                print(f"Failed to open device {opt['device']}")
        except Exception as e:
            print(f"Error with device {opt['device']}: {e}")
    
    # If we get here, try a last resort with libcamera-vid piped to OpenCV
    try:
        print("Attempting to use libcamera-vid as a last resort...")
        print("This method will be added if needed based on your response")
        # Code here would use subprocess to call libcamera-vid and pipe to OpenCV
        # This is a more complex approach we can implement if necessary
    except Exception as e:
        print(f"libcamera-vid attempt failed: {e}")
    
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
                for _ in range(3):  # Try up to 3 times to get a frame
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.1)
            
            if not ret or frame is None or frame.size == 0:
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
