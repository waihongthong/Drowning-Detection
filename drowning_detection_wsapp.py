import cv2
from ultralytics import YOLO
import requests
import time
import json
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request

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

# Camera initialization function that tries multiple methods
def init_camera():
    camera_methods = [
        # Method 1: Standard OpenCV camera
        lambda: cv2.VideoCapture(0),
        
        # Method 2: GStreamer pipeline for Raspberry Pi Camera v2
        lambda: cv2.VideoCapture(
            "libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        ),
        
        # Method 3: Simple V4L2 access
        lambda: cv2.VideoCapture('/dev/video0'),
        
        # Method 4: Alternative GStreamer pipeline
        lambda: cv2.VideoCapture(
            "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        )
    ]
    
    for i, method in enumerate(camera_methods):
        print(f"Trying camera method {i+1}...")
        cap = method()
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully connected to camera using method {i+1}")
                return cap
            else:
                print(f"Method {i+1} opened camera but couldn't read frame")
                cap.release()
        else:
            print(f"Method {i+1} failed to open camera")
    
    print("All camera methods failed!")
    return None

# Initialize camera
cap = init_camera()
if cap is None:
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
            ret, frame = cap.read()
            
            if not ret:
                error_count += 1
                print(f"Failed to grab frame (error {error_count}/{MAX_ERRORS})")
                
                if error_count >= MAX_ERRORS:
                    print("Too many errors, trying to reinitialize camera...")
                    cap.release()
                    cap = init_camera()
                    if cap is None:
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
                results = model(frame, imgsz=640)[0]
                
                for r in results.boxes:
                    cls_id = int(r.cls[0])
                    conf = float(r.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if class_name.lower() == "drowning" and conf > 0.5:
                        print(f"Drowning detected (conf={conf:.2f})")
                        if time.time() - last_alert_time > alert_cooldown:
                            send_alert()
                            last_alert_time = time.time()
            
            # Add status text
            cv2.putText(frame, "Status: Running", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Drowning Detection System", frame)
            
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
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")
