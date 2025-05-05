import cv2
from ultralytics import YOLO
import requests
import time
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in your project directory or provide full path

# Firebase setup
PROJECT_ID = "drowning-notification"
DEVICE_TOKEN = "fidLtsmJTvqYV7a7oYz1Cp:APA91bE_qVjw7hLQt_WjTYkrwNlYWhq6_V4HgvJwrzd7huMzLaR_YK5ejOVtvjGIkkRdoUl4RJnYDNHX0H595yNM449cIPwI1UpIxkwFKuoYyAp3_Eq8498"
SERVICE_ACCOUNT_FILE = "drowning-notification-firebase-adminsdk-fbsvc-55feeeaf68.json"

# Get OAuth2 access token from service account
def get_access_token():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    credentials.refresh(Request())
    return credentials.token

# Send notification via HTTP v1
def send_alert():
    access_token = get_access_token()
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

# Camera setup
cap = cv2.VideoCapture(
    "libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER
)
last_alert_time = 0
alert_cooldown = 10  # seconds

print("Starting detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, imgsz=640)[0]

    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        class_name = model.names[cls_id]

        if class_name.lower() == "drowning" and conf > 0.5:
            print(f"Drowning detected (conf={conf:.2f})")
            if time.time() - last_alert_time > alert_cooldown:
                send_alert()
                last_alert_time = time.time()

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
