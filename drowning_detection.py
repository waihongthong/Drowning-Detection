import cv2
import torch
import requests

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Init camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        if row['name'] == 'drowning':
            print("🚨 Drowning detected!")
            # Send notification via FCM here (replace with your key + device token)
            requests.post("https://fcm.googleapis.com/fcm/send", json={
                "to": "<DEVICE_TOKEN>",
                "notification": {"title": "Drowning Alert", "body": "Possible drowning detected!"},
            }, headers={
                "Authorization": "key=<YOUR_SERVER_KEY>",
                "Content-Type": "application/json"
            })

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
