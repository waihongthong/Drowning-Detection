from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import time

# Load YOLO model
model_path = "best.pt"
model = YOLO(model_path)
print(f"✅ Loaded YOLOv8 model from {model_path}")

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(2)  # Give camera time to warm up

print("🚨 Starting detection... Press 'q' to exit.")

while True:
    frame = picam2.capture_array()

    # Run YOLO detection
    results = model(frame, verbose=False)[0]

    # Draw results on the frame
    annotated_frame = results.plot()

    # Display the frame
    cv2.imshow("Drowning Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# Cleanup
picam2.close()
cv2.destroyAllWindows()
