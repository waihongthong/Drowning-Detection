import os
import sys
import argparse
import time
import gpiozero
from threading import Thread

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Define and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "best_ncnn_model")',
                    required=True)
parser.add_argument('--source', help='Video source, should be "picamera0" for Raspberry Pi camera',
                    default="picamera0")
parser.add_argument('--resolution', help='Resolution in WxH format (example: "1280x720")',
                    default="1280x720")
args = parser.parse_args()

# GPIO pin setup
BUZZER_PIN = 17  # GPIO pin for buzzer
LED_PIN = 18     # GPIO pin for LED

# Create GPIO objects
buzzer = gpiozero.Buzzer(BUZZER_PIN)
led = gpiozero.LED(LED_PIN)

# Initialize GPIO states
buzzer.off()
led.off()

# Function to activate alarm for specified duration
def activate_alarm(duration=10):
    buzzer.on()
    led.on()
    print(f"🚨 DROWNING DETECTED! Alarm activated for {duration} seconds")
    time.sleep(duration)
    buzzer.off()
    led.off()
    print("✅ Alarm deactivated")

# Parse resolution
resolution = args.resolution.split("x")
resW, resH = int(resolution[0]), int(resolution[1])

# Check if model file exists
if not os.path.exists(args.model):
    print(f'❌ ERROR: Model path "{args.model}" is invalid or model was not found.')
    sys.exit(0)

# Load the YOLO model
model = YOLO(args.model)
labels = model.names
print(f"✅ Loaded YOLO model from {args.model}")
print(f"📊 Detection classes: {labels}")

# Initialize Picamera2
if args.source != "picamera0":
    print(f'⚠️ Only Raspberry Pi camera is supported. Ignoring "{args.source}" and using picamera0.')

picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"format": "RGB888", "size": (resW, resH)}))
picam.start()
print(f"📷 Camera initialized at resolution {resW}x{resH}")
time.sleep(2)  # Give camera time to warm up

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize FPS calculation variables
frame_rate_buffer = []
fps_avg_len = 30
avg_frame_rate = 0

# Alarm state control
alarm_active = False
alarm_thread = None

print("🔍 Starting drowning detection... Press 'q' to quit, 's' to pause")

# Main detection loop
while True:
    t_start = time.perf_counter()
    
    # Capture frame
    frame = picam.capture_array()
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    # Extract detections
    detections = results[0].boxes
    
    # Initialize counter for drowning detections
    drowning_count = 0
    total_objects = 0
    
    # Process each detection
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        
        # Get class ID, name and confidence
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        
        # Draw box if confidence threshold is high enough
        if conf > 0.5:
            total_objects += 1
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Check if detection is drowning
            if classname.lower() == "drowning":
                drowning_count += 1
                # Highlight drowning detection with a thicker box
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,0,255), 4)
    
    # Activate alarm if drowning is detected and alarm is not already active
    if drowning_count > 0 and not alarm_active:
        alarm_active = True
        alarm_thread = Thread(target=activate_alarm)
        alarm_thread.daemon = True
        alarm_thread.start()
        
        # Draw drowning alert on screen
        alert_text = "!!! DROWNING DETECTED !!!"
        cv2.putText(frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
        cv2.putText(frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Reset alarm_active flag when alarm thread is finished
    if alarm_active and not alarm_thread.is_alive():
        alarm_active = False
    
    # Calculate and display FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1.0 / (t_stop - t_start)
    
    # Update FPS buffer
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)
    
    # Draw FPS and object count on frame
    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Objects: {total_objects}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Drowning: {drowning_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display alarm status
    if alarm_active:
        cv2.putText(frame, "ALARM ACTIVE", (resW-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Drowning Detection", frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        print("👋 Exiting program...")
        break
    elif key == ord('s'):  # Press 's' to pause
        print("⏸️ Program paused. Press any key to resume.")
        cv2.waitKey(0)
        print("▶️ Program resumed.")
    elif key == ord('p'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('drowning_capture.png', frame)
        print("📸 Frame captured and saved as 'drowning_capture.png'")

# Clean up
print(f"📊 Average FPS: {avg_frame_rate:.2f}")
picam.close()
cv2.destroyAllWindows()
