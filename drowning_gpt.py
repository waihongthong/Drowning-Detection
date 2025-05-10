import os
import sys
import argparse
import time
import gpiozero
from threading import Thread
import RPi.GPIO as GPIO  # Adding direct GPIO access for troubleshooting

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
BUZZER_PIN = 23  # GPIO pin for buzzer
LED_PIN = 17     # GPIO pin for LED

# Initialize both GPIO libraries for redundancy
# 1. Using gpiozero (high-level)
try:
    buzzer = gpiozero.Buzzer(BUZZER_PIN)
    led = gpiozero.LED(LED_PIN)
    buzzer.off()
    led.off()
except Exception as e:
    print(f"⚠️ gpiozero setup error: {e}")

# 2. Using RPi.GPIO directly (lower-level fallback)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(LED_PIN, GPIO.LOW)

print("🔌 Testing GPIO devices...")
# Test both devices quickly at startup
try:
    # Test LED
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(LED_PIN, GPIO.LOW)
    
    # Test Buzzer
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.2) 
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    
    print("✅ GPIO devices tested successfully")
except Exception as e:
    print(f"⚠️ GPIO test error: {e}")


# Function to activate alarm for specified duration
def activate_alarm(duration=10):
    try:
        # Try using gpiozero first
        buzzer.on()
        led.on()
    except:
        # Fallback to direct GPIO control
        pass
    
    # Always use direct GPIO control to ensure devices activate
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    GPIO.output(LED_PIN, GPIO.HIGH)
    
    print(f"🚨 DROWNING DETECTED! Alarm activated for {duration} seconds")
    
    # Keep alarm on for the duration
    time.sleep(duration)
    
    # Turn off devices
    try:
        buzzer.off()
        led.off()
    except:
        pass
    
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.LOW)
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

print("📷 Initializing camera...")
picam = Picamera2()

# Use lowest latency configuration available
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (resW, resH)},
    buffer_count=4,  # More buffers can help with latency
    queue=False,     # Queue=False for lower latency
    controls={"FrameDurationLimits": (33333, 33333)}  # Force ~30fps
)

picam.configure(config)
picam.start()
print(f"📷 Camera initialized at resolution {resW}x{resH}")

# Initialize camera and warm up
start_time = time.time()
while time.time() - start_time < 2:
    # Capture and discard frames during warmup to stabilize the pipeline
    _ = picam.capture_array()
    
print("✅ Camera ready")

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

# Add a debugging function to manually trigger the alarm
def manual_alarm_test():
    print("🧪 Testing alarm manually...")
    test_thread = Thread(target=activate_alarm, args=(2,))
    test_thread.daemon = True
    test_thread.start()

# Main detection loop
print("⚙️ Optimizing for low latency...")

while True:
    t_start = time.perf_counter()
    
    # Capture frame with lowest possible latency
    frame = picam.capture_array()
    
    # Run YOLO detection with optimized settings
    results = model(frame, verbose=False, conf=0.4)  # Lower confidence threshold for better sensitivity
    
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
    
    # Check for key presses - use a shorter wait time for better responsiveness
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
    elif key == ord('t'):  # Press 't' to test the alarm (without drowning detection)
        manual_alarm_test()

# Clean up
print(f"📊 Average FPS: {avg_frame_rate:.2f}")
try:
    buzzer.off()
    led.off()
except:
    pass
    
# Ensure GPIO pins are properly cleaned up
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(LED_PIN, GPIO.LOW)
GPIO.cleanup()

picam.close()
cv2.destroyAllWindows()
print("👋 Program terminated successfully")
