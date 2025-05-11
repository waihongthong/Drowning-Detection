import os
import sys
import argparse
import time
from threading import Thread, Lock

import cv2
import numpy as np
from picamera2 import Picamera2
# Remove the Preview import that's causing issues
try:
    from picamera2.controls import Controls
    CONTROLS_AVAILABLE = True
except ImportError:
    CONTROLS_AVAILABLE = False
    print("⚠️ picamera2.controls module not found. Using default camera settings.")

from ultralytics import YOLO

# Try to import gpiozero for hardware control
try:
    import gpiozero
    GPIO_AVAILABLE = True
    print("✅ GPIO control available")
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ gpiozero module not found. Install with: 'pip install gpiozero' or 'sudo apt-get install python3-gpiozero'")
    print("⚠️ Hardware alerts (buzzer and LED) will be simulated")


# Define and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "best_ncnn_model")',
                    required=True)
parser.add_argument('--source', help='Video source, should be "picamera0" for Raspberry Pi camera',
                    default="picamera0")
parser.add_argument('--resolution', help='Resolution in WxH format (example: "1280x720")',
                    default="1280x720")
parser.add_argument('--confidence', help='Confidence threshold for detections (0.0-1.0)',
                    type=float, default=0.4)
args = parser.parse_args()

# GPIO pin setup
BUZZER_PIN = 23  # GPIO pin for buzzer
LED_PIN = 17     # GPIO pin for LED

# Create GPIO objects if available
if GPIO_AVAILABLE:
    buzzer = gpiozero.Buzzer(BUZZER_PIN)
    led = gpiozero.LED(LED_PIN)
    
    # Initialize GPIO states
    buzzer.off()
    led.off()
else:
    # Create dummy class for simulation
    class DummyGPIO:
        def on(self):
            pass
        
        def off(self):
            pass
    
    buzzer = DummyGPIO()
    led = DummyGPIO()

# Thread synchronization lock
alarm_lock = Lock()

# Function to activate alarm for specified duration
def activate_alarm(duration=10):
    with alarm_lock:
        buzzer.on()
        led.on()
        if GPIO_AVAILABLE:
            print(f"🚨 DROWNING DETECTED! Hardware alarm activated for {duration} seconds")
        else:
            print(f"🚨 DROWNING DETECTED! [SIMULATED ALARM] for {duration} seconds")
    
    time.sleep(duration)
    
    with alarm_lock:
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
print(f"🎯 Detection confidence threshold: {args.confidence}")

# Initialize Picamera2 with optimized settings for lower latency
if args.source != "picamera0":
    print(f'⚠️ Only Raspberry Pi camera is supported. Ignoring "{args.source}" and using picamera0.')

picam = Picamera2()

# Create optimized configuration
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (resW, resH)},
    lores={"size": (640, 480)},  # Lower resolution stream for faster processing if needed
    display="lores",
    buffer_count=1  # Reduce buffer count to minimize latency
)

# Configure and optimize camera settings
picam.configure(config)

# Set controls for better performance if available
if CONTROLS_AVAILABLE:
    try:
        controls = Controls(picam)
        controls.FrameDurationLimits = (33333, 33333)  # Force ~30fps (in microseconds)
        controls.FrameRate = 30.0
        print("✅ Camera controls configured for optimal performance")
    except Exception as e:
        print(f"⚠️ Could not set camera controls: {e}")
        print("⚠️ Using default camera settings")
else:
    print("⚠️ Camera controls not available. Using default settings.")

# Start camera with minimal delays
print(f"📷 Initializing camera at resolution {resW}x{resH}...")
picam.start()

# Quick warmup sequence
for _ in range(5):
    # Capture and discard frames to warm up the camera pipeline
    picam.capture_array()
    time.sleep(0.1)

print("📷 Camera initialized and ready")

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

# Initialize counter for consecutive drowning detections to reduce false positives
consecutive_drowning = 0
DROWNING_THRESHOLD = 2  # Number of consecutive frames with drowning to trigger alarm

print("🔍 Starting drowning detection... Press 'q' to quit, 's' to pause")

# Main detection loop
while True:
    t_start = time.perf_counter()
    
    # Capture frame with minimal processing
    frame = picam.capture_array()
    
    # Run YOLO detection with optimized settings
    results = model(frame, verbose=False, conf=args.confidence)
    
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
        if conf > args.confidence:
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
    
    # Update consecutive drowning counter
    if drowning_count > 0:
        consecutive_drowning += 1
    else:
        consecutive_drowning = 0
    
    # Activate alarm if drowning is detected for consecutive frames and alarm is not already active
    if consecutive_drowning >= DROWNING_THRESHOLD and not alarm_active:
        alarm_active = True
        
        # Save drowning detection image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'drowning_detected_{timestamp}.png'
        cv2.imwrite(filename, frame)
        print(f"📸 Drowning detection captured and saved as '{filename}'")
        
        # Start alarm thread
        alarm_thread = Thread(target=activate_alarm)
        alarm_thread.daemon = True
        alarm_thread.start()
        
        # Draw drowning alert on screen
        alert_text = "!!! DROWNING DETECTED !!!"
        cv2.putText(frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
        cv2.putText(frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Reset alarm_active flag when alarm thread is finished
    if alarm_active and (not alarm_thread or not alarm_thread.is_alive()):
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
    
    # Display consecutive drowning count
    cv2.putText(frame, f'Cons. Drowning: {consecutive_drowning}/{DROWNING_THRESHOLD}', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display alarm status
    if alarm_active:
        cv2.putText(frame, "ALARM ACTIVE", (resW-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame with minimal delay
    cv2.imshow("Drowning Detection", frame)
    
    # Check for key presses with minimal blocking
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        print("👋 Exiting program...")
        break
    elif key == ord('s'):  # Press 's' to pause
        print("⏸️ Program paused. Press any key to resume.")
        cv2.waitKey(0)
        print("▶️ Program resumed.")
    elif key == ord('p'):  # Press 'p' to save a picture of results on this frame
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'drowning_capture_{timestamp}.png'
        cv2.imwrite(filename, frame)
        print(f"📸 Frame captured and saved as '{filename}'")

# Clean up
print(f"📊 Average FPS: {avg_frame_rate:.2f}")
buzzer.off()  # Ensure buzzer is off when exiting
led.off()     # Ensure LED is off when exiting
picam.close()
cv2.destroyAllWindows()
