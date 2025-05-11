import os
import sys
import argparse
import time
import threading
from threading import Thread, Lock
from queue import Queue

import cv2
import numpy as np
from picamera2 import Picamera2
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
                    type=float, default=0.6)  # Increased from 0.4 to reduce false positives
parser.add_argument('--consecutive', help='Number of consecutive detections required to trigger alarm',
                    type=int, default=5)  # Increased from 2 for better reliability
parser.add_argument('--min-area', help='Minimum detection area in pixels',
                    type=int, default=2000)  # Minimum area filter
parser.add_argument('--max-area', help='Maximum detection area in pixels',
                    type=int, default=200000)  # Maximum area filter
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
print(f"🔄 Consecutive detections required: {args.consecutive}")
print(f"📏 Detection area filter: Min={args.min_area}, Max={args.max_area} pixels")

# Initialize Picamera2 with optimized settings for lower latency
if args.source != "picamera0":
    print(f'⚠️ Only Raspberry Pi camera is supported. Ignoring "{args.source}" and using picamera0.')

picam = Picamera2()

# Create optimized configuration for lower latency
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (resW, resH)},
    buffer_count=1  # Minimal buffer for lower latency
)

# Configure camera
picam.configure(config)

# Set controls for better performance if available
if CONTROLS_AVAILABLE:
    try:
        controls = Controls(picam)
        controls.FrameDurationLimits = (33333, 33333)  # Force ~30fps (in microseconds)
        controls.FrameRate = 30.0
        # Add auto-exposure and auto-white-balance for better image quality
        controls.AeEnable = True
        controls.AwbEnable = True
        print("✅ Camera controls configured for optimal performance")
    except Exception as e:
        print(f"⚠️ Could not set camera controls: {e}")
        print("⚠️ Using default camera settings")
else:
    print("⚠️ Camera controls not available. Using default settings.")

# Function to filter detections based on size
def filter_detections(detections, min_area=args.min_area, max_area=args.max_area):
    filtered = []
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        if min_area <= area <= max_area:
            filtered.append(detections[i])
    return filtered

# Start camera with minimal delays
print(f"📷 Initializing camera at resolution {resW}x{resH}...")
picam.start(show_preview=False)  # Don't show preview during startup

# Shorter warm-up sequence
print("Warming up camera...")
for _ in range(3):
    picam.capture_array()
    time.sleep(0.05)

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

# Initialize counter for consecutive drowning detections
consecutive_drowning = 0
DROWNING_THRESHOLD = args.consecutive

# Add confidence tracking for more robust detection
drowning_confidences = []
MIN_AVG_CONFIDENCE = 0.7  # Minimum average confidence to trigger alarm

# Setup processing queue for threaded processing
frame_queue = Queue(maxsize=2)
process_results_queue = Queue(maxsize=2)
processing_active = True

# Function to process frames in a separate thread
def process_frames():
    while processing_active:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
            
        frame = frame_queue.get()
        if frame is None:  # None is sentinel to exit
            break
            
        # Process frame with YOLO
        results = model(frame, verbose=False, conf=args.confidence)
        
        # Filter detections by size
        filtered_detections = filter_detections(results[0].boxes)
        
        # Put results in queue for main thread to display
        process_results_queue.put((frame, filtered_detections))
        frame_queue.task_done()

# Start processing thread
processing_thread = Thread(target=process_frames, daemon=True)
processing_thread.start()

print("🔍 Starting drowning detection... Press 'q' to quit, 's' to pause")

# Main detection loop
running = True
while running:
    t_start = time.perf_counter()
    
    # Capture frame
    frame = picam.capture_array()
    
    # Put frame in queue for processing thread
    if not frame_queue.full():
        frame_queue.put(frame.copy())
    
    # Get processed results if available
    if not process_results_queue.empty():
        processed_frame, detections = process_results_queue.get()
        
        # Initialize counter for drowning detections
        drowning_count = 0
        total_objects = 0
        current_drowning_confidences = []
        
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
                cv2.rectangle(processed_frame, (xmin,ymin), (xmax,ymax), color, 2)
                
                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(processed_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(processed_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Check if detection is drowning
                if classname.lower() == "drowning":
                    drowning_count += 1
                    current_drowning_confidences.append(conf)
                    # Highlight drowning detection with a thicker box
                    cv2.rectangle(processed_frame, (xmin,ymin), (xmax,ymax), (0,0,255), 4)
        
        # Update consecutive drowning counter with confidence checking
        if drowning_count > 0:
            consecutive_drowning += 1
            # Calculate average confidence of current detections
            avg_conf = sum(current_drowning_confidences) / len(current_drowning_confidences)
            drowning_confidences.append(avg_conf)
            if len(drowning_confidences) > DROWNING_THRESHOLD:
                drowning_confidences.pop(0)
        else:
            consecutive_drowning = 0
            drowning_confidences = []
        
        # Calculate average confidence over threshold period
        avg_confidence_threshold = 0
        if len(drowning_confidences) > 0:
            avg_confidence_threshold = sum(drowning_confidences) / len(drowning_confidences)
        
        # Activate alarm if drowning is detected for consecutive frames with sufficient confidence
        # and alarm is not already active
        if (consecutive_drowning >= DROWNING_THRESHOLD and 
            len(drowning_confidences) >= DROWNING_THRESHOLD and 
            avg_confidence_threshold >= MIN_AVG_CONFIDENCE and 
            not alarm_active):
            
            alarm_active = True
            
            # Save drowning detection image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'drowning_detected_{timestamp}.png'
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Drowning detection captured and saved as '{filename}'")
            
            # Start alarm thread
            alarm_thread = Thread(target=activate_alarm)
            alarm_thread.daemon = True
            alarm_thread.start()
            
            # Draw drowning alert on screen
            alert_text = "!!! DROWNING DETECTED !!!"
            cv2.putText(processed_frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
            cv2.putText(processed_frame, alert_text, (int(resW/2)-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
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
        cv2.putText(processed_frame, f'FPS: {avg_frame_rate:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f'Objects: {total_objects}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f'Drowning: {drowning_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display consecutive drowning count and confidence
        cv2.putText(processed_frame, f'Cons. Drowning: {consecutive_drowning}/{DROWNING_THRESHOLD}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f'Conf. Threshold: {avg_confidence_threshold:.2f}/{MIN_AVG_CONFIDENCE}', (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display alarm status
        if alarm_active:
            cv2.putText(processed_frame, "ALARM ACTIVE", (resW-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the processed frame
        cv2.imshow("Drowning Detection", processed_frame)
    else:
        # If no processed frame yet, show raw frame
        cv2.imshow("Drowning Detection", frame)
    
    # Check for key presses with minimal blocking
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        print("👋 Exiting program...")
        running = False
        break
    elif key == ord('s'):  # Press 's' to pause
        print("⏸️ Program paused. Press any key to resume.")
        cv2.waitKey(0)
        print("▶️ Program resumed.")
    elif key == ord('p'):  # Press 'p' to save a picture of results on this frame
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'drowning_capture_{timestamp}.png'
        cv2.imwrite(filename, processed_frame if 'processed_frame' in locals() else frame)
        print(f"📸 Frame captured and saved as '{filename}'")

# Clean up
processing_active = False
if not frame_queue.full():
    frame_queue.put(None)  # Push sentinel to stop thread
processing_thread.join(timeout=1.0)

print(f"📊 Average FPS: {avg_frame_rate:.2f}")
buzzer.off()  # Ensure buzzer is off when exiting
led.off()     # Ensure LED is off when exiting
picam.close()
cv2.destroyAllWindows()
