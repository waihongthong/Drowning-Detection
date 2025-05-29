import os
import sys
import argparse
import time
import threading
import requests
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

# Web server communication
SERVER_URL = "http://localhost:5000/api"

def update_server_status(**kwargs):
    """Send status updates to web server"""
    try:
        response = requests.post(f"{SERVER_URL}/status/update", 
                               json=kwargs, timeout=1)
    except requests.exceptions.RequestException:
        pass  # Server might not be running, continue silently

def trigger_server_alarm(duration=10):
    """Trigger alarm through web server"""
    try:
        response = requests.post(f"{SERVER_URL}/alarm/trigger", 
                               json={'duration': duration}, timeout=2)
    except requests.exceptions.RequestException:
        pass

# Define and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "best_ncnn_model")',
                    required=True)
parser.add_argument('--source', help='Video source, should be "picamera0" for Raspberry Pi camera',
                    default="picamera0")
parser.add_argument('--resolution', help='Resolution in WxH format (example: "640x480")',
                    default="640x480")
parser.add_argument('--confidence', help='Confidence threshold for detections (0.0-1.0)',
                    type=float, default=0.6)
parser.add_argument('--consecutive', help='Number of consecutive detections required to trigger alarm',
                    type=int, default=5)
parser.add_argument('--min-area', help='Minimum detection area in pixels',
                    type=int, default=2000)
parser.add_argument('--max-area', help='Maximum detection area in pixels',
                    type=int, default=200000)
parser.add_argument('--headless', help='Run without display (for server mode)',
                    action='store_true')
args = parser.parse_args()

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

# Initialize Picamera2
if args.source != "picamera0":
    print(f'⚠️ Only Raspberry Pi camera is supported. Ignoring "{args.source}" and using picamera0.')

picam = Picamera2()
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (resW, resH)},
    buffer_count=2
)
picam.configure(config)

if CONTROLS_AVAILABLE:
    try:
        controls = Controls(picam)
        controls.FrameDurationLimits = (33333, 33333)
        controls.FrameRate = 30.0
        controls.AeEnable = True
        controls.AwbEnable = True
        print("✅ Camera controls configured for optimal performance")
    except Exception as e:
        print(f"⚠️ Could not set camera controls: {e}")

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

# Start camera
print(f"📷 Initializing camera at resolution {resW}x{resH}...")
picam.start(show_preview=False)

# Warm up camera
print("Warming up camera...")
for _ in range(5):
    try:
        frame = picam.capture_array()
        if frame is not None and len(frame.shape) == 3:
            print(f"Camera warmed up with frame shape: {frame.shape}")
        time.sleep(0.1)
    except Exception as e:
        print(f"Warning during warmup: {e}")
        time.sleep(0.2)

print("📷 Camera initialized and ready")

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize variables
frame_rate_buffer = []
fps_avg_len = 10
avg_frame_rate = 0
consecutive_drowning = 0
DROWNING_THRESHOLD = args.consecutive
drowning_confidences = []
MIN_AVG_CONFIDENCE = 0.7
alarm_active = False
alarm_thread = None

# Setup processing queue
frame_queue = Queue(maxsize=2)
process_results_queue = Queue(maxsize=2)
processing_active = True

def process_frames():
    """Process frames in separate thread"""
    while processing_active:
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get()
            if frame is None:
                break
                
            try:
                results = model(frame, verbose=False, conf=args.confidence)
                filtered_detections = filter_detections(results[0].boxes)
                
                if not process_results_queue.full():
                    process_results_queue.put((frame, filtered_detections))
            except Exception as e:
                print(f"Error processing frame: {e}")
                if not process_results_queue.full():
                    process_results_queue.put((frame, []))
                    
            frame_queue.task_done()
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(0.1)

# Start processing thread
processing_thread = Thread(target=process_frames, daemon=True)
processing_thread.start()

print("🔍 Starting drowning detection...")
if not args.headless:
    print("Press 'q' to quit, 's' to pause")

# Main detection loop
running = True
status_update_counter = 0

while running:
    t_start = time.perf_counter()
    
    try:
        frame = picam.capture_array()
        
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        if not frame_queue.full():
            frame_queue.put(frame.copy())
    except Exception as e:
        print(f"Error capturing frame: {e}")
        time.sleep(0.1)
        continue
    
    # Process results
    if not process_results_queue.empty():
        processed_frame, detections = process_results_queue.get()
        
        drowning_count = 0
        total_objects = 0
        current_drowning_confidences = []
        
        # Process detections
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            
            if conf > args.confidence:
                total_objects += 1
                
                if not args.headless:
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(processed_frame, (xmin,ymin), (xmax,ymax), color, 2)
                    
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(processed_frame, (xmin, label_ymin-labelSize[1]-10), 
                                 (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(processed_frame, label, (xmin, label_ymin-7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                if classname.lower() == "drowning":
                    drowning_count += 1
                    current_drowning_confidences.append(conf)
                    if not args.headless:
                        cv2.rectangle(processed_frame, (xmin,ymin), (xmax,ymax), (0,0,255), 4)
        
        # Update consecutive drowning counter
        if drowning_count > 0:
            consecutive_drowning += 1
            avg_conf = sum(current_drowning_confidences) / len(current_drowning_confidences)
            drowning_confidences.append(avg_conf)
            if len(drowning_confidences) > DROWNING_THRESHOLD:
                drowning_confidences.pop(0)
        else:
            consecutive_drowning = 0
            drowning_confidences = []
        
        # Calculate average confidence
        avg_confidence_threshold = 0
        if len(drowning_confidences) > 0:
            avg_confidence_threshold = sum(drowning_confidences) / len(drowning_confidences)
        
        # Check for drowning alarm
        drowning_detected = (consecutive_drowning >= DROWNING_THRESHOLD and 
                           len(drowning_confidences) >= DROWNING_THRESHOLD and 
                           avg_confidence_threshold >= MIN_AVG_CONFIDENCE)
        
        if drowning_detected and not alarm_active:
            alarm_active = True
            
            # Save detection image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'drowning_detected_{timestamp}.png'
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Drowning detection saved as '{filename}'")
            
            # Trigger alarm through server
            trigger_server_alarm(10)
            
            if not args.headless:
                alert_text = "!!! DROWNING DETECTED !!!"
                cv2.putText(processed_frame, alert_text, (int(resW/2)-150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
                cv2.putText(processed_frame, alert_text, (int(resW/2)-150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Reset alarm flag
        if alarm_active and not drowning_detected:
            alarm_active = False
        
        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = 1.0 / (t_stop - t_start)
        
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)
        
        # Update server status every 30 frames
        status_update_counter += 1
        if status_update_counter >= 30:
            update_server_status(
                is_detecting=True,
                drowning_detected=drowning_detected,
                last_detection_time=time.time() if drowning_detected else None,
                consecutive_detections=consecutive_drowning,
                confidence=avg_confidence_threshold,
                alarm_active=alarm_active,
                total_objects=total_objects,
                fps=avg_frame_rate
            )
            status_update_counter = 0
        
        if not args.headless:
            # Draw status info
            cv2.putText(processed_frame, f'FPS: {avg_frame_rate:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f'Objects: {total_objects}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f'Drowning: {drowning_count}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f'Cons. Drowning: {consecutive_drowning}/{DROWNING_THRESHOLD}', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f'Conf. Threshold: {avg_confidence_threshold:.2f}/{MIN_AVG_CONFIDENCE}', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if alarm_active:
                cv2.putText(processed_frame, "ALARM ACTIVE", (resW-200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Drowning Detection", processed_frame)
    elif not args.headless:
        cv2.imshow("Drowning Detection", frame)
    
    if not args.headless:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Exiting program...")
            running = False
            break
        elif key == ord('s'):
            print("⏸️ Program paused. Press any key to resume.")
            cv2.waitKey(0)
            print("▶️ Program resumed.")
        elif key == ord('p'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'drowning_capture_{timestamp}.png'
            cv2.imwrite(filename, processed_frame if 'processed_frame' in locals() else frame)
            print(f"📸 Frame captured and saved as '{filename}'")

# Clean up
processing_active = False
if not frame_queue.full():
    frame_queue.put(None)
processing_thread.join(timeout=1.0)

print(f"📊 Average FPS: {avg_frame_rate:.2f}")
picam.close()
if not args.headless:
    cv2.destroyAllWindows()
