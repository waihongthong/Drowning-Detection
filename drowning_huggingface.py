"""Integrated Drowning Detection Web Server for Raspberry Pi
Provides REST API, video streaming, and real-time drowning detection for Flutter mobile app"""

import os
import json
import time
import threading
import argparse
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from queue import Queue
from pymongo import MongoClient
import base64
from datetime import datetime
from threading import Thread
import os
import requests

CLOUD_API_URL = "https://twhhhh-deepsave.hf.space/detect"  
CLOUD_ENABLED = True
PROCESS_EVERY_N_FRAMES = 3  # Reduced frequency to avoid overloading
CLOUD_TIMEOUT = 8  # Increased timeout for better reliability

# Cloud processing variables
cloud_processing = False
last_cloud_result = None
cloud_frame_counter = 0

MONGODB_URI = "mongodb+srv://waihong0717:0717Waihong!@cluster0.zicfsjv.mongodb.net/fyp" 
DB_NAME = "drowning_detection"
COLLECTION_NAME = "detection_history"
IMAGES_DIR = "detection_images"

try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("✅ MongoDB connection established")
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    MONGODB_AVAILABLE = False

# Create images directory if it doesn't exist
os.makedirs(IMAGES_DIR, exist_ok=True)

try:
    from picamera2.controls import Controls
    CONTROLS_AVAILABLE = True
except ImportError:
    CONTROLS_AVAILABLE = False

# Try to import gpiozero for hardware control
try:
    import gpiozero
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Global variables
camera = None
model = None
detection_active = False
camera_active = False
detection_status = {
    'is_detecting': False,
    'drowning_detected': False,
    'last_detection_time': None,
    'consecutive_detections': 0,
    'confidence': 0.0,
    'alarm_active': False,
    'total_objects': 0,
    'fps': 0.0
}

# Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
STREAM_FPS = 30

# Detection Configuration (can be modified via API)
detection_config = {
    'confidence_threshold': 0.7,
    'consecutive_threshold': 2,
    'min_area': 2000,
    'max_area': 200000,
    'min_avg_confidence': 0.7
}

# GPIO setup
BUZZER_PIN = 18
LED_PIN = 17

if GPIO_AVAILABLE:
    buzzer = gpiozero.Buzzer(BUZZER_PIN)
    led = gpiozero.LED(LED_PIN)
    buzzer.off()
    led.off()
else:
    class DummyGPIO:
        def on(self): pass
        def off(self): pass
    buzzer = DummyGPIO()
    led = DummyGPIO()

# Thread locks
camera_lock = threading.Lock()
status_lock = threading.Lock()
model_lock = threading.Lock()

# Detection state variables
consecutive_drowning = 0
drowning_confidences = []
frame_rate_buffer = []
fps_avg_len = 10

# Processing queue
processed_frame_queue = Queue(maxsize=2)

def send_frame_to_cloud(frame, frame_id):
    """Send frame to cloud API asynchronously"""
    global last_cloud_result, cloud_processing
    
    try:
        cloud_processing = True
        
        # Fix 1: Consistent resize to match cloud API expectations
        # Test with your Hugging Face space to confirm the expected input size
        small_frame = cv2.resize(frame, (640, 480)) 
        
        # Encode as JPEG with higher quality for better detection
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Increased from 75
        ret, buffer = cv2.imencode('.jpg', small_frame, encode_params)
        
        if not ret:
            print("❌ Failed to encode frame")
            return
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "data": img_base64,
            "confidence_threshold": detection_config['confidence_threshold']
        }
        
        # Try primary endpoint first
        try:
            print(f"🔄 Sending to: {CLOUD_API_URL}")
            
            response = requests.post(
                    CLOUD_API_URL,
                    json=payload,
                    timeout=CLOUD_TIMEOUT,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"📊 Raw response: {result}")
                    
                    # Fix 3: Enhanced result processing
                    result['frame_id'] = frame_id
                    result['timestamp'] = time.time()
                    result['original_frame_shape'] = frame.shape
                    result['endpoint_used'] = CLOUD_API_URL
                    result['processed_frame_size'] = small_frame.shape
                    
                    last_cloud_result = result
                    
                    # Better detection counting
                    detections = result.get('detections', result.get('predictions', []))
                    print(f"✅ Cloud processed frame {frame_id}: {len(detections)} detections")
                    
                    return True
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"❌ Response text: {response.text[:500]}")
                    
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout after {CLOUD_TIMEOUT} seconds")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error to {primary_endpoint['url']}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
        
        # If primary endpoint fails, create fallback result
        print(f"❌ Cloud endpoint failed for frame {frame_id}")
        last_cloud_result = {
            'success': False,
            'detections': [],
            'frame_id': frame_id,
            'timestamp': time.time(),
            'error': 'Cloud API request failed'
        }
        return False
            
    except Exception as e:
        print(f"❌ Critical cloud request error: {e}")
        last_cloud_result = {
            'success': False,
            'detections': [],
            'frame_id': frame_id,
            'timestamp': time.time(),
            'error': str(e)
        }
        return False
    finally:
        cloud_processing = False
        
def apply_cloud_detections_to_frame(frame, cloud_result):
    """Apply cloud detection results to current frame"""
    if not cloud_result:
        cv2.putText(frame, 'Cloud: No result', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame, 0, 0, False
    
    # Fix 4: Better error handling and result parsing
    if not cloud_result.get('success', True):
        error_msg = cloud_result.get('error', 'Unknown error')
        cv2.putText(frame, f'Cloud Error: {error_msg[:20]}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return frame, 0, 0, False
    
    # Try different response formats from Hugging Face
    detections = []
    if 'detections' in cloud_result:
        detections = cloud_result['detections']
    elif 'predictions' in cloud_result:
        detections = cloud_result['predictions']
    elif 'results' in cloud_result:
        detections = cloud_result['results']
    elif isinstance(cloud_result, list):
        detections = cloud_result
    
    if not detections:
        cv2.putText(frame, 'Cloud: No detections', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame, 0, 0, False
    
    drowning_count = 0
    total_objects = len(detections)
    
    # Fix 5: Improved coordinate scaling
    processed_size = cloud_result.get('processed_frame_size', [320, 320])
    if len(processed_size) >= 2:
        scale_x = frame.shape[1] / processed_size[1]
        scale_y = frame.shape[0] / processed_size[0]
    else:
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
    
    print(f"📊 Processing {len(detections)} detections with scale ({scale_x:.2f}, {scale_y:.2f})")
    
    for i, detection in enumerate(detections):
        try:
            # Handle different coordinate formats
            coords = None
            confidence = 0.0
            class_name = 'unknown'
            
            # Parse coordinates
            if 'bbox' in detection and isinstance(detection['bbox'], list):
                coords = detection['bbox']
            elif 'box' in detection:
                coords = detection['box']
            elif all(k in detection for k in ['x1', 'y1', 'x2', 'y2']):
                coords = [detection['x1'], detection['y1'], detection['x2'], detection['y2']]
            elif all(k in detection for k in ['x', 'y', 'w', 'h']):
                x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
                coords = [x - w/2, y - h/2, x + w/2, y + h/2]
            
            if not coords or len(coords) < 4:
                print(f"⚠️ Invalid coordinates in detection {i}: {detection}")
                continue
            
            # Parse confidence
            confidence = detection.get('confidence', detection.get('score', 0.0))
            
            # Parse class
            class_name = detection.get('class', detection.get('label', 'object'))
            
            # Scale coordinates
            x1, y1, x2, y2 = coords
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            # Check if detection is drowning
            is_drowning = any(keyword in class_name.lower() for keyword in ['drown', 'drowning'])
            
            # Draw bounding box
            color = (0, 0, 255) if is_drowning else (0, 255, 0)
            thickness = 3 if is_drowning else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Ensure label fits in frame
            label_y = max(y1, label_size[1] + 5)
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(frame, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Count drowning detections
            if is_drowning and confidence > detection_config['confidence_threshold']:
                drowning_count += 1
                print(f"🚨 Drowning detected: {class_name} with confidence {confidence:.2f}")
                
        except Exception as e:
            print(f"❌ Error processing detection {i}: {e}")
            print(f"❌ Detection data: {detection}")
            continue
    
    # Determine if drowning detected
    drowning_detected = drowning_count > 0
    
    # Add status overlay
    cv2.putText(frame, f'Cloud: {total_objects} objects, {drowning_count} drowning', 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame, drowning_count, total_objects, drowning_detected

def load_yolo_model(model_path):
    """Load the YOLO model for drowning detection"""
    global model
    try:
        if model_path.endswith('.pt'):
            if not os.path.exists(model_path):
                print(f'❌ ERROR: Model path "{model_path}" is invalid or model was not found.')
                return False
                
            with model_lock:
                model = YOLO(model_path)
                labels = model.names
                print(f"✅ Loaded YOLO model from {model_path}")
                print(f"📊 Detection classes: {labels}")
                return True
        
        # Handle NCNN format
        elif model_path.endswith('.param') or 'ncnn' in model_path.lower():
            print("⚠️ NCNN format detected. This requires OpenCV DNN backend.")
            print("💡 For hybrid cloud mode, local model is optional.")
            print("🌐 Cloud processing will handle AI detection.")
            return False
            
        else:
            print(f"❌ Unsupported model format: {model_path}")
            print("💡 Supported formats: .pt (PyTorch)")
            return False
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def filter_detections(detections, min_area=None, max_area=None):
    """Filter detections based on size"""
    if min_area is None:
        min_area = detection_config['min_area']
    if max_area is None:
        max_area = detection_config['max_area']
        
    filtered = []
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        if min_area <= area <= max_area:
            filtered.append(detections[i])
    return filtered

def initialize_camera():
    """Initialize Picamera2 for streaming"""
    global camera, camera_active
    try:
        with camera_lock:
            if camera is not None:
                return True
                
            print("🔄 Initializing camera...")
            camera = Picamera2()
            
            # Try different configurations if the first one fails
            try:
                config = camera.create_preview_configuration(
                    main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                    buffer_count=4  # Increased buffer count
                )
                camera.configure(config)
            except Exception as e:
                print(f"Failed with RGB888, trying XRGB8888: {e}")
                try:
                    config = camera.create_preview_configuration(
                        main={"format": "XRGB8888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                        buffer_count=4
                    )
                    camera.configure(config)
                except Exception as e2:
                    print(f"Failed with XRGB8888, trying YUV420: {e2}")
                    config = camera.create_preview_configuration(
                        main={"format": "YUV420", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                        buffer_count=4
                    )
                    camera.configure(config)
            
            if CONTROLS_AVAILABLE:
                try:
                    controls = Controls(camera)
                    controls.FrameDurationLimits = (33333, 66666)  # 15-30fps range
                    controls.FrameRate = 20.0
                    controls.AeEnable = True
                    controls.AwbEnable = True
                except Exception as e:
                    print(f"Could not set camera controls: {e}")
            
            camera.start(show_preview=False)
            
            # Warm up camera with more attempts
            print("🔄 Warming up camera...")
            for i in range(10):
                try:
                    frame = camera.capture_array()
                    if frame is not None and frame.size > 0:
                        print(f"✅ Camera frame {i+1}/10 captured successfully")
                        time.sleep(0.2)
                    else:
                        print(f"⚠️ Empty frame {i+1}/10")
                        time.sleep(0.5)
                except Exception as e:
                    print(f"⚠️ Frame {i+1}/10 capture failed: {e}")
                    time.sleep(0.5)
                    
            camera_active = True
            print("✅ Camera initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Failed to initialize camera: {e}")
        camera_active = False
        return False

def cleanup_camera():
    """Clean up camera resources"""
    global camera, camera_active
    try:
        with camera_lock:
            if camera is not None:
                camera.close()
                camera = None
                camera_active = False
                print("✅ Camera cleaned up")
    except Exception as e:
        print(f"Error cleaning up camera: {e}")

def process_detection(frame):
    """Process frame for drowning detection"""
    global consecutive_drowning, drowning_confidences, model
    
    if model is None:
        cv2.putText(frame, 'LOCAL MODEL: OFF', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, 'CLOUD MODE: ACTIVE', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, 0, 0, False
    
    try:
        with model_lock:
            results = model(frame, verbose=False, conf=detection_config['confidence_threshold'])
        
        # Filter detections by size
        filtered_detections = filter_detections(results[0].boxes)
        
        # Process detections
        drowning_count = 0
        total_objects = 0
        current_drowning_confidences = []
        
        # Set bounding box colors
        bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                      (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
        
        # Process each detection
        for i in range(len(filtered_detections)):
            # Get bounding box coordinates
            xyxy_tensor = filtered_detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            # Get class ID, name and confidence
            classidx = int(filtered_detections[i].cls.item())
            classname = model.names[classidx]
            conf = filtered_detections[i].conf.item()
            
            # Draw box if confidence threshold is high enough
            if conf > detection_config['confidence_threshold']:
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
                    current_drowning_confidences.append(conf)
                    
                    # 🚨 IMMEDIATE GPIO RESPONSE FOR HIGH-CONFIDENCE DETECTIONS
                    if conf > 0.8:
                        # Immediate 1-second pulse for any high-confidence detection
                        buzzer.on()
                        led.on()
                        threading.Timer(1.0, lambda: [buzzer.off(), led.off()]).start()
                        print(f"⚡ IMMEDIATE ALERT: High confidence drowning detected ({conf:.2f})")
                    
                    # Highlight drowning detection with a thicker box
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,0,255), 4)
        
        # Update consecutive drowning counter with confidence checking
        if drowning_count > 0:
            consecutive_drowning += 1
            # Calculate average confidence of current detections
            avg_conf = sum(current_drowning_confidences) / len(current_drowning_confidences)
            drowning_confidences.append(avg_conf)
            if len(drowning_confidences) > detection_config['consecutive_threshold']:
                drowning_confidences.pop(0)
        else:
            consecutive_drowning = 0
            drowning_confidences = []
        
        # Calculate average confidence over threshold period
        avg_confidence_threshold = 0
        if len(drowning_confidences) > 0:
            avg_confidence_threshold = sum(drowning_confidences) / len(drowning_confidences)
        
        # Check if drowning should be triggered (for sustained alarm)
        drowning_detected = (consecutive_drowning >= detection_config['consecutive_threshold'] and 
                           len(drowning_confidences) >= detection_config['consecutive_threshold'] and 
                           avg_confidence_threshold >= detection_config['min_avg_confidence'])
        
        return frame, drowning_count, total_objects, drowning_detected
        
    except Exception as e:
        print(f"Error processing detection: {e}")
        return frame, 0, 0, False

def generate_test_frame():
    """Generate a test frame when camera is not available"""
    # Create a simple test image
    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    frame[:] = (64, 128, 64)  # Dark green background
    
    # Add some text
    cv2.putText(frame, 'Camera Test Frame', (50, CAMERA_HEIGHT//2 - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Time: {datetime.now().strftime("%H:%M:%S")}', 
                (50, CAMERA_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}', 
                (50, CAMERA_HEIGHT//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def save_detection_to_database(image_path, detection_data):
    """Save detection data and image to MongoDB"""
    if not MONGODB_AVAILABLE:
        return None
    
    try:
        # Read and encode image as base64
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_filename': os.path.basename(image_path),
            'image_data': image_data,  # Base64 encoded image
            'confidence': detection_data.get('confidence', 0.0),
            'consecutive_detections': detection_data.get('consecutive_detections', 0),
            'total_objects': detection_data.get('total_objects', 0),
            'message': 'Drowning detected by AI system',
            'camera_resolution': f'{CAMERA_WIDTH}x{CAMERA_HEIGHT}',
            'detection_config': detection_config.copy()
        }
        
        # Insert into MongoDB
        result = collection.insert_one(detection_record)
        print(f"✅ Detection saved to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Error saving to MongoDB: {e}")
        return None

def generate_frames():
    """Generate video frames for streaming with detection overlay"""
    global camera, detection_status, cloud_frame_counter, last_cloud_result
    
    frame_count = 0
    last_frame_time = time.time()
    last_cloud_time = 0
    
    while True:
        try:
            # Capture frame
            frame = None
            if camera_active and camera is not None:
                with camera_lock:
                    frame = camera.capture_array()
                    
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = generate_test_frame()
            else:
                frame = generate_test_frame()
            
            if frame is None:
                frame = generate_test_frame()
            
            # Initialize detection variables
            drowning_count = 0
            total_objects = 0
            drowning_detected = False
            
            # Process with cloud API
            if detection_active and CLOUD_ENABLED:
                current_time = time.time()
                
                # Send frame to cloud every N frames and not too frequently
                if (cloud_frame_counter % PROCESS_EVERY_N_FRAMES == 0 and 
                    not cloud_processing and 
                    (current_time - last_cloud_time) > 2.0):  # At least 2 seconds between requests
                    
                    Thread(
                        target=send_frame_to_cloud, 
                        args=(frame.copy(), frame_count),
                        daemon=True
                    ).start()
                    last_cloud_time = current_time
                
                # Apply latest cloud results
                if last_cloud_result:
                    frame, drowning_count, total_objects, drowning_detected = apply_cloud_detections_to_frame(
                        frame, last_cloud_result
                    )
                
                cloud_frame_counter += 1
                
                # Add cloud processing overlay
                if cloud_processing:
                    cv2.putText(frame, 'Cloud: Processing...', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    last_result_age = current_time - last_cloud_result.get('timestamp', 0) if last_cloud_result else 999
                    if last_result_age < 30:  # Show age if result is recent
                        cv2.putText(frame, f'Cloud: {last_result_age:.1f}s ago', (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            elif detection_active and not CLOUD_ENABLED:
                # Fallback to local processing
                frame, drowning_count, total_objects, drowning_detected = process_detection(frame)
            
            # Update status
            current_time = time.time()
            with status_lock:
                detection_status['total_objects'] = total_objects
                detection_status['is_detecting'] = detection_active
                detection_status['confidence'] = (
                    max([d.get('confidence', 0) for d in last_cloud_result.get('detections', [])], default=0)
                    if last_cloud_result else 0
                )
                
                if drowning_detected:
                    detection_status['drowning_detected'] = True
                    detection_status['last_detection_time'] = current_time
                    detection_status['consecutive_detections'] += 1
                    
                    # Save detection image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'cloud_drowning_{timestamp}.jpg'
                    image_path = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Save to database
                    save_detection_to_database(image_path, {
                        'confidence': detection_status['confidence'],
                        'consecutive_detections': detection_status['consecutive_detections'],
                        'total_objects': total_objects
                    })
                    
                    # Trigger alarm
                    threading.Thread(target=activate_alarm, args=(10,), daemon=True).start()
                else:
                    detection_status['consecutive_detections'] = 0
            
            # Add enhanced overlay information
            overlay_y = CAMERA_HEIGHT - 90
            cv2.putText(frame, f'Cloud: {"ON" if CLOUD_ENABLED else "OFF"}', 
                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if CLOUD_ENABLED else (0, 0, 255), 2)
            
            cv2.putText(frame, f'Objects: {total_objects}', 
                       (10, overlay_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if drowning_detected:
                cv2.putText(frame, '🚨 DROWNING DETECTED', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_fps = 30 / (current_time - last_frame_time) if frame_count > 30 else 0
                detection_status['fps'] = current_fps
                print(f"📊 Frame {frame_count}: FPS: {current_fps:.1f}, Objects: {total_objects}, Drowning: {drowning_detected}")
                last_frame_time = current_time
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                       frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.1)

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    with status_lock:
        status = detection_status.copy()
        status['camera_active'] = camera_active
        return jsonify(status)

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """Start drowning detection"""
    global detection_active
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model loaded. Please load a model first.'}), 400
    
    detection_active = True
    with status_lock:
        detection_status['is_detecting'] = True
    
    return jsonify({'success': True, 'message': 'Drowning detection started'})

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """Stop drowning detection"""
    global detection_active, consecutive_drowning, drowning_confidences
    
    detection_active = False
    consecutive_drowning = 0
    drowning_confidences = []
    
    with status_lock:
        detection_status['is_detecting'] = False
        detection_status['drowning_detected'] = False
        detection_status['consecutive_detections'] = 0
        detection_status['confidence'] = 0.0
    
    return jsonify({'success': True, 'message': 'Drowning detection stopped'})

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load YOLO model for detection"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'success': False, 'error': 'Model path is required'}), 400
        
        if load_yolo_model(model_path):
            return jsonify({'success': True, 'message': f'Model loaded successfully from {model_path}'})
        else:
            return jsonify({'success': False, 'error': 'Failed to load model'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/detection', methods=['GET', 'POST'])
def detection_configuration():
    """Get or update detection configuration"""
    global detection_config
    
    if request.method == 'GET':
        return jsonify(detection_config)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # Update configuration
            if 'confidence_threshold' in data:
                detection_config['confidence_threshold'] = float(data['confidence_threshold'])
            if 'consecutive_threshold' in data:
                detection_config['consecutive_threshold'] = int(data['consecutive_threshold'])
            if 'min_area' in data:
                detection_config['min_area'] = int(data['min_area'])
            if 'max_area' in data:
                detection_config['max_area'] = int(data['max_area'])
            if 'min_avg_confidence' in data:
                detection_config['min_avg_confidence'] = float(data['min_avg_confidence'])
            
            return jsonify({'success': True, 'message': 'Configuration updated', 'config': detection_config})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/device/info')
def get_device_info():
    """Get device information"""
    return jsonify({
        'device_name': 'Raspberry Pi Drowning Detector',
        'ip_address': request.host.split(':')[0],
        'camera_resolution': f'{CAMERA_WIDTH}x{CAMERA_HEIGHT}',
        'gpio_available': GPIO_AVAILABLE,
        'model_loaded': model is not None,
        'detection_active': detection_active,
        'camera_active': camera_active,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/alarm/trigger', methods=['POST'])
def trigger_alarm():
    """Manually trigger alarm for testing"""
    try:
        duration = request.json.get('duration', 5) if request.json else 5
        threading.Thread(target=activate_alarm, args=(duration,), daemon=True).start()
        return jsonify({'success': True, 'message': f'Alarm triggered for {duration} seconds'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm():
    """Stop active alarm"""
    try:
        buzzer.off()
        led.off()
        with status_lock:
            detection_status['alarm_active'] = False
        return jsonify({'success': True, 'message': 'Alarm stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print(f"📹 Video feed requested from {request.remote_addr}")
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera streaming"""
    if initialize_camera():
        return jsonify({'success': True, 'message': 'Camera started'})
    else:
        return jsonify({'success': False, 'error': 'Failed to start camera'}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera streaming"""
    cleanup_camera()
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/camera/restart', methods=['POST'])
def restart_camera():
    """Restart camera"""
    print("🔄 Restarting camera...")
    cleanup_camera()
    time.sleep(2)
    if initialize_camera():
        return jsonify({'success': True, 'message': 'Camera restarted successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to restart camera'}), 500

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test API connection"""
    return jsonify({
        'success': True,
        'message': 'Drowning Detection Server is running',
        'timestamp': datetime.now().isoformat(),
        'server_ip': request.host,
        'model_loaded': model is not None,
        'detection_active': detection_active,
        'camera_active': camera_active
    })

def activate_alarm(duration=10):
    """Activate alarm for specified duration"""
    try:
        buzzer.on()
        led.on()
        with status_lock:
            detection_status['alarm_active'] = True
        
        print(f"🚨 Alarm activated for {duration} seconds")
        time.sleep(duration)
        
        buzzer.off()
        led.off()
        with status_lock:
            detection_status['alarm_active'] = False
        
        print("✅ Alarm deactivated")
    except Exception as e:
        print(f"Error in alarm: {e}")

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'detection_active': detection_active,
        'camera_active': camera_active
    })

@app.route('/api/history')
def get_detection_history():
    """Get detection history from database"""
    try:
        if not MONGODB_AVAILABLE:
            return jsonify({'success': False, 'error': 'Database not available'}), 500
        
        # Get recent detections (last 30 days)
        thirty_days_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        detections = list(collection.find(
            {'timestamp': {'$gte': thirty_days_ago.isoformat()}},
            {'image_data': 0}  # Exclude base64 data from list
        ).sort('timestamp', -1).limit(100))
        
        # Convert ObjectId to string for JSON serialization
        for detection in detections:
            detection['_id'] = str(detection['_id'])
        
        return jsonify({
            'success': True,
            'history': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detection/image/<filename>')
def get_detection_image(filename):
    """Serve detection images"""
    try:
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/image_data/<filename>')
def get_detection_image_data(filename):
    """Get detection image as base64 data from database"""
    try:
        if not MONGODB_AVAILABLE:
            return jsonify({'error': 'Database not available'}), 500
        
        detection = collection.find_one({'image_filename': filename})
        if detection and 'image_data' in detection:
            return jsonify({
                'success': True,
                'image_data': detection['image_data'],
                'filename': filename
            })
        else:
            return jsonify({'error': 'Image not found in database'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/delete/<detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """Delete a detection record"""
    try:
        if not MONGODB_AVAILABLE:
            return jsonify({'error': 'Database not available'}), 500
        
        from bson.objectid import ObjectId
        result = collection.delete_one({'_id': ObjectId(detection_id)})
        
        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'Detection deleted'})
        else:
            return jsonify({'error': 'Detection not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/cloud/status')
def cloud_status():
        """Get cloud processing status"""
        return jsonify({
        'cloud_enabled': CLOUD_ENABLED,
        'cloud_processing': cloud_processing,
        'last_result_time': last_cloud_result.get('timestamp') if last_cloud_result else None,
        'api_url': CLOUD_API_URL,
        'process_every_n_frames': PROCESS_EVERY_N_FRAMES,
        'timeout': CLOUD_TIMEOUT,
        'last_endpoint_used': last_cloud_result.get('endpoint_used') if last_cloud_result else None
    })

@app.route('/api/cloud/toggle', methods=['POST'])
def toggle_cloud():
        """Toggle cloud processing on/off"""
        global CLOUD_ENABLED
        CLOUD_ENABLED = not CLOUD_ENABLED
        return jsonify({
            'success': True, 
            'cloud_enabled': CLOUD_ENABLED,
            'message': f'Cloud processing {"enabled" if CLOUD_ENABLED else "disabled"}'
        })

@app.route('/api/cloud/test', methods=['POST'])
def test_cloud_api():
    """Test cloud API connection"""
    try:
        print("🧪 Testing cloud API connection...")
        
        # Create a more realistic test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:200, 100:200] = (255, 0, 0)  # Red square
        test_frame[300:400, 400:500] = (0, 255, 0)  # Green square
        
        # Add text to make it more interesting
        cv2.putText(test_frame, 'TEST FRAME', (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Test the cloud API
        success = send_frame_to_cloud(test_frame, 'api_test')
        
        if success and last_cloud_result and last_cloud_result.get('frame_id') == 'api_test':
            return jsonify({
                'success': True,
                'message': 'Cloud API test successful',
                'result': {
                    'endpoint_used': last_cloud_result.get('endpoint_used'),
                    'detections_count': len(last_cloud_result.get('detections', [])),
                    'response_time': f"{time.time() - last_cloud_result.get('timestamp', 0):.2f}s",
                    'frame_processed': last_cloud_result.get('processed_frame_size'),
                    'raw_response_keys': list(last_cloud_result.keys())
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Cloud API test failed',
                'error': last_cloud_result.get('error') if last_cloud_result else 'No response'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Cloud API test error: {str(e)}'
        })

@app.route('/api/cloud/debug')
def debug_cloud():
    """Debug endpoint to check cloud response format"""
    return jsonify({
        'last_cloud_result': last_cloud_result,
        'cloud_processing': cloud_processing,
        'cloud_enabled': CLOUD_ENABLED,
        'api_url': CLOUD_API_URL,
        'detection_config': detection_config
    })

if __name__ == '__main__':
    # Parse command line arguments for model loading
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=False)
    parser.add_argument('--auto-start', help='Auto-start detection after loading model', 
                       action='store_true', default=True)
    parser.add_argument('--test-mode', help='Run in test mode without camera', 
                       action='store_true', default=False)
    args = parser.parse_args()
    
    print("🚀 Starting Integrated Drowning Detection Web Server...")
    print("📱 Flutter app can connect to: http://10.207.200.86:5000")
    print("📹 Video stream available at: http://10.207.200.86:5000/video_feed")
    print("🔌 API endpoints available at: http://10.207.200.86:5000/api/")
    
    # Initialize camera on startup (unless in test mode)
    if not args.test_mode:
        initialize_camera()
    else:
        print("⚠️ Running in test mode - camera disabled")
    
    # Load model if provided
    if args.model:
        if load_yolo_model(args.model):
            print(f"✅ Model loaded from {args.model}")
            if args.auto_start:
                detection_active = True
                print("✅ Detection started automatically")
        else:
            print(f"❌ Failed to load model from {args.model}")
    else:
        print("ℹ️ No model specified. Use /api/model/load to load a model later.")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    finally:
        cleanup_camera()
        buzzer.off()
        led.off()
