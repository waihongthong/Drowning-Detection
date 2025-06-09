"""Integrated Drowning Detection Web Server for Raspberry Pi
Provides REST API, video streaming, and real-time drowning detection for Flutter mobile app"""

import os
import json
import time
import threading
import argparse
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from queue import Queue
from pymongo import MongoClient
import base64
from datetime import datetime
import os


HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/your-drowning-model"
HUGGINGFACE_API_KEY = "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"  
USE_CLOUD_INFERENCE = True  # Toggle between local and cloud inference
CLOUD_INFERENCE_BATCH_SIZE = 1  # Process frames in batches
FRAME_SKIP_RATIO = 3
MONGODB_URI = "mongodb+srv://waihong0717:0717Waihong!@cluster0.zicfsjv.mongodb.net/fyp" 
DB_NAME = "drowning_detection"
COLLECTION_NAME = "detection_history"
IMAGES_DIR = "detection_images"

# Cloud inference queue
cloud_inference_queue = queue.Queue(maxsize=10)
cloud_results_queue = queue.Queue(maxsize=10)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

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
BUZZER_PIN = 23
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

def upload_model_to_huggingface():
    """
    Instructions to upload your YOLO model to Hugging Face:
    
    1. Install required packages:
       pip install huggingface_hub transformers
    
    2. Create a new model repository on Hugging Face:
       - Go to https://huggingface.co/new
       - Choose "Model" and create a new repository
    
    3. Upload your model files:
    """
    
    upload_script = '''
# Run this script to upload your model to Hugging Face

from huggingface_hub import HfApi, Repository
import os

# Initialize Hugging Face API
api = HfApi()

# Your model details
model_name = "your-username/drowning-detection-yolo"
model_path = "/path/to/your/best.pt"  # Your YOLO model file

# Create repository if it doesn't exist
try:
    api.create_repo(repo_id=model_name, repo_type="model")
except:
    pass  # Repository might already exist

# Upload the model file
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.pt",
    repo_id=model_name,
    commit_message="Upload drowning detection YOLO model"
)

# Create a model card (README.md)
model_card = """
---
tags:
- yolo
- object-detection
- drowning-detection
- computer-vision
license: mit
---

# Drowning Detection YOLO Model

This model detects drowning incidents in video streams.

## Usage

```python
from ultralytics import YOLO
model = YOLO('model.pt')
results = model('path/to/image.jpg')
```
"""

with open("README.md", "w") as f:
    f.write(model_card)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=model_name,
    commit_message="Add model documentation"
)

print(f"Model uploaded to: https://huggingface.co/{model_name}")
'''
    
    return upload_script

def load_yolo_model(model_path):
    """Load the YOLO model for drowning detection"""
    global model
    try:
        if not os.path.exists(model_path):
            print(f'❌ ERROR: Model path "{model_path}" is invalid or model was not found.')
            return False
            
        with model_lock:
            model = YOLO(model_path)
            labels = model.names
            print(f"✅ Loaded YOLO model from {model_path}")
            print(f"📊 Detection classes: {labels}")
            return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
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

async def cloud_inference_worker():
    """Worker function for cloud inference"""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                if not cloud_inference_queue.empty():
                    frame_data = cloud_inference_queue.get_nowait()
                    frame_id, frame_bytes = frame_data
                    
                    # Send frame to Hugging Face API
                    result = await send_frame_to_huggingface(session, frame_bytes)
                    
                    # Put result in results queue
                    if not cloud_results_queue.full():
                        cloud_results_queue.put_nowait((frame_id, result))
                    
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Cloud inference error: {e}")
                await asyncio.sleep(1)

async def send_frame_to_huggingface(session, frame_bytes):
    """Send frame to Hugging Face Inference API"""
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        
        async with session.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            data=frame_bytes,
            timeout=aiohttp.ClientTimeout(total=5)  # 5 second timeout
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                return parse_huggingface_response(result)
            else:
                print(f"HF API error: {response.status}")
                return None
                
    except Exception as e:
        print(f"HF API request error: {e}")
        return None

def parse_huggingface_response(hf_result):
    """Parse Hugging Face API response to match YOLO format"""
    try:
        detections = []
        
        # HF returns results in different format, adapt as needed
        for detection in hf_result:
            if detection.get('label', '').lower() == 'drowning':
                bbox = detection.get('box', {})
                confidence = detection.get('score', 0)
                
                # Convert to YOLO-like format
                detection_obj = {
                    'class': 'drowning',
                    'confidence': confidence,
                    'bbox': [
                        bbox.get('xmin', 0),
                        bbox.get('ymin', 0),
                        bbox.get('xmax', 0),
                        bbox.get('ymax', 0)
                    ]
                }
                detections.append(detection_obj)
        
        return detections
        
    except Exception as e:
        print(f"Error parsing HF response: {e}")
        return []

def process_detection(frame):
    """Process frame for drowning detection"""
    global consecutive_drowning, drowning_confidences
    
    try:
        # Convert frame to bytes for cloud inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        frame_bytes = img_buffer.getvalue()
        
        # Send to cloud inference queue (non-blocking)
        frame_id = int(time.time() * 1000)  # Unique frame ID
        if not cloud_inference_queue.full():
            cloud_inference_queue.put_nowait((frame_id, frame_bytes))
        
        # Check for results from previous frames
        detections = []
        if not cloud_results_queue.empty():
            try:
                result_frame_id, cloud_detections = cloud_results_queue.get_nowait()
                if cloud_detections:
                    detections = cloud_detections
            except:
                pass
        
        # Process detections (similar to original logic)
        drowning_count = 0
        total_objects = len(detections)
        current_drowning_confidences = []
        
        for detection in detections:
            if detection['class'].lower() == 'drowning':
                drowning_count += 1
                confidence = detection['confidence']
                current_drowning_confidences.append(confidence)
                
                # Draw bounding box
                bbox = detection['bbox']
                cv2.rectangle(frame, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             (0, 0, 255), 2)
                
                # Add label
                label = f'Drowning: {int(confidence*100)}%'
                cv2.putText(frame, label, 
                           (int(bbox[0]), int(bbox[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Immediate alert for high confidence
                if confidence > 0.8:
                    buzzer.on()
                    led.on()
                    threading.Timer(1.0, lambda: [buzzer.off(), led.off()]).start()
        
        # Update consecutive drowning logic
        if drowning_count > 0:
            consecutive_drowning += 1
            avg_conf = sum(current_drowning_confidences) / len(current_drowning_confidences)
            drowning_confidences.append(avg_conf)
            if len(drowning_confidences) > detection_config['consecutive_threshold']:
                drowning_confidences.pop(0)
        else:
            consecutive_drowning = 0
            drowning_confidences = []
        
        # Check drowning detection
        avg_confidence_threshold = (sum(drowning_confidences) / len(drowning_confidences) 
                                  if drowning_confidences else 0)
        
        drowning_detected = (consecutive_drowning >= detection_config['consecutive_threshold'] and 
                           len(drowning_confidences) >= detection_config['consecutive_threshold'] and 
                           avg_confidence_threshold >= detection_config['min_avg_confidence'])
        
        return frame, drowning_count, total_objects, drowning_detected
        
    except Exception as e:
        print(f"Cloud detection error: {e}")
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
    def generate_frames_optimized():
    """Optimized frame generation with cloud inference"""
    global camera, detection_status, frame_rate_buffer, consecutive_drowning, drowning_confidences, camera_active
    
    frame_count = 0
    last_frame_time = time.time()
    last_fps_time = time.time()
    
    # Start cloud inference worker if using cloud
    if USE_CLOUD_INFERENCE:
        asyncio.create_task(cloud_inference_worker())
    
    while True:
        try:
            frame_start_time = time.perf_counter()
            frame = None
            
            # Get frame (same as original)
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if camera_active and camera is not None:
                        with camera_lock:
                            frame = camera.capture_array()
                        if frame is not None and frame.size > 0:
                            break
                    else:
                        frame = generate_test_frame()
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        frame = generate_test_frame()
            
            if frame is None or frame.size == 0:
                frame = generate_test_frame()
            
            # Frame format conversion (same as original)
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3 and camera_active:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if frame.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            
            # Process detection with optimizations
            drowning_count = 0
            total_objects = 0
            drowning_detected = False
            
            if detection_active and model is not None:
                # Decide whether to use cloud or local inference
                if USE_CLOUD_INFERENCE:
                    # Skip frames for cloud inference to reduce latency
                    if frame_count % FRAME_SKIP_RATIO == 0:
                        frame, drowning_count, total_objects, drowning_detected = process_detection(frame)
                else:
                    # Use local inference (original method)
                    frame, drowning_count, total_objects, drowning_detected = process_detection(frame)
            
            # Update status and generate frame output (same as original)
            current_time = time.time()
            with status_lock:
                detection_status['total_objects'] = total_objects
                detection_status['consecutive_detections'] = consecutive_drowning
                detection_status['is_detecting'] = detection_active
                
                if drowning_confidences:
                    detection_status['confidence'] = sum(drowning_confidences) / len(drowning_confidences)
                else:
                    detection_status['confidence'] = 0.0
                
                # Handle drowning detection
                if drowning_detected and not detection_status['alarm_active']:
                    detection_status['drowning_detected'] = True
                    detection_status['last_detection_time'] = time.time()
                    detection_status['consecutive_detections'] = consecutive_drowning
                    detection_status['confidence'] = sum(drowning_confidences) / len(drowning_confidences) if drowning_confidences else 0
                    
                    # Save image and trigger alarm
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'drowning_detected_{timestamp}.jpg'
                    image_path = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    detection_data = {
                        'confidence': detection_status['confidence'],
                        'consecutive_detections': consecutive_drowning,
                        'total_objects': total_objects
                    }
                    save_detection_to_database(image_path, detection_data)
                    threading.Thread(target=activate_alarm, args=(10,), daemon=True).start()
            
            # Calculate FPS
            frame_end_time = time.perf_counter()
            frame_duration = frame_end_time - frame_start_time
            instantaneous_fps = 1.0 / frame_duration if frame_duration > 0 else 0
            
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(instantaneous_fps)
            avg_fps = np.mean(frame_rate_buffer) if frame_rate_buffer else 0
            
            if current_time - last_fps_time >= 1.0:
                with status_lock:
                    detection_status['fps'] = round(avg_fps, 1)
                last_fps_time = current_time
            
            # Draw overlay information
            overlay_color = (0, 255, 255) if not USE_CLOUD_INFERENCE else (255, 165, 0)  # Orange for cloud
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30), font, font_scale, overlay_color, thickness)
            cv2.putText(frame, f'Objects: {total_objects}', (10, 60), font, font_scale, overlay_color, thickness)
            cv2.putText(frame, f'Mode: {"Cloud" if USE_CLOUD_INFERENCE else "Local"}', (10, 90), font, font_scale, overlay_color, thickness)
            
            if detection_active:
                cv2.putText(frame, f'Drowning: {drowning_count}', (10, 120), font, font_scale, overlay_color, thickness)
                cv2.putText(frame, f'Consecutive: {consecutive_drowning}/{detection_config["consecutive_threshold"]}', 
                           (10, 150), font, font_scale, overlay_color, thickness)
            
            # Drowning alert overlay
            if detection_status.get('drowning_detected', False):
                alert_text = "!!! DROWNING DETECTED !!!"
                cv2.putText(frame, alert_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(frame, alert_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Camera status
            status_y = CAMERA_HEIGHT - 30
            if camera_active:
                cv2.putText(frame, "CAMERA: LIVE", (10, status_y), font, font_scale, (0, 255, 0), thickness)
            else:
                cv2.putText(frame, "CAMERA: TEST MODE", (10, status_y), font, font_scale, overlay_color, thickness)
            
            # Encode and yield frame
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Reduced quality for faster encoding
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame_bytes = buffer.tobytes()
                frame_count += 1
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                       frame_bytes + b'\r\n')
            
            # Adaptive frame rate control
            if USE_CLOUD_INFERENCE:
                # Faster frame rate for cloud inference since processing is offloaded
                target_fps = 15
            else:
                target_fps = STREAM_FPS
            
            target_frame_time = 1.0 / target_fps
            sleep_time = target_frame_time - frame_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.5)

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
        
@app.route('/api/inference/mode', methods=['GET', 'POST'])
def inference_mode():
    """Get or set inference mode (local vs cloud)"""
    global USE_CLOUD_INFERENCE
    
    if request.method == 'GET':
        return jsonify({
            'current_mode': 'cloud' if USE_CLOUD_INFERENCE else 'local',
            'huggingface_configured': bool(HUGGINGFACE_API_KEY and HUGGINGFACE_API_URL)
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        mode = data.get('mode', 'local')
        
        if mode == 'cloud':
            if not HUGGINGFACE_API_KEY or not HUGGINGFACE_API_URL:
                return jsonify({
                    'success': False, 
                    'error': 'Hugging Face API not configured'
                }), 400
            USE_CLOUD_INFERENCE = True
        else:
            USE_CLOUD_INFERENCE = False
        
        return jsonify({
            'success': True,
            'message': f'Inference mode set to {mode}',
            'current_mode': 'cloud' if USE_CLOUD_INFERENCE else 'local'
        })

if __name__ == '__main__':
    # Parse command line arguments for model loading
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=False)
    parser.add_argument('--auto-start', help='Auto-start detection after loading model', 
                       action='store_true', default=False)
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
