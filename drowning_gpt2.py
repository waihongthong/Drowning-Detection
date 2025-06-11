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
import os

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

def process_detection(frame):
    """Process frame for drowning detection"""
    global consecutive_drowning, drowning_confidences, model
    
    if model is None:
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
    global camera, detection_status, frame_rate_buffer, consecutive_drowning, drowning_confidences, camera_active
    
    frame_count = 0
    last_frame_time = time.time()
    last_fps_time = time.time()
    
    while True:
        try:
            frame_start_time = time.perf_counter()
            frame = None
            
            # Get frame from camera with retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if camera_active and camera is not None:
                        with camera_lock:
                            frame = camera.capture_array()
                            
                        # Validate frame
                        if frame is not None and frame.size > 0:
                            break
                        else:
                            print(f"⚠️ Empty frame on retry {retry + 1}")
                            time.sleep(0.1)
                    else:
                        frame = generate_test_frame()
                        break
                        
                except Exception as e:
                    print(f"⚠️ Frame capture error (retry {retry + 1}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(0.2)
                    else:
                        frame = generate_test_frame()
            
            # Ensure we have a valid frame
            if frame is None or frame.size == 0:
                frame = generate_test_frame()
            
            # Convert frame format properly
            original_shape = frame.shape
            try:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        # Picamera2 gives RGB, convert to BGR for OpenCV
                        if camera_active:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                # Resize frame to match expected dimensions
                if frame.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                    
            except Exception as e:
                print(f"⚠️ Frame conversion error: {e}")
                frame = generate_test_frame()
            
            # Process detection ONLY if active and model loaded
            drowning_count = 0
            total_objects = 0
            drowning_detected = False
            
            if detection_active and model is not None:
                try:
                    frame, drowning_count, total_objects, drowning_detected = process_detection(frame)
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Update detection status in thread-safe manner
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
                    
                    # Enhanced image saving with better naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'drowning_detected_{timestamp}.jpg'
                    image_path = os.path.join(IMAGES_DIR, filename)
                    
                    # Save image with higher quality
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"📸 Drowning detection captured and saved as '{image_path}'")
                    
                    # Save to database
                    detection_data = {
                        'confidence': detection_status['confidence'],
                        'consecutive_detections': consecutive_drowning,
                        'total_objects': total_objects
                    }
                    db_id = save_detection_to_database(image_path, detection_data)
                    
                    # Trigger alarm
                    threading.Thread(target=activate_alarm, args=(10,), daemon=True).start()
                    
                elif not drowning_detected and detection_status['drowning_detected']:
                    # Clear drowning detection when no longer detected
                    detection_status['drowning_detected'] = False
                    print("✅ Drowning detection cleared")
            
            # Calculate accurate FPS
            frame_end_time = time.perf_counter()
            frame_duration = frame_end_time - frame_start_time
            instantaneous_fps = 1.0 / frame_duration if frame_duration > 0 else 0
            
            # Update FPS buffer
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(instantaneous_fps)
            avg_fps = np.mean(frame_rate_buffer) if frame_rate_buffer else 0
            
            # Update FPS in status every second
            if current_time - last_fps_time >= 1.0:
                with status_lock:
                    detection_status['fps'] = round(avg_fps, 1)
                last_fps_time = current_time
            
            # Draw overlay information
            overlay_color = (0, 255, 255)  # Yellow
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Status overlay
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30), font, font_scale, overlay_color, thickness)
            cv2.putText(frame, f'Objects: {total_objects}', (10, 60), font, font_scale, overlay_color, thickness)
            cv2.putText(frame, f'Frame: {frame_count}', (10, 90), font, font_scale, overlay_color, thickness)
            
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
            
            # Encode frame with consistent quality
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 85,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                print("⚠️ Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Debug output every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                actual_fps = 30 / (current_time - last_frame_time) if frame_count > 30 else 0
                print(f"📊 Frame {frame_count}: {len(frame_bytes)} bytes, "
                      f"Actual FPS: {actual_fps:.1f}, Avg FPS: {avg_fps:.1f}, "
                      f"Objects: {total_objects}, Detection: {detection_active}")
                last_frame_time = current_time
            
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            # Control frame rate - aim for consistent timing
            target_frame_time = 1.0 / STREAM_FPS
            sleep_time = target_frame_time - frame_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            # Generate error frame
            error_frame = generate_test_frame()
            cv2.putText(error_frame, f'ERROR: {str(e)[:40]}', (10, CAMERA_HEIGHT//2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                       frame_bytes + b'\r\n')
            
            time.sleep(0.5)  # Wait on error

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
