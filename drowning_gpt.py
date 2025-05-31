#!/usr/bin/env python3
"""
Integrated Drowning Detection Web Server for Raspberry Pi
Provides REST API, video streaming, and real-time drowning detection for Flutter mobile app
"""

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
STREAM_FPS = 15

# Detection Configuration (can be modified via API)
detection_config = {
    'confidence_threshold': 0.6,
    'consecutive_threshold': 5,
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
    global camera
    try:
        with camera_lock:
            if camera is not None:
                return True
                
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                buffer_count=2
            )
            camera.configure(config)
            
            if CONTROLS_AVAILABLE:
                try:
                    controls = Controls(camera)
                    controls.FrameDurationLimits = (33333, 33333)  # ~30fps
                    controls.FrameRate = 30.0
                    controls.AeEnable = True
                    controls.AwbEnable = True
                except Exception as e:
                    print(f"Could not set camera controls: {e}")
            
            camera.start(show_preview=False)
            
            # Warm up camera
            for _ in range(5):
                frame = camera.capture_array()
                time.sleep(0.1)
                
            print("✅ Camera initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Failed to initialize camera: {e}")
        return False

def cleanup_camera():
    """Clean up camera resources"""
    global camera
    try:
        with camera_lock:
            if camera is not None:
                camera.close()
                camera = None
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
        
        # Check if drowning should be triggered
        drowning_detected = (consecutive_drowning >= detection_config['consecutive_threshold'] and 
                           len(drowning_confidences) >= detection_config['consecutive_threshold'] and 
                           avg_confidence_threshold >= detection_config['min_avg_confidence'])
        
        return frame, drowning_count, total_objects, drowning_detected
        
    except Exception as e:
        print(f"Error processing detection: {e}")
        return frame, 0, 0, False

def generate_frames():
    """Generate video frames for streaming with detection overlay"""
    global camera, detection_status, frame_rate_buffer, consecutive_drowning, drowning_confidences
    
    while True:
        try:
            t_start = time.perf_counter()
            
            with camera_lock:
                if camera is None:
                    time.sleep(0.1)
                    continue
                    
                frame = camera.capture_array()
                
                # Convert frame if necessary
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process detection if active and model is loaded
            drowning_count = 0
            total_objects = 0
            drowning_detected = False
            
            if detection_active and model is not None:
                frame, drowning_count, total_objects, drowning_detected = process_detection(frame)
                
                # Handle drowning detection
                with status_lock:
                    if drowning_detected and not detection_status['alarm_active']:
                        detection_status['drowning_detected'] = True
                        detection_status['last_detection_time'] = time.time()
                        detection_status['consecutive_detections'] = consecutive_drowning
                        detection_status['confidence'] = sum(drowning_confidences) / len(drowning_confidences) if drowning_confidences else 0
                        
                        # Save detection image
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f'drowning_detected_{timestamp}.png'
                        cv2.imwrite(filename, frame)
                        print(f"📸 Drowning detection captured and saved as '{filename}'")
                        
                        # Trigger alarm
                        threading.Thread(target=activate_alarm, args=(10,), daemon=True).start()
                    
                    elif not drowning_detected:
                        detection_status['drowning_detected'] = False
                    
                    detection_status['total_objects'] = total_objects
                    detection_status['consecutive_detections'] = consecutive_drowning
                    if drowning_confidences:
                        detection_status['confidence'] = sum(drowning_confidences) / len(drowning_confidences)
            
            # Calculate FPS
            t_stop = time.perf_counter()
            frame_rate_calc = 1.0 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0
            
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_fps = np.mean(frame_rate_buffer)
            
            # Add status overlay
            with status_lock:
                detection_status['fps'] = avg_fps
                detection_status['is_detecting'] = detection_active
                
                # Draw status information
                cv2.putText(frame, f'FPS: {avg_fps:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Objects: {total_objects}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if detection_active:
                    cv2.putText(frame, f'Drowning: {drowning_count}', 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f'Consecutive: {consecutive_drowning}/{detection_config["consecutive_threshold"]}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if detection_status['drowning_detected']:
                    # Draw drowning alert
                    alert_text = "!!! DROWNING DETECTED !!!"
                    cv2.putText(frame, alert_text, (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                    cv2.putText(frame, alert_text, (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if detection_status['alarm_active']:
                    cv2.putText(frame, "ALARM ACTIVE", (CAMERA_WIDTH-200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1.0 / STREAM_FPS)  # Control frame rate
            
        except Exception as e:
            print(f"Error generating frame: {e}")
            time.sleep(0.1)

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    with status_lock:
        return jsonify(detection_status)

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
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test API connection"""
    return jsonify({
        'success': True,
        'message': 'Drowning Detection Server is running',
        'timestamp': datetime.now().isoformat(),
        'server_ip': request.host,
        'model_loaded': model is not None,
        'detection_active': detection_active
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
        'camera_active': camera is not None
    })

if __name__ == '__main__':
    # Parse command line arguments for model loading
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=False)
    parser.add_argument('--auto-start', help='Auto-start detection after loading model', 
                       action='store_true', default=False)
    args = parser.parse_args()
    
    print("🚀 Starting Integrated Drowning Detection Web Server...")
    print("📱 Flutter app can connect to: http://192.168.0.16:5000")
    print("📹 Video stream available at: http://192.168.0.16:5000/video_feed")
    print("🔌 API endpoints available at: http://192.168.0.16:5000/api/")
    
    # Initialize camera on startup
    initialize_camera()
    
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
