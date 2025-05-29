#!/usr/bin/env python3
"""
Drowning Detection Web Server for Raspberry Pi
Provides REST API and video streaming for Flutter mobile app
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from picamera2 import Picamera2
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

def generate_frames():
    """Generate video frames for streaming"""
    global camera, detection_status
    
    while True:
        try:
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
            
            # Add status overlay
            with status_lock:
                # Draw status information
                cv2.putText(frame, f'FPS: {detection_status["fps"]:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Objects: {detection_status["total_objects"]}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if detection_status['drowning_detected']:
                    # Draw drowning alert
                    alert_text = "!!! DROWNING DETECTED !!!"
                    cv2.putText(frame, alert_text, (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
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

@app.route('/api/device/info')
def get_device_info():
    """Get device information"""
    return jsonify({
        'device_name': 'Raspberry Pi Drowning Detector',
        'ip_address': request.host.split(':')[0],
        'camera_resolution': f'{CAMERA_WIDTH}x{CAMERA_HEIGHT}',
        'gpio_available': GPIO_AVAILABLE,
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
        'server_ip': request.host
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

def update_detection_status(**kwargs):
    """Update detection status from external detection script"""
    with status_lock:
        detection_status.update(kwargs)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 Starting Drowning Detection Web Server...")
    print("📱 Flutter app can connect to: http://192.168.0.16:5000")
    print("📹 Video stream available at: http://192.168.0.16:5000/video_feed")
    print("🔌 API endpoints available at: http://192.168.0.16:5000/api/")
    
    # Initialize camera on startup
    initialize_camera()
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    finally:
        cleanup_camera()
        buzzer.off()
        led.off()
