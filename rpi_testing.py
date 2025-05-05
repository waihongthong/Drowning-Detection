import time
import threading
import subprocess
import RPi.GPIO as GPIO
from ultralytics import YOLO
import cv2
import numpy as np
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

# GPIO Setup
LED_PIN = 17     # GPIO pin for LED
BUZZER_PIN = 18  # GPIO pin for buzzer
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # Disable warnings for pins already in use
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# Create PWM object for buzzer (frequency 2000 Hz)
buzzer_pwm = GPIO.PWM(BUZZER_PIN, 2000)

# Global variables for web streaming
latest_frame = None
latest_frame_lock = threading.Lock()
web_server = None
web_server_thread = None

# Function to trigger alarm for 10 seconds
def trigger_alarm(duration=10):
    print("🚨 ALARM: Drowning detected!")
    buzzer_pwm.start(50)  # Start buzzer (50% duty cycle)
    GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on LED
    time.sleep(duration)
    buzzer_pwm.stop()  # Stop buzzer
    GPIO.output(LED_PIN, GPIO.LOW)  # Turn off LED
    print("Alarm stopped")

class LibCameraHandler:
    """Use libcamera-still for capturing frames from Raspberry Pi camera"""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame = None
        self.last_frame_time = 0
        
    def initialize(self):
        try:
            # Check if libcamera-still is available
            result = subprocess.run(['which', 'libcamera-still'], 
                                  capture_output=True, text=True)
            if not result.stdout:
                print("❌ libcamera-still not found")
                return False
                
            # Take a test image to verify camera works
            test_path = "test_libcamera.jpg"
            print("Testing libcamera-still...")
            result = subprocess.run([
                'libcamera-still', 
                '-t', '1000',  # Timeout 1 second
                '--width', str(self.width),
                '--height', str(self.height),
                '-o', test_path
            ], capture_output=True)
            
            if result.returncode != 0:
                print(f"❌ libcamera-still test failed: {result.stderr.decode()}")
                return False
                
            if not os.path.exists(test_path):
                print("❌ libcamera-still did not create test image")
                return False
                
            test_img = cv2.imread(test_path)
            if test_img is None or test_img.size == 0:
                print("❌ Failed to read test image")
                return False
                
            print(f"✅ Successfully tested camera with libcamera-still")
            self.last_frame_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Error initializing LibCamera: {e}")
            return False
    
    def read_frame(self):
        try:
            # Capture a frame using libcamera-still
            temp_path = "temp_frame.jpg"
            result = subprocess.run([
                'libcamera-still', 
                '-t', '500',  # Timeout 500ms
                '--immediate',  # Take picture immediately
                '--width', str(self.width),
                '--height', str(self.height),
                '--nopreview',
                '-o', temp_path
            ], capture_output=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to capture frame: {result.stderr.decode()}")
                return False, None
                
            frame = cv2.imread(temp_path)
            if frame is not None and frame.size > 0:
                self.frame = frame
                self.last_frame_time = time.time()
                return True, frame
            else:
                print("❌ Failed to read captured frame")
                return False, None
                
        except Exception as e:
            print(f"❌ Error capturing frame: {e}")
            return False, None
    
    def is_healthy(self):
        """Check if camera is healthy based on recent frames"""
        # If we haven't gotten a frame in 10 seconds, camera is unhealthy
        return (time.time() - self.last_frame_time) < 10  # More tolerance for libcamera
    
    def release(self):
        # Nothing to release for libcamera-still approach
        pass

class RaspiStillHandler:
    """Use raspistill for capturing frames from Raspberry Pi camera (legacy)"""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame = None
        self.last_frame_time = 0
        
    def initialize(self):
        try:
            # Check if raspistill is available
            result = subprocess.run(['which', 'raspistill'], 
                                  capture_output=True, text=True)
            if not result.stdout:
                print("❌ raspistill not found")
                return False
                
            # Take a test image to verify camera works
            test_path = "test_raspistill.jpg"
            print("Testing raspistill...")
            result = subprocess.run([
                'raspistill', 
                '-t', '1000',  # Timeout 1 second
                '-w', str(self.width),
                '-h', str(self.height),
                '-o', test_path
            ], capture_output=True)
            
            if result.returncode != 0:
                print(f"❌ raspistill test failed: {result.stderr.decode()}")
                return False
                
            if not os.path.exists(test_path):
                print("❌ raspistill did not create test image")
                return False
                
            test_img = cv2.imread(test_path)
            if test_img is None or test_img.size == 0:
                print("❌ Failed to read test image")
                return False
                
            print(f"✅ Successfully tested camera with raspistill")
            self.last_frame_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RaspiStill: {e}")
            return False
    
    def read_frame(self):
        try:
            # Capture a frame using raspistill
            temp_path = "temp_frame.jpg"
            result = subprocess.run([
                'raspistill', 
                '-t', '300',  # Timeout 300ms
                '-n',        # No preview
                '-w', str(self.width),
                '-h', str(self.height),
                '-o', temp_path
            ], capture_output=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to capture frame: {result.stderr.decode()}")
                return False, None
                
            frame = cv2.imread(temp_path)
            if frame is not None and frame.size > 0:
                self.frame = frame
                self.last_frame_time = time.time()
                return True, frame
            else:
                print("❌ Failed to read captured frame")
                return False, None
                
        except Exception as e:
            print(f"❌ Error capturing frame: {e}")
            return False, None
    
    def is_healthy(self):
        """Check if camera is healthy based on recent frames"""
        # If we haven't gotten a frame in 10 seconds, camera is unhealthy
        return (time.time() - self.last_frame_time) < 10  # More tolerance for raspistill
    
    def release(self):
        # Nothing to release for raspistill approach
        pass

def try_cv2_camera():
    """Try to initialize camera using OpenCV's VideoCapture with improved settings"""
    print("\nTrying OpenCV camera access...")
    
    # Try V4L2 backend explicitly first (recommended for Raspberry Pi)
    print("Trying V4L2 backend explicitly...")
    try:
        # Use VideoCapture with explicit V4L2 backend
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Set explicit camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("✅ Successfully opened camera with V4L2 backend")
                return cap
            else:
                print("⚠️ Camera opened with V4L2 but couldn't grab frame")
                cap.release()
        else:
            print("❌ Could not open camera with V4L2 backend")
    except Exception as e:
        print(f"❌ Error with V4L2 backend: {e}")
    
    # First try direct device paths
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            print(f"Trying {device_path}...")
            try:
                cap = cv2.VideoCapture(device_path)
                
                # Set explicit camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Successfully opened {device_path}")
                        return cap
                    else:
                        print(f"⚠️ Device {device_path} opened but couldn't grab frame")
                        cap.release()
                else:
                    print(f"❌ Could not open {device_path}")
            except Exception as e:
                print(f"❌ Error with {device_path}: {e}")
    
    # Then try indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        try:
            cap = cv2.VideoCapture(i)
            
            # Set explicit camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ Successfully opened camera index {i}")
                    return cap
                else:
                    print(f"⚠️ Camera {i} opened but couldn't grab frame")
                    cap.release()
            else:
                print(f"❌ Could not open camera index {i}")
        except Exception as e:
            print(f"❌ Error with camera index {i}: {e}")
    
    print("❌ Failed to initialize any camera with OpenCV")
    return None

class OpenCVCameraHandler:
    """Wrapper class for OpenCV camera with health monitoring"""
    def __init__(self, cap):
        self.cap = cap
        self.last_frame_time = time.time()
        
    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.last_frame_time = time.time()
                return True, frame
            else:
                return False, None
        except Exception as e:
            print(f"❌ Error reading OpenCV frame: {e}")
            return False, None
    
    def is_healthy(self):
        """Check if camera is healthy based on recent frames"""
        if not self.cap or not self.cap.isOpened():
            return False
        # If we haven't gotten a frame in 3 seconds, camera is unhealthy
        return (time.time() - self.last_frame_time) < 3
    
    def release(self):
        if self.cap:
            try:
                self.cap.release()
                print("✅ OpenCV camera resources released")
            except Exception as e:
                print(f"❌ Error releasing OpenCV camera: {e}")
            self.cap = None

def initialize_camera():
    """Try to initialize any available camera method"""
    # List available devices
    print("Available video devices:")
    subprocess.run(['ls', '-l', '/dev/video*'], stderr=subprocess.STDOUT)
    
    # Try LibCamera first (for Pi OS Bullseye and newer)
    print("\nTrying LibCamera...")
    camera = LibCameraHandler()
    if camera.initialize():
        print("✅ LibCamera initialized successfully!")
        return camera, "libcamera"
    
    # Try legacy RaspiStill (for older Pi OS)
    print("\nTrying legacy RaspiStill...")
    camera = RaspiStillHandler()
    if camera.initialize():
        print("✅ RaspiStill initialized successfully!")
        return camera, "raspistill"
    
    # Fallback to OpenCV
    print("\nFalling back to OpenCV camera...")
    cap = try_cv2_camera()
    if cap is not None:
        print("✅ OpenCV camera initialized successfully!")
        return OpenCVCameraHandler(cap), "opencv"
    
    # No camera available
    print("❌ Failed to initialize any camera method")
    return None, None

def save_detection_image(frame, results, drowning_detected):
    """Save detection image with bounding boxes and information"""
    # Create a copy of the frame
    output_frame = frame.copy()
    
    # Process results (same as in main loop)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = result.names[class_id]
            
            if class_name == 'drowning' and confidence > 0.5:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box and label
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"DROWNING {confidence:.2f}"
                cv2.putText(output_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Add status text
    status = "DROWNING DETECTED" if drowning_detected else "Monitoring"
    cv2.putText(output_frame, "Status: " + status, 
              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
              (0, 0, 255) if drowning_detected else (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_frame, timestamp, 
              (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
              (255, 255, 255), 2)
    
    # Save the image
    if drowning_detected:
        filename = f"drowning_alert_{int(time.time())}.jpg"
    else:
        filename = f"detection_{int(time.time())}.jpg"
    
    cv2.imwrite(filename, output_frame)
    print(f"Image saved: {filename}")
    return filename

# Web streaming handler
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
        
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # HTML page with auto-refresh
            self.wfile.write(bytes('''
            <html>
            <head>
                <title>Raspberry Pi Drowning Detection</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; background-color: #f0f0f0; }
                    h1 { color: #333; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                    .status { font-size: 20px; margin: 20px 0; }
                    .footer { margin-top: 30px; font-size: 14px; color: #777; }
                </style>
                <script>
                    function refreshImage() {
                        document.getElementById('stream').src = "/stream?" + new Date().getTime();
                    }
                    setInterval(refreshImage, 1000);  // Refresh image every 1 second
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Drowning Detection Monitor</h1>
                    <div class="status">Status: Running</div>
                    <img id="stream" src="/stream" />
                    <div class="footer">
                        <p>Raspberry Pi Drowning Detection System</p>
                        <p id="time"></p>
                        <script>
                            function updateTime() {
                                document.getElementById('time').innerHTML = new Date().toLocaleString();
                            }
                            setInterval(updateTime, 1000);
                            updateTime();
                        </script>
                    </div>
                </div>
            </body>
            </html>
            ''', 'utf-8'))
            
        elif self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            # Return the latest frame as JPEG
            with latest_frame_lock:
                if latest_frame is not None:
                    # Convert the OpenCV frame to JPEG
                    _, buffer = cv2.imencode('.jpg', latest_frame)
                    self.wfile.write(buffer.tobytes())
                else:
                    # If no frame is available, send a placeholder image
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No camera feed available", (120, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    self.wfile.write(buffer.tobytes())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Suppress HTTP request logs to keep console clean
        return

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def get_local_ip():
    """Get the local IP address of the Raspberry Pi"""
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

def start_web_server():
    """Start the web server in a separate thread"""
    global web_server, web_server_thread
    
    ip = get_local_ip()
    port = 8000
    
    try:
        web_server = StreamingServer((ip, port), StreamingHandler)
        web_server_thread = threading.Thread(target=web_server.serve_forever)
        web_server_thread.daemon = True
        web_server_thread.start()
        print(f"✅ Web server started - Access at: http://{ip}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to start web server: {e}")
        return False

def stop_web_server():
    """Stop the web server thread"""
    global web_server
    if web_server:
        web_server.shutdown()
        web_server = None
        print("✅ Web server stopped")

def main():
    global latest_frame
    
    print("Starting drowning detection system...")
    
    # Create output directory if it doesn't exist
    os.makedirs("detections", exist_ok=True)
    
    # Check if display is available
    display_available = os.environ.get('DISPLAY') is not None
    
    # Start web server for remote viewing
    print("Starting web server for remote viewing...")
    start_web_server()
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('best.pt')  # Or use 'yolov8n.pt' for a pretrained nano model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        GPIO.cleanup()
        stop_web_server()
        return
    
    # Initialize camera
    camera, camera_type = initialize_camera()
    if not camera:
        print("❌ Could not initialize any camera. Exiting.")
        GPIO.cleanup()
        stop_web_server()
        return
    
    print(f"✅ Using {camera_type} camera")
    
    # Create window for display if display is available
    if display_available:
        window_name = "Drowning Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Resizable window
        print("✅ Display window created")
    else:
        print("⚠️ No display detected. Running in headless mode.")
        print(f"View the camera feed at: http://{get_local_ip()}:8000")
    
    # Variables for detection
    alarm_active = False
    last_camera_check = time.time()
    camera_recovery_attempts = 0
    last_save_time = 0  # For periodic image saving
    
    print("Camera view started. Press 'q' to quit.")
    
    try:
        while True:
            # Camera health check and recovery (every 10 seconds)
            if time.time() - last_camera_check > 10:
                last_camera_check = time.time()
                if not camera.is_healthy():
                    print("⚠️ Camera appears unhealthy, attempting recovery...")
                    camera_recovery_attempts += 1
                    
                    # Release current camera
                    camera.release()
                    
                    # Try to reinitialize camera
                    if camera_recovery_attempts < 5:  # Limit recovery attempts
                        camera, camera_type = initialize_camera()
                        if camera:
                            print(f"✅ Camera recovered using {camera_type}")
                        else:
                            print("❌ Could not recover camera. Will retry...")
                            time.sleep(2)  # Wait before next attempt
                    else:
                        print("❌ Maximum camera recovery attempts reached. Exiting.")
                        break
            
            # Capture frame
            if not camera:
                print("❌ No active camera. Attempting recovery...")
                time.sleep(1)
                last_camera_check = 0  # Force camera check on next loop
                continue
                
            ret, frame = camera.read_frame()
                
            if not ret or frame is None:
                print("❌ Failed to capture frame, retrying...")
                time.sleep(0.5)
                continue
            
            # Update latest frame for web streaming
            with latest_frame_lock:
                latest_frame = frame.copy()
            
            # Run YOLOv8 detection
            results = model(frame, verbose=False)
            
            # Process results
            drowning_detected = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = model.names[class_id]
                    
                    if class_name == 'drowning' and confidence > 0.5:
                        drowning_detected = True
                        print(f"⚠️ Drowning detected with confidence {confidence:.2f}")
            
            # Create a display frame with detections
            display_frame = frame.copy()
            
            # Draw detections on display frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = model.names[class_id]
                    
                    if class_name == 'drowning' and confidence > 0.5:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw bounding box and label
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"DROWNING {confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Add status text
            status = "DROWNING DETECTED" if drowning_detected else "Monitoring"
            cv2.putText(display_frame, "Status: " + status, 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                      (0, 0, 255) if drowning_detected else (0, 255, 0), 2)
            
            # Add camera info
            camera_info = f"Camera: {camera_type}"
            cv2.putText(display_frame, camera_info, 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, "Time: " + timestamp, 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add web address for remote viewing
            if camera_type == "libcamera" or camera_type == "raspistill":
                web_info = f"Web view: http://{get_local_ip()}:8000"
                cv2.putText(display_frame, web_info, 
                          (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          (255, 255, 255), 2)
            
            # Display the frame if display is available
            if display_available:
                cv2.imshow(window_name, display_frame)
            
            # Trigger alarm if drowning detected (and alarm not already active)
            if drowning_detected and not alarm_active:
                alarm_active = True
                # Save detection image when drowning is detected
                save_detection_image(frame, results, drowning_detected)
                threading.Thread(target=trigger_alarm).start()
            
            # Reset alarm status after it finishes
            if not drowning_detected and alarm_active:
                alarm_active = False
            
            # Periodically save detection images
            if time.time() - last_save_time > 60:  # Every 60 seconds
                last_save_time = time.time()
                save_detection_image(frame, results, drowning_detected)
            
            # Check for key press if display is available
            if display_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting due to 'q' key press...")
                    break
                elif key == ord('s'):
                    # Save current frame on 's' key press
                    save_detection_image(frame, results, drowning_detected)
                    print("Frame saved manually")
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt...")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cleanup
        if camera:
            camera.release()
        
        if display_available:
            cv2.destroyAllWindows()
        
        # Stop web server
        stop_web_server()
        
        buzzer_pwm.stop()
        GPIO.cleanup()
        print("✅ Resources released.")

if __name__ == "__main__":
    main()
