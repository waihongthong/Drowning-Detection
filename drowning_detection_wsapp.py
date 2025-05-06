import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import json
import os
import subprocess
import threading
import signal
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Check if we're on Raspberry Pi and import picamera2 if available
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
    print("Picamera2 module found, will use it for camera access")
except ImportError:
    HAS_PICAMERA2 = False
    print("Picamera2 module not found, will use OpenCV for camera access")

# Load YOLOv8 model
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"Warning: Model file {model_path} not found. Please check the path.")
    exit(1)

try:
    model = YOLO(model_path)
    print(f"Successfully loaded YOLO model from {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Firebase setup
PROJECT_ID = "drowning-notification"
DEVICE_TOKEN = "fidLtsmJTvqYV7a7oYz1Cp:APA91bE_qVjw7hLQt_WjTYkrwNlYWhq6_V4HgvJwrzd7huMzLaR_YK5ejOVtvjGIkkRdoUl4RJnYDNHX0H595yNM449cIPwI1UpIxkwFKuoYyAp3_Eq8498"
SERVICE_ACCOUNT_FILE = "drowning-notification-firebase-adminsdk-fbsvc-55feeeaf68.json"

# Get OAuth2 access token from service account
def get_access_token():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"]
        )
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

# Send notification via HTTP v1
def send_alert():
    try:
        access_token = get_access_token()
        if not access_token:
            print("Failed to get access token, can't send alert")
            return False
            
        url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; UTF-8",
        }
        message = {
            "message": {
                "token": DEVICE_TOKEN,
                "notification": {
                    "title": "🚨 Drowning Alert",
                    "body": "Drowning detected by Raspberry Pi!"
                }
            }
        }
        response = requests.post(url, headers=headers, json=message)
        print("Notification sent:", response.status_code, response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending alert: {e}")
        return False

# Try to configure libcamera for V4L2 compatibility
def setup_libcamera_compatibility():
    try:
        # Check if the bcm2835-v4l2 module is loaded
        modprobe_result = subprocess.run(["lsmod"], 
                                         capture_output=True, 
                                         text=True)
        
        if "bcm2835_v4l2" not in modprobe_result.stdout:
            print("Loading bcm2835-v4l2 module for libcamera compatibility...")
            subprocess.run(["sudo", "modprobe", "bcm2835-v4l2"], 
                           capture_output=True)
        
        # Set reasonable timeout for libcamera operations
        os.environ['LIBCAMERA_TIMEOUT_MS'] = '5000'
        
    except Exception as e:
        print(f"Note: Could not set up libcamera compatibility: {e}")

# Improved class to handle libcamera-vid process and pipe
class LibcameraVideoCapture:
    def __init__(self, width=640, height=480, framerate=15):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.process = None
        self.latest_frame = None
        self.running = False
        self.frame_available = threading.Event()
        self.lock = threading.Lock()
        self.frame_size = width * height * 3 // 2  # YUV420 size
        
    def start(self):
        try:
            # Command to run libcamera-vid and output to stdout
            cmd = [
                "libcamera-vid",
                "--width", str(self.width),
                "--height", str(self.height),
                "--framerate", str(self.framerate),
                "--codec", "yuv420",
                "--timeout", "0",  # Run continuously
                "--output", "-"  # Output to stdout
            ]
            
            print(f"Starting libcamera-vid with command: {' '.join(cmd)}")
            
            # Start the process with a large buffer
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 10  # Very large buffer
            )
            
            self.running = True
            
            # Start a thread to read frames
            self.thread = threading.Thread(target=self._read_frames)
            self.thread.daemon = True
            self.thread.start()
            
            time.sleep(3)  # Give more time to start up
            
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"Error: libcamera-vid exited prematurely with code {self.process.poll()}")
                error_output = self.process.stderr.read().decode('utf-8', errors='replace')
                print(f"Error output: {error_output}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error starting libcamera-vid: {e}")
            self.stop()
            return False
    
    def _read_frames(self):
        """Thread function that continuously reads frames from the pipe"""
        buffer = bytearray()
        
        try:
            while self.running:
                if self.process.poll() is not None:
                    print(f"Warning: libcamera-vid process exited with code {self.process.poll()}")
                    self.running = False
                    break
                
                # Read a chunk of data
                try:
                    chunk = self.process.stdout.read(4096)  # Read in smaller chunks
                    if not chunk:  # End of stream
                        if self.running:
                            print("End of stream detected from libcamera-vid")
                            time.sleep(0.1)
                            continue
                        break
                    
                    buffer.extend(chunk)
                    
                    # Check if we have a complete frame
                    while len(buffer) >= self.frame_size:
                        # Extract a frame
                        frame_data = buffer[:self.frame_size]
                        buffer = buffer[self.frame_size:]
                        
                        # Convert YUV420 to numpy array
                        yuv = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
                        
                        # Convert YUV to BGR
                        with self.lock:
                            try:
                                self.latest_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                                self.frame_available.set()
                            except cv2.error as e:
                                print(f"OpenCV error processing frame: {e}")
                        
                except Exception as e:
                    print(f"Error reading from pipe: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in frame reading thread: {e}")
        finally:
            self.running = False
    
    def read(self):
        """Returns (success, frame) similar to OpenCV's VideoCapture"""
        # Wait for a frame to be available (with timeout)
        if self.frame_available.wait(timeout=2.0):
            with self.lock:
                if self.latest_frame is not None:
                    self.frame_available.clear()
                    return True, self.latest_frame.copy()
        
        return False, None
    
    def isOpened(self):
        """Check if camera is working"""
        return self.running and self.process is not None and self.process.poll() is None
        
    def stop(self):
        """Stop the libcamera-vid process"""
        self.running = False
        if self.process:
            try:
                # Send SIGTERM signal
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # If it doesn't terminate, force kill
                    self.process.kill()
                    self.process.wait(timeout=1)
            except Exception as e:
                print(f"Error stopping libcamera-vid process: {e}")
        
        self.process = None
        self.latest_frame = None
        print("libcamera-vid process stopped")

# Function to try direct v4l2 device access as a fallback
def try_v4l2_devices():
    # List all video devices
    try:
        devices = os.listdir('/dev')
        video_devices = [d for d in devices if d.startswith('video')]
        
        for device in video_devices:
            device_path = f"/dev/{device}"
            print(f"Trying V4L2 device: {device_path}")
            
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # Test if we can get a frame
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Successfully opened {device_path}")
                        return cap
                    time.sleep(0.5)
                
                cap.release()
                print(f"Could not read frames from {device_path}")
    except Exception as e:
        print(f"Error scanning V4L2 devices: {e}")
    
    return None

# Initialize camera using picamera2 if available, otherwise fall back to OpenCV
def init_camera():
    # Try to set up libcamera compatibility
    setup_libcamera_compatibility()
    
    # First, try picamera2 (preferred for Raspberry Pi)
    if HAS_PICAMERA2:
        try:
            print("Initializing camera with picamera2...")
            picam2 = Picamera2()
            # Configure camera
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(2)  # Give camera time to warm up
            
            # Test capture
            test_frame = picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print("picamera2 initialized successfully")
                return picam2, True, "picamera2"  # True indicates it's picamera2
            else:
                print("picamera2 started but couldn't capture valid frame")
                picam2.stop()
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
    
    # Try libcamera-vid directly
    print("Trying libcamera-vid approach...")
    libcam = LibcameraVideoCapture(width=640, height=480, framerate=15)
    if libcam.start():
        # Test if we can get a frame
        for _ in range(5):  # More attempts
            ret, test_frame = libcam.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print("libcamera-vid initialized successfully")
                return libcam, False, "libcamera-vid"  # False because it's not picamera2
            time.sleep(1)
        
        print("libcamera-vid started but couldn't capture valid frame")
        libcam.stop()
    
    # Try direct V4L2 device access
    v4l2_cap = try_v4l2_devices()
    if v4l2_cap is not None:
        return v4l2_cap, False, "opencv"
    
    # Fall back to OpenCV with various options
    print("Trying OpenCV camera methods...")
    
    # Try different camera options
    camera_options = [
        {"device": 0, "api": cv2.CAP_V4L2},
        {"device": 0, "api": cv2.CAP_ANY},
        {"device": "libcamera:///dev/video0", "api": cv2.CAP_V4L2},
        {"device": "/dev/video0", "api": cv2.CAP_V4L2},
        {"device": "device=/dev/video0,format=YUY2", "api": cv2.CAP_V4L2}
    ]
    
    for opt in camera_options:
        try:
            print(f"Trying camera: {opt['device']} with API: {opt['api']}")
            cap = cv2.VideoCapture(opt['device'], opt['api'])
            
            if cap.isOpened():
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # Test read with multiple attempts
                for attempt in range(5):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Successfully connected to camera using {opt['device']}")
                        print(f"Frame shape: {test_frame.shape}")
                        return cap, False, "opencv"  # False indicates it's OpenCV
                    time.sleep(0.5)
                
                print(f"Device {opt['device']} opened but couldn't read valid frames")
                cap.release()
            else:
                print(f"Failed to open device {opt['device']}")
        except Exception as e:
            print(f"Error with device {opt['device']}: {e}")
    
    print("All camera methods failed!")
    return None, False, None

# Handle exit gracefully
running = True
def signal_handler(sig, frame):
    global running
    print("Received signal to terminate")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize camera
camera, is_picamera, camera_type = init_camera()
if camera is None:
    print("Failed to initialize any camera. Exiting.")
    exit(1)

print(f"Successfully initialized camera using {camera_type}")

last_alert_time = 0
alert_cooldown = 10  # seconds
frame_count = 0
error_count = 0
MAX_ERRORS = 10  # Increased tolerance for errors

print("Starting detection... Press 'q' to exit.")
no_frame_count = 0
MAX_NO_FRAME = 15  # Maximum consecutive frames without data before reinitialization

try:
    while running:
        try:
            # Get frame from appropriate camera
            if is_picamera:
                frame = camera.capture_array()
                ret = True
            else:
                for _ in range(3):  # Try up to 3 times to get a frame
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.1)
            
            if not ret or frame is None or frame.size == 0:
                error_count += 1
                no_frame_count += 1
                print(f"Failed to grab frame (error {error_count}/{MAX_ERRORS}, no frame {no_frame_count}/{MAX_NO_FRAME})")
                
                # If we've had too many consecutive frames without data, reinitialize
                if no_frame_count >= MAX_NO_FRAME:
                    print("Too many consecutive frames without data, reinitializing camera...")
                    if camera_type == "libcamera-vid":
                        camera.stop()
                    elif not is_picamera and camera_type == "opencv":
                        camera.release()
                    elif is_picamera:
                        camera.stop()
                        
                    camera, is_picamera, camera_type = init_camera()
                    if camera is None:
                        print("Camera reinitialization failed, retrying in 5 seconds...")
                        time.sleep(5)
                        camera, is_picamera, camera_type = init_camera()
                        if camera is None:
                            print("Second reinitialization attempt failed. Exiting.")
                            break
                    no_frame_count = 0
                    error_count = 0
                
                # If we've had some errors but not consecutive no-frames, just wait and continue
                if error_count >= MAX_ERRORS and no_frame_count < MAX_NO_FRAME:
                    print("Waiting for camera to recover...")
                    time.sleep(2)
                    error_count = 0
                
                time.sleep(0.5)
                continue
            
            # Reset error counts when we successfully get a frame
            error_count = 0
            no_frame_count = 0
            frame_count += 1
            
            # Only process every 3rd frame to reduce CPU load
            if frame_count % 3 == 0:
                # Ensure frame is in correct format for YOLO
                if is_picamera:
                    # picamera2 returns RGB, convert to BGR for OpenCV
                    frame_for_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_for_display = frame.copy()
                
                results = model(frame, imgsz=640)[0]
                
                # Process detections
                for r in results.boxes:
                    cls_id = int(r.cls[0])
                    conf = float(r.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_for_display, f"{class_name} {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if class_name.lower() == "drowning" and conf > 0.5:
                        print(f"Drowning detected (conf={conf:.2f})")
                        if time.time() - last_alert_time > alert_cooldown:
                            send_alert()
                            last_alert_time = time.time()
            
                # Add status text
                cv2.putText(frame_for_display, f"Status: Running ({camera_type})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show FPS
                cv2.putText(frame_for_display, f"Frame: {frame_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Drowning Detection System", frame_for_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested exit")
                break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            error_count += 1
            time.sleep(1)

except Exception as e:
    print(f"Unhandled exception: {e}")
finally:
    # Clean up
    if camera is not None:
        if camera_type == "libcamera-vid":
            camera.stop()
        elif not is_picamera and camera_type == "opencv":
            camera.release()
        elif is_picamera:
            camera.stop()
    cv2.destroyAllWindows()
    print("Program terminated")
