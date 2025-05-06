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

# Global variables
running = True
model = None
PROJECT_ID = "drowning-notification"
DEVICE_TOKEN = "fidLtsmJTvqYV7a7oYz1Cp:APA91bE_qVjw7hLQt_WjTYkrwNlYWhq6_V4HgvJwrzd7huMzLaR_YK5ejOVtvjGIkkRdoUl4RJnYDNHX0H595yNM449cIPwI1UpIxkwFKuoYyAp3_Eq8498"
SERVICE_ACCOUNT_FILE = "drowning-notification-firebase-adminsdk-fbsvc-55feeeaf68.json"

# Load YOLO model
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return None
    
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Get Firebase access token
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

# Send notification
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

# Class for libcamera video capture
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
            # Command to run libcamera-vid
            cmd = [
                "libcamera-vid",
                "--width", str(self.width),
                "--height", str(self.height),
                "--framerate", str(self.framerate),
                "--codec", "yuv420",
                "--timeout", "0",
                "--output", "-"
            ]
            
            print(f"Starting libcamera-vid with command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 10
            )
            
            self.running = True
            
            # Start thread to read frames
            self.thread = threading.Thread(target=self._read_frames)
            self.thread.daemon = True
            self.thread.start()
            
            time.sleep(2)  # Give time to start up
            
            # Check if process is running
            if self.process.poll() is not None:
                print(f"Error: libcamera-vid exited with code {self.process.poll()}")
                error_output = self.process.stderr.read().decode('utf-8', errors='replace')
                print(f"Error output: {error_output}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error starting libcamera-vid: {e}")
            self.stop()
            return False
    
    def _read_frames(self):
        buffer = bytearray()
        
        while self.running:
            if self.process and self.process.poll() is not None:
                print(f"Warning: libcamera-vid process exited with code {self.process.poll()}")
                self.running = False
                break
            
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    if self.running:
                        time.sleep(0.1)
                        continue
                    break
                
                buffer.extend(chunk)
                
                # Process complete frames
                while len(buffer) >= self.frame_size:
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
    
    def read(self):
        if self.frame_available.wait(timeout=1.0):
            with self.lock:
                if self.latest_frame is not None:
                    self.frame_available.clear()
                    return True, self.latest_frame.copy()
        
        return False, None
    
    def isOpened(self):
        return self.running and self.process is not None and self.process.poll() is None
        
    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                print(f"Error stopping libcamera-vid process: {e}")
        
        self.process = None
        self.latest_frame = None
        print("libcamera-vid process stopped")

# Initialize camera
def init_camera():
    # Try picamera2 first
    if HAS_PICAMERA2:
        try:
            print("Initializing camera with picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            
            test_frame = picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print("picamera2 initialized successfully")
                return picam2, True, "picamera2"
            else:
                picam2.stop()
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
    
    # Try libcamera-vid approach
    print("Trying libcamera-vid approach...")
    libcam = LibcameraVideoCapture(width=640, height=480, framerate=15)
    if libcam.start():
        for _ in range(3):
            ret, test_frame = libcam.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print("libcamera-vid initialized successfully")
                return libcam, False, "libcamera-vid"
            time.sleep(1)
        
        libcam.stop()
    
    # Fall back to OpenCV
    print("Trying OpenCV camera...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("OpenCV camera initialized successfully")
                return cap, False, "opencv"
            cap.release()
    except Exception as e:
        print(f"OpenCV camera error: {e}")
    
    return None, False, None

# Signal handler for graceful exit
def signal_handler(sig, frame):
    global running
    print("Received signal to terminate")
    running = False

# Main function
def main():
    global running
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load the model
    model = load_model()
    if model is None:
        print("Failed to load YOLO model. Exiting.")
        return
    
    # Initialize camera
    camera, is_picamera, camera_type = init_camera()
    if camera is None:
        print("Failed to initialize any camera. Exiting.")
        return
    
    print(f"Successfully initialized camera using {camera_type}")
    
    # Main detection loop
    last_alert_time = 0
    alert_cooldown = 10  # seconds
    frame_count = 0
    error_count = 0
    no_frame_count = 0
    
    print("Starting detection... Press 'q' to exit.")
    
    try:
        while running:
            try:
                # Get frame
                if is_picamera:
                    frame = camera.capture_array()
                    ret = True
                else:
                    ret, frame = camera.read()
                
                if not ret or frame is None or frame.size == 0:
                    error_count += 1
                    no_frame_count += 1
                    print(f"Failed to grab frame (attempt {no_frame_count}/10)")
                    
                    # Reinitialize camera if needed
                    if no_frame_count >= 10:
                        print("Reinitializing camera...")
                        if camera_type == "libcamera-vid":
                            camera.stop()
                        elif not is_picamera:
                            camera.release()
                        elif is_picamera:
                            camera.stop()
                            
                        camera, is_picamera, camera_type = init_camera()
                        if camera is None:
                            print("Camera reinitialization failed. Exiting.")
                            break
                        no_frame_count = 0
                        error_count = 0
                    
                    time.sleep(0.5)
                    continue
                
                # Reset error counts
                error_count = 0
                no_frame_count = 0
                frame_count += 1
                
                # Only process every 3rd frame for performance
                if frame_count % 3 == 0:
                    # Prepare frame for display
                    if is_picamera:
                        frame_for_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_for_display = frame.copy()
                    
                    # Run YOLO detection
                    results = model(frame, imgsz=640)[0]
                    
                    # Process detections
                    for r in results.boxes:
                        cls_id = int(r.cls[0])
                        conf = float(r.conf[0])
                        class_name = model.names[cls_id]
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, r.xyxy[0])
                        cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame_for_display, f"{class_name} {conf:.2f}", 
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Send alert if drowning detected
                        if class_name.lower() == "drowning" and conf > 0.5:
                            print(f"Drowning detected (conf={conf:.2f})")
                            if time.time() - last_alert_time > alert_cooldown:
                                send_alert()
                                last_alert_time = time.time()
                
                    # Show status information
                    cv2.putText(frame_for_display, f"Camera: {camera_type}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_for_display, f"Frame: {frame_count}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Drowning Detection", frame_for_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested exit")
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                error_count += 1
                time.sleep(1)
                
                if error_count > 20:
                    print("Too many errors, exiting")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        if camera is not None:
            if camera_type == "libcamera-vid":
                camera.stop()
            elif not is_picamera:
                camera.release()
            elif is_picamera:
                camera.stop()
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()
