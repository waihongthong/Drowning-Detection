import time
import threading
import subprocess
import RPi.GPIO as GPIO
from ultralytics import YOLO
import cv2
import numpy as np
import os

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

# Function to trigger alarm for 10 seconds
def trigger_alarm(duration=10):
    print("🚨 ALARM: Drowning detected!")
    buzzer_pwm.start(50)  # Start buzzer (50% duty cycle)
    GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on LED
    time.sleep(duration)
    buzzer_pwm.stop()  # Stop buzzer
    GPIO.output(LED_PIN, GPIO.LOW)  # Turn off LED
    print("Alarm stopped")

# Check if we're running in headless mode (without display)
HEADLESS_MODE = True  # Set to True to run without GUI display

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
            # Try multiple frames to ensure stability
            success_count = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.1)
            
            if success_count >= 3:  # At least 3 successful frames
                print(f"✅ Successfully opened camera with V4L2 backend ({success_count}/5 test frames)")
                cv2.imwrite("test_cv2_v4l2.jpg", frame)
                return cap
            else:
                print("⚠️ Camera opened with V4L2 but returned inconsistent frames")
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
            cap = cv2.VideoCapture(device_path)
            
            # Set explicit camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if cap.isOpened():
                # Try multiple frames to ensure stability
                success_count = 0
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success_count += 1
                    time.sleep(0.1)
                
                if success_count >= 3:  # At least 3 successful frames
                    print(f"✅ Successfully opened {device_path} with {success_count}/5 test frames")
                    cv2.imwrite(f"test_cv2_{i}.jpg", frame)
                    return cap
                else:
                    print(f"⚠️ Device {device_path} opened but returned inconsistent frames")
                    cap.release()
            else:
                print(f"❌ Could not open {device_path}")
    
    # Then try indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        # Set explicit camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if cap.isOpened():
            # Try multiple frames to ensure stability
            success_count = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.1)
            
            if success_count >= 3:  # At least 3 successful frames
                print(f"✅ Successfully opened camera index {i} with {success_count}/5 test frames")
                cv2.imwrite(f"test_cv2_idx_{i}.jpg", frame)
                return cap
            else:
                print(f"⚠️ Camera {i} opened but returned inconsistent frames")
                cap.release()
    
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
    # Try LibCamera first (for Pi OS Bullseye and newer)
    print("Trying LibCamera...")
    camera = LibCameraHandler()
    if camera.initialize():
        return camera, "libcamera"
    
    # Try legacy RaspiStill (for older Pi OS)
    print("Trying legacy RaspiStill...")
    camera = RaspiStillHandler()
    if camera.initialize():
        return camera, "raspistill"
    
    # Fallback to OpenCV
    print("Trying OpenCV camera...")
    cap = try_cv2_camera()
    if cap is not None:
        return OpenCVCameraHandler(cap), "opencv"
    
    # No camera available
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

def main():
    print("Starting drowning detection system...")
    print("Running in headless mode (no GUI display)" if HEADLESS_MODE else "Running with GUI display")
    
    # Create output directory if it doesn't exist
    os.makedirs("detections", exist_ok=True)
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('best.pt')  # Or use 'yolov8n.pt' for a pretrained nano model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        GPIO.cleanup()
        return
    
    # Initialize camera
    camera, camera_type = initialize_camera()
    if not camera:
        print("❌ Could not initialize any camera. Exiting.")
        GPIO.cleanup()
        return
    
    print(f"✅ Using {camera_type} camera")
    
    # Variable to track if alarm is active
    alarm_active = False
    last_camera_check = time.time()
    camera_recovery_attempts = 0
    last_save_time = 0  # For periodic image saving (headless mode)
    
    print("Starting detection loop. Press Ctrl+C to quit.")
    
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
            
            # Print a periodic status message
            if int(time.time()) % 10 == 0:  # Every 10 seconds
                print(f"✅ Camera running: {camera_type}")
            
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
                        # Details are logged when saving image
            
            # Trigger alarm if drowning detected (and alarm not already active)
            if drowning_detected and not alarm_active:
                alarm_active = True
                # Save detection image when drowning is detected
                save_detection_image(frame, results, drowning_detected)
                threading.Thread(target=trigger_alarm).start()
            
            # Reset alarm status after it finishes
            if not drowning_detected and alarm_active:
                alarm_active = False
            
            # In headless mode, periodically save detection images
            if HEADLESS_MODE and time.time() - last_save_time > 30:  # Every 30 seconds
                last_save_time = time.time()
                save_detection_image(frame, results, drowning_detected)
            
            # In display mode, show the frame
            if not HEADLESS_MODE:
                # Create a frame with detections for display
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
                        (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 2)
                
                try:
                    # Display the frame
                    cv2.imshow("Drowning Detection", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"⚠️ Display error: {e}")
                    print("Switching to headless mode...")
                    HEADLESS_MODE = True
            
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
        
        # Only try to close windows in display mode
        if not HEADLESS_MODE:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        buzzer_pwm.stop()
        GPIO.cleanup()
        print("✅ Resources released.")

if __name__ == "__main__":
    main()
