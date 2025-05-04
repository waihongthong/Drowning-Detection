import time
import threading
import RPi.GPIO as GPIO
from ultralytics import YOLO
import cv2
import numpy as np
import os

# ----- Check if picamera2 is available -----
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA2 = True
    print("✅ Successfully imported picamera2")
except ImportError:
    HAVE_PICAMERA2 = False
    print("❌ Could not import picamera2. Will try fallback methods.")

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

class PiCamera2Handler:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        self.width = width
        self.height = height
        self.frame = None
        
    def initialize(self):
        try:
            # Configure the camera
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            
            # Start the camera
            self.picam2.start()
            time.sleep(2)  # Give camera time to warm up
            
            # Try capturing a test frame
            test_frame = self.picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print("✅ PiCamera2 initialized successfully!")
                cv2.imwrite("camera_test.jpg", test_frame)
                print("✅ Test image saved as camera_test.jpg")
                return True
            else:
                print("❌ PiCamera2 returned empty test frame")
                return False
        except Exception as e:
            print(f"❌ Error initializing PiCamera2: {e}")
            return False
    
    def read_frame(self):
        try:
            self.frame = self.picam2.capture_array()
            # Convert from RGB to BGR for OpenCV compatibility
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            return True, self.frame
        except Exception as e:
            print(f"❌ Error capturing frame: {e}")
            return False, None
    
    def release(self):
        if self.picam2:
            self.picam2.close()
            print("✅ PiCamera2 resources released")

def try_cv2_camera():
    """Try to initialize camera using OpenCV's VideoCapture"""
    print("\nTrying OpenCV camera access...")
    
    # First try direct device path
    for i in range(10):  # Try first 10 video devices
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            print(f"Trying {device_path}...")
            cap = cv2.VideoCapture(device_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Successfully opened {device_path}")
                    # Save test image
                    cv2.imwrite(f"test_cv2_{i}.jpg", frame)
                    print(f"✅ Test image saved as test_cv2_{i}.jpg")
                    return cap
                cap.release()
    
    # Then try indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully opened camera index {i}")
                # Save test image
                cv2.imwrite(f"test_cv2_idx_{i}.jpg", frame)
                print(f"✅ Test image saved as test_cv2_idx_{i}.jpg")
                return cap
            cap.release()
    
    print("❌ Failed to initialize any camera with OpenCV")
    return None

def main():
    print("Starting drowning detection system...")
    
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
    camera = None
    using_picamera2 = False
    using_opencv = False
    
    # Try PiCamera2 first (recommended for newer Raspberry Pi)
    if HAVE_PICAMERA2:
        print("Initializing PiCamera2...")
        camera = PiCamera2Handler()
        if camera.initialize():
            using_picamera2 = True
        else:
            print("❌ Failed to initialize PiCamera2, trying OpenCV...")
            camera = None
    
    # Fallback to OpenCV if PiCamera2 fails or isn't available
    if camera is None:
        print("Trying OpenCV camera...")
        cap = try_cv2_camera()
        if cap is not None:
            using_opencv = True
            print("✅ Using OpenCV camera")
        else:
            print("❌ Could not initialize any camera. Exiting.")
            GPIO.cleanup()
            return
    
    # Variable to track if alarm is active
    alarm_active = False
    
    print("Starting detection loop. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            if using_picamera2:
                ret, frame = camera.read_frame()
            elif using_opencv:
                ret, frame = cap.read()
            else:
                print("❌ No camera method available. Exiting.")
                break
                
            if not ret or frame is None:
                print("❌ Failed to capture frame, retrying...")
                time.sleep(0.5)
                continue
            
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
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"DROWNING {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        drowning_detected = True
            
            # Trigger alarm if drowning detected (and alarm not already active)
            if drowning_detected and not alarm_active:
                alarm_active = True
                threading.Thread(target=trigger_alarm).start()
            
            # Reset alarm status after it finishes
            if not drowning_detected and alarm_active:
                alarm_active = False
            
            # Add status text
            status = "DROWNING DETECTED" if drowning_detected else "Monitoring"
            cv2.putText(frame, "Status: " + status, 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                      (0, 0, 255) if drowning_detected else (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Drowning Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt...")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cleanup
        if using_picamera2 and camera:
            camera.release()
        elif using_opencv:
            cap.release()
        
        cv2.destroyAllWindows()
        buzzer_pwm.stop()
        GPIO.cleanup()
        print("✅ Resources released.")

if __name__ == "__main__":
    main()
