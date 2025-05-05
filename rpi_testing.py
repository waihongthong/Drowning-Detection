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
from io import BytesIO

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

def try_cv2_camera():
    """Try to initialize camera using OpenCV's VideoCapture with improved settings"""
    print("\nTrying OpenCV camera access...")
    
    # Try different camera indices
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        # Set explicit camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if cap.isOpened():
            # Try to grab a test frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✅ Successfully opened camera index {i}")
                return cap
            else:
                print(f"⚠️ Camera {i} opened but couldn't grab frame")
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
    global latest_frame
    
    print("Starting drowning detection system...")
    
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
    cap = try_cv2_camera()
    if not cap:
        print("❌ Could not initialize camera. Exiting.")
        GPIO.cleanup()
        return
    
    camera = OpenCVCameraHandler(cap)
    print("✅ Camera initialized successfully!")
    
    # Create window for display
    window_name = "Drowning Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Resizable window
    
    # Variables for detection
    alarm_active = False
    
    print("Camera view started. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
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
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, 
                      (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                      (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save image", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add buzzer and LED status
            buzzer_status = "ON" if alarm_active else "OFF"
            cv2.putText(display_frame, f"Buzzer & LED: {buzzer_status}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                      (0, 0, 255) if alarm_active else (255, 255, 255), 2)
            
            # Display the frame
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
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting due to 'q' key press...")
                break
            
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
        
        cv2.destroyAllWindows()
        
        buzzer_pwm.stop()
        GPIO.cleanup()
        print("✅ Resources released.")

if __name__ == "__main__":
    main()
