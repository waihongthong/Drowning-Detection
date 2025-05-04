import cv2
import time
import threading
import RPi.GPIO as GPIO
from ultralytics import YOLO

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

def test_cameras():
    """Test available camera indices and return the first working one"""
    for index in range(3):  # Try indices 0, 1, 2
        print(f"Testing camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully connected to camera at index {index}")
                cap.release()
                return index
            cap.release()
    return None

# Function to initialize camera with retry mechanism
def initialize_camera():
    # First try to find a working camera
    camera_index = test_cameras()
    
    if camera_index is None:
        print("No working cameras found. Please check connections.")
        return None

    # Try to open the camera
    print(f"Opening camera with index {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verify camera opened successfully
    if not cap.isOpened():
        print(f"Failed to open camera with index {camera_index}")
        return None
    
    # Try to read a test frame
    for _ in range(5):  # Try up to 5 times
        ret, frame = cap.read()
        if ret:
            print("Camera initialized successfully!")
            return cap
        print("Failed to capture frame, retrying...")
        time.sleep(1)
    
    print("Could not capture frames after multiple attempts")
    cap.release()
    return None

# Main function
def main():
    print("Starting drowning detection system...")
    
    # Load YOLOv8 model (replace 'best.pt' with your trained model)
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('best.pt')  # Or use 'yolov8n.pt' for a pretrained nano model
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        print("Error: Could not initialize camera. Exiting.")
        return
    
    # Variable to track if alarm is active
    alarm_active = False
    
    print("Starting detection loop. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                time.sleep(0.5)
                continue
                
            # Run YOLOv8 detection
            results = model(frame, verbose=False)  # Disable logging
            
            # Reset detection status
            drowning_detected = False
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = model.names[class_id]
                    
                    if class_name == 'drowning' and confidence > 0.5:  # Add confidence threshold
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
    
            # Add status text to frame
            cv2.putText(frame, "Status: " + ("DROWNING DETECTED" if drowning_detected else "Monitoring"), 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                      (0, 0, 255) if drowning_detected else (0, 255, 0), 2)
    
            # Display the frame (press 'q' to quit)
            cv2.imshow("Drowning Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        buzzer_pwm.stop()
        GPIO.cleanup()
        print("Resources released.")

if __name__ == "__main__":
    main()
