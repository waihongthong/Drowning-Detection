import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO

# GPIO Setup
LED_PIN = 17     # GPIO pin for LED
BUZZER_PIN = 18  # GPIO pin for buzzer
GPIO.setmode(GPIO.BCM)
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

# Load YOLOv8 model (replace 'best.pt' with your trained model)
print("Loading YOLOv8 model...")
model = YOLO('best.pt')  # Or use 'yolov8n.pt' for a pretrained nano model
print("Model loaded!")

# Initialize camera (0 for Pi Camera, 1 for USB webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Variable to track if alarm is active
alarm_active = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Run YOLOv8 detection
        results = model(frame, verbose=False)  # Disable logging
        
        # Reset detection status
        drowning_detected = False
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                if model.names[class_id] == 'drowning':  # Check for 'drowning' class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "DROWNING", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    drowning_detected = True

        # Trigger alarm if drowning detected (and alarm not already active)
        if drowning_detected and not alarm_active:
            alarm_active = True
            threading.Thread(target=trigger_alarm).start()
        
        # Reset alarm status after it finishes
        if not drowning_detected and alarm_active:
            alarm_active = False

        # Display the frame (press 'q' to quit)
        cv2.imshow("Drowning Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Small delay to reduce CPU usage
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    buzzer_pwm.stop()
    GPIO.cleanup()
    print("Resources released.")