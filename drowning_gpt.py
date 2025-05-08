import cv2
import numpy as np
import time
import subprocess
import os
import tempfile
from ultralytics import YOLO

def main():
    # Load YOLO model
    model_path = "best.pt"
    model = YOLO(model_path)
    print(f"✅ Loaded YOLOv8 model from {model_path}")

    # Set up dimensions
    width, height = 640, 480
    
    # Create a temporary file for frames
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "frame.jpg")
    
    print("📷 Setting up libcamera...")
    
    # Start the libcamera preview with a low framerate
    # The camera will continuously update the same jpg file
    cmd = [
        "libcamera-still",
        "--width", str(width),
        "--height", str(height),
        "--output", temp_file,
        "--timelapse", "100",  # Take a picture every 100ms
        "--nopreview"
    ]
    
    # Start the libcamera process
    camera_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the first image to be captured
    print("🔄 Waiting for camera to initialize...")
    while not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
        time.sleep(0.5)
    
    # Give a bit more time for stable operation
    time.sleep(2)
    print("🚨 Starting detection... Press 'q' to exit.")
    
    # Create window for display
    cv2.namedWindow("Drowning Detection", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read the current frame from the file
            frame = cv2.imread(temp_file)
            
            if frame is None:
                print("⚠️ Could not read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Run YOLO detection
            results = model(frame, verbose=False)[0]
            
            # Draw results on the frame
            annotated_frame = results.plot()
            
            # Display the frame
            cv2.imshow("Drowning Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                break
            
            # Brief sleep to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("👋 Keyboard interrupt detected, exiting...")
    
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        camera_process.terminate()
        camera_process.wait(timeout=5)
        cv2.destroyAllWindows()
        
        # Remove the temporary file and directory
        if os.path.exists(temp_file):
            os.remove(temp_file)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
