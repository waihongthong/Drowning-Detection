import cv2
import numpy as np
import time
import subprocess
import os
import signal
from threading import Thread
from ultralytics import YOLO

class LibcameraCapture:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.process = None
        self.running = False
        self.frame = None
        self.pipe_path = "/tmp/libcamera_pipe"
        
    def start(self):
        # Create a named pipe if it doesn't exist
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)
            
        # Start libcamera-vid process outputting to the pipe
        cmd = [
            "libcamera-vid",
            "--width", str(self.width),
            "--height", str(self.height),
            "--output", self.pipe_path,
            "--codec", "yuv420",
            "--timeout", "0",
            "--nopreview"
        ]
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.running = True
        
        # Start a thread to read frames from the pipe
        self.thread = Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()
        
        # Give time for everything to initialize
        time.sleep(2)
        
    def _read_frames(self):
        # Open the named pipe for reading
        pipe_fd = os.open(self.pipe_path, os.O_RDONLY)
        frame_size = self.width * self.height * 3 // 2  # YUV420 size
        
        while self.running:
            try:
                # Read raw YUV data from the pipe
                yuv_data = os.read(pipe_fd, frame_size)
                
                if len(yuv_data) != frame_size:
                    continue
                    
                # Convert YUV to BGR for OpenCV
                y = np.frombuffer(yuv_data[:self.width*self.height], dtype=np.uint8).reshape((self.height, self.width))
                u = np.frombuffer(yuv_data[self.width*self.height:self.width*self.height*5//4], 
                                   dtype=np.uint8).reshape((self.height//2, self.width//2))
                v = np.frombuffer(yuv_data[self.width*self.height*5//4:], 
                                   dtype=np.uint8).reshape((self.height//2, self.width//2))
                
                # Resize U and V to match Y dimensions
                u_resized = cv2.resize(u, (self.width, self.height))
                v_resized = cv2.resize(v, (self.width, self.height))
                
                # Stack the channels and convert from YUV to BGR
                yuv = cv2.merge([y, u_resized, v_resized])
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                self.frame = bgr
                
            except Exception as e:
                print(f"Error reading frame: {e}")
                break
                
        os.close(pipe_fd)
        
    def read(self):
        return self.frame is not None, self.frame
        
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.process:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=1.0)
            
        # Clean up pipe
        if os.path.exists(self.pipe_path):
            os.unlink(self.pipe_path)

def main():
    # Load YOLO model
    model_path = "best.pt"
    model = YOLO(model_path)
    print(f"✅ Loaded YOLOv8 model from {model_path}")

    # Initialize libcamera capture
    cap = LibcameraCapture(width=640, height=480)
    cap.start()
    print("📷 Camera initialized and started")

    # Create window for display
    cv2.namedWindow("Drowning Detection", cv2.WINDOW_NORMAL)
    print("🚨 Starting detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠️ No frame received, waiting...")
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

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
