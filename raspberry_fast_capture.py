import cv2
import numpy as np
import subprocess as sp
import time
import atexit
import picamera
import picamera.array

# Video capture parameters
(w, h) = (640, 240)
fps = 90  # Adjusted for Raspberry Pi 4
max_frames = 300

def capture_frames():
    frames = []
    
    try:
        with picamera.PiCamera() as camera:
            # Configure camera settings
            camera.resolution = (w, h)
            camera.framerate = fps
            camera.exposure_mode = 'sports'  # Fast motion mode
            
            # Create a video port capture object
            with picamera.array.PiRGBArray(camera) as stream:
                print("Recording...")
                start_time = time.time()
                frame_count = 0
                
                for frame in camera.capture_continuous(stream, format='bgr', use_video_port=True):
                    # Convert to grayscale
                    gray_frame = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
                    
                    # Append frame
                    frames.append(gray_frame)
                    frame_count += 1
                    
                    # Clear the stream for the next frame
                    stream.seek(0)
                    stream.truncate()
                    
                    # Stop after max frames
                    if frame_count >= max_frames:
                        break
                
                end_time = time.time()
                print(f"Captured {frame_count} frames in {end_time - start_time:.2f} seconds")
                print(f"Actual FPS: {frame_count / (end_time - start_time):.2f}")
    
    except Exception as e:
        print(f"Error capturing frames: {e}")
        return []
    
    return frames

def save_video(frames):
    if not frames:
        print("No frames to save")
        return
    
    print("Writing frames to disk...")
    out = cv2.VideoWriter("slow_motion.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h), isColor=False)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print("Video saved as slow_motion.avi")

def display_frames(frames):
    print("Displaying frames...")
    for frame in frames:
        cv2.imshow("Slow Motion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()

def main():
    # Capture frames
    frames = capture_frames()
    
    # Save video
    save_video(frames)
    
    # Display frames
    display_frames(frames)

if __name__ == "__main__":
    main()
