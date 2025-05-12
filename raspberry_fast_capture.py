import cv2
import numpy as np
import subprocess as sp
import time
import atexit

def capture_high_speed_video():
    # Video capture parameters
    (w, h) = (640, 240)
    bytesPerFrame = w * h
    fps = 90  # Adjusted for more reliable capture

    # Updated command for modern Raspberry Pi
    videoCmd = [
        "libcamera-vid", 
        "-t", "0",  # Run indefinitely
        "--width", str(w), 
        "--height", str(h), 
        "--framerate", str(fps), 
        "--codec", "yuv420", 
        "--output", "-"
    ]

    frames = []
    max_frames = 300
    N_frames = 0

    try:
        # Start the camera process
        cameraProcess = sp.Popen(videoCmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
        atexit.register(cameraProcess.terminate)

        print("Recording...")
        start_time = time.time()

        while True:
            # Read a frame
            rawFrame = cameraProcess.stdout.read(bytesPerFrame)
            if not rawFrame:
                break

            # Convert raw bytes to numpy array
            frame = np.frombuffer(rawFrame, dtype=np.uint8)
            frame = frame.reshape((h, w))

            # Optional frame processing 
            # Uncomment and modify as needed
            # frame = cv2.Canny(frame, 50, 150)  # Edge detection example

            frames.append(frame)
            N_frames += 1

            if N_frames >= max_frames:
                break

        end_time = time.time()
        
        # Calculate and print actual frame rate
        elapsed_seconds = end_time - start_time
        print(f"Capture complete. Actual frame rate: {N_frames / elapsed_seconds:.2f} fps")

        # Save frames as video
        print("Writing frames to video file...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter("high_speed_capture.avi", fourcc, 30, (w, h), isColor=False)
        
        for frame in frames:
            out.write(frame)
        out.release()

        # Display frames
        print("Displaying captured frames...")
        for frame in frames:
            cv2.imshow("High-Speed Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cv2.destroyAllWindows()
        if 'cameraProcess' in locals():
            cameraProcess.terminate()

if __name__ == "__main__":
    capture_high_speed_video()
