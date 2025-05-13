from picamera2 import Picamera2
import cv2
import time
import numpy as np

def main():
    # Initialize Picamera2
    picam2 = Picamera2()

    # Set resolution and configure camera
    width, height = 640, 240
    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "XRGB8888"}
    )
    picam2.configure(config)
    picam2.start()

    # Frame capture settings
    max_frames = 300
    frames = []
    print("Recording...")

    start_time = time.time()

    for i in range(max_frames):
        # Always get the latest frame
        frame = picam2.capture_array()

        # Optional: convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save frame to list
        frames.append(gray)

        # Display real-time frame
        cv2.imshow("Live Feed", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    fps = len(frames) / (end_time - start_time)
    print(f"Done! Captured {len(frames)} frames at {fps:.2f} fps.")

    # Stop camera
    picam2.stop()

    # Save as video file
    print("Saving video...")
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
        out.write(rgb)
    out.release()

    print("Playback video...")
    for f in frames:
        cv2.imshow("Playback", f)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
