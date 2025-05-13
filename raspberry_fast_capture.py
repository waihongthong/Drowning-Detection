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

    # Frame capture setup
    frames = []
    print("Recording... Press 'q' to stop.")

    frame_count = 0
    prev_time = time.time()

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Overlay FPS on frame
        display_frame = gray.copy()
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

        # Show real-time feed
        cv2.imshow("Live Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop and summarize
    total_time = time.time() - prev_time
    avg_fps = frame_count / total_time
    print(f"Done! Captured {frame_count} frames. Average FPS: {avg_fps:.2f}")

    picam2.stop()

    # Save video
    print("Saving video...")
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        out.write(rgb)
    out.release()

    # Playback
    print("Playback video... Press 'q' to quit.")
    for f in frames:
        cv2.imshow("Playback", f)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
