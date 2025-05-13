from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 240), "format": "XRGB8888"}))
picam2.start()

frames = []
max_frames = 300

print("Recording...")
start = time.time()

for _ in range(max_frames):
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

end = time.time()
fps = len(frames) / (end - start)
print(f"Done! Result: {fps:.2f} fps")

# Save video
out = cv2.VideoWriter("slow_motion.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 240))
for f in frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
out.release()

# Display video
for f in frames:
    cv2.imshow("Slow Motion", f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
