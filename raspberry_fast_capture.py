import cv2
import time

# Capture parameters
width = 640
height = 240
max_frames = 300

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust as needed

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frames = []
print("Recording...")

start_time = time.time()

for _ in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

end_time = time.time()
cap.release()

elapsed = end_time - start_time
print(f"Done! Result: {len(frames) / elapsed:.2f} fps")

# Write to AVI file
print("Writing frames to disk...")
out = cv2.VideoWriter("slow_motion.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
for frame in frames:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    out.write(rgb_frame)
out.release()

# Display video
print("Display frames with OpenCV...")
for frame in frames:
    cv2.imshow("Slow Motion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
