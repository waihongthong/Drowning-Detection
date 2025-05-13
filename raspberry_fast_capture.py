from picamera2 import Picamera2
import cv2
import time

class CameraHandler:
    def __init__(self):
        self.picam2 = Picamera2()

    def global_camera_info(self):
        # This is an example function returning dummy camera information.
        # You should adjust this to reflect your system's camera info.
        return [
            {'Num': 0, 'Name': 'Camera 1'},
            {'Num': 1, 'Name': 'Camera 2'}
        ]
    
    def get_camera(self, camera_num):
        # Retrieve the camera info based on camera_num
        camera_info = self.global_camera_info()

        if camera_num >= 0 and camera_num < len(camera_info):
            return camera_info[camera_num]['Num']
        else:
            print(f"Invalid camera_num {camera_num}, defaulting to camera 0.")
            return camera_info[0]['Num']  # Default to the first camera if invalid index

    def start_camera(self, camera_num):
        # Select the correct camera
        camera_num = self.get_camera(camera_num)

        # Configure the camera for 640x240 resolution
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (640, 240), "format": "XRGB8888"}))
        self.picam2.start()

        frames = []
        max_frames = 300
        print("Recording...")

        start_time = time.time()

        for _ in range(max_frames):
            frame = self.picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        end_time = time.time()
        fps = len(frames) / (end_time - start_time)
        print(f"Done! Result: {fps:.2f} fps")

        # Save frames to an AVI file
        print("Writing frames to disk...")
        out = cv2.VideoWriter("slow_motion.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 240))
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(rgb_frame)
        out.release()

        # Display frames using OpenCV
        print("Displaying frames with OpenCV...")
        for frame in frames:
            cv2.imshow("Slow Motion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    camera_handler = CameraHandler()

    # Choose the camera index you want to use, e.g., 0 for the first camera
    camera_num = 0  # Change this if you have multiple cameras
    camera_handler.start_camera(camera_num)
