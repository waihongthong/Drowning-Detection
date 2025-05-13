import cv2
import numpy as np
import subprocess as sp
import time
import os
import signal
import atexit
import sys

def capture_high_speed_video():
    # Configuration for Pi 4 with Camera v2.1
    # The Camera v2.1 can handle higher resolutions
    (w, h) = (640, 480)  # Standard resolution for v2.1
    fps = 40  # Camera v2.1 can handle higher frame rates on Pi 4
    duration = 10  # Record for 10 seconds
    
    print(f"Raspberry Pi 4 + Camera v2.1 High Speed Capture")
    print(f"Resolution: {w}x{h}, Target FPS: {fps}, Duration: {duration}s")
    
    # Method 1: PiCamera method (legacy but reliable)
    try:
        from picamera import PiCamera
        use_picamera = True
        print("Using PiCamera library")
    except ImportError:
        use_picamera = False
        print("PiCamera not available, falling back to libcamera")
    
    # Prepare output directory
    output_dir = "camera_output"
    os.makedirs(output_dir, exist_ok=True)
    
    frames = []
    
    if use_picamera:
        try:
            import io
            from picamera import PiCamera
            
            # Initialize camera
            camera = PiCamera()
            camera.resolution = (w, h)
            camera.framerate = fps
            
            # Let camera warm up
            print("Warming up camera...")
            time.sleep(2)
            
            # Setup in-memory stream
            stream = io.BytesIO()
            
            print(f"Starting capture with PiCamera...")
            start_time = time.time()
            frame_count = 0
            
            # Capture frames for specified duration
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                # Get frame data
                stream.seek(0)
                data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                frames.append(image)
                
                # Reset stream for next frame
                stream.seek(0)
                stream.truncate()
                
                frame_count += 1
                
                # Print progress
                if frame_count % 10 == 0:
                    current_time = time.time() - start_time
                    print(f"Captured {frame_count} frames ({frame_count/current_time:.1f} fps)")
                
                # Check if we've reached the time limit
                if time.time() - start_time >= duration:
                    break
            
            camera.close()
            
        except Exception as e:
            print(f"PiCamera method failed: {e}")
            use_picamera = False
    
    # Method 2: libcamera method (newer API)
    if not use_picamera or not frames:
        print("\nUsing libcamera method...")
        
        # Test if camera is accessible
        test_file = f"{output_dir}/test_image.jpg"
        test_cmd = f"libcamera-still -t 1000 -o {test_file}"
        
        print(f"Testing camera with: {test_cmd}")
        os.system(test_cmd)
        
        if os.path.exists(test_file):
            print(f"Camera test successful, image saved to {test_file}")
        else:
            print("Camera test failed, please check camera connection")
            return
        
        # Setup raw video capture
        print("\nCapturing raw video...")
        raw_video_file = f"{output_dir}/raw_capture.h264"
        
        # Use libcamera-vid to capture raw video
        vid_cmd = [
            "libcamera-vid",
            "--width", str(w),
            "--height", str(h),
            "--framerate", str(fps),
            "--output", raw_video_file,
            "--timeout", str(duration * 1000),  # ms
            "--nopreview"
        ]
        
        print(f"Running: {' '.join(vid_cmd)}")
        capture_process = sp.run(vid_cmd)
        
        if capture_process.returncode != 0:
            print(f"Video capture failed with exit code {capture_process.returncode}")
            return
        
        print(f"Raw video saved to {raw_video_file}")
        
        # Now extract frames from the raw video using OpenCV
        print("\nExtracting frames from video...")
        cap = cv2.VideoCapture(raw_video_file)
        
        if not cap.isOpened():
            print(f"Failed to open video file: {raw_video_file}")
            return
        
        # Get video properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {total_frames} frames at {actual_fps} fps")
        
        # Read all frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if len(frame.shape) == 3:  # Convert to grayscale if color
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            frames.append(frame)
            frame_count += 1
            
            if frame_count % 20 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"Extracted {frame_count} frames")
    
    # Process the captured frames
    if not frames:
        print("No frames were captured!")
        return
    
    print(f"\nCapture complete: {len(frames)} frames")
    
    # Save first and last frame for inspection
    first_frame_file = f"{output_dir}/first_frame.jpg"
    last_frame_file = f"{output_dir}/last_frame.jpg"
    
    cv2.imwrite(first_frame_file, frames[0])
    cv2.imwrite(last_frame_file, frames[-1])
    
    print(f"Saved first frame to {first_frame_file}")
    print(f"Saved last frame to {last_frame_file}")
    
    # Save as video
    output_video = f"{output_dir}/high_speed_capture.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]), isColor=False)
    
    print(f"\nSaving frames to video...")
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video}")
    
    # Optional: Create slow-motion version
    slow_motion = f"{output_dir}/slow_motion.avi"
    slow_out = cv2.VideoWriter(slow_motion, fourcc, 10, (frames[0].shape[1], frames[0].shape[0]), isColor=False)
    
    for frame in frames:
        slow_out.write(frame)
    
    slow_out.release()
    print(f"Slow motion video saved to {slow_motion}")

    # If we can display frames, do so
    try:
        # Check if display is available
        test_frame = np.zeros((100, 100), dtype=np.uint8)
        cv2.imshow("Test", test_frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        print("\nPress ESC to stop playback")
        
        # Show frames
        for frame in frames:
            cv2.imshow("Captured Frames", frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key
                break
                
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Unable to display frames: {e}")
        print("Videos have been saved - you can view them on another device")

if __name__ == "__main__":
    try:
        capture_high_speed_video()
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Program complete")
