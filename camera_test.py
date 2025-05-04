import cv2
import time
import os

def test_camera_indices():
    """Test all possible camera indices and save sample images"""
    print("\n=== CAMERA INDEX TEST ===")
    working_indices = []
    
    # Try different camera indices
    for i in range(3):  # Try indices 0, 1, 2
        print(f"\nTesting camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if not cap.isOpened():
            print(f"❌ Could not open camera with index {i}")
        else:
            print(f"✅ Successfully opened camera with index {i}")
            
            # Try to read multiple frames (sometimes the first few can fail)
            success = False
            for attempt in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Successfully captured frame from camera {i} (attempt {attempt+1})")
                    # Save the frame to confirm it works
                    filename = f'camera_test_{i}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved test image as {filename}")
                    
                    # Display image dimensions
                    height, width = frame.shape[:2]
                    print(f"📏 Image dimensions: {width}x{height}")
                    
                    working_indices.append(i)
                    success = True
                    break
                else:
                    print(f"❌ Failed to capture frame from camera {i} (attempt {attempt+1})")
                    time.sleep(0.5)
            
            if not success:
                print(f"❌ Could not capture any frames from camera {i} after multiple attempts")
        
        cap.release()
        time.sleep(1)
    
    return working_indices

def check_camera_devices():
    """Check for available camera devices in the system"""
    print("\n=== SYSTEM CAMERA CHECK ===")
    
    # Check for video devices
    print("\nChecking for video devices:")
    try:
        devices = os.popen("ls -l /dev/video*").read().strip()
        if devices:
            print("✅ Found video devices:")
            print(devices)
        else:
            print("❌ No video devices found in /dev")
    except:
        print("❌ Error checking video devices")
    
    # Check Pi Camera specifically
    print("\nChecking for Raspberry Pi Camera:")
    try:
        camera_status = os.popen("vcgencmd get_camera").read().strip()
        print(f"📊 Pi Camera status: {camera_status}")
        
        if "supported=1 detected=1" in camera_status:
            print("✅ Pi Camera is detected and supported")
        else:
            print("⚠️ Pi Camera may not be properly connected or enabled")
            print("   Run 'sudo raspi-config' and enable camera in Interfacing Options")
    except:
        print("❌ Error checking Pi Camera status")

def test_camera_stream(camera_index=0, duration=10):
    """Test camera stream for a specific duration"""
    print(f"\n=== TESTING CAMERA STREAM (INDEX {camera_index}) ===")
    print(f"Streaming for {duration} seconds. Press 'q' to quit early.")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera with index {camera_index}")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            time.sleep(0.5)
            continue
        
        frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(f"Camera Test (Index {camera_index})", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Camera test complete")
    print(f"📊 Captured {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"📊 Average FPS: {fps:.2f}")
    
    return True

def main():
    print("===== RASPBERRY PI CAMERA DIAGNOSTIC TOOL =====")
    print("This script will help diagnose camera connection issues")
    
    # Check system camera devices
    check_camera_devices()
    
    # Test camera indices
    working_indices = test_camera_indices()
    
    if working_indices:
        print(f"\n✅ Working camera indices found: {working_indices}")
        
        # Test stream with first working camera
        test_camera_stream(working_indices[0], duration=10)
        
        print("\n✅ DIAGNOSIS COMPLETE")
        print(f"Your camera is working at index {working_indices[0]}")
        print(f"Use this index in your drowning detection script:")
        print(f"cap = cv2.VideoCapture({working_indices[0]})")
    else:
        print("\n❌ NO WORKING CAMERAS FOUND")
        print("Please check:")
        print("1. Camera connections (USB or ribbon cable)")
        print("2. Run 'sudo raspi-config' to enable Pi Camera")
        print("3. Try rebooting your Raspberry Pi")

if __name__ == "__main__":
    main()