import cv2
import numpy as np
import time

def test_basic_capture():
    """Test raw camera capture performance"""
    print("Testing basic camera capture...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual camera properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Camera reports: {actual_fps} FPS at {actual_width}x{actual_height}")
    
    # Test raw capture speed
    start_time = time.time()
    frame_count = 0
    
    while frame_count < 100:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            # Just show the frame - no processing
            cv2.imshow('Raw Capture Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    print(f"Raw capture: {actual_fps:.1f} FPS")
    
    cap.release()
    cv2.destroyAllWindows()
    return actual_fps

def test_fullscreen_scaling():
    """Test impact of fullscreen scaling"""
    print("\nTesting fullscreen scaling...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow('Fullscreen Test', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Fullscreen Test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    start_time = time.time()
    frame_count = 0
    
    while frame_count < 100:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            # Get screen dimensions and scale
            try:
                rect = cv2.getWindowImageRect('Fullscreen Test')
                screen_width = rect[2] if rect[2] > 0 else 1920
                screen_height = rect[3] if rect[3] > 0 else 1080
            except:
                screen_width = 1920
                screen_height = 1080
            
            # Scale and crop as in original
            frame_h, frame_w = frame.shape[:2]
            scale_w = screen_width / frame_w
            scale_h = screen_height / frame_h
            scale = max(scale_w, scale_h)
            
            new_width = int(frame_w * scale)
            new_height = int(frame_h * scale)
            result_scaled = cv2.resize(frame, (new_width, new_height))
            
            y_offset = (new_height - screen_height) // 2
            x_offset = (new_width - screen_width) // 2
            result_cropped = result_scaled[y_offset:y_offset+screen_height, x_offset:x_offset+screen_width]
            
            cv2.imshow('Fullscreen Test', result_cropped)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    print(f"Fullscreen scaling: {actual_fps:.1f} FPS")
    
    cap.release()
    cv2.destroyAllWindows()
    return actual_fps

def test_camera_formats():
    """Test different camera formats and resolutions"""
    print("\nTesting camera formats...")
    cap = cv2.VideoCapture(0)
    
    # Test different resolutions
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p")
    ]
    
    for width, height, name in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"{name}: Set {width}x{height}, got {actual_w}x{actual_h} @ {actual_fps} FPS")
    
    cap.release()

def test_threading_impact():
    """Test if threading improves capture"""
    print("\nTesting threaded vs non-threaded capture...")
    
    # First test non-threaded
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    start_time = time.time()
    frame_count = 0
    
    while frame_count < 100:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            # Simulate some processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Non-threaded', gray)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    elapsed = time.time() - start_time
    non_threaded_fps = frame_count / elapsed
    print(f"Non-threaded with processing: {non_threaded_fps:.1f} FPS")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Performance Testing for Glitchy Silhouette")
    print("==========================================")
    
    # Run tests
    raw_fps = test_basic_capture()
    fullscreen_fps = test_fullscreen_scaling()
    test_camera_formats()
    test_threading_impact()
    
    print("\nSummary:")
    print(f"Raw capture: {raw_fps:.1f} FPS")
    print(f"Fullscreen: {fullscreen_fps:.1f} FPS")
    print(f"Performance loss from fullscreen: {((raw_fps - fullscreen_fps) / raw_fps * 100):.1f}%")

if __name__ == "__main__":
    main()