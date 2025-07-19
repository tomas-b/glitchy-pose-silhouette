import cv2
import numpy as np
import time

def time_operations():
    """Time individual operations to find bottlenecks"""
    print("Timing individual operations on 1920x1080 frame:")
    print("=" * 50)
    
    # Test data
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
    
    operations = []
    
    # 1. Color conversion
    start = time.time()
    for _ in range(10):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t = (time.time() - start) / 10 * 1000
    operations.append(("Color conversion (BGR->Gray)", t))
    
    # 2. MOG2 background subtraction
    mog2 = cv2.createBackgroundSubtractorMOG2()
    start = time.time()
    for _ in range(10):
        fg_mask = mog2.apply(frame, learningRate=0.0)
    t = (time.time() - start) / 10 * 1000
    operations.append(("MOG2 background subtraction", t))
    
    # 3. Gaussian blur
    start = time.time()
    for _ in range(10):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    t = (time.time() - start) / 10 * 1000
    operations.append(("Gaussian blur (5x5)", t))
    
    # 4. Canny edge detection
    start = time.time()
    for _ in range(10):
        edges = cv2.Canny(gray, 50, 150)
    t = (time.time() - start) / 10 * 1000
    operations.append(("Canny edge detection", t))
    
    # 5. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    start = time.time()
    for _ in range(10):
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    t = (time.time() - start) / 10 * 1000
    operations.append(("Morphological ops (open+close)", t))
    
    # 6. Soft mask creation (Gaussian blur on mask)
    start = time.time()
    for _ in range(10):
        mask_float = mask.astype(np.float32) / 255.0
        soft_mask = cv2.GaussianBlur(mask_float, (41, 41), 20/3)
        soft_mask = (soft_mask * 255).astype(np.uint8)
    t = (time.time() - start) / 10 * 1000
    operations.append(("Soft mask creation", t))
    
    # 7. Frame resize for display
    start = time.time()
    for _ in range(10):
        resized = cv2.resize(frame, (2560, 1440))
    t = (time.time() - start) / 10 * 1000
    operations.append(("Frame resize (1920x1080->2560x1440)", t))
    
    # 8. Random pixel effect
    motion_pixels = mask > 128
    num_pixels = np.sum(motion_pixels)
    start = time.time()
    for _ in range(10):
        result = frame.copy()
        if num_pixels > 0:
            random_colors = np.random.randint(0, 255, (num_pixels, 3), dtype=np.uint8)
            result[motion_pixels] = random_colors
    t = (time.time() - start) / 10 * 1000
    operations.append((f"Random pixels ({num_pixels} pixels)", t))
    
    # 9. MediaPipe pose (if available)
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()
        for _ in range(10):
            results = pose.process(rgb_frame)
        t = (time.time() - start) / 10 * 1000
        operations.append(("MediaPipe pose detection", t))
        pose.close()
    except:
        operations.append(("MediaPipe pose detection", "Not available"))
    
    # Print results sorted by time
    print("\nOperation timings (milliseconds per frame):")
    print("-" * 50)
    
    total_time = 0
    for name, time_ms in sorted(operations, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
        if isinstance(time_ms, (int, float)):
            print(f"{name:<40} {time_ms:>6.1f} ms")
            total_time += time_ms
        else:
            print(f"{name:<40} {time_ms:>6}")
    
    print("-" * 50)
    print(f"{'TOTAL:':<40} {total_time:>6.1f} ms")
    print(f"{'Theoretical max FPS:':<40} {1000/total_time if total_time > 0 else 0:>6.1f} fps")
    
    # Check OpenCV optimization
    print("\n" + "=" * 50)
    print("OpenCV Build Information:")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Number of threads: {cv2.getNumThreads()}")
    print(f"CPU optimization: {cv2.getCPUTickCount()}")
    
    # Try to check for optimizations
    build_info = cv2.getBuildInformation()
    optimizations = ["AVX", "AVX2", "SSE", "NEON", "OpenCL", "TBB", "OpenMP"]
    print("\nEnabled optimizations:")
    for opt in optimizations:
        if opt in build_info:
            print(f"  - {opt}: Yes")

if __name__ == "__main__":
    time_operations()