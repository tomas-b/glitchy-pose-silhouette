import cv2
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

# Import the main processor
import sys
sys.path.append('.')
from glitchy_silhouette import GlitchySilhouetteProcessor, ThreadedCamera

def profile_main_loop():
    """Profile the main processing loop"""
    camera = ThreadedCamera(0).start()
    processor = GlitchySilhouetteProcessor()
    
    # Let camera warm up
    time.sleep(1)
    
    # Skip countdown for profiling
    processor.countdown_active = False
    processor.learning_background = False
    processor.ready_for_detection = True
    processor.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    frame_count = 0
    start_time = time.time()
    
    while frame_count < 100:  # Profile 100 frames
        frame = camera.read()
        if frame is not None:
            result = processor.process_frame(frame, 0)
            frame_count += 1
            
            if result is not None:
                cv2.imshow('Profile Test', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_count} frames in {elapsed:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.1f}")
    
    camera.stop()
    cv2.destroyAllWindows()

def check_cpu_usage():
    """Check what's using CPU"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print("\nSystem Info:")
    print(f"CPU Count: {psutil.cpu_count()} cores")
    print(f"CPU Freq: {psutil.cpu_freq().current:.0f} MHz")
    
    # Monitor for a few seconds
    print("\nCPU Usage during processing:")
    for i in range(5):
        cpu_percent = process.cpu_percent(interval=1)
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f} MB")

def time_operations():
    """Time individual operations"""
    print("\nTiming individual operations:")
    
    # Test frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
    
    # Time MOG2
    mog2 = cv2.createBackgroundSubtractorMOG2()
    start = time.time()
    for _ in range(10):
        fg_mask = mog2.apply(frame, learningRate=0.0)
    mog2_time = (time.time() - start) / 10
    print(f"MOG2 background subtraction: {mog2_time*1000:.1f} ms")
    
    # Time Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = time.time()
    for _ in range(10):
        edges = cv2.Canny(gray, 50, 150)
    canny_time = (time.time() - start) / 10
    print(f"Canny edge detection: {canny_time*1000:.1f} ms")
    
    # Time morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    start = time.time()
    for _ in range(10):
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph_time = (time.time() - start) / 10
    print(f"Morphological operations: {morph_time*1000:.1f} ms")
    
    # Time resize operation (for fullscreen)
    start = time.time()
    for _ in range(10):
        resized = cv2.resize(frame, (2560, 1440))  # Typical scaled size
    resize_time = (time.time() - start) / 10
    print(f"Frame resize for fullscreen: {resize_time*1000:.1f} ms")
    
    # Time random pixel generation
    motion_pixels = mask > 128
    num_pixels = np.sum(motion_pixels)
    start = time.time()
    for _ in range(10):
        random_colors = np.random.randint(0, 255, (num_pixels, 3), dtype=np.uint8)
    random_time = (time.time() - start) / 10
    print(f"Random pixel generation ({num_pixels} pixels): {random_time*1000:.1f} ms")
    
    # Total estimated frame time
    total_time = mog2_time + canny_time + morph_time + resize_time + random_time
    print(f"\nEstimated total per frame: {total_time*1000:.1f} ms ({1/total_time:.1f} FPS max)")

if __name__ == "__main__":
    print("Performance Analysis for Glitchy Silhouette")
    print("=" * 50)
    
    check_cpu_usage()
    time_operations()
    
    # Profile with cProfile
    print("\nProfiling main loop...")
    pr = cProfile.Profile()
    pr.enable()
    
    profile_main_loop()
    
    pr.disable()
    
    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\nTop time-consuming functions:")
    print(s.getvalue())