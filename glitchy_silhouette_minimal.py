import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue

class ThreadedCamera:
    """High-performance threaded camera capture"""
    def __init__(self, src=0):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Try different queue sizes
        self.q = Queue(maxsize=1)  # Reduced from 2
        self.running = True
        self.dropped_frames = 0
        
    def start(self):
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if self.q.full():
                    # Drop the oldest frame
                    try:
                        self.q.get_nowait()
                        self.dropped_frames += 1
                    except:
                        pass
                self.q.put(frame)
                    
    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.capture.release()

def main():
    print("Minimal Glitchy Silhouette - Performance Test")
    print("Press 'q' to quit")
    print()
    
    # Test 1: Threaded camera with minimal processing
    camera = ThreadedCamera(0).start()
    time.sleep(1)  # Let camera warm up
    
    # FPS tracking
    fps_start = time.time()
    frame_count = 0
    display_fps = 0
    
    # Create window once
    cv2.namedWindow('Minimal Test', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Minimal Test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get screen dimensions once
    try:
        rect = cv2.getWindowImageRect('Minimal Test')
        screen_width = rect[2] if rect[2] > 0 else 1920
        screen_height = rect[3] if rect[3] > 0 else 1080
    except:
        screen_width = 1920
        screen_height = 1080
    
    while True:
        frame = camera.read()
        if frame is not None:
            frame_count += 1
            
            # Update FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                display_fps = 30 / elapsed
                fps_start = time.time()
                print(f"FPS: {display_fps:.1f}, Dropped: {camera.dropped_frames}")
            
            # Minimal processing - just FPS display
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Direct display without scaling
            cv2.imshow('Minimal Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop()
    cv2.destroyAllWindows()
    print(f"\nTotal dropped frames: {camera.dropped_frames}")

if __name__ == "__main__":
    main()