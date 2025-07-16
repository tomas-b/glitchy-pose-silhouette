import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue
import random

class ThreadedCamera:
    """High-performance threaded camera capture"""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Optimize camera settings
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.q = Queue(maxsize=2)
        self.running = True
        
    def start(self):
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if not self.q.full():
                    self.q.put(frame)
                    
    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None
        
    def stop(self):
        self.running = False
        self.capture.release()

class GlitchySilhouetteProcessor:
    """Advanced background subtraction with edge detection for clean silhouettes"""
    
    def __init__(self):
        # State management for countdown -> learning -> detection phases
        self.countdown_active = True
        self.learning_background = False
        self.ready_for_detection = False
        
        # Countdown parameters
        self.countdown_start = time.time()
        self.countdown_duration = 5.0  # 5 seconds
        
        # Background learning parameters
        self.learning_start = None
        self.learning_duration = 3.0  # 3 seconds to learn background
        self.learning_frames_needed = 60  # frames to process for learning
        self.learning_frames_processed = 0
        
        # MOG2 Background Subtractor (will be initialized after countdown)
        self.bg_subtractor = None
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.blur_kernel = 5
        
        # Morphological operations for clean masks
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Glitch effect parameters
        self.glitch_mode = 0  # 0=random pixels, 1=blocks, 2=scanlines, 3=datamosh
        self.glitch_intensity = 1.0
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        
        print("🎨 Glitchy Silhouette Processor initialized")
        print("Countdown phase active - get ready!")
        
    def process_frame(self, frame):
        """Main processing pipeline with countdown -> learning -> detection phases"""
        if frame is None:
            return None
            
        # Performance tracking
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            self.current_fps = 30 / (time.time() - self.fps_start)
            self.fps_start = time.time()
        
        current_time = time.time()
        
        # Phase 1: Countdown
        if self.countdown_active:
            remaining_time = self.countdown_duration - (current_time - self.countdown_start)
            if remaining_time > 0:
                return self.draw_countdown(frame, remaining_time)
            else:
                # Countdown finished, start background learning
                self.countdown_active = False
                self.learning_background = True
                self.learning_start = current_time
                self.learning_frames_processed = 0
                # Initialize MOG2 now
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=200,           # Frames to learn background
                    varThreshold=10,       # Sensitivity (lower = more sensitive)
                    detectShadows=True     # Remove shadows automatically
                )
                print("🎬 Countdown complete! Learning background...")
        
        # Phase 2: Background Learning
        if self.learning_background:
            # Feed frames to MOG2 for learning
            if self.bg_subtractor is not None:
                self.bg_subtractor.apply(frame)
                self.learning_frames_processed += 1
            
            # Check if learning is complete
            learning_elapsed = current_time - self.learning_start
            if (self.learning_frames_processed >= self.learning_frames_needed or 
                learning_elapsed >= self.learning_duration):
                self.learning_background = False
                self.ready_for_detection = True
                print("✨ Background learning complete! Ready for consistent detection!")
            
            return self.draw_learning_progress(frame)
        
        # Phase 3: Motion Detection and Glitch Effects
        if self.ready_for_detection and self.bg_subtractor is not None:
            # Step 1: Background subtraction using MOG2 (NO LEARNING - frozen background!)
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0.0)
            
            # Step 2: Edge detection on original frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # Step 3: Combine background mask with edges (weighted blend)
            combined_mask = cv2.addWeighted(fg_mask, 0.7, edges, 0.3, 0)
            
            # Step 4: Morphological operations for clean silhouettes
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            
            # Step 5: Find contours and filter by size
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create clean mask with only significant contours
            clean_mask = np.zeros_like(combined_mask)
            min_contour_area = 500
            
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    cv2.fillPoly(clean_mask, [contour], 255)
            
            # Step 6: Apply glitchy effects to silhouette areas
            result = self.apply_glitch_effect(frame, clean_mask)
            
            # Step 7: Add debug info
            self.add_debug_overlay(result, len(contours), np.sum(clean_mask > 0))
            
            return result
        
        # Fallback: just show the frame
        return frame
    
    def apply_glitch_effect(self, frame, mask):
        """Apply different glitch effects to detected silhouettes"""
        result = frame.copy()
        
        if self.glitch_mode == 0:
            # Random colored pixels
            result = self.glitch_random_pixels(result, mask)
        elif self.glitch_mode == 1:
            # Glitch blocks
            result = self.glitch_blocks(result, mask)
        elif self.glitch_mode == 2:
            # Scan lines
            result = self.glitch_scanlines(result, mask)
        elif self.glitch_mode == 3:
            # Datamosh effect
            result = self.glitch_datamosh(result, mask)
            
        return result
    
    def glitch_random_pixels(self, frame, mask):
        """Original random pixel effect but cleaner"""
        result = frame.copy()
        motion_pixels = mask > 0
        
        if np.any(motion_pixels):
            # Generate random colors for all motion pixels at once (vectorized!)
            random_colors = np.random.randint(0, 255, (np.sum(motion_pixels), 3), dtype=np.uint8)
            result[motion_pixels] = random_colors
            
        return result
    
    def glitch_blocks(self, frame, mask):
        """Rectangular glitch blocks"""
        result = frame.copy()
        h, w = mask.shape
        block_size = int(20 * self.glitch_intensity)
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block_mask = mask[y:y+block_size, x:x+block_size]
                if np.any(block_mask > 0):
                    # Random color for entire block
                    color = [random.randint(0, 255) for _ in range(3)]
                    result[y:y+block_size, x:x+block_size] = color
                    
        return result
    
    def glitch_scanlines(self, frame, mask):
        """Horizontal scanline effect"""
        result = frame.copy()
        h, w = mask.shape
        line_height = max(1, int(5 * self.glitch_intensity))
        
        for y in range(0, h, line_height * 2):
            line_mask = mask[y:y+line_height]
            if np.any(line_mask > 0):
                # Shift pixels horizontally with random colors
                shift = random.randint(-20, 20)
                color = [random.randint(0, 255) for _ in range(3)]
                result[y:y+line_height] = color
                
        return result
    
    def glitch_datamosh(self, frame, mask):
        """Datamosh-style compression artifacts"""
        result = frame.copy()
        
        # Create displacement field for motion areas
        motion_areas = mask > 0
        if np.any(motion_areas):
            # Random displacement
            displacement = np.random.randint(-10, 10, motion_areas.shape, dtype=np.int8)
            
            # Apply displacement (simplified datamosh)
            rows, cols = np.where(motion_areas)
            for r, c in zip(rows[::5], cols[::5]):  # Sample for performance
                new_r = np.clip(r + displacement[r, c], 0, frame.shape[0] - 1)
                new_c = np.clip(c + displacement[r, c], 0, frame.shape[1] - 1)
                result[r, c] = frame[new_r, new_c]
                
        return result
    
    def draw_countdown(self, frame, remaining_time):
        """Draw countdown overlay on frame - FULLSCREEN camera with overlay"""
        h, w = frame.shape[:2]
        
        # Start with full camera frame
        result = frame.copy()
        
        # Create dark overlay for better text visibility
        dark_overlay = result.copy()
        dark_overlay[:] = [0, 0, 0]  # Black overlay
        cv2.addWeighted(dark_overlay, 0.6, result, 0.4, 0, result)
        
        # Countdown number
        countdown_num = int(remaining_time) + 1
        pulse = 0.8 + 0.4 * abs(np.sin(time.time() * 3))  # Pulsing effect
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) * 0.01 * pulse
        thickness = max(2, int(font_scale * 2))
        
        text = str(countdown_num)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Draw countdown number
        cv2.putText(result, text, (text_x, text_y), font, font_scale, (0, 100, 255), thickness)
        
        # Instructions
        instruction_text = "Get out of frame! Starting in..."
        inst_size = cv2.getTextSize(instruction_text, font, 0.8, 2)[0]
        inst_x = (w - inst_size[0]) // 2
        inst_y = text_y - text_size[1] - 30
        cv2.putText(result, instruction_text, (inst_x, inst_y), font, 0.8, (255, 255, 255), 2)
        
        return result
    
    def draw_learning_progress(self, frame):
        """Draw background learning progress - FULLSCREEN camera with overlay"""
        h, w = frame.shape[:2]
        
        # Start with full camera frame  
        result = frame.copy()
        
        # Progress calculation
        progress = min(1.0, self.learning_frames_processed / self.learning_frames_needed)
        
        # Progress bar
        bar_width = w // 2
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 60
        
        # Background bar
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 100), -1)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Learning background... {int(progress * 100)}%"
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = bar_y - 10
        cv2.putText(result, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
        
        return result

    def add_debug_overlay(self, frame, contour_count, silhouette_pixels):
        """Add debug information to frame"""
        cv2.rectangle(frame, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 160), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, 35), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Glitch Mode: {self.get_glitch_name()}", (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Silhouettes: {contour_count}", (20, 85), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pixels: {silhouette_pixels}", (20, 110), font, 0.7, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, f"Controls: SPACE=Mode WASD=Sens/Intensity R=Reset", (20, 135), font, 0.5, (200, 200, 200), 1)
        
        # Show current sensitivity if available
        if self.bg_subtractor is not None:
            sensitivity = self.bg_subtractor.getVarThreshold()
            cv2.putText(frame, f"Sensitivity: {sensitivity:.0f} | Intensity: {self.glitch_intensity:.1f}", (20, 155), font, 0.5, (255, 255, 0), 1)
    
    def get_glitch_name(self):
        names = ["Random Pixels", "Glitch Blocks", "Scan Lines", "Datamosh"]
        return names[self.glitch_mode]
    
    def adjust_sensitivity(self, delta):
        """Adjust motion detection sensitivity"""
        if self.bg_subtractor is not None:
            current_threshold = self.bg_subtractor.getVarThreshold()
            new_threshold = max(5, min(200, current_threshold + delta))
            self.bg_subtractor.setVarThreshold(new_threshold)
            print(f"Sensitivity: {new_threshold} (lower = more sensitive)")
        else:
            print("Background subtractor not initialized yet")
    
    def cycle_glitch_mode(self):
        """Switch between glitch effects"""
        self.glitch_mode = (self.glitch_mode + 1) % 4
        print(f"Glitch Mode: {self.get_glitch_name()}")
    
    def adjust_glitch_intensity(self, delta):
        """Adjust glitch effect intensity"""
        self.glitch_intensity = max(0.1, min(3.0, self.glitch_intensity + delta))
        print(f"Glitch Intensity: {self.glitch_intensity:.1f}")
    
    def reset_to_countdown(self):
        """Reset the entire process back to countdown phase"""
        self.countdown_active = True
        self.learning_background = False
        self.ready_for_detection = False
        self.countdown_start = time.time()
        self.learning_start = None
        self.learning_frames_processed = 0
        self.bg_subtractor = None
        print("🔄 Reset to countdown phase - get ready!")

def main():
    print("🚀 Starting Glitchy Silhouette Replacement")
    print("Using MOG2 background subtraction for professional results!")
    print()
    print("Controls:")
    print("  SPACE - Cycle glitch effects")
    print("  W/S   - Adjust motion sensitivity (up/down)") 
    print("  A/D   - Adjust glitch intensity (left/right)")
    print("  R     - Reset background model")
    print("  ESC/Q - Quit")
    print()
    
    # Initialize camera and processor
    camera = ThreadedCamera(0).start()
    processor = GlitchySilhouetteProcessor()
    
    # Let camera warm up
    time.sleep(1)
    
    print("🎬 Camera ready! MOG2 learning background...")
    print("Move around after a few seconds to see clean silhouette detection!")
    
    while True:
        frame = camera.read()
        if frame is not None:
            # Process frame for glitchy silhouettes
            result = processor.process_frame(frame)
            
            if result is not None:
                # Display fullscreen
                cv2.namedWindow('Glitchy Silhouettes', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Glitchy Silhouettes', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Glitchy Silhouettes', result)
        
        # Handle controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            processor.cycle_glitch_mode()
        elif key == ord('r'):  # R
            processor.reset_to_countdown()
        elif key == ord('w') or key == ord('W'):  # W = Up sensitivity
            processor.adjust_sensitivity(-10)
        elif key == ord('s') or key == ord('S'):  # S = Down sensitivity
            processor.adjust_sensitivity(10)
        elif key == ord('a') or key == ord('A'):  # A = Left intensity
            processor.adjust_glitch_intensity(-0.2)
        elif key == ord('d') or key == ord('D'):  # D = Right intensity
            processor.adjust_glitch_intensity(0.2)
        elif key != 255:  # Debug: show what key was pressed
            print(f"Key pressed: {key}")
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    print("👋 Glitchy silhouettes complete!")

if __name__ == "__main__":
    main()
