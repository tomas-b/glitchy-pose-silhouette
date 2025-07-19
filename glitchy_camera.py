#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["opencv-python", "numpy", "mediapipe"]
# ///

import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue
import random
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not installed - pose estimation disabled")

class ThreadedCamera:
    """High-performance threaded camera capture"""
    def __init__(self, src=0):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        # Optimize camera settings
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.q = Queue(maxsize=1)  # Reduce queue size to avoid latency
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
                    # Drop old frame to keep latency low
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
        self.capture.release()
    
    def switch_source(self, new_src):
        """Switch to a different camera source"""
        # Stop current capture
        self.running = False
        # Wait for thread to finish
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.capture.release()
        
        # Clear queue
        while not self.q.empty():
            self.q.get()
        
        # Small delay to ensure camera is fully released
        time.sleep(0.5)
        
        # Start new capture
        self.src = new_src
        self.capture = cv2.VideoCapture(new_src)
        # Re-apply camera settings
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check if camera opened successfully
        if not self.capture.isOpened():
            print(f"⚠️  Failed to open camera {new_src}, falling back to camera 0")
            self.src = 0
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Restart capture thread
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        
        print(f"📷 Switched to camera {self.src}")

class GlitchySilhouetteProcessor:
    """Advanced background subtraction with edge detection for clean silhouettes"""
    
    def __init__(self):
        # State management for countdown -> learning -> detection phases
        self.countdown_active = True
        self.learning_background = False
        self.ready_for_detection = False
        
        # Detect available cameras
        self.available_cameras = self.detect_cameras()
        
        # Countdown parameters
        self.countdown_start = time.time()
        self.countdown_duration = 5.0  # 5 seconds
        
        # Background learning parameters
        self.learning_start = None
        self.learning_duration = 3.0  # 3 seconds to learn background
        self.learning_frames_needed = 60  # frames to process for learning
        self.learning_frames_processed = 0
        
        # Store ACTUAL background image for manual comparison
        self.background_image = None
        self.use_manual_bg = True  # Use manual background comparison instead of MOG2
        
        # MOG2 Background Subtractor (will be initialized after countdown)
        self.bg_subtractor = None
        
        # Adaptive edge detection parameters
        self.adaptive_canny = True
        self.blur_kernel = 5
        
        # Multi-scale detection
        self.use_multiscale = False  # Disable for performance
        self.scales = [1.0]  # Just full res for now
        
        # Morphological operations for clean masks
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Glitch effect parameters
        self.glitch_mode = 0  # 0=random pixels, 1=blocks, 2=scanlines, 3=datamosh
        self.glitch_intensity = 1.0
        
        # Soft thresholding parameters
        self.soft_threshold = True
        self.soft_radius = 20  # pixels for soft edge falloff
        
        # Pose estimation parameters
        self.use_pose_estimation = MEDIAPIPE_AVAILABLE
        self.pose_detected = False
        self.pose_landmarks = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0=fastest, 1=balanced, 2=accurate
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("🤖 MediaPipe Pose estimation enabled!")
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        
        print("🎨 Advanced Glitchy Silhouette Processor initialized")
        print("Using KNN background subtraction + Multi-scale adaptive Canny")
        print("Countdown phase active - get ready!")
        print(f"📷 Found {len(self.available_cameras)} cameras")
        
    def process_frame(self, frame, camera_src=0):
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
                # Initialize MOG2 with better parameters
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,           # More frames for stable background
                    varThreshold=16,       # Good default sensitivity
                    detectShadows=True     # Remove shadows automatically
                )
                print("🎬 Countdown complete! Learning background...")
        
        # Phase 2: Background Learning
        if self.learning_background:
            # Feed frames to MOG2 AND capture final background
            if self.bg_subtractor is not None:
                self.bg_subtractor.apply(frame, learningRate=0.1)
                self.learning_frames_processed += 1
                # Capture the background image at the end
                if self.learning_frames_processed == self.learning_frames_needed - 1:
                    self.background_image = frame.copy()
                    print("📸 Background image CAPTURED!")
            
            # Check if learning is complete
            learning_elapsed = current_time - self.learning_start
            if (self.learning_frames_processed >= self.learning_frames_needed or 
                learning_elapsed >= self.learning_duration):
                self.learning_background = False
                self.ready_for_detection = True
                print("✨ NOW COMPARING AGAINST FROZEN BACKGROUND FOREVER!")
            
            return self.draw_learning_progress(frame)
        
        # Phase 3: Motion Detection and Glitch Effects  
        if self.ready_for_detection and self.background_image is not None:
            # Step 1: Pose estimation (if enabled)
            if self.use_pose_estimation:
                self.detect_pose(frame)
            
            # Step 2: Compare current frame against FROZEN background image
            fg_mask = self.compare_against_frozen_background(frame, self.background_image)
            
            # Step 3: Advanced edge detection
            edges = self.advanced_edge_detection(frame)
            
            # Step 3: Combine background mask with edges (weighted blend)
            combined_mask = cv2.addWeighted(fg_mask, 0.8, edges, 0.2, 0)
            
            # Step 4: Morphological operations for clean silhouettes
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            
            # Step 5: Find contours and filter by size
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create clean mask with only significant contours
            clean_mask = np.zeros_like(combined_mask)
            min_contour_area = 3000  # Larger = ignore small lighting changes
            
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    cv2.fillPoly(clean_mask, [contour], 255)
            
            # Step 6: Apply soft thresholding for gradual edges
            if self.soft_threshold:
                soft_mask = self.create_soft_mask(clean_mask)
            else:
                soft_mask = clean_mask
            
            # Step 7: Apply glitchy effects to silhouette areas
            result = self.apply_glitch_effect(frame, soft_mask)
            
            # Step 8: Draw pose landmarks if detected
            if self.use_pose_estimation and self.pose_detected:
                result = self.draw_pose_landmarks(result)
            
            # Step 9: Add debug info
            self.add_debug_overlay(result, len(contours), np.sum(clean_mask > 0), camera_src)
            
            return result
        
        # Fallback: just show the frame
        return frame
    
    def create_soft_mask(self, binary_mask):
        """Create soft-edged mask with gradual falloff"""
        # Convert binary mask to float
        mask_float = binary_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for soft edges
        soft_mask = cv2.GaussianBlur(mask_float, (self.soft_radius * 2 + 1, self.soft_radius * 2 + 1), self.soft_radius / 3)
        
        # Convert back to 0-255 range
        soft_mask = (soft_mask * 255).astype(np.uint8)
        
        return soft_mask
    
    def compare_against_frozen_background(self, current_frame, background_frame):
        """Compare current frame against NEVER-CHANGING background image"""
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple absolute difference
        diff = cv2.absdiff(current_gray, bg_gray)
        
        # Threshold the difference - anything different from background
        _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        return fg_mask
    
    def detect_pose(self, frame):
        """Detect pose using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose detection
        results = self.pose.process(rgb_frame)
        
        # Store results
        self.pose_detected = results.pose_landmarks is not None
        self.pose_landmarks = results.pose_landmarks
    
    def draw_pose_landmarks(self, frame):
        """Draw pose landmarks on frame"""
        if not (MEDIAPIPE_AVAILABLE and self.pose_detected and self.pose_landmarks):
            return frame
            
        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        annotated_frame = frame.copy()
        
        # Draw connections with custom style
        mp_drawing.draw_landmarks(
            annotated_frame,
            self.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 255),  # Yellow landmarks
                thickness=2,
                circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 255),  # Magenta connections
                thickness=2
            )
        )
        
        return annotated_frame

    def advanced_edge_detection(self, frame):
        """Multi-scale adaptive edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        if self.use_multiscale:
            # Multi-scale edge detection
            edges_combined = np.zeros_like(gray)
            
            for scale in self.scales:
                # Resize for different scales
                if scale != 1.0:
                    h, w = gray.shape
                    scaled_gray = cv2.resize(blurred, (int(w * scale), int(h * scale)))
                else:
                    scaled_gray = blurred
                
                # Adaptive Canny thresholds
                if self.adaptive_canny:
                    median = np.median(scaled_gray)
                    sigma = 0.33
                    lower = int(max(0, (1.0 - sigma) * median))
                    upper = int(min(255, (1.0 + sigma) * median))
                    lower = max(lower, 30)
                    upper = max(upper, 60)
                else:
                    lower, upper = 50, 150
                
                # Detect edges
                edges_scale = cv2.Canny(scaled_gray, lower, upper)
                
                # Resize back if needed
                if scale != 1.0:
                    edges_scale = cv2.resize(edges_scale, (w, h))
                
                # Combine scales
                edges_combined = cv2.addWeighted(edges_combined, 0.7, edges_scale, 0.3, 0)
            
            return edges_combined
        else:
            # Single scale adaptive Canny
            if self.adaptive_canny:
                median = np.median(blurred)
                sigma = 0.33
                lower = int(max(0, (1.0 - sigma) * median))
                upper = int(min(255, (1.0 + sigma) * median))
                lower = max(lower, 30)
                upper = max(upper, 60)
                return cv2.Canny(blurred, lower, upper)
            else:
                return cv2.Canny(blurred, 50, 150)

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
        """Random pixel effect with soft blending - FAST vectorized version"""
        result = frame.copy()
        
        if self.soft_threshold and len(mask.shape) == 2:
            # FAST vectorized soft blending
            alpha = mask.astype(np.float32) / 255.0
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            
            # Generate random colors for entire frame
            random_frame = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
            
            # Vectorized blend: result = alpha * random + (1-alpha) * original
            result = (alpha_3d * random_frame + (1 - alpha_3d) * result).astype(np.uint8)
        else:
            # Original hard threshold version
            motion_pixels = mask > 0
            if np.any(motion_pixels):
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

    def add_debug_overlay(self, frame, contour_count, silhouette_pixels, camera_src=0):
        """Add debug information to frame"""
        # Larger box to fit camera list
        box_height = 210 + len(self.available_cameras) * 20
        cv2.rectangle(frame, (10, 10), (450, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, box_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Camera: {camera_src}", (20, 35), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Glitch Mode: {self.get_glitch_name()}", (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Silhouettes: {contour_count}", (20, 85), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pixels: {silhouette_pixels}", (20, 110), font, 0.7, (255, 255, 255), 2)
        
        # Pose detection info
        if self.use_pose_estimation:
            pose_status = "✓ Detected" if self.pose_detected else "✗ Not found"
            cv2.putText(frame, f"Pose: {pose_status}", (20, 135), font, 0.7, (0, 255, 0) if self.pose_detected else (0, 0, 255), 2)
        
        # Controls
        cv2.putText(frame, f"Controls: 0-9=Camera SPACE=Mode WASD=Sens/Intensity R=Reset", (20, 160), font, 0.5, (200, 200, 200), 1)
        
        # Show current sensitivity if available
        if self.bg_subtractor is not None:
            sensitivity = self.bg_subtractor.getVarThreshold()
            cv2.putText(frame, f"MOG2 Var: {sensitivity:.0f} | Intensity: {self.glitch_intensity:.1f}", (20, 180), font, 0.5, (255, 255, 0), 1)
        
        # Camera list
        cameras_y = 200
        cv2.putText(frame, "AVAILABLE CAMERAS:", (20, cameras_y), font, 0.6, (255, 255, 0), 2)
        cameras_y += 20
        
        for cam in self.available_cameras:
            # Highlight current camera
            color = (0, 255, 0) if cam['index'] == camera_src else (200, 200, 200)
            cam_text = f"[{cam['index']}] {cam['name']}"
            cv2.putText(frame, cam_text, (20, cameras_y), font, 0.5, color, 1)
            cameras_y += 20
    
    def get_glitch_name(self):
        names = ["Random Pixels", "Glitch Blocks", "Scan Lines", "Datamosh"]
        return names[self.glitch_mode]
    
    def adjust_sensitivity(self, delta):
        """Adjust motion detection sensitivity for MOG2"""
        if self.bg_subtractor is not None:
            current_threshold = self.bg_subtractor.getVarThreshold()
            new_threshold = max(5, min(200, current_threshold + delta))
            self.bg_subtractor.setVarThreshold(new_threshold)
            print(f"MOG2 Sensitivity: {new_threshold} (lower = more sensitive)")
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
    
    def detect_cameras(self):
        """Detect all available cameras and their properties"""
        cameras = []
        max_tested = 10  # Test up to 10 camera indices
        consecutive_failures = 0
        
        for i in range(max_tested):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        # Get camera properties
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Try to get camera name (not always available)
                        backend = cap.getBackendName()
                        
                        camera_info = {
                            'index': i,
                            'name': f"Camera {i} ({backend})",
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': fps
                        }
                        cameras.append(camera_info)
                        consecutive_failures = 0
                    cap.release()
                else:
                    consecutive_failures += 1
                    # Stop after 3 consecutive failures
                    if consecutive_failures >= 3 and len(cameras) > 0:
                        break
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= 3 and len(cameras) > 0:
                    break
        
        # Always ensure at least camera 0 is in the list
        if len(cameras) == 0:
            cameras.append({
                'index': 0,
                'name': "Camera 0 (Default)",
                'resolution': "Unknown",
                'fps': 30.0
            })
        
        return cameras

def main():
    print("🚀 Starting Glitchy Silhouette Replacement")
    print("Using MOG2 background subtraction for professional results!")
    print()
    print("Controls:")
    print("  0-9   - Switch camera source")
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
            result = processor.process_frame(frame, camera.src)
            
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
        elif ord('0') <= key <= ord('9'):  # Number keys for camera switching
            camera_index = key - ord('0')
            # Check if this camera exists
            if any(cam['index'] == camera_index for cam in processor.available_cameras):
                if camera.src != camera_index:
                    camera.switch_source(camera_index)
                    processor.reset_to_countdown()
            else:
                print(f"⚠️  Camera {camera_index} not available")
        elif key != 255:  # Debug: show what key was pressed
            print(f"Key pressed: {key}")
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    print("👋 Glitchy silhouettes complete!")

if __name__ == "__main__":
    main()