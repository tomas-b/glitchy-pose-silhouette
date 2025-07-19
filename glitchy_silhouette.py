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
        # Try to set camera to 16:9 aspect ratio (common for fullscreen)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
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
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Check if camera opened successfully
        if not self.capture.isOpened():
            print(f"⚠️  Failed to open camera {new_src}, falling back to camera 0")
            self.src = 0
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
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
        self.use_manual_bg = False  # Use manual background comparison instead of MOG2
        
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
        self.glitch_mode = 0  # 0=random pixels, 1=blocks
        self.glitch_intensity = 1.0
        
        # Soft thresholding parameters
        self.soft_threshold = False  # Disable by default for performance
        self.soft_radius = 10  # Smaller radius for performance
        
        # Pose estimation parameters
        self.use_pose_estimation = MEDIAPIPE_AVAILABLE
        self.pose_detected = False
        self.pose_landmarks = None
        
        # Toggle flags
        self.use_filter_effects = True  # Toggle for filter effects
        self.show_debug_info = True  # Toggle for debug overlay
        
        # Hand gesture detection
        self.left_hand_raised = False
        self.right_hand_raised = False
        self.current_skeleton_color = "normal"  # normal, left_hand, right_hand, both_hands
        
        # Head color effect parameters
        self.head_effect_active = False
        self.head_effect_start_time = None
        self.head_effect_duration = 5.0  # 5 seconds
        self.head_effect_radius = 80  # pixels around head
        self.effect_screenshots_taken = False  # Track if we've taken screenshots
        self.body_bbox = None  # Store body bounding box
        self.cooldown_active = False
        self.cooldown_start_time = None
        self.cooldown_duration = 5.0  # 5 seconds cooldown after photos
        
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
            print("🙋 Hand gesture detection: Raise hands above head to change skeleton color!")
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        self.last_frame_time = time.time()
        self.frame_times = []  # Rolling average
        self.pose_frame_skip = 0  # Skip pose detection on some frames
        
        print("🎨 Advanced Glitchy Silhouette Processor initialized")
        print("Using KNN background subtraction + Multi-scale adaptive Canny")
        print("Countdown phase active - get ready!")
        print(f"📷 Found {len(self.available_cameras)} cameras")
        
    def process_frame(self, frame, camera_src=0):
        """Main processing pipeline with countdown -> learning -> detection phases"""
        if frame is None:
            return None
            
        # Performance tracking - accurate per-frame timing
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Keep rolling average of last 30 frames
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate FPS from average frame time
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
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
                    varThreshold=200,      # Much higher threshold to reduce sensitivity
                    detectShadows=False    # Disable shadow detection for better performance
                )
                # Set additional parameters for better filtering
                self.bg_subtractor.setBackgroundRatio(0.9)  # Stricter background model
                self.bg_subtractor.setComplexityReductionThreshold(0.2)
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
                print("✨ NOW COMPARING AGAINST FROZEN BACKGROUND!")
            
            return self.draw_learning_progress(frame)
        
        # Phase 3: Motion Detection and Glitch Effects  
        if self.ready_for_detection:
            # Debug: Check if background image was captured
            if self.background_image is None:
                print("⚠️ ERROR: Background image not captured! Using current frame as background.")
                self.background_image = frame.copy()
                
            if self.background_image is not None:
                # Check if cooldown has finished
                if self.cooldown_active and self.cooldown_start_time:
                    cooldown_elapsed = current_time - self.cooldown_start_time
                    if cooldown_elapsed >= self.cooldown_duration:
                        # Cooldown finished
                        self.cooldown_active = False
                        self.cooldown_start_time = None
                        print("✅ Cooldown finished! Pose detection re-enabled.")
                
                # If no effects are needed, skip processing (but still do minimal detection)
                if not self.use_filter_effects and not self.use_pose_estimation:
                    result = frame.copy()
                    self.add_debug_overlay(result, 0, 0, camera_src)
                    return result
                
                # Step 1: Pose estimation (if enabled) - skip some frames for performance
                if self.use_pose_estimation:
                    self.pose_frame_skip += 1
                    if self.pose_frame_skip >= 2:  # Process every 2nd frame
                        self.detect_pose(frame)
                        self.pose_frame_skip = 0
                
                # Step 2: Use direct background comparison with color difference
                if self.use_manual_bg:
                    # Simple and FAST difference detection
                    diff = cv2.absdiff(frame, self.background_image)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                else:
                    # Fall back to MOG2
                    fg_mask = self.bg_subtractor.apply(frame, learningRate=0.0)
                    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
                # Step 3: Advanced edge detection
                edges = self.advanced_edge_detection(frame)
                
                # Step 3.5: Combine background mask with edges (weighted blend)
                combined_mask = cv2.addWeighted(fg_mask, 0.8, edges, 0.2, 0)
            
                # Apply another threshold to clean up
                _, combined_mask = cv2.threshold(combined_mask, 100, 255, cv2.THRESH_BINARY)
                
                # Step 4: Morphological operations for clean silhouettes
                # Larger kernel for stronger noise removal
                large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, large_kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            
                # Step 5: Find contours and filter by size
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create clean mask with only significant contours
                clean_mask = np.zeros_like(combined_mask)
                min_contour_area = 15000  # Even larger to ignore minor changes
                
                for contour in contours:
                    if cv2.contourArea(contour) > min_contour_area:
                        cv2.fillPoly(clean_mask, [contour], 255)
            
                # Step 6: Apply soft thresholding for gradual edges
                if self.soft_threshold:
                    soft_mask = self.create_soft_mask(clean_mask)
                else:
                    soft_mask = clean_mask
                
                # Store current motion mask
                self._current_motion_mask = soft_mask
            
                # Step 7: Apply glitchy effects to silhouette areas
                if self.use_filter_effects:
                    # When MediaPipe is off, don't apply mask restriction
                    if not self.use_pose_estimation:
                        # Apply effects to full silhouette without skeleton bounding
                        result = self.apply_glitch_effect(frame, soft_mask)
                    elif self.pose_detected and self.pose_landmarks:
                        # MediaPipe is on and pose detected - use bounding box
                        bbox = self.get_skeleton_bbox(frame.shape[:2])
                        if bbox:
                            # Create mask for bounding box area
                            bbox_mask = np.zeros_like(soft_mask)
                            x, y, w, h = bbox
                            bbox_mask[y:y+h, x:x+w] = 255
                            
                            # Apply effects only where both masks overlap
                            combined_mask = cv2.bitwise_and(soft_mask, bbox_mask)
                            result = self.apply_glitch_effect(frame, combined_mask)
                        else:
                            result = frame.copy()
                    else:
                        # MediaPipe is on but no pose detected
                        result = self.apply_glitch_effect(frame, soft_mask)
                else:
                    # Filter effects are off
                    result = frame.copy()
            
                # Step 8: Apply head color effect if active and handle screenshots
                if self.head_effect_active:
                    result = self.apply_head_color_effect(result, frame)
            
                # Step 9: Draw pose landmarks if detected
                if self.use_pose_estimation and self.pose_detected:
                    result = self.draw_pose_landmarks(result)
                    
                    # Draw bounding box outline when pose is detected
                    bbox = self.get_skeleton_bbox(frame.shape[:2])
                    if bbox and not self.head_effect_active:
                        x, y, w, h = bbox
                        # Draw a subtle outline to show the effect area
                        cv2.rectangle(result, (x, y), (x + w, y + h), (100, 100, 100), 2)
            
                # Step 10: Add debug info
                self.add_debug_overlay(result, len(contours), np.sum(clean_mask > 0), camera_src)
                
                # Apply cooldown overlay if active
                if self.cooldown_active and self.cooldown_start_time:
                    cooldown_elapsed = current_time - self.cooldown_start_time
                    if cooldown_elapsed < self.cooldown_duration:
                        remaining_cooldown = self.cooldown_duration - cooldown_elapsed
                        result = self.draw_cooldown_message(result, remaining_cooldown)
                
                return result
        
        # Fallback: just show the frame
        return frame
    
    def create_soft_mask(self, binary_mask):
        """Create soft-edged mask with gradual falloff - optimized"""
        # Skip if radius is too small
        if self.soft_radius < 3:
            return binary_mask
            
        # Downsample for blur, then upsample - much faster
        h, w = binary_mask.shape
        scale = 0.25  # Process at 1/4 resolution
        small_h, small_w = int(h * scale), int(w * scale)
        
        # Downsample
        small_mask = cv2.resize(binary_mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Blur at lower resolution
        kernel_size = max(3, int(self.soft_radius * scale) * 2 + 1)
        soft_small = cv2.GaussianBlur(small_mask, (kernel_size, kernel_size), self.soft_radius * scale / 3)
        
        # Upsample back
        soft_mask = cv2.resize(soft_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return soft_mask
    
    def compare_against_frozen_background(self, current_frame, background_frame):
        """Compare current frame against NEVER-CHANGING background image"""
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple absolute difference
        diff = cv2.absdiff(current_gray, bg_gray)
        
        # Threshold the difference using adjustable sensitivity
        # Get current threshold from MOG2 (even though we're not using MOG2 for detection)
        threshold = 30  # default
        if self.bg_subtractor is not None:
            threshold = int(self.bg_subtractor.getVarThreshold())
        
        _, fg_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        return fg_mask
    
    def detect_pose(self, frame):
        """Detect pose using MediaPipe and analyze hand positions - optimized"""
        if not MEDIAPIPE_AVAILABLE:
            return
        
        # Process at lower resolution for performance
        h, w = frame.shape[:2]
        scale = 0.5  # Process at half resolution
        small_h, small_w = int(h * scale), int(w * scale)
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Process pose detection
        results = self.pose.process(rgb_frame)
        
        # Store results (landmarks are normalized 0-1, so they work at any resolution)
        self.pose_detected = results.pose_landmarks is not None
        self.pose_landmarks = results.pose_landmarks
        
        # Analyze hand positions if pose detected
        if self.pose_detected:
            self.analyze_hand_positions()
    
    def analyze_hand_positions(self):
        """Check if hands are raised above head"""
        if not self.pose_landmarks:
            return
            
        landmarks = self.pose_landmarks.landmark
        
        # Key landmarks (MediaPipe pose landmark indices)
        # 0: Nose, 15: Left wrist, 16: Right wrist
        nose_y = landmarks[0].y
        left_wrist_y = landmarks[15].y
        right_wrist_y = landmarks[16].y
        
        # Check if hands are raised (lower Y value = higher on screen)
        self.left_hand_raised = left_wrist_y < nose_y
        self.right_hand_raised = right_wrist_y < nose_y
        
        # Determine skeleton color based on hand positions
        if self.left_hand_raised and self.right_hand_raised:
            self.current_skeleton_color = "both_hands"
        elif self.left_hand_raised:
            self.current_skeleton_color = "left_hand"
        elif self.right_hand_raised:
            self.current_skeleton_color = "right_hand"
        else:
            self.current_skeleton_color = "normal"
        
        # Activate head effect when hand is raised (not during cooldown)
        if not self.cooldown_active:
            if (self.left_hand_raised or self.right_hand_raised) and not self.head_effect_active:
                self.head_effect_active = True
                self.head_effect_start_time = time.time()
                self.effect_screenshots_taken = False
                print("🎨 Photo countdown started!")
                print("📸 5-second countdown → Flash → Photo!")
            elif not (self.left_hand_raised or self.right_hand_raised) and self.head_effect_active:
                # Deactivate if hands are lowered
                self.head_effect_active = False
                self.head_effect_start_time = None
                self.effect_screenshots_taken = False
                print("🎨 Photo countdown cancelled")
    
    def get_skeleton_colors(self):
        """Get colors based on current gesture state"""
        if self.current_skeleton_color == "both_hands":
            return (0, 255, 0), (0, 255, 0)  # Green landmarks and connections
        elif self.current_skeleton_color == "left_hand":
            return (255, 0, 0), (255, 100, 0)  # Blue landmarks, cyan connections
        elif self.current_skeleton_color == "right_hand":
            return (0, 0, 255), (0, 100, 255)  # Red landmarks, orange connections
        else:
            return (0, 255, 255), (255, 0, 255)  # Yellow landmarks, magenta connections (original)

    def draw_pose_landmarks(self, frame):
        """Draw pose landmarks on frame with gesture-based colors"""
        if not (MEDIAPIPE_AVAILABLE and self.pose_detected and self.pose_landmarks):
            return frame
            
        # Get colors based on current hand gestures
        landmark_color, _ = self.get_skeleton_colors()
        
        # Draw only the landmark points (no connections)
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        for landmark in self.pose_landmarks.landmark:
            if landmark.visibility > 0.5:  # Only draw visible landmarks
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 3, landmark_color, -1)
                cv2.circle(annotated_frame, (x, y), 4, (0, 0, 0), 1)  # Black outline
        
        return annotated_frame
    
    def draw_cooldown_message(self, frame, remaining_time):
        """Draw simple countdown timer"""
        result = frame.copy()
        
        # Draw countdown timer text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Next photo in: {int(remaining_time + 1)}s"
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        
        # Position at top center
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 50
        
        # Draw text with background for visibility
        cv2.rectangle(result, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(result, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
        
        return result
    
    def get_skeleton_bbox(self, frame_shape):
        """Calculate bounding box from skeleton landmarks"""
        if not (self.pose_detected and self.pose_landmarks):
            return None
            
        h, w = frame_shape
        landmarks = self.pose_landmarks.landmark
        
        # Get all visible landmark positions
        xs = []
        ys = []
        for landmark in landmarks:
            if landmark.visibility > 0.5:  # Only use visible landmarks
                xs.append(int(landmark.x * w))
                ys.append(int(landmark.y * h))
        
        if not xs or not ys:
            return None
            
        # Calculate bounding box with some padding
        padding = 30
        x_min = min(xs) - padding
        y_min = min(ys) - padding
        x_max = max(xs) + padding
        y_max = max(ys) + padding
        
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        
        # Make 10% bigger on all sides
        expansion = 0.1
        x_min = int(x_min - width * expansion / 2)
        y_min = int(y_min - height * expansion / 2)
        x_max = int(x_max + width * expansion / 2)
        y_max = int(y_max + height * expansion / 2)
        
        # Ensure bounds are within frame
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def take_effect_screenshots(self, original_frame, effect_frame):
        """Take screenshots of both original and effect frames"""
        import os
        
        # Create screenshots directory if it doesn't exist
        screenshots_dir = "./screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
            print(f"📁 Created directory: {screenshots_dir}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save original camera frame
        original_filename = os.path.join(screenshots_dir, f"{timestamp}_original.png")
        cv2.imwrite(original_filename, original_frame)
        print(f"📸 Saved original: {original_filename}")
        
        # Save frame with effects
        effect_filename = os.path.join(screenshots_dir, f"{timestamp}_effect.png")
        cv2.imwrite(effect_filename, effect_frame)
        print(f"📸 Saved with effects: {effect_filename}")
    
    def apply_head_color_effect(self, frame, original_frame):
        """Draw bounding box around detected pose for photo countdown"""
        if not self.head_effect_start_time:
            return frame
        
        # Check if effect duration has expired
        elapsed_time = time.time() - self.head_effect_start_time
        if elapsed_time > self.head_effect_duration:
            self.head_effect_active = False
            self.head_effect_start_time = None
            self.effect_screenshots_taken = False
            # Start cooldown period
            self.cooldown_active = True
            self.cooldown_start_time = time.time()
            print("🎨 Photo taken!")
            print("⏸️  Starting 5-second cooldown...")
            return frame
        
        # Take screenshots at the end
        if elapsed_time >= self.head_effect_duration - 0.1 and not self.effect_screenshots_taken:
            self.take_effect_screenshots(original_frame, frame)
            self.effect_screenshots_taken = True
        
        result = frame.copy()
        
        # Draw skeleton bounding box if pose is detected
        if self.pose_detected and self.pose_landmarks:
            bbox = self.get_skeleton_bbox(frame.shape[:2])
            if bbox:
                x, y, w, h = bbox
                # Red countdown box
                box_color = (0, 0, 255)  # BGR format, red
                cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 6)
                
                # Add countdown text
                remaining = self.head_effect_duration - elapsed_time
                countdown_text = f"{int(remaining + 1)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                thickness = 3
                text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y - 10
                cv2.putText(result, countdown_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        
        # Flash white at the end
        if 4.9 <= elapsed_time <= 5.0:
            flash_intensity = 1.0 - ((elapsed_time - 4.9) / 0.1)
            white_overlay = np.ones_like(result) * 255
            result = cv2.addWeighted(result, 1.0 - flash_intensity, white_overlay, flash_intensity, 0)
        
        return result

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
            
        return result
    
    def glitch_random_pixels(self, frame, mask):
        """Random pixel effect - heavily optimized version"""
        result = frame.copy()
        
        # Downsample mask for performance
        h, w = mask.shape
        scale = 0.5  # Process at half resolution
        small_h, small_w = int(h * scale), int(w * scale)
        
        # Work on downsampled version
        small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Only process where mask is active
        motion_pixels = small_mask > 200  # Higher threshold
        if np.any(motion_pixels):
            # Limit number of pixels to process
            num_pixels = np.sum(motion_pixels)
            if num_pixels > 50000:  # Cap at 50k pixels
                # Randomly sample pixels to process
                indices = np.where(motion_pixels)
                sample_size = 50000
                selected = np.random.choice(len(indices[0]), sample_size, replace=False)
                motion_pixels = np.zeros_like(motion_pixels)
                motion_pixels[indices[0][selected], indices[1][selected]] = True
                num_pixels = sample_size
            
            # Generate random colors
            random_colors = np.random.randint(0, 255, (num_pixels, 3), dtype=np.uint8)
            small_frame[motion_pixels] = random_colors
            
            # Upscale back with nearest neighbor for pixelated effect
            result = cv2.resize(small_frame, (w, h), interpolation=cv2.INTER_NEAREST)
                
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
        if not self.show_debug_info:
            return
            
        # Larger box to fit all hotkeys and camera list
        box_height = 280 + len(self.available_cameras) * 18  # Adjust height for camera list
        cv2.rectangle(frame, (10, 10), (500, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, box_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Camera: {camera_src}", (20, 35), font, 0.7, (0, 255, 0), 2)
        mode_text = "Lab Color Diff" if self.use_manual_bg else "MOG2"
        cv2.putText(frame, f"Mode: {mode_text} | Glitch: {self.get_glitch_name()} | Effects: {'ON' if self.use_filter_effects else 'OFF'}", (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Silhouettes: {contour_count}", (20, 85), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pixels: {silhouette_pixels}", (20, 110), font, 0.7, (255, 255, 255), 2)
        
        # Pose detection info
        mediapipe_status = "ON" if self.use_pose_estimation else "OFF"
        if self.use_pose_estimation:
            pose_status = "✓ Detected" if self.pose_detected else "✗ Not found"
            cv2.putText(frame, f"MediaPipe: {mediapipe_status} | Pose: {pose_status}", (20, 135), font, 0.7, (0, 255, 0) if self.pose_detected else (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"MediaPipe: {mediapipe_status}", (20, 135), font, 0.7, (100, 100, 100), 2)
            
            # Hand gesture status
            if self.pose_detected:
                gesture_text = f"Hands: L{'↑' if self.left_hand_raised else '↓'} R{'↑' if self.right_hand_raised else '↓'} ({self.current_skeleton_color})"
                cv2.putText(frame, gesture_text, (20, 160), font, 0.5, (255, 255, 255), 1)
                
                # Head effect status
                if self.head_effect_active and self.head_effect_start_time:
                    elapsed = time.time() - self.head_effect_start_time
                    remaining = max(0, self.head_effect_duration - elapsed)
                    effect_text = f"Head Effect: {remaining:.1f}s remaining"
                    cv2.putText(frame, effect_text, (20, 185), font, 0.5, (255, 200, 0), 1)
        
        # Controls section with clearer hotkey display
        y_offset = 205 if (self.pose_detected and self.head_effect_active) else 180
        
        # Title for controls
        cv2.putText(frame, "HOTKEYS:", (20, y_offset), font, 0.6, (255, 255, 0), 2)
        
        # Individual hotkeys with better formatting
        control_y = y_offset + 20
        cv2.putText(frame, "[1/2/3] Switch Camera", (20, control_y), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[SPACE] Cycle Effects", (250, control_y), font, 0.5, (200, 200, 200), 1)
        
        control_y += 18
        cv2.putText(frame, "[M] Toggle MediaPipe", (20, control_y), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[F] Toggle Filters", (250, control_y), font, 0.5, (200, 200, 200), 1)
        
        control_y += 18
        cv2.putText(frame, "[W/S] Sensitivity ↑/↓", (20, control_y), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[A/D] Intensity ←/→", (250, control_y), font, 0.5, (200, 200, 200), 1)
        
        control_y += 18
        cv2.putText(frame, "[R] Reset", (20, control_y), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[H] Hide This", (120, control_y), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[B] BG Mode", (250, control_y), font, 0.5, (200, 200, 200), 1)
        
        control_y += 18
        cv2.putText(frame, "[Q/ESC] Quit", (20, control_y), font, 0.5, (200, 200, 200), 1)
        
        # Show current sensitivity if available
        if self.bg_subtractor is not None:
            sensitivity = self.bg_subtractor.getVarThreshold()
            cv2.putText(frame, f"Sensitivity: {sensitivity:.0f} | Intensity: {self.glitch_intensity:.1f}", (20, control_y + 10), font, 0.5, (255, 255, 0), 1)
        
        # Camera list section
        cameras_y = control_y + 30
        cv2.putText(frame, "AVAILABLE CAMERAS:", (20, cameras_y), font, 0.6, (255, 255, 0), 2)
        cameras_y += 20
        
        for cam in self.available_cameras:
            # Highlight current camera
            color = (0, 255, 0) if cam['index'] == camera_src else (200, 200, 200)
            cam_text = f"[{cam['index']}] {cam['name']} - {cam['resolution']} @ {cam['fps']:.0f}fps"
            cv2.putText(frame, cam_text, (20, cameras_y), font, 0.5, color, 1)
            cameras_y += 18
    
    def get_glitch_name(self):
        names = ["Random Pixels", "Glitch Blocks"]
        return names[self.glitch_mode]
    
    def adjust_sensitivity(self, delta):
        """Adjust motion detection sensitivity for MOG2"""
        if self.bg_subtractor is not None:
            current_threshold = self.bg_subtractor.getVarThreshold()
            new_threshold = max(5, min(200, current_threshold + delta))
            self.bg_subtractor.setVarThreshold(new_threshold)
            # Sensitivity updated
        else:
            pass  # Not initialized yet
    
    def cycle_glitch_mode(self):
        """Switch between glitch effects"""
        self.glitch_mode = (self.glitch_mode + 1) % 2
        # Mode cycled
    
    def adjust_glitch_intensity(self, delta):
        """Adjust glitch effect intensity"""
        self.glitch_intensity = max(0.1, min(3.0, self.glitch_intensity + delta))
        # Intensity adjusted
    
    def reset_to_countdown(self):
        """Reset the entire process back to countdown phase"""
        self.countdown_active = True
        self.learning_background = False
        self.ready_for_detection = False
        self.countdown_start = time.time()
        self.learning_start = None
        self.learning_frames_processed = 0
        self.bg_subtractor = None
        # Reset to countdown
    
    def toggle_mediapipe(self):
        """Toggle MediaPipe pose estimation on/off"""
        if MEDIAPIPE_AVAILABLE:
            self.use_pose_estimation = not self.use_pose_estimation
            if not self.use_pose_estimation:
                self.pose_detected = False
                self.pose_landmarks = None
            print(f"🤖 MediaPipe: {'ON' if self.use_pose_estimation else 'OFF'}")
        else:
            print("⚠️  MediaPipe not available")
    
    def toggle_filter_effects(self):
        """Toggle filter effects on/off"""
        self.use_filter_effects = not self.use_filter_effects
        print(f"🎨 Filter Effects: {'ON' if self.use_filter_effects else 'OFF'}")
    
    def toggle_debug_info(self):
        """Toggle debug info display on/off"""
        self.show_debug_info = not self.show_debug_info
        print(f"📊 Debug Info: {'ON' if self.show_debug_info else 'OFF'}")
    
    def toggle_background_mode(self):
        """Toggle between manual background comparison and MOG2"""
        self.use_manual_bg = not self.use_manual_bg
        mode_name = "Lab Color Difference" if self.use_manual_bg else "MOG2"
        print(f"🎨 Background Detection Mode: {mode_name}")
        print(f"  {'Direct comparison against frozen background' if self.use_manual_bg else 'Gaussian Mixture Model (MOG2)'}")
    
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
    print("  0-9   - Switch camera source (number keys)")
    print("  M     - Toggle MediaPipe pose detection")
    print("  F     - Toggle filter effects")
    print("  H     - Toggle debug info overlay")
    print("  B     - Toggle background detection mode (Lab/MOG2)")
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
    
    # Create window first
    cv2.namedWindow('Glitchy Silhouettes', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Glitchy Silhouettes', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("🎬 Camera ready! MOG2 learning background...")
    print("Move around after a few seconds to see clean silhouette detection!")
    print("Press Q or ESC to quit")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"✓ Processing frame {frame_count}...")
            frame = camera.read()
            if frame is not None:
                # Process frame for glitchy silhouettes
                result = processor.process_frame(frame, camera.src)
                
                if result is not None and len(result.shape) == 3:
                    # Get screen dimensions
                    try:
                        rect = cv2.getWindowImageRect('Glitchy Silhouettes')
                        screen_width = rect[2] if rect[2] > 0 else 1920
                        screen_height = rect[3] if rect[3] > 0 else 1080
                    except:
                        # Fallback dimensions if getWindowImageRect fails
                        screen_width = 1920
                        screen_height = 1080
                    
                    # Calculate scaling to fill screen while maintaining aspect ratio
                    frame_h, frame_w = result.shape[:2]
                    scale_w = screen_width / frame_w
                    scale_h = screen_height / frame_h
                    scale = max(scale_w, scale_h)  # Use max to fill screen (may crop)
                    
                    # Scale the frame
                    new_width = int(frame_w * scale)
                    new_height = int(frame_h * scale)
                    result_scaled = cv2.resize(result, (new_width, new_height))
                    
                    # Crop to fit screen exactly
                    y_offset = (new_height - screen_height) // 2
                    x_offset = (new_width - screen_width) // 2
                    result_cropped = result_scaled[y_offset:y_offset+screen_height, x_offset:x_offset+screen_width]
                    
                    cv2.imshow('Glitchy Silhouettes', result_cropped)
                else:
                    # Show a black screen if no result
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "No camera feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('Glitchy Silhouettes', black_frame)
        
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
            elif key == ord('m') or key == ord('M'):  # Toggle MediaPipe
                processor.toggle_mediapipe()
            elif key == ord('f') or key == ord('F'):  # Toggle filter effects
                processor.toggle_filter_effects()
            elif key == ord('h') or key == ord('H'):  # Toggle debug info
                processor.toggle_debug_info()
            elif key == ord('b') or key == ord('B'):  # Toggle background mode
                processor.toggle_background_mode()
            # Other keys ignored
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("👋 Glitchy silhouettes complete!")

if __name__ == "__main__":
    main()
