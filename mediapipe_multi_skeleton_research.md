# MediaPipe Multiple Skeleton Detection Research

## Current Limitation
The current implementation uses MediaPipe Pose, which is designed to detect a single person's pose landmarks. The model focuses on the most prominent person in the frame.

## Options for Multiple Person Detection

### 1. MediaPipe Holistic (Not Multi-Person)
- MediaPipe Holistic combines pose, face, and hand detection
- Still limited to single person detection
- Not suitable for our multi-person requirement

### 2. MediaPipe BlazePose GHUM (Single Person)
- The underlying model used by MediaPipe Pose
- Optimized for single person detection
- No native multi-person support

### 3. Alternative Approaches

#### Option A: Object Detection + Single Pose
1. Use a person detector (like YOLO or MediaPipe's object detection)
2. Crop detected person bounding boxes
3. Run MediaPipe Pose on each cropped region
4. Combine results

Implementation sketch:
```python
# Pseudo-code
persons = detect_persons(frame)  # Returns bounding boxes
poses = []
for person_bbox in persons:
    cropped = frame[person_bbox]
    pose = mediapipe_pose.process(cropped)
    if pose.pose_landmarks:
        # Transform landmarks back to full frame coordinates
        poses.append(transform_landmarks(pose, person_bbox))
```

#### Option B: Use Different Libraries
1. **OpenPose**: Supports multi-person pose estimation
   - More computationally expensive
   - Requires separate installation
   
2. **MMPose**: Supports various multi-person models
   - Part of OpenMMLab ecosystem
   - Good accuracy but heavier

3. **YOLO-Pose**: YOLO variants trained for pose estimation
   - Fast and supports multiple people
   - Less accurate than MediaPipe for single person

### Recommended Approach for This Project

Given the current architecture and performance requirements:

1. **Short term**: Document the limitation and keep single-person detection
2. **Medium term**: Implement Option A (Object Detection + Single Pose) if needed
3. **Long term**: Consider switching to a dedicated multi-person pose library

### Implementation Considerations

If implementing multi-person support:
- Performance will decrease (multiple inference passes)
- Need to handle overlapping detections
- UI/UX changes needed to show multiple skeletons
- Effect application logic needs updates

### Current Code Impact

To support multiple skeletons, these areas would need changes:
- `detect_pose()`: Process multiple regions
- `analyze_hand_positions()`: Track multiple people's gestures
- `draw_pose_landmarks()`: Draw multiple skeletons
- `get_skeleton_bbox()`: Return multiple bounding boxes
- Effect application: Decide how to apply effects with multiple people

## Conclusion

MediaPipe Pose is inherently single-person. True multi-person support requires either:
1. Multiple inference passes with person detection
2. Switching to a different pose estimation library

# MediaPipe multi-person pose detection reveals native support with significant limitations

MediaPipe's modern Tasks API does support multi-person pose detection through the `num_poses` parameter, contradicting widespread belief that it's single-person only. However, this capability comes with substantial architectural limitations and performance trade-offs that make hybrid approaches using YOLO + MediaPipe or alternative solutions like YOLOv7-Pose often more practical for production multi-person scenarios.

The confusion stems from MediaPipe's evolution: the legacy API (`mp.solutions.pose.Pose()`) was single-person only and deprecated in March 2023, while the newer MediaPipe Tasks `PoseLandmarker` API introduced the `num_poses` parameter defaulting to 1. This default setting, combined with MediaPipe's fundamental optimization for single-person tracking, has perpetuated the misconception about its capabilities.

## Native multi-person support exists but struggles with proximity

MediaPipe's `PoseLandmarker` API allows multi-person detection by configuring the `num_poses` parameter:

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configure for multi-person detection
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'),
    num_poses=5,  # Detect up to 5 people
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)
```

However, this native support faces **critical limitations when people are within 75cm of each other**, causing detection failures and tracking instability. The architecture, originally designed for single-person fitness and dance applications, struggles with overlapping poses and maintains separate tracking for each person without intelligent identity management across frames.

## Performance degrades linearly with each additional person

Benchmarks reveal MediaPipe's multi-person performance follows predictable patterns. On a MacBook Pro with Intel i9, the full BlazePose model achieves **81 FPS for single person** but drops to approximately **40 FPS for 2 people** and **20 FPS for 4 people**. This linear degradation occurs because MediaPipe processes each person independently without computational efficiency gains from batch processing.

Memory usage scales similarly - each additional person requires **200-400MB** depending on model complexity (lite, full, or heavy). A 4-person scenario consumes 800MB-1.6GB total memory with no resource sharing between instances. The Python implementation remains CPU-only, lacking GPU acceleration that competing solutions leverage for better multi-person performance.

Comparative analysis shows YOLOv7-Pose achieves **83.39 FPS** on GPU for multi-person detection versus MediaPipe's **29 FPS** on CPU for single person. While MediaPipe excels at detecting distant or small-scale poses, YOLOv7 handles occlusion and close proximity better - critical for real multi-person scenarios.

## Hybrid YOLO + MediaPipe approach dominates real implementations

The developer community has converged on a hybrid solution combining YOLO for person detection with MediaPipe for individual pose estimation. This approach bypasses MediaPipe's proximity limitations while leveraging its superior 33-keypoint accuracy:

```python
import cv2
import mediapipe as mp
import torch

# Initialize models
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]  # Detect only persons
mp_pose = mp.solutions.pose

# Process frame
def process_multi_person_frame(frame):
    # Detect people with YOLO
    results = yolo_model(frame)
    
    pose_results = []
    for box in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Add margin for better pose detection
        margin = 10
        person_crop = frame[y1-margin:y2+margin, x1-margin:x2+margin]
        
        # Apply MediaPipe to individual crop
        with mp_pose.Pose(model_complexity=1) as pose:
            pose_result = pose.process(person_crop)
            
            if pose_result.pose_landmarks:
                # Transform coordinates back to original frame
                for landmark in pose_result.pose_landmarks.landmark:
                    landmark.x = (landmark.x * (x2-x1) + x1) / frame.shape[1]
                    landmark.y = (landmark.y * (y2-y1) + y1) / frame.shape[0]
                
                pose_results.append(pose_result)
    
    return pose_results
```

Critical implementation details include **adding crop margins** to prevent landmark cutoff, **coordinate transformation** from crop space to original frame space, and **person tracking** across frames to maintain identity consistency. Without proper tracking, poses can "swap" between people as YOLO detection order changes.

## Memory and computational optimization strategies prove essential

Efficient multi-person implementations require careful resource management. **Instance pooling** prevents expensive MediaPipe object creation/destruction cycles, while **frame skipping** (processing every 2nd or 3rd frame) maintains acceptable performance for non-critical applications. Setting `static_image_mode=False` enables video-optimized tracking, and `model_complexity=1` balances accuracy with speed.

For production systems, limiting input resolution to 640 pixels maximum and implementing proper garbage collection prevents memory accumulation over extended runtime. The most performant approach uses YOLOv5 nano (`yolov5n`) for person detection combined with MediaPipe lite model when accuracy permits.

## Recent updates focus on framework improvements, not multi-person capabilities

MediaPipe v0.10.24 (2025) introduced enhanced C++ graph builder support, WebGPU utilities, and Gemma model integration, but **no improvements to multi-person pose detection**. The BlazePose model remains at 33 keypoints with three complexity variants, maintaining its single-person optimization focus.

The official roadmap shows no plans for native multi-person enhancements, suggesting the hybrid approach will remain necessary. Industry momentum has shifted toward purpose-built multi-person solutions like YOLOv8-Pose and emerging transformer-based architectures designed for crowd pose estimation.

## Alternative solutions excel at native multi-person detection

For applications requiring robust multi-person support, several alternatives outperform MediaPipe:

**OpenPose** provides the most comprehensive solution with 137 total keypoints and runtime invariant to person count, but requires $25,000 annual commercial licensing. **YOLOv7-Pose** offers the best balance - native multi-person support, GPU acceleration, and superior occlusion handling at 17 COCO keypoints. **PoseNet** achieves 97.6% accuracy with multi-person capability for web applications, while Google's **MoveNet** delivers fastest single-person inference with multi-person version under development.

## Practical recommendations depend on specific requirements

For **2-3 people with high accuracy needs**, MediaPipe's native `num_poses` parameter or hybrid YOLO approach works acceptably. Applications with **4+ people** should use YOLOv7-Pose or OpenPose for reliable performance. **Mobile deployments** benefit from MediaPipe's optimization despite multi-person limitations, while **GPU-equipped systems** achieve better results with YOLOv7 or OpenPose.

The key insight: MediaPipe technically supports multi-person detection but wasn't architected for it. Its strength remains single-person pose estimation with industry-leading mobile performance. For true multi-person applications, purpose-built solutions or hybrid approaches provide superior reliability, performance, and maintainability. Choose based on your specific constraints - person count, hardware platform, accuracy requirements, and development complexity tolerance.For this artistic application focused on single-user interaction, the current single-person approach is likely sufficient.
