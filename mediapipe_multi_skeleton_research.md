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

For this artistic application focused on single-user interaction, the current single-person approach is likely sufficient.