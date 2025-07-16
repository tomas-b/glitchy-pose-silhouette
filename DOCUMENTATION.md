# Glitchy Silhouette - Technical Documentation

## Overview

Glitchy Silhouette is an interactive art installation that combines real-time computer vision, pose detection, and glitch effects to create a unique photo booth experience. The system detects human silhouettes and applies dynamic visual effects, with an integrated gesture-controlled photo capture feature.

## Features

### Core Functionality
- **Real-time silhouette detection** using MOG2 background subtraction
- **Pose estimation** with MediaPipe for skeleton tracking
- **Glitch effects** applied to detected motion areas
- **Gesture-controlled photo booth** triggered by raising hand above head
- **Automatic screenshot capture** with countdown timer

### Visual Effects
1. **Random Pixels** - Randomized color pixels in motion areas
2. **Glitch Blocks** - Rectangular color blocks
3. **Scan Lines** - Horizontal line distortions
4. **Datamosh** - Compression artifact simulation

## Technical Architecture

### Main Components

#### 1. ThreadedCamera
- Provides high-performance camera capture using threading
- Maintains a small buffer to reduce latency
- Optimized for 30 FPS capture

#### 2. GlitchySilhouetteProcessor
The main processing class that handles:
- Background learning and subtraction
- Motion detection and edge enhancement
- Pose estimation and gesture recognition
- Effect application and rendering

### Processing Pipeline

1. **Initialization Phase** (5 seconds)
   - Countdown display to allow user to exit frame
   - Background model initialization

2. **Background Learning** (3 seconds)
   - MOG2 background subtractor learns static background
   - Captures reference frame for comparison

3. **Detection Phase**
   - Motion detection using frozen background comparison
   - Edge detection for enhanced silhouette quality
   - Pose estimation for skeleton tracking
   - Effect application within bounding box

4. **Photo Booth Mode**
   - Triggered by hand raised above head
   - 5-second countdown with red bounding box
   - White flash and dual screenshot capture
   - 5-second cooldown with timer display

## Key Algorithms

### Background Subtraction
- Uses MOG2 with `learningRate=0.0` for frozen background
- Combined with Canny edge detection for cleaner masks
- Morphological operations for noise reduction

### Pose Detection
- MediaPipe Pose with 33 body landmarks
- Gesture recognition based on wrist-to-nose position comparison
- Bounding box calculation from visible landmarks

### Effect Masking
- Effects applied only within skeleton bounding box
- Soft thresholding with Gaussian blur for smooth edges
- Optimized rendering using NumPy vectorization

## Performance Optimizations

1. **Threaded camera capture** - Decouples capture from processing
2. **Selective effect application** - Only processes areas within bounding box
3. **Vectorized operations** - NumPy-based calculations for speed
4. **Simplified cooldown** - No heavy fade effects during cooldown
5. **Frame-based state management** - Efficient state tracking

## Controls

- **SPACE** - Cycle through glitch effects
- **W/S** - Adjust motion sensitivity (MOG2 variance threshold)
- **A/D** - Adjust glitch intensity
- **R** - Reset to countdown phase
- **Q/ESC** - Quit application

## Configuration Parameters

### Motion Detection
- `varThreshold`: 135 (default) - Higher = less sensitive
- `min_contour_area`: 3000 - Minimum area for valid detection

### Effects
- `glitch_intensity`: 0.1 to 3.0 - Effect strength multiplier
- `soft_radius`: 20 pixels - Soft edge falloff distance

### Photo Booth
- `countdown_duration`: 5 seconds before photo
- `cooldown_duration`: 5 seconds after photo
- `head_effect_radius`: Not used in final version

## Output

Screenshots are saved with timestamps:
- `screenshot_original_YYYYMMDD_HHMMSS.png` - Raw camera frame
- `screenshot_effect_YYYYMMDD_HHMMSS.png` - Frame with effects

## Dependencies

- OpenCV (cv2) - Computer vision operations
- NumPy - Numerical computations
- MediaPipe - Pose estimation
- Threading - Concurrent camera capture

## Usage

```bash
python glitchy_silhouette.py
```

Or with uv (self-contained):
```bash
./glitchy_pose_selfcontained.py
```