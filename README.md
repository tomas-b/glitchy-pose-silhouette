# Glitchy Camera - Incremental Feature Versions

This directory contains incremental versions of glitchy_camera.py, each building on the previous one:

## Base Version
- **glitchy_camera.py** - The original working version with `--camera` command line argument

## Feature Versions

### 1. Toggles (glitchy_camera_1_toggles.py)
- Added M key to toggle MediaPipe pose detection on/off
- Added F key to toggle filter effects on/off  
- Added H key to toggle debug overlay on/off
- Shows toggle states in debug overlay

### 2. Hand Gestures (glitchy_camera_2_hand_gestures.py)
- Detects when hands are raised above head
- Changes skeleton colors based on gestures:
  - Yellow/Magenta: Normal (hands down)
  - Blue/Cyan: Left hand raised
  - Red/Orange: Right hand raised
  - Green: Both hands raised
- Shows hand status in debug overlay

### 3. Photo Booth (glitchy_camera_3_photo_booth.py)
- Raise hand to trigger 5-second photo countdown
- Red bounding box appears around pose during countdown
- White flash effect at photo capture
- 5-second cooldown period between photos
- Shows countdown timer and cooldown status

### 4. Screenshots (glitchy_camera_4_screenshots.py)
- Saves photos to `./screenshots/` directory
- Captures both original and effect versions
- Filenames use timestamp format: `YYYYMMDD_HHMMSS_original.png`
- Directory created automatically if it doesn't exist

## Usage

All versions support the `--camera` argument:

```bash
./glitchy_camera_4_screenshots.py --camera 3
```

## Controls (Full Feature Set)

- **SPACE** - Cycle glitch effects (Random Pixels, Blocks, Scanlines, Datamosh)
- **W/S** - Adjust motion sensitivity (up/down)
- **A/D** - Adjust glitch intensity (left/right)
- **M** - Toggle MediaPipe pose detection
- **F** - Toggle filter effects
- **H** - Toggle debug overlay
- **R** - Reset to countdown phase
- **ESC/Q** - Quit

🤚 **Raise hand above head for photo booth mode!**
