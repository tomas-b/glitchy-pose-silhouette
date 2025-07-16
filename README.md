# Glitchy Silhouette

Real-time silhouette detection with glitch effects and photo booth mode.

## Setup

```bash
git clone https://github.com/malohuandus/test.git
cd test
uv sync
uv run python glitchy_silhouette.py
```

## Usage

1. Stand in front of camera
2. Wait for 5-second countdown
3. Move to see glitch effects on your silhouette
4. Raise hand above head to take a photo

## Controls

- **SPACE** - Change effect
- **W/S** - Adjust sensitivity  
- **A/D** - Adjust intensity
- **R** - Reset
- **Q** - Quit

## Photo Booth

When you raise your hand:
- Red box appears with 5-second countdown
- White flash = photo taken
- Saves two images: original and with effects
- 5-second cooldown before next photo

## Effects

- Random Pixels - Colorful noise
- Glitch Blocks - Rectangle distortions  
- Scan Lines - Horizontal interference
- Datamosh - Compression artifacts

## Technical Details

- Uses OpenCV for motion detection (MOG2 background subtraction)
- MediaPipe for pose/skeleton tracking
- Effects only apply within detected body area for performance
- Threaded camera capture for smooth operation

## Requirements

- Python 3.8+
- Webcam
- uv package manager