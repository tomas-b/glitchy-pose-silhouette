# Glitchy Silhouette <�

An interactive art installation that transforms your movements into glitchy digital art. Features real-time motion detection, pose tracking, and a gesture-controlled photo booth mode.

## ( Features

- **Real-time silhouette detection** with customizable glitch effects
- **Pose tracking** with skeleton visualization
- **Photo booth mode** - Raise your hand to trigger a countdown and capture
- **Multiple glitch effects** - Random pixels, blocks, scan lines, and datamosh
- **Performance optimized** - Effects only applied within detected body area

## 🚀 Quick Start

### Setup and run:
```bash
# Clone the repository
git clone https://github.com/malohuandus/test.git
cd test

# Sync dependencies with uv
uv sync

# Run the application
uv run python glitchy_silhouette.py
```

### Using self-contained script:
```bash
# Make executable
chmod +x glitchy_pose_selfcontained.py

# Run directly (uv required)
./glitchy_pose_selfcontained.py
```

## 📸 Photo Booth Mode

1. **Raise your hand above your head** to trigger photo mode
2. **Red bounding box** appears with 5-second countdown
3. **White flash** indicates photo capture
4. **5-second cooldown** before next photo

Photos are saved as:
- `screenshot_original_*.png` - Raw camera frame
- `screenshot_effect_*.png` - Frame with effects

## <� Controls

| Key | Action |
|-----|--------|
| **SPACE** | Cycle glitch effects |
| **W/S** | Adjust sensitivity |
| **A/D** | Adjust intensity |
| **R** | Reset background |
| **Q/ESC** | Quit |

## =� Requirements

- Python 3.8+
- Webcam
- OpenCV
- NumPy
- MediaPipe

## 📖 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical information.

## 🎨 Effects Gallery

### Random Pixels
Transforms detected motion into colorful pixel noise

### Glitch Blocks
Creates rectangular color distortions

### Scan Lines
Horizontal interference patterns

### Datamosh
Simulates video compression artifacts

## > Contributing

Feel free to submit issues and enhancement requests!

## =� License

MIT License - See LICENSE file for details