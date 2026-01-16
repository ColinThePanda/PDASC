# PDASC

## Panda ASCII

**High-performance terminal ASCII art converter for images, videos, and live camera feeds**

[![PyPI version](https://badge.fury.io/py/pdasc.svg)](https://pypi.org/project/PDASC/)
[![Python](https://img.shields.io/pypi/pyversions/pdasc?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/PDASC/)
[![License](https://img.shields.io/github/license/ColinThePanda/PDASC?style=flat-square)](LICENSE)

[Installation](#installation) • [Quick Start](#quick-start) • [Usage](#usage) • [Documentation](#documentation)

---

![Demo Image](docs/demo_image.png)

![Demo Video](docs/demo_video.gif)

## Overview

PDASC transforms multimedia content into colored ASCII art with hardware-accelerated processing. Features include Numba JIT compilation for near-C performance, a custom `.asc` file format with Zstandard compression for instant playback, and GPU-accelerated rendering for web deployment with edge detection support.

### Key Capabilities

- **Multiple Input Sources**: Static images, video files, live camera feeds, pre-encoded `.asc` files
- **Hardware Acceleration**: Numba JIT compilation and OpenGL shader-based rendering
- **Advanced Processing**: Customizable character sets from TrueType fonts, adjustable ASCII density and block size
- **Edge Detection**: Multi-pass rendering pipeline with Sobel edge detection for enhanced detail preservation
- **Audio Support**: Synchronized PCM16 audio in `.asc` format with real-time streaming
- **Instant Playback**: Pre-rendered ANSI sequences with Zstandard compression (5-38× compression ratio)
- **Web Interface**: Interactive image converter and GPU-processed video player

## Installation

### Install with pip

```bash
pip install pdasc
```

### Alternative: Install with pipx

```bash
pipx install pdasc
```

Note that pipx requires a C compiler to be installed.

### Prerequisites

**Required:**

- Python 3.10 or higher
- FFmpeg ([download](https://ffmpeg.org))

**Platform-specific FFmpeg installation:**

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

**Terminal requirements:**

- 24-bit color support (most modern terminals)
- Test with: `printf "\x1b[38;2;255;100;0mTRUECOLOR\x1b[0m\n"`

## Quick Start

```bash
# Display an image
pdasc play image.png

# Play a video with audio
pdasc play video.mp4

# Stream from webcam
pdasc play camera

# Encode for instant playback
pdasc encode video.mp4 -o output.asc
pdasc play output.asc

# Launch web interface
pdasc website

# GPU-accelerated web video with edge detection
pdasc website video.mp4
```

## Usage

### Commands

#### `play` - Display images, videos, camera, or `.asc` files

```bash
pdasc play <input> [options]

# Examples
pdasc play photo.png -n 70 -b 4          # High detail image
pdasc play video.mp4 --no-audio          # Silent video
pdasc play camera -c 1 -n 32 -b 8        # Webcam with custom settings
pdasc website video.mp4 -n 32            # GPU-rendered web video
```

**Input types:**

- Image file path (PNG, JPG, GIF, BMP, TIFF, WebP)
- Video file path (MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V)
- `camera` for webcam input
- `.asc` file path
- `.asc.mp4` file path (GPU-processed)

#### `encode` - Convert to `.asc` format

```bash
pdasc encode <input> -o <output.asc> [options]

# Examples
pdasc encode video.mp4 -o video.asc      # Encode with defaults
pdasc encode image.png -o img.asc --no-color
pdasc encode large.mp4 -o out.asc -b 16  # Larger blocks = smaller file
```

#### `website` - Launch web interface

```bash
pdasc website [input]

# Examples
pdasc website                    # Interactive image converter
pdasc website video.mp4          # GPU-rendered video player with edge detection
pdasc website output.asc.mp4     # Play existing web video
```

### Options

| Option         | Short | Default            | Description                                               |
| -------------- | ----- | ------------------ | --------------------------------------------------------- |
| `--block-size` | `-b`  | `8`                | Pixels per character (2-32, must divide image dimensions) |
| `--num-ascii`  | `-n`  | `8`                | Number of ASCII characters (2-95)                         |
| `--font`       | `-f`  | `CascadiaMono.ttf` | TrueType font path for character generation               |
| `--no-color`   |       |                    | Disable color output (grayscale only)                     |
| `--no-audio`   |       |                    | Disable audio playback                                    |
| `--camera`     | `-c`  | `0`                | Camera index for camera input                             |
| `--output`     | `-o`  | `ascii_out.asc`    | Output file path                                          |
| `--debug`      |       |                    | Show FPS and debug info                                   |
| `--port`       | `-p`  | `5000`             | Web server port (website only)                            |

**Note on block size:** Must be a factor of both image width and height. Common values: 2, 4, 8, 16, 32.

**Note on website mode:** GPU rendering with `website` only supports `-n` parameter, not `-b`.

## Documentation

### GPU-Accelerated Video Processing

The web video player uses a multi-pass OpenGL shader pipeline for real-time ASCII art rendering with edge detection, inspired by [Acerola's ASCII shader technique](https://www.youtube.com/watch?v=gg40RWiaHRY).

The core concept works in two stages:

**Fill ASCII (luminance-based):**

1. Extract luminance from the input frame
2. Quantize to 10 discrete brightness levels
3. Map each level to an ASCII character based on visual density (` .;coPO?@█`)

**Edge ASCII (edge-based):**

1. Apply Difference of Gaussians (DoG) filter to detect edges
2. Use Sobel filter to compute edge angles
3. Quantize angles into 4 directions: horizontal (0°), vertical (90°), and diagonals (45°, 135°)
4. Determine the dominant edge direction in each 8×8 pixel block
5. Map the dominant direction to an edge character

The original technique uses a compute shader that dispatches a thread group for each 8×8 pixel chunk, with each thread analyzing edge directions and writing to shared memory to find the dominant edge type. This implementation uses a simplified approach with iterative downsampling passes instead.

**Fixed character set:** The shader uses 4 edge ASCII characters and 8 fill ASCII characters. This configuration is not user-configurable.

#### Shader Passes

The rendering pipeline consists of these sequential passes:

```txt
Input Frame
    ↓
1. Luminance Extraction
    ↓
2. Gaussian Blur (Horizontal)
    ↓
3. Gaussian Blur (Vertical) + DoG Edge Detection
    ↓
4. Sobel Edge Detection (Horizontal)
    ↓
5. Sobel Edge Detection (Vertical)
    ↓
6. Color/Luminance Packing
    ↓
7. Downsampling Chain (1/2 → 1/4 → 1/8)
    ↓
8. ASCII Character Selection
    ↓
Output Video (.asc.mp4)
```

Each 8×8 pixel block is analyzed to determine whether to use an edge character (based on edge threshold and direction) or a fill character (based on luminance).

**Example (Acerola's original implementation):**

![Acerola ASCII Example](docs/acerola_ascii.gif)

_Note: This example showcases Acerola's original shader._

### The .asc File Format

The `.asc` (ASCII Container) format stores pre-rendered ANSI escape sequences compressed with Zstandard, enabling instant playback with zero conversion overhead.

#### Storage Characteristics

The format prioritizes playback performance over storage efficiency, typically requiring ~20× more storage than the original before compression. Zstandard compression (level 5) reduces this significantly:

**Example (5326 frames, 720p, block-size 4, num-ascii 32):**

- Original H.264 video: 48.3 MB
- Uncompressed ANSI: 3.62 GB
- Compressed colored: 722.15 MB (~5× compression)
- Compressed grayscale: 93.55 MB (~38.7× compression)
- Audio PCM16: 35.85 MB

#### Format Specification

**Header (24 bytes)**

| Offset | Size | Type   | Description                           |
| ------ | ---- | ------ | ------------------------------------- |
| 0x00   | 4    | char   | Magic: "ASCI"                         |
| 0x04   | 2    | uint16 | Version (currently 2)                 |
| 0x06   | 2    | uint16 | Flags (IS_VIDEO=0x01, HAS_AUDIO=0x02) |
| 0x08   | 4    | float  | FPS                                   |
| 0x0C   | 4    | uint32 | Frame count                           |
| 0x10   | 8    | -      | Reserved                              |

**Frame Index Section**

- Array of uint32 values, one per frame, storing uncompressed lengths

**Compressed Frame Data**

- 4-byte uint32: compressed data size
- Zstandard-compressed concatenated frames (ANSI strings)

**Audio Section (optional)**

- 4-byte uint32: audio data size
- 1-byte uint8: format (1 = PCM16)
- 4-byte uint32: sample rate
- 1-byte uint8: channels
- Raw PCM16 audio data

### How It Works

#### Character Mapping

The system analyzes TrueType fonts to create optimal character-to-brightness mappings:

1. Render each printable ASCII character at 48×48 pixels
2. Calculate average luminance per character
3. Map characters to quantized luminance values
4. Create uniform grayscale ramp (e.g., " .:-=+\*#%@")

#### Processing Pipeline

```txt
Input Image/Frame
    ↓
Divide into blocks (e.g., 8×8 pixels)
    ↓
Compute average color per block (Numba-accelerated)
    ↓
Calculate luminance: L = 0.2126R + 0.7152G + 0.0722B
    ↓
Map luminance to character index
    ↓
Generate ANSI escape sequence with RGB color
    ↓
Compress with Zstandard (encode) or Render (play)
```

#### Numba Acceleration

Core processing loops are JIT-compiled with Numba for near-C performance:

```python
@njit(parallel=True, fastmath=True, cache=True)
def compute_blocks(img, cs, gray_levels, color):
    # Parallel block processing
    # Significantly faster than pure Python
```

#### GPU Rendering for Web

Browser playback uses ModernGL with OpenGL shaders (browsers cannot efficiently update thousands of span elements at 30+ FPS):

- Fragment shader processes each pixel in parallel
- Converts RGB to ASCII character lookup
- Applies edge detection for enhanced detail
- Outputs standard `.asc.mp4` video file for smooth browser playback

### Performance Optimization

#### Block Size vs Quality

| Block Size | Resolution | Performance | Use Case                       |
| ---------- | ---------- | ----------- | ------------------------------ |
| 4×4        | Very High  | Good        | High-quality images and videos |
| 8×8        | High       | Better      | **Default, recommended**       |
| 16×16      | Medium     | Best        | Lower-end hardware             |
| 32×32      | Low        | Fastest     | Very limited hardware          |

#### ASCII Character Count

| Count | Detail | Use Case                         |
| ----- | ------ | -------------------------------- |
| 4-16  | Low    | Artistic effect, retro aesthetic |
| 32-64 | Medium | **Good balance**                 |
| 70-95 | High   | Maximum detail preservation      |

#### Best Practices

1. **Real-time (webcam)**: Use `-b 8 -n 32`
2. **High-quality images**: Use `-b 4 -n 70`
3. **Video playback**: Always encode to `.asc` first
4. **Web playback**: Use `website` command with GPU acceleration
5. **Storage constraints**: Use grayscale (`--no-color`) for 5-8× smaller files

## Gallery

### High-Resolution Image

```bash
pdasc play landscape.png -n 50 -b 8
```

![Landscape Example](docs/landscape.png)

### Real-Time Webcam

```bash
pdasc play camera -b 12 -n 40
```

![Webcam Example](docs/camera.gif)

### Grayscale Art

```bash
pdasc play artwork.png --no-color -n 95 -b 2
```

![Grayscale Example](docs/painting_grayscale.png)

## Troubleshooting

### Camera Issues

```bash
# Error: "Could not open camera 0"
# Try different camera index:
pdasc play camera -c 1

# Linux: Check available cameras
ls /dev/video*
```

### Video Playback

**Frames dropping or stuttering:**

- Increase block size: `-b 16`
- Reduce ASCII density: `-n 32`
- Encode to `.asc` format first (don't convert in real-time)

**Audio out of sync:**

- Always encode to `.asc` format for perfect synchronization

### Terminal Display

**Colors not displaying:**

- Verify 24-bit color support: `printf "\x1b[38;2;255;100;0mTRUECOLOR\x1b[0m\n"`
- Try different terminal (Windows Terminal, iTerm2, Alacritty)

**Display cut off:**

- Maximize terminal window
- Use smaller block sizes for more content in limited space

### FFmpeg

**FFmpeg not found:**

- Install FFmpeg and ensure it's in PATH
- Verify: `ffmpeg -version`

**Out of memory during encoding:**

- Use larger block sizes (`-b 16` or `-b 32`)
- Close other applications

### Block Size Errors

```bash
# Error: Block size must be a factor of image dimensions
# For 1920×1080: valid sizes include 2, 4, 5, 8, 10, 12, 15, 16, 20, 24
pdasc play image.png -b 8  # Works for most images
```

## License

MIT License - Copyright (c) 2026 Colin Politi (ColinThePanda)

See [LICENSE](LICENSE) for full text.

## Links

- **PyPI**: https://pypi.org/project/PDASC/
- **GitHub**: https://github.com/ColinThePanda/PDASC
- **Issues**: https://github.com/ColinThePanda/PDASC/issues

## Credits

GPU-accelerated ASCII video shader based on ["I Tried Turning Games Into Text"](https://www.youtube.com/watch?v=gg40RWiaHRY) by Acerola ([Garrett Gunnell](https://github.com/GarrettGunnell)).

---

<div align="center">

### Made for Intersession 2026

_Transform your terminal into a multimedia canvas_

</div>
