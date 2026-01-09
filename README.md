# AsciiImageCLI

A high-performance terminal-based ASCII art converter that transforms images, videos, and live camera feeds into colored ASCII art. Features hardware-accelerated processing with Numba, custom font-based character mapping, and a proprietary `.asc` file format with Zstandard compression for instant playback.

![Demo Image](docs/demo_image.png)
*Example: High-resolution image converted to colored ASCII art*

![Demo Video](docs/demo_video.gif)
*Example: Video playback with synchronized audio*

## Features

- **Multiple Input Sources**
  - Static images (PNG, JPG, GIF, BMP, TIFF, WebP)
  - Video files (MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V)
  - Live camera feed with real-time processing
  - Pre-encoded `.asc` files for instant playback

- **Advanced Processing**
  - Hardware-accelerated conversion using Numba JIT compilation
  - Customizable character sets generated from any TrueType font
  - Configurable ASCII density (number of characters used)
  - Adjustable block size for quality vs. performance tuning
  - Full RGB color support with terminal ANSI 24-bit color codes

- **Audio Support**
  - Synchronized audio playback for videos
  - Real-time audio streaming during live camera mode
  - PCM16 audio encoding in `.asc` format

- **Instant Playback**
  - Custom `.asc` (ASCII Container) binary format with Zstandard compression
  - Pre-rendered ANSI escape sequences stored directly
  - Minimal processing overhead during playback (decompression only)
  - Frame-accurate synchronization with audio

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Play Command](#play-command)
  - [Encode Command](#encode-command)
  - [Command Options](#command-options)
- [The .asc File Format](#the-asc-file-format)
- [How It Works](#how-it-works)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- A modern terminal with 24-bit color support (most modern terminals)
- TrueType font file (default: `CascadiaMono.ttf`)
- FFmpeg (for video processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Fast numerical operations
- `Pillow` - Image processing
- `numba` - JIT compilation for performance
- `sounddevice` - Audio playback
- `opencv-python` - Camera and video processing
- `zstandard` - Fast compression/decompression

### Font Setup

The project requires a TrueType font to generate the ASCII character mapping. Place your font file (e.g., `CascadiaMono.ttf`) in the project root directory, or specify a custom path using the `--font` option.

**Recommended fonts:**
- Cascadia Mono
- JetBrains Mono
- Fira Code
- Any monospace TrueType font

## Quick Start

```bash
# Display an image
python main.py play image.jpg

# Play a video with audio
python main.py play video.mp4

# Use your webcam
python main.py play camera

# Encode a video to .asc format for instant playback
python main.py encode video.mp4 -o output.asc

# Play the encoded file (blazing fast!)
python main.py play output.asc
```

## Usage

The CLI has two main commands: `play` (for displaying/playing) and `encode` (for creating `.asc` files).

### Play Command

Display images, play videos, stream from camera, or play `.asc` files.

```bash
python main.py play <input> [options]
```

**Input types:**
- Path to an image file
- Path to a video file
- Path to a `.asc` file
- `camera` for webcam input

**Examples:**

```bash
# Display an image
python main.py play photo.png

# Play a video without audio
python main.py play movie.mp4 --no-audio

# Use webcam with higher ASCII density
python main.py play camera -n 70

# Play with larger block size (lower resolution, better performance)
python main.py play video.mp4 -b 16

# Play with custom font
python main.py play image.jpg -f "JetBrainsMono.ttf"

# Use a different camera (default is 0)
python main.py play camera -c 1
```

### Encode Command

Convert images or videos to the `.asc` format for instant playback with minimal processing overhead.

```bash
python main.py encode <input> -o <output.asc> [options]
```

**Examples:**

```bash
# Encode a video with default settings
python main.py encode movie.mp4 -o movie.asc

# Encode an image without color
python main.py encode photo.jpg -o photo.asc --no-color

# Encode with custom ASCII density
python main.py encode video.mp4 -o video.asc -n 70

# Encode with larger blocks for better performance
python main.py encode large_video.mp4 -o output.asc -b 16
```

### Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--block-size` | `-b` | `8` | Size of character blocks (pixels per character). Higher = lower resolution but better performance |
| `--num-ascii` | `-n` | `8` | Number of ASCII characters to use (2-95). Higher = more detail in grayscale gradients |
| `--font` | `-f` | `CascadiaMono.ttf` | Path to TrueType font file for character generation |
| `--no-color` | - | `False` | Disable color output (grayscale only) |
| `--no-audio` | - | `False` | Disable audio playback for videos |
| `--camera` | `-c` | `0` | Camera index when using camera input |
| `--output` | `-o` | `ascii_out.mp4` | Output file path (encode command only) |

## The .asc File Format

The `.asc` (ASCII Container) format is a custom binary format designed for instant playback of ASCII art animations. It stores pre-rendered ANSI escape sequences compressed with Zstandard, enabling near-instant decompression and zero conversion overhead during playback.

### Design Philosophy

The `.asc` format prioritizes **instant playback** over storage efficiency. By pre-rendering all frames as complete ANSI strings during encoding and compressing them with Zstandard (level 5), playback requires only decompression—no conversion processing. The format balances reasonable file sizes with minimal CPU overhead during playback, making it ideal for smooth, frame-accurate video playback in terminal environments.

### Format Specification

#### Header (24 bytes)

| Offset | Size | Type   | Description                                     |
|--------|------|--------|-------------------------------------------------|
| 0x00   | 4    | char   | Magic number: "ASII" (ASCII with compression)   |
| 0x04   | 2    | uint16 | Version number (currently 2)                    |
| 0x06   | 2    | uint16 | Flags (bit field)                               |
| 0x08   | 4    | float  | FPS (frames per second)                         |
| 0x0C   | 4    | uint32 | Frame count                                     |
| 0x10   | 8    | -      | Reserved for future use                         |

#### Flags (bit field)

```
Bit 0: IS_VIDEO  (0x01) - Multiple frames (video)
Bit 1: HAS_AUDIO (0x02) - Audio data is present
Bits 2-15: Reserved
```

#### Frame Index Section

Following the header, frame lengths are stored for quick random access:

| Offset   | Size     | Type   | Description                              |
|----------|----------|--------|------------------------------------------|
| variable | 4        | uint32 | Frame 1 uncompressed length (bytes)      |
| +4       | 4        | uint32 | Frame 2 uncompressed length (bytes)      |
| ...      | ...      | ...    | ... (one entry per frame)                |

#### Compressed Frame Data

After the frame index:

| Offset   | Size     | Type   | Description                              |
|----------|----------|--------|------------------------------------------|
| variable | 4        | uint32 | Compressed data size (bytes)             |
| +4       | variable | bytes  | Zstandard-compressed frame data          |

The compressed data contains all frames concatenated together. Each frame string contains:
- ANSI 24-bit color escape sequences (`\033[38;2;R;G;Bm`)
- ASCII characters (typically doubled for better aspect ratio)
- Newline characters for row separation
- ANSI reset sequences

**Example frame string structure (before compression):**
```
\033[38;2;45;67;89m##\033[38;2;50;70;92m##...\n\033[38;2;42;65;88m##...
```

#### Audio Section (optional)

If `HAS_AUDIO` flag is set, audio data follows the compressed frames:

| Offset   | Size     | Type   | Description                    |
|----------|----------|--------|--------------------------------|
| variable | 4        | uint32 | Audio data size in bytes       |
| +4       | 1        | uint8  | Audio format (1 = PCM16)       |
| +5       | 4        | uint32 | Sample rate (Hz)               |
| +9       | 1        | uint8  | Number of channels             |
| +10      | variable | bytes  | Raw PCM16 audio data           |

### Compression Benefits

Zstandard compression (level 5) provides excellent compression ratios for ANSI strings due to:
- Repetitive color escape sequences
- Similar character patterns across frames
- Predictable structure of ANSI codes

**Typical compression ratios:**
- Text-heavy content: 10:1 to 20:1
- Color-rich video: 5:1 to 10:1
- Grayscale content: 15:1 to 25:1

### File Size Characteristics

With Zstandard compression, `.asc` files are now **comparable or smaller** than equivalent video files:

#### Example: 1920×1080 video, 8×8 blocks, 30 FPS, 10 seconds

- Character grid: 240×135 = 32,400 positions
- Uncompressed per frame: ~500 KB - 2 MB
- Compressed per frame: ~50 KB - 200 KB (typical 10:1 ratio)
- 300 frames compressed: **15 MB - 60 MB** (without audio)
- Original H.264 video: ~20-50 MB

**Storage ratio: 0.5-2× of equivalent video** (vs 3-10× uncompressed)

### Why Use .asc Files?

Despite requiring decompression, `.asc` files offer significant advantages:

1. **Minimal processing overhead** - Only decompression, no conversion
2. **Frame-perfect synchronization** - Eliminates conversion lag
3. **Consistent performance** - Predictable CPU usage
4. **Instant startup** - Fast initial decompression
5. **Reasonable file sizes** - Competitive with video formats
6. **Predictable playback** - Same experience on all hardware

The format is ideal for scenarios where smooth, reliable playback is more important than real-time conversion.

## How It Works

### 1. Character Mapping Generation

The system analyzes a TrueType font to create an optimal character-to-brightness mapping:

```python
# Generate luminance values for each printable ASCII character
color_ramp = generate_color_ramp(font_path="CascadiaMono.ttf")

# Select N characters with the most uniform brightness distribution
char_map = get_charmap(color_ramp, levels=8)
# Result: " .:-=+*#%@" (from darkest to brightest)
```

Each character is rendered at 48×48 pixels, and its average luminance is computed. Characters are then sorted and quantized to create a uniform grayscale ramp.

### 2. Image Processing Pipeline

```
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
Assemble complete frame string
    ↓
Compress with Zstandard (encode) or Render to terminal (play)
```

### 3. Numba Acceleration

The core processing loop is JIT-compiled with Numba for near-C performance:

```python
@njit(parallel=True, fastmath=True, cache=True)
def compute_blocks(img, cs, gray_levels, color):
    # Parallel processing of image blocks
    # Significantly faster than pure Python
    ...
```

### 4. Terminal Rendering

Output uses ANSI escape sequences for:
- 24-bit true color: `\033[38;2;{r};{g};{b}m`
- Cursor positioning: `\033[H` (home position)
- Alternate screen buffer: `\033[?1049h` (enter) / `\033[?1049l` (exit)
- Cursor visibility: `\033[?25l` (hide) / `\033[?25h` (show)

### 5. Encoding Process

When encoding to `.asc` format:

1. Process each video frame through the conversion pipeline
2. Render each frame to a complete ANSI string
3. Compress all frames together using Zstandard level 5
4. Write frame length index for random access
5. Write compressed frame data
6. Optionally extract and append PCM16 audio data
7. Store metadata (FPS, frame count, flags) in header

### 6. Playback Process

When playing `.asc` files:

1. Read header to determine FPS and frame count
2. Decompress all frames using Zstandard (fast one-time operation)
3. Write strings directly to terminal at target FPS
4. Stream audio in parallel if present
5. No conversion or re-processing required

## Performance Optimization

### Block Size vs. Quality

| Block Size | Resolution | Performance      | Use Case                         |
|------------|------------|------------------|----------------------------------|
| 4×4        | Very High  | Good             | High-quality images and videos   |
| 8×8        | High       | Better           | Default, recommended             |
| 16×16      | Medium     | Best             | Lower-end hardware               |
| 32×32      | Low        | Fastest          | Very limited hardware            |

**Note:** Block size affects encoding time and output quality. Playback performance depends mainly on decompression speed.

### Number of ASCII Characters

| Num ASCII | Detail | Character Set Size | Use Case                    |
|-----------|--------|--------------------|-----------------------------|
| 8-16      | Low    | Small              | Artistic effect, retro look |
| 32-64     | Medium | Medium             | Good balance                |
| 70-95     | High   | Large              | Maximum detail preservation |

### Best Practices

1. **For webcam/real-time**: Use `-b 8 -n 32` for smooth live processing
2. **For high-quality images**: Use `-b 4 -n 70` for maximum detail
3. **For video playback**: Always encode to `.asc` first for best performance
4. **For storage constraints**: Use larger block sizes during encoding (compression helps significantly)
5. **For terminal size constraints**: Match block size to terminal dimensions

## Examples

### Example 1: Portrait Photo

```bash
python main.py play portrait.png -n 70 -b 8
```

![Portrait Example](docs/example_portrait.png)
*High-quality portrait with 70 characters*

### Example 2: Landscape Video

```bash
# First encode for instant playback
python main.py encode landscape.mp4 -o landscape.asc -n 50 -b 8

# Then play with minimal processing overhead
python main.py play landscape.asc
```

![Landscape Example](docs/example_landscape.gif)
*Landscape video with synchronized audio*

### Example 3: Webcam Streaming

```bash
python main.py play camera -b 12 -n 40
```

![Webcam Example](docs/example_webcam.gif)
*Real-time webcam feed at 30 FPS*

### Example 4: Grayscale Art

```bash
python main.py play artwork.png --no-color -n 95
```

![Grayscale Example](docs/example_grayscale.png)
*Black and white image with maximum character variety*

## Troubleshooting

### Camera Not Working

```bash
# Error: "Could not open camera 0"
# Solution: Try different camera index
python main.py play camera -c 1

# On Linux, check available cameras:
ls /dev/video*
```

### Video Playback Issues

**Frames dropping or stuttering:**
- Increase block size during encoding: `-b 16`
- Reduce ASCII density: `-n 32`
- Ensure you're playing from `.asc` file, not converting in real-time

**Audio out of sync:**
- This typically occurs when processing frames in real-time
- Always encode to `.asc` format first for perfect synchronization

### Terminal Issues

**Colors not displaying:**
- Ensure your terminal supports 24-bit color
- Test with: `printf "\x1b[38;2;255;100;0mTRUECOLOR\x1b[0m\n"`
- Try different terminal emulators (Windows Terminal, iTerm2, Alacritty)

**Display cut off:**
- The program automatically scales to fit terminal size
- Maximize your terminal window for best results
- Use smaller block sizes for more content in limited space

### Encoding Errors

**FFmpeg not found:**
- Install FFmpeg: 
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from ffmpeg.org

**Out of memory during encoding:**
- Process videos in smaller chunks
- Use larger block sizes to reduce frame string size
- Close other applications to free RAM

**Compression takes too long:**
- This is normal for long videos with high detail
- Consider using larger block sizes (`-b 16` or `-b 32`)
- Compression is a one-time cost for much faster playback

### Font Errors

```bash
# Error: "Font file 'CascadiaMono.ttf' not found"
# Solution: Specify full path to font
python main.py play image.png -f "/usr/share/fonts/truetype/cascadia/CascadiaMono.ttf"

# Or place font in project directory
```

## Technical Details

### Dependencies

- **NumPy**: Multi-dimensional array operations and efficient data structures
- **Pillow (PIL)**: Image loading, manipulation, and resizing
- **Numba**: Just-In-Time compilation for performance-critical loops
- **sounddevice**: Cross-platform audio I/O
- **OpenCV (cv2)**: Video decoding and camera capture
- **FFmpeg**: Video/audio extraction and processing (external dependency)
- **zstandard**: Fast compression/decompression library

### System Requirements

- **CPU**: Any modern multi-core processor (Numba utilizes all cores)
- **RAM**: 4GB minimum, 8GB+ recommended for HD video encoding
- **Storage**: Reasonable space for `.asc` files (typically 0.5-2× source video size)
- **Terminal**: Any terminal emulator with 24-bit true color support

### Platform Support

- **Linux**: Full support (tested on Ubuntu 20.04+)
- **macOS**: Full support (tested on macOS 11+)
- **Windows**: Supported with Windows Terminal or WSL

## Project Structure

```
AsciiImageCLI/
├── main.py                   # CLI entry point and argument parsing
├── ascii_converter.py        # Core conversion engine with Numba
├── ascii_displayer.py        # Terminal rendering and playback
├── ascii_file_encoding.py    # .asc format encoder/decoder with Zstandard
├── audio_player.py           # Audio playback handling
├── generate_color_ramp.py    # Font analysis and charmap generation
├── utils.py                  # Helper functions (color packing, etc.)
├── video_extracter.py        # FFmpeg-based video/audio extraction
├── requirements.txt          # Python dependencies
├── CascadiaMono.ttf          # Default monospace font
└── README.md                 # This file
```

## License

MIT License

Copyright (c) 2026 ColinThePanda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
