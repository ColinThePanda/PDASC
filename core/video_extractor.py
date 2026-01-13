import subprocess
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional

def extract_video(video_path: str, audio_rate=44100) -> Tuple[float, Generator[Image.Image, None, None], Optional[Generator[np.ndarray, None, None]]]:
    """
    Extracts video frames and audio from a video file as generators.

    Args:
        video_path: Path to video
        audio_rate: Sample rate for audio (default 44100 Hz)

    Returns:
        fps: float
        frame_gen: generator yielding PIL.Image frames
        audio_gen: generator yielding np.ndarray audio chunks (shape: [N, 2]) or None if no audio
    """

    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"{video_path} does not exist")

    # Probe video resolution and FPS
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(video_file)
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    info: Dict[str, Any] = json.loads(probe.stdout or "{}")
    streams = info.get('streams', [])
    if not streams:
        raise RuntimeError("No video streams found or FFprobe failed")

    stream: Dict[str, Any] = streams[0]
    orig_width = int(stream['width'])
    orig_height = int(stream['height'])

    # Parse FPS
    num, den = map(int, stream.get('r_frame_rate', '30/1').split('/'))
    fps = round(num / den)

    # Downscale to 720p if necessary
    if orig_height > 720:
        height = 720
        width = int(orig_width * (720 / orig_height))
        # Make width even for compatibility
        width = width - (width % 2)
        scale_filter = f"scale={width}:{height}"
    else:
        width = orig_width
        height = orig_height
        scale_filter = None

    # Video command raw RGB24
    video_cmd = [
        "ffmpeg",
        "-i", str(video_file),
    ]
    
    if scale_filter:
        video_cmd.extend(["-vf", scale_filter])
    
    video_cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an",  # No audio
        "-"
    ])
    
    video_proc = subprocess.Popen(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Audio: PCM16 stereo
    audio_cmd = [
        "ffmpeg",
        "-i", str(video_file),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "2",
        "-ar", str(audio_rate),
        "-vn",  # No video
        "-"
    ]
    audio_proc = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = width * height * 3
    audio_chunk_size = audio_rate * 2 * 2  # 1 second buffer: 2 bytes/sample, 2 channels

    def frame_generator():
        try:
            if video_proc.stdout is not None:
                while True:
                    raw = video_proc.stdout.read(frame_size)
                    if not raw:
                        break
                    if len(raw) < frame_size:
                        continue
                    img_np = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    yield Image.fromarray(img_np, "RGB")
        finally:
            if video_proc.stdout is not None:
                video_proc.stdout.close()
            video_proc.wait()

    def audio_generator():
        try:
            if audio_proc.stdout is not None:
                while True:
                    raw = audio_proc.stdout.read(audio_chunk_size)
                    if not raw:
                        break
                    valid_size = (len(raw) // 4) * 4
                    if valid_size == 0:
                        continue
                    audio_np = np.frombuffer(raw[:valid_size], dtype=np.int16).reshape(-1, 2)
                    yield audio_np
        finally:
            if audio_proc.stdout is not None:
                audio_proc.stdout.close()
            audio_proc.wait()

    return fps, frame_generator(), audio_generator()