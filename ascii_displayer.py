from PIL import Image
from ascii_converter import AsciiConverter
from utils import unpack_int24_array
import numpy as np
import time
import sys
import signal
from audio_player import AudioPlayer
from ascii_file_encoding import AsciiDecoder, AsciiEncoder

class AsciiDisplayer:
    def __init__(self, converter: AsciiConverter):
        self.converter: AsciiConverter = converter
    
    def color_text(self, text: str, r: int, g: int, b: int):
        r = max(min(r, 255), 0)
        g = max(min(g, 255), 0)
        b = max(min(b, 255), 0)
        return f"\033[38;2;{r};{g};{b}m{text}"

    def render_ascii(self, ascii_array: np.ndarray, colored: bool):
        """
        Render a structured ASCII array to a colored string for terminal.
        
        ascii_array: np.ndarray with dtype [('char','<U1'),('color',np.uint32)]
        
        Returns: str with ANSI color codes (or plain text if not colored)
        """

        lines = []
        if colored:
            for row in ascii_array:
                chars = np.char.multiply(row['char'], 2)
                r, g, b = unpack_int24_array(row['color'])
                line = ""
                last_color = None
                for ch, ri, gi, bi in zip(chars, r, g, b):
                    color = (ri, gi, bi)
                    if color != last_color:
                        line += f"\033[38;2;{ri};{gi};{bi}m"
                        last_color = color
                    line += ch
                lines.append(line)
            return "\033[1;40m" + "\n".join(lines) + "\033[0m"
        else:
            for row in ascii_array:
                chars = np.char.multiply(row['char'], 2)
                line = "".join(chars)
                lines.append(line)
            return "\n".join(lines)
    
    def render_image(self, image: Image.Image, color: bool = True):
        ascii = self.converter.get_ascii(image, color)
        frame = self.render_ascii(ascii, color)
        sys.stdout.write(f"\033[H{frame}")
        sys.stdout.flush()
    
    def display_image(self, image: Image.Image, color: bool = True):
        import shutil
        
        print("\033[?1049h\033[?25l\033[H\033[2J", end="")
        sys.stdout.flush()

        def cleanup():
            print("\033[?25h\033[?1049l", end="")
            sys.stdout.flush()
        
        def signal_handler(sig, frame):
            cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            term_cols, term_rows = shutil.get_terminal_size()
            
            img_width, img_height = image.size
            chars_wide = img_width // self.converter.chunk_size
            chars_tall = img_height // self.converter.chunk_size
            
            needed_cols = chars_wide * 2
            needed_rows = chars_tall
            
            # Scale down if image would be larger than terminal because would cut off
            if needed_cols > term_cols or needed_rows > (term_rows - 2):
                scale_factor = min(
                    term_cols / needed_cols,
                    (term_rows - 2) / needed_rows
                )
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.render_image(image, color)
            input()
        finally:
            cleanup()

    def display_video(self, video_path: str, play_audio: bool = True, color: bool = True):
        from video_extracter import extract_video

        # Setup terminal
        print("\033[?1049h\033[?25l\033[H\033[2J", end="")
        sys.stdout.flush()

        def cleanup():
            print("\033[?25h\033[?1049l", end="")
            sys.stdout.flush()
        
        def signal_handler(sig, frame):
            """"Handles CTRL-C and exiting to properly close video"""
            cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            fps, frame_gen, audio_gen = extract_video(video_path)
            
            player = None
            if play_audio and audio_gen is not None:
                player = AudioPlayer(audio_gen)
                player.start()
            
            frame_time = 1.0 / fps
            start_time = time.time()
            frame_idx = 0
            
            for frame in frame_gen:
                elapsed = time.time() - start_time
                target_frame = int(elapsed * fps)
                
                frame_idx += 1
                
                # Skip frame if behind
                if frame_idx - 1 < target_frame:
                    continue
                
                self.render_image(frame, color)
                
                # Sleep if extra time to cap fps to video frame rate
                target_time = start_time + frame_idx * frame_time
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        finally:
            if player:
                player.stream.stop()
                player.stream.close()
            cleanup()
    
    def display_asc_file(self, asc_path: str, play_audio: bool = True):
        """Display a pre-encoded .asc file - BLAZING FAST!"""
        decoder = AsciiDecoder()
        decoder.read(asc_path)
        
        # Setup terminal
        print("\033[?1049h\033[?25l\033[H\033[2J", end="")
        sys.stdout.flush()

        def cleanup():
            print("\033[?25h\033[?1049l", end="")
            sys.stdout.flush()
        
        def signal_handler(sig, frame):
            cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            player = None
            if play_audio and decoder.has_audio and decoder.audio_data:
                def audio_gen():
                    if decoder.audio_data:
                        chunk_size = decoder.audio_rate * 2 * decoder.audio_channels
                        for i in range(0, len(decoder.audio_data), chunk_size):
                            chunk = decoder.audio_data[i:i+chunk_size]
                            valid_size = (len(chunk) // 4) * 4
                            if valid_size > 0:
                                audio_np = np.frombuffer(chunk[:valid_size], dtype=np.int16).reshape(-1, decoder.audio_channels)
                                yield audio_np
                
                player = AudioPlayer(audio_gen(), samplerate=decoder.audio_rate, channels=decoder.audio_channels)
                player.start()
            
            if decoder.is_video:
                # Video playback - just write pre-rendered strings!
                frame_time = 1.0 / decoder.fps
                start_time = time.time()
                
                for frame_idx, frame_str in enumerate(decoder.frames):
                    elapsed = time.time() - start_time
                    target_frame = int(elapsed * decoder.fps)
                    
                    # Skip frame if behind
                    if frame_idx < target_frame:
                        continue
                    
                    # Just write the pre-rendered string - NO PROCESSING!
                    sys.stdout.write(f"\033[H{frame_str}")
                    sys.stdout.flush()
                    
                    # Sleep to maintain frame rate
                    target_time = start_time + (frame_idx + 1) * frame_time
                    sleep_time = target_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            else:
                # Single image - just write it!
                sys.stdout.write(f"\033[H{decoder.frames[0]}")
                sys.stdout.flush()
                input()
        
        finally:
            if player:
                player.stream.stop()
                player.stream.close()
            cleanup()
            
    def display_camera(self, camera_index: int = 0, color: bool = True):
        """Display live camera feed as ASCII art"""
        import cv2
        import os
        
        # Suppress OpenCV errors temporarily
        devnull = open(os.devnull, 'w')
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        
        try:
            # Try to open camera before setting up terminal for better errors
            cap = cv2.VideoCapture(camera_index)
            
            # read test frame
            ret, test_frame = cap.read()
        finally:
            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            devnull.close()
        
        if not ret or test_frame is None:
            cap.release()
            print(f"Error: Could not open camera {camera_index}")
            print("Make sure a camera is connected and is available")
            return
        
        # Setup terminal
        print("\033[?1049h\033[?25l\033[H\033[2J", end="")
        sys.stdout.flush()
        
        def cleanup():
            print("\033[?25h\033[?1049l", end="")
            sys.stdout.flush()
        
        def signal_handler(sig, frame):
            cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Get camera FPS (default 30)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 60:
            fps = 30
        
        frame_time = 1.0 / fps
        
        try:
            last_time = time.time()
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                image = Image.fromarray(frame_rgb)
                
                self.render_image(image, color)
                
                elapsed = time.time() - last_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
                
        finally:
            cap.release()
            cleanup()