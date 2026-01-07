from PIL import Image
from ascii_converter import AsciiConverter
from utils import unpack_int24_array
import numpy as np
import time
import sys
from audio_player import AudioPlayer

class AsciiDisplayer:
    def __init__(self, converter: AsciiConverter):
        self.converter: AsciiConverter = converter
    
    def color_text(self, text: str, r: int, g: int, b: int):
        r = max(min(r, 255), 0)
        g = max(min(g, 255), 0)
        b = max(min(b, 255), 0)
        return f"\033[38;2;{r};{g};{b}m{text}"

    def render_ascii(self, ascii_array: np.ndarray):
        """
        Render a structured ASCII array to a colored string for terminal.
        
        ascii_array: np.ndarray with dtype [('char','<U1'),('color',np.uint32)]
        
        Returns: str with ANSI color codes
        """

        lines = []
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
        return "\n".join(lines) + "\033[0m"
    
    def display_image(self, image: Image.Image):
        ascii = self.converter.get_ascii(image)
        frame = self.render_ascii(ascii)
        sys.stdout.write(f"\033[H{frame}")
        sys.stdout.flush()

    def display_video(self, video_path: str, play_audio: bool = True):
        from video_extracter import extract_video
        import signal
        import sys

        # Setup terminal
        print("\033[?1049h\033[?25l\033[H\033[2J", end="") # seperate buffer, hide cursor, move cursor home, clear screen
        sys.stdout.flush()

        def cleanup():
            print("\033[?25h\033[?1049l", end="") # restore cursor and restore buffer
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
                
                self.display_image(frame)
                
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