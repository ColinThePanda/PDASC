import struct
import os
from utils import format_file_size
from typing import List, Optional

class AsciiEncoder:
    """Encoder for .asc (ASCII Container) file format - stores pre-rendered ANSI strings"""
    
    MAGIC = b'ASII'
    VERSION = 2
    
    # Flag bits
    FLAG_IS_VIDEO = 1 << 0
    FLAG_HAS_AUDIO = 1 << 1
    
    def __init__(self):
        self.frames = []  # Store actual ANSI strings
        self.fps = 30.0
        self.has_audio = False
        self.audio_data = None
        self.audio_rate = 44100
        self.audio_channels = 2
    
    def add_rendered_frame(self, ansi_string: str):
        """
        Add a pre-rendered frame (ANSI escape sequence string)
        
        Args:
            ansi_string: Complete ANSI string ready to display
        """
        self.frames.append(ansi_string)
    
    def set_audio(self, audio_data: bytes, sample_rate: int = 44100, channels: int = 2):
        """Set audio data (raw PCM16)"""
        self.has_audio = True
        self.audio_data = audio_data
        self.audio_rate = sample_rate
        self.audio_channels = channels
    
    def write(self, output_path: str):
        """Write encoded data to file"""
        if not self.frames:
            raise ValueError("No frames added")
        
        with open(output_path, 'wb') as f:
            flags = 0
            is_video = len(self.frames) > 1
            
            if is_video:
                flags |= self.FLAG_IS_VIDEO
            if self.has_audio:
                flags |= self.FLAG_HAS_AUDIO
            
            # Write header (24 bytes)
            header = struct.pack(
                '!4sHHfI8s',
                self.MAGIC,                 # Magic number (4 bytes)
                self.VERSION,               # Version (2 bytes)
                flags,                      # Flags (2 bytes)
                self.fps,                   # FPS (4 bytes)
                len(self.frames),  # Frame count (4 bytes)
                b'\x00' * 8                 # Reserved (8 bytes)
            )
            f.write(header)
            
            # Write frames as length-prefixed strings
            for frame_str in self.frames:
                frame_bytes = frame_str.encode('utf-8')
                f.write(struct.pack('!I', len(frame_bytes)))
                f.write(frame_bytes)
            
            # Write audio if present
            if self.has_audio and self.audio_data:
                audio_header = struct.pack(
                    '!IBI',
                    len(self.audio_data),  # Audio data size
                    1,                     # Audio format (1 = PCM16)
                    self.audio_rate,       # Sample rate
                )
                f.write(audio_header)
                f.write(struct.pack('!B', self.audio_channels))
                f.write(self.audio_data)
    
    def encode_image_to_asc(self, image_path: str, output_path: str, color: bool = True, converter=None, displayer=None):
        """Encode a single image to .asc format"""
        from PIL import Image
        
        if not converter or not displayer:
            from ascii_converter import AsciiConverter
            from ascii_displayer import AsciiDisplayer
            converter = converter or AsciiConverter()
            displayer = displayer or AsciiDisplayer(converter)
        
        # Load and convert image
        image = Image.open(image_path)
        ascii_array = converter.get_ascii(image, color)
        
        # Render to ANSI string
        ansi_string = displayer.render_ascii(ascii_array, color)
        
        self.fps = 1.0
        self.add_rendered_frame(ansi_string)
        self.write(output_path)
        
        print(f"Encoded image to {output_path}")
        print(f"File size: {format_file_size(os.path.getsize(output_path))}")
    
    def encode_video_to_asc(self, video_path: str, output_path: str, 
                           play_audio: bool = True, color: bool = True, 
                           converter=None, displayer=None):
        """Encode a video to .asc format"""
        from video_extracter import extract_video
        import threading
        
        if not converter:
            from ascii_converter import AsciiConverter
            converter = converter or AsciiConverter()
        if not displayer:
            from ascii_displayer import AsciiDisplayer
            displayer = displayer or AsciiDisplayer(converter)
        
        print(f"Encoding {video_path}...")
        
        # Extract video
        fps, frame_gen, audio_gen = extract_video(video_path)
        self.fps = fps
        
        # Collect audio in background
        audio_chunks = []
        audio_thread = None
        
        if play_audio and audio_gen is not None:
            def collect_audio():
                for chunk in audio_gen:
                    audio_chunks.append(chunk.tobytes())
            
            audio_thread = threading.Thread(target=collect_audio, daemon=True)
            audio_thread.start()
        
        # Encode frames
        frame_count = 0
        try:
            for frame in frame_gen:
                # Convert to ASCII and render to ANSI string
                ascii_array = converter.get_ascii(frame, color)
                ansi_string = displayer.render_ascii(ascii_array, color)
                
                self.add_rendered_frame(ansi_string)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Encoded {frame_count} frames...", end='\r')
            
            print(f"\nEncoded {frame_count} frames total")
            
            # Wait for audio
            if audio_thread:
                audio_thread.join(timeout=5.0)
                
                if audio_chunks:
                    audio_data = b''.join(audio_chunks)
                    self.set_audio(audio_data)
                    print(f"Added {format_file_size(len(audio_data))} of audio")
            
            self.write(output_path)
            
            file_size = os.path.getsize(output_path)
            print(f"Saved to {output_path}")
            print(f"File size: {format_file_size(file_size)}")
            print(f"⚠️  Large file size is expected for instant playback!")
            
        except Exception as e:
            print(f"\nError during encoding: {e}")
            raise


class AsciiDecoder:
    """Decoder for .asc (ASCII Container) file format - reads pre-rendered frames"""
    
    def __init__(self):
        self.fps = 0.0
        self.has_audio = False
        self.is_video = False
        self.frames = []  # Pre-rendered ANSI strings
        self.audio_data = None
        self.audio_rate = 44100
        self.audio_channels = 2
    
    def read(self, input_path: str):
        """Read and decode .asc file"""
        with open(input_path, 'rb') as f:
            # Read header (24 bytes)
            header_data = f.read(24)
            if len(header_data) < 24:
                raise ValueError("Invalid file: header too short")
            
            (magic, version, flags, fps, frame_count, reserved) = struct.unpack(
                '!4sHHfI8s', header_data
            )
            
            if magic != b'ASCR':
                raise ValueError(f"Invalid file format: expected 'ASCR', got {magic}")
            
            if version != 2:
                raise ValueError(f"Unsupported version: {version}")
            
            # Parse flags
            self.is_video = bool(flags & AsciiEncoder.FLAG_IS_VIDEO)
            self.has_audio = bool(flags & AsciiEncoder.FLAG_HAS_AUDIO)
            
            self.fps = fps
            
            # Read all frames
            self.frames = []
            for _ in range(frame_count):
                frame_size = struct.unpack('!I', f.read(4))[0]
                frame_bytes = f.read(frame_size)
                frame_str = frame_bytes.decode('utf-8')
                self.frames.append(frame_str)
            
            # Read audio if present
            if self.has_audio:
                audio_size, audio_format, self.audio_rate = struct.unpack('!IBI', f.read(9))
                self.audio_channels = struct.unpack('!B', f.read(1))[0]
                
                if audio_format != 1:
                    raise ValueError(f"Unsupported audio format: {audio_format}")
                
                self.audio_data = f.read(audio_size)
    
    def get_frame(self, index: int) -> str:
        """Get a pre-rendered frame by index"""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range")
        return self.frames[index]
    
    def get_all_frames(self) -> List[str]:
        """Get all pre-rendered frames"""
        return self.frames