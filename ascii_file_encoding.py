import struct
import numpy as np
from typing import List, Tuple, Optional
import os
from ascii_converter import AsciiConverter
from utils import format_file_size

class AsciiEncoder:
    """Encoder for .asc (ASCII Container) file format"""
    
    MAGIC = b'ASCI'
    VERSION = 1
    
    # Flag bits
    FLAG_HAS_COLOR = 1 << 0
    FLAG_IS_VIDEO = 1 << 1
    FLAG_HAS_AUDIO = 1 << 2
    
    def __init__(self):
        self.frames = []
        self.charmap = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.chunk_size = 8
        self.has_audio = False
        self.audio_data = None
        self.audio_rate = 44100
        self.audio_channels = 2
        self.converter = None
    
    def add_frame(self, char_indices: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Add a frame to the encoder.
        
        Args:
            char_indices: np.ndarray of shape (H, W) with dtype uint8 (indices into charmap)
            colors: Optional np.ndarray of shape (H, W) with dtype uint32 (packed RGB as 0xRRGGBB)
        """
        if not self.frames:
            self.height, self.width = char_indices.shape
        
        self.frames.append((char_indices, colors))
    
    def set_metadata(self, charmap: str, fps: float = 30.0, chunk_size: int = 8):
        """Set encoding metadata"""
        self.charmap = charmap
        self.fps = fps
        self.chunk_size = chunk_size
    
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
        if not self.charmap:
            raise ValueError("Charmap not set")
        
        with open(output_path, 'wb') as f:
            flags = 0
            has_color = self.frames[0][1] is not None
            is_video = len(self.frames) > 1
            
            if has_color:
                flags |= self.FLAG_HAS_COLOR
            if is_video:
                flags |= self.FLAG_IS_VIDEO
            if self.has_audio:
                flags |= self.FLAG_HAS_AUDIO
            
            # Write header (30 bytes)
            header = struct.pack(
                '!4sHHHHfIBB8s',
                self.MAGIC,           # Magic number (4 bytes)
                self.VERSION,         # Version (2 bytes)
                flags,                # Flags (2 bytes)
                self.width,           # Width (2 bytes)
                self.height,          # Height (2 bytes)
                self.fps,             # FPS * (4 bytes)
                len(self.frames),     # Frame count (4 bytes)
                len(self.charmap),    # Num ASCII chars (1 byte)
                self.chunk_size,      # Chunk size (1 byte)
                b'\x00' * 8           # Reserved (8 bytes)
            )
            f.write(header)
            
            charmap_bytes = self.charmap.encode('utf-8')
            f.write(struct.pack('!H', len(charmap_bytes)))
            f.write(charmap_bytes)
            
            for char_indices, colors in self.frames:
                char_data = char_indices.astype(np.uint8).tobytes()
                
                if has_color and colors is not None:
                    r = ((colors >> 16) & 0xFF).astype(np.uint8)
                    g = ((colors >> 8) & 0xFF).astype(np.uint8)
                    b = (colors & 0xFF).astype(np.uint8)
                    
                    # Stack and flatten to get RGB bytes in order
                    color_data = np.stack([r, g, b], axis=-1).tobytes()
                    frame_data = char_data + color_data
                else:
                    frame_data = char_data
                
                f.write(struct.pack('!I', len(frame_data)))
                f.write(frame_data)
            
            if self.has_audio and self.audio_data:
                audio_header = struct.pack(
                    '!IBI',
                    len(self.audio_data),      # Audio data size
                    1,                         # Audio format (1 = PCM16)
                    self.audio_rate,           # Sample rate
                )
                f.write(audio_header)
                f.write(struct.pack('!B', self.audio_channels))
                f.write(self.audio_data)
    
    def encode_image_to_asc(self, image_path: str, output_path: str, color: bool = True, converter: AsciiConverter | None = None):
        """
        Encode a single image to .asc format
        
        Args:
            image_path: Path to input image file
            output_path: Path to output .asc file
            color: Whether to encode color information
        """
        from PIL import Image
        
        if not converter:
            if not self.converter:
                self.converter = AsciiConverter()
                converter = self.converter
            else:
                converter = self.converter
        
        # Load and convert image
        image = Image.open(image_path)
        ascii_array = converter.get_ascii(image, color)
        
        # Extract char indices
        char_indices = np.array([
            converter.char_map.index(c) 
            for row in ascii_array['char'] 
            for c in row
        ], dtype=np.uint8)
        char_indices = char_indices.reshape(ascii_array.shape)
        
        # Extract colors if present
        colors = ascii_array['color'] if color else None
        
        self.set_metadata(
            charmap=converter.char_map,
            fps=1.0,  # irrelevant for images
            chunk_size=converter.chunk_size
        )
        self.add_frame(char_indices, colors)
        self.write(output_path)
        
        print(f"Encoded image to {output_path}")
        print(f"File size: {format_file_size(os.path.getsize(output_path))}")

    def encode_video_to_asc(self, video_path: str, output_path: str, 
                        play_audio: bool = True, color: bool = True, converter: AsciiConverter | None = None):
        """
        Encode a video to .asc format
        
        Args:
            video_path: Path to input video file
            output_path: Path to output .asc file
            play_audio: Whether to include audio in encoding
            color: Whether to encode color information
        """
        from video_extracter import extract_video
        import threading
        
        if not converter:
            if not self.converter:
                self.converter = AsciiConverter()
                converter = self.converter
            else:
                converter = self.converter
        
        print(f"Encoding {video_path}...")
        
        # Extract video
        fps, frame_gen, audio_gen = extract_video(video_path)
        
        self.set_metadata(
            charmap=converter.char_map,
            fps=fps,
            chunk_size=converter.chunk_size
        )
        
        audio_chunks = []
        audio_thread = None
        
        if play_audio and audio_gen is not None:
            def collect_audio():
                for chunk in audio_gen:
                    audio_chunks.append(chunk.tobytes())
            
            audio_thread = threading.Thread(target=collect_audio, daemon=True)
            audio_thread.start()
        
        frame_count = 0
        try:
            for frame in frame_gen:
                ascii_array = converter.get_ascii(frame, color)
                
                char_indices = np.array([
                    converter.char_map.index(c) 
                    for row in ascii_array['char'] 
                    for c in row
                ], dtype=np.uint8)
                char_indices = char_indices.reshape(ascii_array.shape)
                
                colors = ascii_array['color'] if color else None
                
                self.add_frame(char_indices, colors)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Encoded {frame_count} frames...", end='\r')
            
            print(f"\nEncoded {frame_count} frames total")
            
            if audio_thread:
                audio_thread.join(timeout=5.0)
                
                if audio_chunks:
                    audio_data = b''.join(audio_chunks)
                    self.set_audio(audio_data)
                    print(f"Added {len(audio_data):,} bytes of audio")
            
            self.write(output_path)
            
            file_size = os.path.getsize(output_path)
            print(f"Saved to {output_path}")
            print(f"File size: {format_file_size(file_size)}")
            
        except Exception as e:
            print(f"\nError during encoding: {e}")
            raise


class AsciiDecoder:
    """Decoder for .asc (ASCII Container) file format"""
    
    def __init__(self):
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.charmap = ""
        self.chunk_size = 8
        self.has_color = False
        self.has_audio = False
        self.is_video = False
        self.frames = []
        self.audio_data = None
        self.audio_rate = 44100
        self.audio_channels = 2
    
    def read(self, input_path: str):
        """Read and decode .asc file"""
        with open(input_path, 'rb') as f:
            # Read header (30 bytes)
            header_data = f.read(30)
            if len(header_data) < 30:
                raise ValueError("Invalid file: header too short")
            
            (magic, version, flags, width, height, fps,
             frame_count, num_ascii, chunk_size, reserved) = struct.unpack(
                '!4sHHHHfIBB8s', header_data
            )
            
            if magic != b'ASCI':
                raise ValueError(f"Invalid file format: expected 'ASCI', got {magic}")
            
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")
            
            # Parse flags using bitmask with bitwise and for bool
            self.has_color = bool(flags & AsciiEncoder.FLAG_HAS_COLOR)
            self.is_video = bool(flags & AsciiEncoder.FLAG_IS_VIDEO)
            self.has_audio = bool(flags & AsciiEncoder.FLAG_HAS_AUDIO)
            
            self.width = width
            self.height = height
            self.fps = fps
            self.chunk_size = chunk_size
            
            charmap_len = struct.unpack('!H', f.read(2))[0]
            self.charmap = f.read(charmap_len).decode('utf-8')
            
            self.frames = []
            for _ in range(frame_count):
                frame_size = struct.unpack('!I', f.read(4))[0]
                frame_data = f.read(frame_size)
                
                char_size = width * height
                char_bytes = frame_data[:char_size]
                char_indices = np.frombuffer(char_bytes, dtype=np.uint8).reshape(height, width)
                
                # Read 24 bit colors
                if self.has_color:
                    color_bytes = frame_data[char_size:]
                    expected_color_size = char_size * 3  # 3 bytes per pixel
                    
                    if len(color_bytes) != expected_color_size:
                        raise ValueError(f"Invalid color data size: expected {expected_color_size}, got {len(color_bytes)}")
                    
                    color_data = np.frombuffer(color_bytes, dtype=np.uint8).reshape(height, width, 3)
                    r = color_data[:, :, 0].astype(np.uint32)
                    g = color_data[:, :, 1].astype(np.uint32)
                    b = color_data[:, :, 2].astype(np.uint32)
                    
                    colors = (r << 16) | (g << 8) | b
                else:
                    colors = None
                
                self.frames.append((char_indices, colors))
            
            if self.has_audio:
                audio_size, audio_format, self.audio_rate = struct.unpack('!IBI', f.read(9))
                self.audio_channels = struct.unpack('!B', f.read(1))[0]
                
                if audio_format != 1:
                    raise ValueError(f"Unsupported audio format: {audio_format}")
                
                self.audio_data = f.read(audio_size)
    
    def get_frame(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get a specific frame by index"""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range")
        return self.frames[index]
    
    def get_all_frames(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get all frames"""
        return self.frames
    
    def to_ascii_array(self, frame_index: int) -> np.ndarray:
        """
        Convert a frame to the structured ASCII array format.
        Returns: np.ndarray with dtype [('char','<U1'),('color',np.uint32)]
        """
        char_indices, colors = self.get_frame(frame_index)
        
        char_map_arr = np.array(list(self.charmap), dtype='<U1')
        chars = char_map_arr[char_indices]
        
        dtype = np.dtype([('char', '<U1'), ('color', np.uint32)])
        out = np.empty(chars.shape, dtype=dtype)
        out['char'] = chars
        
        if colors is not None:
            out['color'] = colors
        else:
            out['color'] = np.zeros_like(char_indices, dtype=np.uint32)
        
        return out