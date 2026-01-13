from PIL import Image
import moderngl
import numpy as np
from .generate_color_ramp import generate_color_ramp, get_charmap, render_charmap
from .video_extractor import extract_video
import time
import subprocess
import os

class VideoAsciiConverter:
    def __init__(self, fragment_shader_src: str, ascii_img: Image.Image, colored: bool = True):
        self.colored = colored
        
        self.ctx = moderngl.create_standalone_context()

        vertices = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            dtype="f4",
        )

        vertex_shader = """
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main() {
                uv = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """

        self.prog = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader_src
        )
        
        self.prog["colored"] = self.colored
        
        width, height = ascii_img.size
        ascii_img_data = ascii_img.tobytes()
        self.ascii_tex = self.ctx.texture((width, height), 1, ascii_img_data)
        self.ascii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ascii_tex.use(1)
        self.prog["ascii_map"] = 1
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_vert")
        
        # Pre-allocate reusable resources
        self.current_texture = None
        self.current_framebuffer = None
        self.current_size = None

    def _ensure_resources(self, width, height):
        """Reuse texture and framebuffer if size matches"""
        if self.current_size != (width, height):
            # Release old resources
            if self.current_texture:
                self.current_texture.release()
            if self.current_framebuffer:
                self.current_framebuffer.release()
            
            # Create new ones
            self.current_texture = self.ctx.texture((width, height), 4)
            self.current_framebuffer = self.ctx.simple_framebuffer((width, height))
            self.current_size = (width, height)

    def process_frame(self, image: Image.Image):
        try:
            # Avoid conversion if already RGBA
            if image.mode != "RGBA":
                img = image.convert("RGBA")
            else:
                img = image
                
            input_width, input_height = img.size
            
            # Reuse texture and framebuffer
            self._ensure_resources(input_width, input_height)
            
            if not self.current_texture or not self.current_framebuffer:
                raise ValueError("Failed to create texture and framebuffer")
            
            # Write directly to existing texture instead of creating new one
            self.current_texture.write(img.tobytes())
            self.current_texture.use(0)
            self.prog["tex"] = 0

            self.current_framebuffer.use()
            self.vao.render(moderngl.TRIANGLE_STRIP)

            # Read once into bytes
            data = self.current_framebuffer.read(components=4)
            result_img = Image.frombytes("RGBA", (input_width, input_height), data)

            return result_img

        except Exception as e:
            print(f"\nError processing image:")
            print(e)
            return None
    
    def cleanup(self):
        """Call this when done processing all frames"""
        if self.current_texture:
            self.current_texture.release()
        if self.current_framebuffer:
            self.current_framebuffer.release()


def feed_audio(audio_gen, ffmpeg_process):
    """Thread function to feed audio to ffmpeg"""
    try:
        if ffmpeg_process.stdin and audio_gen:
            for audio_chunk in audio_gen:
                ffmpeg_process.stdin.write(audio_chunk.tobytes())
    except (OSError, BrokenPipeError):
        pass

def process_video(converter: VideoAsciiConverter, video_path: str, output_path: str = "", audio: bool = True) -> str:
    fps, frame_gen, audio_gen = extract_video(video_path)

    if output_path == "":
        parts = os.path.splitext(os.path.basename(video_path))
        output_path = os.path.join("output", f"ascii_{parts[0]}.asc{parts[-1]}")
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))

    # Build ffmpeg command based on audio parameter
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        # Video input from stdin
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1280x720',  # Will be updated after first frame
        '-pix_fmt', 'rgba',
        '-r', str(fps),
        '-i', '-',
    ]
    
    if audio:
        # Audio input from the original file (force MP4 format)
        ffmpeg_cmd.extend([
            '-f', 'mp4',
            '-i', video_path,
            # Map video from stdin, audio from file
            '-map', '0:v',
            '-map', '1:a?',
        ])
    
    # Video and audio encoding settings
    ffmpeg_cmd.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
    ])
    
    if audio:
        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
        ])
    
    ffmpeg_cmd.extend([
        '-f', 'mp4',  # Force MP4 output format
        output_path
    ])

    ffmpeg_process = None
    start = time.time()
    frame_count = 0

    try:
        for frame in frame_gen:
            out_frame = converter.process_frame(frame)
            
            if out_frame is None:
                continue
            
            # Initialize ffmpeg process after first frame to get correct dimensions
            if ffmpeg_process is None:
                width, height = out_frame.size
                ffmpeg_cmd[7] = f'{width}x{height}'
                ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Write frame to ffmpeg stdin
            try:
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.write(out_frame.tobytes())
            except (OSError, BrokenPipeError) as e:
                print(f"\nError writing to ffmpeg: {e}")
                break
                
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...", end='\r')

    finally:
        # Close ffmpeg stdin and wait for process to finish
        if ffmpeg_process:
            if ffmpeg_process.stdin:
                try:
                    ffmpeg_process.stdin.close()
                except:
                    pass
            ffmpeg_process.wait()
        
        converter.cleanup()
        elapsed = time.time() - start
        print(f"\nCompleted: {elapsed:.2f}s for {frame_count} frames ({frame_count/elapsed:.1f} fps)")
        print(f"Output saved to: {output_path}")
        return output_path