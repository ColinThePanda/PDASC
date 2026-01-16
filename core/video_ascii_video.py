"""
ASCII Art Video Converter - Professional Integration
Implements multi-pass ASCII art rendering with edge detection for video processing.
"""

from PIL import Image
import moderngl
import numpy as np
import time
import subprocess
import os


class VideoAsciiConverter:
    """
    Real-time ASCII art video converter using multi-pass rendering pipeline.
    
    Pipeline stages:
    1. Luminance extraction
    2. Gaussian blur (horizontal + vertical) for DoG edge detection
    3. Sobel edge detection (horizontal + vertical)
    4. Color/luminance packing
    5. Downsampling chain (1/2, 1/4, 1/8)
    6. ASCII character selection based on edges and luminance
    """
    
    def __init__(self, shader_dir_path: str, ascii_img: Image.Image, edges_img: Image.Image, colored: bool = True):
        """
        Initialize the ASCII art converter.
        
        Args:
            shader_dir_path: Path to directory containing shader files
            ascii_img: Texture atlas containing ASCII characters for fill (80x8 expected)
            edges_img: Texture atlas containing edge characters (32x8 expected)
            colored: Whether to preserve color from original video (default: True)
        """
        self.colored = colored
        self.ctx = moderngl.create_context(standalone=True)
        
        # Rendering parameters
        self.num_ascii = 8
        self.gaussian_kernel_size = 2
        self.stdev = 2.0
        self.stdev_scale = 1.6
        self.tau = 1.0
        self.threshold = 0.005
        self.invert = False
        self.edge_threshold = 8
        self.exposure = 1.0
        self.attenuation = 1.0
        self.view_grid = False
        self.debug_edges = False
        self.no_edges = False
        self.no_fill = False
        
        # Setup rendering pipeline
        self._setup_shaders(shader_dir_path)
        self._setup_ascii_textures(ascii_img, edges_img)
        self._setup_quad()
        
        # Resource management
        self.current_size = None
        self.textures = {}
        self.framebuffers = {}
        self.vaos = {}  # Cache VAOs per program
        
    def _setup_shaders(self, shader_dir_path: str):
        """Load and compile all shader programs."""
        
        vertex_shader = """
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            uv = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        """
        
        # Load fragment shaders from files
        shader_files = {
            'luminance': 'luminance.frag',
            'blur_h': 'gaussian_h.frag',
            'blur_v': 'gaussian_v.frag',
            'sobel_h': 'sobel_h.frag',
            'sobel_v': 'sobel_v.frag',
            'pack': 'combine.frag',
            'downsample': 'downscale.frag',
            'ascii': 'ascii.frag'
        }
        
        self.programs = {}
        for name, filename in shader_files.items():
            shader_path = os.path.join(shader_dir_path, filename)
            with open(shader_path, 'r') as f:
                fragment_shader = f.read()
            self.programs[name] = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        
        print(f"Loaded {len(self.programs)} shader programs")
    
    def _setup_ascii_textures(self, ascii_img: Image.Image, edges_img: Image.Image):
        """Setup ASCII character texture atlases"""
        
        if ascii_img.mode != 'L':
            ascii_img = ascii_img.convert('L')
        ascii_data = np.array(ascii_img, dtype='u1')
        width, height = ascii_img.size
        
        self.ascii_tex = self.ctx.texture((width, height), 1, ascii_data.tobytes())
        self.ascii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        if edges_img.mode != 'L':
            edges_img = edges_img.convert('L')
        edges_data = np.array(edges_img, dtype='u1')
        width, height = edges_img.size
        
        self.edges_tex = self.ctx.texture((width, height), 1, edges_data.tobytes())
        self.edges_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        print(f"ASCII texture: {self.ascii_tex.size[0]}x{self.ascii_tex.size[1]}, Edge texture: {self.edges_tex.size[0]}x{self.edges_tex.size[1]}")
    
    def _setup_quad(self):
        """Create fullscreen quad geometry"""
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
    
    def _create_vao(self, program):
        """Create or retrieve cached VAO for a specific shader program"""
        program_id = id(program)
        if program_id not in self.vaos:
            self.vaos[program_id] = self.ctx.simple_vertex_array(program, self.vbo, 'in_vert')
        return self.vaos[program_id]
    
    def _ensure_resources(self, width: int, height: int):
        """
        Ensure all textures and framebuffers are allocated for given dimensions.
        Reuses existing resources if dimensions match
        """
        if self.current_size == (width, height):
            return
        
        # Clean up old resources
        for tex in self.textures.values():
            tex.release()
        for fbo in self.framebuffers.values():
            fbo.release()
        
        self.textures.clear()
        self.framebuffers.clear()
        
        # Create processing textures
        self.textures['input'] = self.ctx.texture((width, height), 3, dtype='f4')
        self.textures['luminance'] = self.ctx.texture((width, height), 1, dtype='f4')
        self.textures['blur_temp'] = self.ctx.texture((width, height), 2, dtype='f4')
        self.textures['dog'] = self.ctx.texture((width, height), 1, dtype='f4')
        self.textures['sobel_temp'] = self.ctx.texture((width, height), 2, dtype='f4')
        self.textures['sobel'] = self.ctx.texture((width, height), 4, dtype='f4')
        self.textures['packed'] = self.ctx.texture((width, height), 4, dtype='f4')
        
        # Create downsampled textures
        self.textures['down1'] = self.ctx.texture((width // 2, height // 2), 4, dtype='f4')
        self.textures['down2'] = self.ctx.texture((width // 4, height // 4), 4, dtype='f4')
        self.textures['down3'] = self.ctx.texture((width // 8, height // 8), 4, dtype='f4')
        
        # Result texture
        self.textures['result'] = self.ctx.texture((width, height), 4, dtype='f4')
        
        # Set filters
        for name, tex in self.textures.items():
            if name in ['sobel', 'result']:
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            else:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            tex.repeat_x = False
            tex.repeat_y = False

        
        # Create framebuffers
        for name, tex in self.textures.items():
            if name != 'input':
                self.framebuffers[name] = self.ctx.framebuffer(color_attachments=[tex])
        
        self.current_size = (width, height)
        print(f"Allocated resources for {width}x{height}")
    
    def process_frame(self, image: Image.Image) -> Image.Image | None:
        """
        Process a single video frame through the ASCII art pipeline.
        
        Args:
            image: Input PIL Image (RGB or RGBA)
            
        Returns:
            Processed PIL Image with ASCII art effect
        """
        try:
            if image.mode == 'RGBA':
                # Create white background for proper alpha blending
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                img = background
            elif image.mode != 'RGB':
                img = image.convert('RGB')
            else:
                img = image
            
            width, height = img.size
            self._ensure_resources(width, height)
            
            img_data = np.array(img, dtype='f4') / 255.0
            self.textures['input'].write(img_data.tobytes())
            
            try:
                # Pass 1: Luminance Extraction
                self._render_pass('luminance', {'tex': (self.textures['input'], 0)})
                
                # Pass 2: Horizontal Gaussian Blur
                self._render_pass('blur_h', {
                    'tex': (self.textures['luminance'], 0),
                    'texel_size': (1.0/width, 1.0/height),
                    'sigma': self.stdev,
                    'k': self.stdev_scale,
                    'kernel_size': self.gaussian_kernel_size
                }, target='blur_temp')
                
                # Pass 3: Vertical Gaussian Blur + DoG
                self._render_pass('blur_v', {
                    'tex': (self.textures['blur_temp'], 0),
                    'texel_size': (1.0/width, 1.0/height),
                    'sigma': self.stdev,
                    'k': self.stdev_scale,
                    'tau': self.tau,
                    'threshold_val': self.threshold,
                    'kernel_size': self.gaussian_kernel_size,
                    'invert': 1 if self.invert else 0
                }, target='dog')
                
                # Pass 4: Sobel Horizontal
                self._render_pass('sobel_h', {
                    'tex': (self.textures['dog'], 0),
                    'texel_size': (1.0/width, 1.0/height)
                }, target='sobel_temp')
                
                # Pass 5: Sobel Vertical
                self._render_pass('sobel_v', {
                    'tex': (self.textures['sobel_temp'], 0),
                    'texel_size': (1.0/width, 1.0/height)
                }, target='sobel')
                
                # Pass 6: Pack Color + Luminance
                self._render_pass('pack', {
                    'tex': (self.textures['input'], 0),
                    'lum_tex': (self.textures['luminance'], 1)
                }, target='packed')
                
                # Pass 7: Downsampling Chain
                self._render_pass('downsample', {'tex': (self.textures['packed'], 0)}, target='down1')
                self._render_pass('downsample', {'tex': (self.textures['down1'], 0)}, target='down2')
                self._render_pass('downsample', {'tex': (self.textures['down2'], 0)}, target='down3')
                
                # Pass 8: ASCII Rendering
                self._render_pass('ascii', {
                    'sobel_tex': (self.textures['sobel'], 0),
                    'edge_ascii_tex': (self.edges_tex, 1),
                    'ascii_tex': (self.ascii_tex, 2),
                    'luminance_tex': (self.textures['down3'], 3),
                    'no_edges': self.no_edges,
                    'no_fill': self.no_fill,
                    'edge_threshold': self.edge_threshold,
                    'exposure': self.exposure,
                    'attenuation': self.attenuation
                }, target='result')
                
            except Exception as render_error:
                print(f"\nRender Pass error: {render_error}")
                import traceback
                traceback.print_exc()
                return None
            
            # Read result
            result_data = self.textures['result'].read()
            result_array = np.frombuffer(result_data, dtype='f4').reshape((height, width, 4))
            
            # Convert to RGB image
            if self.colored:
                # Use RGB channels from result
                result_rgb = (np.clip(result_array[:, :, :3], 0, 1) * 255).astype('u1')
            else:
                # Use grayscale (replicate single channel)
                gray = (np.clip(result_array[:, :, 0], 0, 1) * 255).astype('u1')
                result_rgb = np.stack([gray, gray, gray], axis=-1)
            
            return Image.fromarray(result_rgb, 'RGB')
            
        except Exception as e:
            print(f"\nError processing frame: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _render_pass(self, program_name: str, uniforms: dict, target: str | None = None):
        """
        Execute a single rendering pass.
        
        Args:
            program_name: Name of shader program to use
            uniforms: Dictionary of uniform name -> (texture, unit) or value
            target: Name of target framebuffer (uses program_name if None)
        """
        if target is None:
            target = program_name
        
        program = self.programs[program_name]
        fbo = self.framebuffers[target]
        
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        
        # Bind textures and set uniforms
        for name, value in uniforms.items():
            if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], 'use'):
                # Texture uniform: (texture, unit)
                tex, unit = value
                tex.use(unit)
                program[name] = unit
            else:
                # Scalar/vector uniform (including tuples like texel_size)
                program[name] = value
        
        # Render
        vao = self._create_vao(program)
        vao.render(moderngl.TRIANGLE_STRIP)
        # Don't release - we're caching VAOs now
    
    def cleanup(self):
        """Release all GPU resources."""
        for vao in self.vaos.values():
            vao.release()
        for tex in self.textures.values():
            tex.release()
        for fbo in self.framebuffers.values():
            fbo.release()
        if hasattr(self, 'ascii_tex'):
            self.ascii_tex.release()
        if hasattr(self, 'edges_tex'):
            self.edges_tex.release()
        if hasattr(self, 'vbo'):  
            self.vbo.release()  
        for prog in self.programs.values():  
            prog.release()  
        self.vaos.clear()
        self.textures.clear()
        self.framebuffers.clear()
        self.programs.clear() 


def process_video(converter: VideoAsciiConverter, video_path: str, 
                 output_path: str = "", audio: bool = True, fps: float | None = None) -> str:
    """
    Process a video file with ASCII art effect.
    
    Args:
        converter: VideoAsciiConverter instance
        video_path: Path to input video file
        output_path: Path to output video (auto-generated if empty)
        audio: Whether to preserve audio track
        fps: Override frame rate (uses source fps if None)
        
    Returns:
        Path to output video file
    """
    from .video_extractor import extract_video
    
    # Extract video frames and audio
    source_fps, frame_gen, audio_gen = extract_video(video_path)
    if fps is None:
        fps = source_fps
    
    # Generate output path
    if output_path == "":
        parts = os.path.splitext(os.path.basename(video_path))
        output_path = f"ascii_{parts[0]}.asc{parts[-1]}"
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        # Video input from stdin
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1280x720',  # Will be updated after first frame
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
    ]
    
    if audio:
        # Audio input from original file
        ffmpeg_cmd.extend([
            '-i', video_path,
            '-map', '0:v',
            '-map', '1:a?',
        ])
    
    # Encoding settings
    ffmpeg_cmd.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',  # Changed from 'medium' to 'ultrafast' for faster encoding
        '-crf', '23',
        '-threads', '0',  # Use all available CPU threads
    ])
    
    if audio:
        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
        ])
    
    ffmpeg_cmd.extend(['-f', 'mp4', output_path])
    
    ffmpeg_process = None
    start_time = time.time()
    frame_count = 0
    
    try:
        print(f"Processing video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Frame rate: {fps} fps")
        
        for frame in frame_gen:
            # Process frame
            out_frame = converter.process_frame(frame)
            
            if out_frame is None:
                continue
            
            # Initialize ffmpeg after first frame
            if ffmpeg_process is None:
                width, height = out_frame.size
                ffmpeg_cmd[7] = f'{width}x{height}'
                ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8  # Large buffer to prevent blocking
                )
                print(f"Resolution: {width}x{height}")
            
            # Write frame to ffmpeg
            try:
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.write(out_frame.tobytes())
            except (OSError, BrokenPipeError) as e:
                print(f"\nError writing to ffmpeg: {e}")
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames ({fps_actual:.1f} fps)...", end='\r')
        
    finally:
        # Finalize video encoding
        if ffmpeg_process:
            if ffmpeg_process.stdin:
                try:
                    ffmpeg_process.stdin.close()
                except:
                    pass
            
            # Wait for ffmpeg to finish
            # Wait for ffmpeg to finish
            ffmpeg_process.communicate()
            # Wait for ffmpeg to finish
            ffmpeg_process.communicate()
            if ffmpeg_process.returncode != 0:
                print(f"\nFFmpeg failed with return code: {ffmpeg_process.returncode}")
        
        # Cleanup
        converter.cleanup()
        
        # Final statistics
        elapsed = time.time() - start_time
        print()
        print(f"Completed: {elapsed:.2f}s for {frame_count} frames")
        print(f"Average: {frame_count/elapsed:.1f} fps")
        print(f"Output: {output_path}")
        
        return output_path