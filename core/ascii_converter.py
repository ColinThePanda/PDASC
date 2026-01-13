from PIL import Image
import numpy as np
from .generate_color_ramp import generate_color_ramp, get_charmap
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def compute_blocks(img: np.ndarray, cs: int, gray_levels: int, color: bool):
    h, w, _ = img.shape
    bh, bw = h // cs, w // cs

    char_idx = np.empty((bh, bw), np.int32)
    colors = np.empty((bh, bw), np.uint32)  # Always allocate

    inv_area = 1.0 / (cs * cs)

    for by in prange(bh):
        for bx in range(bw):
            r = g = b = 0.0
            
            if color:
                for y in range(cs):
                    for x in range(cs):
                        px = img[by*cs + y, bx*cs + x]
                        r += px[0]
                        g += px[1]
                        b += px[2]

                r *= inv_area
                g *= inv_area
                b *= inv_area
            else:
                # compute grayscale
                for y in range(cs):
                    for x in range(cs):
                        px = img[by*cs + y, bx*cs + x]
                        r += px[0]
                        g += px[1]
                        b += px[2]

                r *= inv_area
                g *= inv_area
                b *= inv_area

            # Grayscale -> char index
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            gi = int(lum * gray_levels)
            if gi >= gray_levels:
                gi = gray_levels - 1
            char_idx[by, bx] = gi
            
            if color:
                # Convert to 8-bit RGB
                ri = np.uint32(r * 255.0)
                gi_color = np.uint32(g * 255.0)
                bi = np.uint32(b * 255.0)

                # Pack RGB -> 0xRRGGBB
                colors[by, bx] = (ri << 16) | (gi_color << 8) | bi
            else:
                colors[by, bx] = 0  # Dummy value when no color
    
    return char_idx, colors

class AsciiConverter:
    def __init__(self, num_ascii: int = 8, chunk_size: int = 8, font_path: str = "CascadiaMono.ttf"):
        self.font_path = font_path
        self.chunk_size: int = chunk_size
        self.char_map = get_charmap(generate_color_ramp(font_path=self.font_path), num_ascii)
        self.num_ascii: int = min(num_ascii, len(self.char_map))
    
    def regen_charmap(self):
        self.char_map = get_charmap(generate_color_ramp(font_path=self.font_path), self.num_ascii)
    
    def quantize_grayscale(self, image: Image.Image) -> Image.Image:
        img_data = np.array(image, dtype=np.float32) / 255.0
        height, width, channels = img_data.shape
        out_data = np.zeros_like(img_data)
        
        for y in range(height):
            for x in range(width):
                in_pix = img_data[y, x]
                gray = sum([0.2126 * in_pix[0], 0.7152 * in_pix[1], 0.0722 * in_pix[2]])
                quantized = np.floor(gray * self.num_ascii) / (self.num_ascii - 1)
                out_data[y, x] = [quantized, quantized, quantized]
        
        out_data = (out_data * 255).astype(np.uint8)
        
        return Image.fromarray(out_data, "RGB")
    
    def get_ascii(self, image: Image.Image, color: bool = True) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32) / 255.0
        cs = self.chunk_size

        char_idx, colors = compute_blocks(
            img,
            cs,
            self.num_ascii,
            color
        )

        # map char indices to chars
        char_map_arr = np.array(list(self.char_map), dtype='<U1')
        chars = char_map_arr[char_idx]

        # Structured array
        dtype = np.dtype([('char', '<U1'), ('color', np.uint32)])
        out = np.empty(chars.shape, dtype=dtype)
        out['char'] = chars
        out['color'] = colors if color else np.zeros_like(colors, dtype=np.uint32)

        return out