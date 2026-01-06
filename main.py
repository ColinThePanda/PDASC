from PIL import Image
import numpy as np
from generate_color_ramp import generate_color_ramp, get_charmap
import time

def pack_int24(color: tuple[int, int, int]) -> int:
    return (color[0] << 16) | (color[1] << 8) | color[2]

def unpack_int24(packed: int) -> tuple[int, int, int]:
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)

def color_text(text: str, r: int, g: int, b: int):
    r = max(min(r, 255), 0)
    g = max(min(g, 255), 0)
    b = max(min(b, 255), 0)
    return f"\033[38;2;{r};{g};{b}m{text}"

class AsciiConverter:
    def __init__(self, quantization_levels: int = 8, chunk_size: int = 8):
        self.quantization_levels: int = quantization_levels
        self.chunk_size: int = chunk_size
        self.char_map = get_charmap(generate_color_ramp(), quantization_levels)
    
    def quantize_grayscale(self, image: Image.Image) -> Image.Image:
        img_data = np.array(image, dtype=np.float32) / 255.0
        height, width, channels = img_data.shape
        out_data = np.zeros_like(img_data)
        
        for y in range(height):
            for x in range(width):
                in_pix = img_data[y, x]
                gray = sum([0.2126 * in_pix[0], 0.7152 * in_pix[1], 0.0722 * in_pix[2]])
                quantized = np.floor(gray * self.quantization_levels) / (self.quantization_levels - 1)
                out_data[y, x] = [quantized, quantized, quantized]
        
        out_data = (out_data * 255).astype(np.uint8)
        
        return Image.fromarray(out_data, "RGB")
    
    def get_ascii(self, image: Image.Image):
        # Quantize Grayscale
        img_data = np.array(image, dtype=np.float32) / 255.0
        height, width, channels = img_data.shape
        
        out: list[list[tuple[str, int]]] = [
            [('', 0) for _ in range(width // self.chunk_size)]
            for _ in range(height // self.chunk_size)
        ]
        
        for y in range(0, height, self.chunk_size):
            for x in range(0, width, self.chunk_size):
                chunk = img_data[y:y + self.chunk_size, x:x + self.chunk_size]
                chunk_gray = 0.2126 * chunk[:, :, 0] + 0.7152 * chunk[:, :, 1] + 0.0722 * chunk[:, :, 2]
                mean_lum = np.mean(chunk_gray)
                mean_rgb = np.mean(chunk, axis=(0, 1))
                char = self.char_map[int(mean_lum * self.quantization_levels)]
                r, g, b = (int(c * 255) for c in mean_rgb)
                int_color = pack_int24((r, g, b))
                out[y//8][x//8] = (char, int_color)
        
        return out

def display_ascii(ascii: list[list[tuple[str, int]]]):
    for row in ascii:
        doubled_colored = "".join([color_text(char[0] * 2, *unpack_int24(char[1])) for char in row])
        print(f"{doubled_colored}\033[0m")

if __name__ == "__main__":
    img = Image.open("test.png").convert('RGB')
    converter = AsciiConverter()
    start = time.time()
    ascii = converter.get_ascii(img)
    display_ascii(ascii)
    print(f"Total time took: {time.time() - start}")