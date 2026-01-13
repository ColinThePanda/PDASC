from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def generate_color_ramp(font_size: int = 32, image_size: int = 48, font_path: str = "CascadiaMono.ttf", chars: list[str] = [chr(i) for i in range(32, 127)]):
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at {font_path}")
    
    font = ImageFont.truetype(font_path, font_size)
    results = []
    
    for char in chars:
        img = Image.new("L", (image_size, image_size), 255)
        draw = ImageDraw.Draw(img)
        
        draw.text((0, 0), char, fill=0, font=font)
        
        img_data = np.asarray(img, dtype=np.float32) / 255.0
        total_luminance = img_data.mean()
        
        results.append((char, total_luminance))
    
    lums = [lum for _, lum in results]
    min_lum = min(lums)
    max_lum = max(lums)

    # Normalize 1-0
    results = [
        (char, 1.0 - (lum - min_lum) / (max_lum - min_lum))
        for char, lum in results
    ]

    return results

def get_charmap(color_ramp: list[tuple[str, float]], levels: int = 8):
    if levels < 2:
        raise ValueError("levels must be >= 2")
    
    if levels >= len(color_ramp):
        color_ramp.sort(key=lambda x: x[1])
        return "".join([char[0] for char in color_ramp])
    
    quantized_values = [i/levels for i in range(levels + 1)]
    
    out_ramp = []
    for value in quantized_values:
        best_char = (color_ramp[0][0], abs(color_ramp[0][1] - value))
        for char, lum in color_ramp:
            distance = abs(value - lum)
            if distance < best_char[1] and char not in out_ramp:
                best_char = (char, distance)
        out_ramp.append(best_char[0])
    return "".join(out_ramp)

def render_charmap(charmap: str, font_path="font8x8.ttf", font_size=8, padding=0):
    char_width = font_size
    char_height = font_size
    w = len(charmap) * (char_width + padding)
    h = char_height + 2 * padding

    img = Image.new("L", (w, h), 0)  # black background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    for i, ch in enumerate(charmap):
        x = i * (char_width + padding) + padding
        y = padding
        draw.text((x, y), ch, fill=255, font=font)

    return img