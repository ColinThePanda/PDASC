import numpy as np

def pack_int24(color: tuple[int, int, int]) -> int:
    return (color[0] << 16) | (color[1] << 8) | color[2]

def unpack_int24(packed: int) -> tuple[int, int, int]:
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)

def pack_int24_chunk(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b

def unpack_int24_array(colors : np.ndarray):
    r = (colors >> 16) & 0xFF
    g = (colors >> 8) & 0xFF
    b = colors & 0xFF
    return r, g, b

def format_file_size(size_bytes: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"