from ascii_converter import AsciiConverter
from ascii_displayer import AsciiDisplayer
import numpy as np

if __name__ == "__main__":
    converter = AsciiConverter(16, 8)
    displayer = AsciiDisplayer(converter)
    displayer.display_video("inputs/rickroll.mp4")