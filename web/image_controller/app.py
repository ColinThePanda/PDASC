import sys
import os
# add root to path to import from core
sys.path.insert(0, os.path.join(__file__, "../../../"))

from flask import Flask, render_template, request
from PIL import Image
from core import AsciiConverter, AsciiDisplayer
import base64
from io import BytesIO

class ImageServer:
    def __init__(self) -> None:
        self.app = Flask(__name__)
        self._setup_app()
        self.converter = AsciiConverter()
        self.displayer = AsciiDisplayer(self.converter)
        self.colored = True

    def _html_from_ansi(self, ansi: str) -> str:
        if not self.colored:
            # Escape HTML characters and replace newlines with <br> for plain ASCII
            import html
            escaped = html.escape(ansi)
            return f'<span>{escaped.replace("\n", "<br>")}</span>'
        
        import re
        
        html_parts = []
        current_color = None
        
        # Pattern to match ANSI escape codes
        ansi_pattern = re.compile(r'\x1b\[([0-9;]+)m')
        
        last_end = 0
        for match in ansi_pattern.finditer(ansi):
            # Add any text before this escape code
            text = ansi[last_end:match.start()]
            if text:
                if current_color:
                    html_parts.append(f'<span style="color:{current_color}">{text}</span>')
                else:
                    html_parts.append(text)
            
            # Parse the escape code
            codes = match.group(1).split(';')
            i = 0
            while i < len(codes):
                code = codes[i]
                
                if code == '0' or code == '':  # Reset
                    current_color = None
                elif code == '38' and i + 2 < len(codes) and codes[i + 1] == '2':  # RGB foreground
                    r, g, b = codes[i + 2], codes[i + 3], codes[i + 4]
                    current_color = f'rgb({r},{g},{b})'
                    i += 4
                
                i += 1
            
            last_end = match.end()
        
        # Add any remaining text
        text = ansi[last_end:]
        if text:
            if current_color:
                html_parts.append(f'<span style="color:{current_color}">{text}</span>')
            else:
                html_parts.append(text)
        
        return ''.join(html_parts).replace('\n', '<br>')
    
    def _setup_app(self):
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
        @self.app.route('/')
        def index():
            return render_template(
                "index.html"
            )
        
        @self.app.route('/ascii', methods=['POST'])
        def ascii():
            # Get form data
            print(f"Form: {request.form}")
            print(f"Files: {request.files}")
            num_ascii = request.form.get('num_ascii')
            block_size = request.form.get('block_size')
            colored = request.form.get('colored')
            image_file  = request.files.get('image')  # This is base64 encoded
            
            if not num_ascii or not block_size or not colored or not image_file:
                return "Error one of the inputs is not defined"
            
            self.converter.num_ascii = int(num_ascii)
            self.converter.chunk_size = int(block_size)
            self.colored = bool(int(colored))
            image = Image.open(image_file.stream)
            self.converter.regen_charmap()
            ascii = self.converter.get_ascii(image, self.colored)
            frame = self.displayer.render_ascii(ascii, self.colored)
            return self._html_from_ansi(frame)
    
    def run(self):
        self.app.run()