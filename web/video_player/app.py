from flask import Flask, render_template, Response, send_file, abort
from pathlib import Path
import subprocess
import os


def create_video_server(video_path, title="ASCII Video Player", autoplay=False):
    """
    Create a Flask app that serves a video file.
    
    Args:
        video_path: Path to video file (str or Path)
        title: Page title (default: "ASCII Video Player")
        autoplay: Whether to autoplay video (default: False)
    
    Returns:
        Flask app instance
    
    Example:
        app = create_video_server("output.mp4")
        app.run(host='0.0.0.0', port=5000)
    """
    video_path = Path(video_path).resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Serve the video player page"""
        return render_template(
            "index.html",
            title=title,
            autoplay=autoplay
        )
    
    @app.route('/video')
    def video():
        """Serve the video file"""
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True  # Enable range requests for seeking
        )
    
    return app