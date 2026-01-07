from ascii_converter import AsciiConverter
from ascii_displayer import AsciiDisplayer
from PIL import Image
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Convert images and videos to ASCII art in the terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
%(prog)s image.png
%(prog)s video.mp4 --no-audio
%(prog)s image.jpg --block-size 16 --num-ascii 70
%(prog)s camera -b 4 -n 32
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or video file or \"camera\" to view the camera"
    )
    
    parser.add_argument(
        "-b", "--block-size",
        type=int,
        default=8,
        help="Size of character blocks (default: 8)"
    )
    
    parser.add_argument(
        "-n", "--num-ascii",
        type=int,
        default=128,
        help="Number of ASCII characters to use (default: 128)"
    )
    
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Index of camera to use (default: 0)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback for videos"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output (not yet implemented)"
    )
    
    args = parser.parse_args()
    
    if args.input != "camera":
        # Validate input file
        if not os.path.exists(args.input):
            print(f"Error: File '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
        
        # Determine file type
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
        ext = os.path.splitext(args.input)[-1].lower()
        
        if ext not in video_extensions and ext not in image_extensions:
            print(f"Error: Unsupported file extension '{ext}'", file=sys.stderr)
            print(f"Supported: {', '.join(sorted(video_extensions | image_extensions))}", file=sys.stderr)
            sys.exit(1)
    
    # Create converter and displayer
    converter = AsciiConverter(num_ascii=args.num_ascii, chunk_size=args.block_size)
    displayer = AsciiDisplayer(converter)
    
    # Process file
    try:
        if args.input == "camera":
            print("")
            displayer.display_camera(camera_index=args.camera, color=not args.no_color)
        elif ext in video_extensions:
            print(f"Playing video: {args.input}")
            displayer.display_video(
                video_path=str(args.input),
                play_audio=not args.no_audio,
                color=not args.no_color
            )
        else:
            print(f"Displaying image: {args.input}")
            displayer.display_image(image=Image.open(args.input), color=not args.no_color)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()