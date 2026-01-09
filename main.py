from ascii_converter import AsciiConverter
from ascii_displayer import AsciiDisplayer
from ascii_file_encoding import AsciiEncoder
from PIL import Image
import argparse
import sys
import os

def add_common_args(parser):
    """Add arguments common to both play and encode"""
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
        "-f", "--font",
        type=str,
        default="CascadiaMono.ttf",
        help="Path to font file to create the ASCII character set from"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback for videos"
    )

def cmd_play(args):
    """Play command - display images/videos/camera"""
    # Validate font
    if not os.path.exists(args.font):
        print(f"Error: Font file '{args.font}' not found", file=sys.stderr)
        sys.exit(1)
    
    converter = AsciiConverter(num_ascii=args.num_ascii, chunk_size=args.block_size, font_path=args.font)
    displayer = AsciiDisplayer(converter)
    
    try:
        if args.input == "camera":
            print(f"Starting camera {args.camera} (press Ctrl+C to stop)")
            displayer.display_camera(camera_index=args.camera, color=not args.no_color)
        else:
            # Validate input file
            if not os.path.exists(args.input):
                print(f"Error: File '{args.input}' not found", file=sys.stderr)
                sys.exit(1)
            
            # Determine file type
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
            
            ext = os.path.splitext(args.input)[-1].lower()
            
            if ext not in video_extensions and ext not in image_extensions and ext != '.asc':
                print(f"Error: Unsupported file extension '{ext}'", file=sys.stderr)
                print(f"Supported: .asc, {', '.join(sorted(video_extensions | image_extensions))}", file=sys.stderr)
                sys.exit(1)
            
            if ext == '.asc':
                print(f"Playing .asc file: {args.input}")
                displayer.display_asc_file(args.input, not args.no_audio)
            elif ext in video_extensions:
                print(f"Playing video: {args.input}")
                displayer.display_video(
                    video_path=args.input,
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

def cmd_encode(args):
    """Encode command - save encoded ASCII to file"""
    # Validate font
    if not os.path.exists(args.font):
        print(f"Error: Font file '{args.font}' not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Encoding {args.input} to {args.output}")
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
    
    encoder = AsciiEncoder()
    converter = AsciiConverter(num_ascii=args.num_ascii, chunk_size=args.block_size, font_path=args.font)
    
    if ext in video_extensions:
        encoder.encode_video_to_asc(args.input, args.output, not args.no_audio, not args.no_color, converter)
    else:
        encoder.encode_image_to_asc(args.input, args.output, not args.no_color, converter)
    
def main():
    parser = argparse.ArgumentParser(
        description="Convert images and videos to ASCII art",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Play subcommand
    play_parser = subparsers.add_parser(
        'play',
        help='Display images/videos/camera as ASCII art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s play image.png
  %(prog)s play video.mp4 --no-audio
  %(prog)s play camera -c 0
  %(prog)s play image.jpg -b 16 -n 70
        """
    )
    
    play_parser.add_argument(
        "input",
        type=str,
        help='Path to input file or "camera" for camera input'
    )
    
    add_common_args(play_parser)
    
    play_parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index when using camera input (default: 0)"
    )
    
    play_parser.set_defaults(func=cmd_play)
    
    # Encode subcommand
    encode_parser = subparsers.add_parser(
        'encode',
        help='Encode video/image to compressed ASCII file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s encode video.mp4 -o output.asc
  %(prog)s encode image.png -o output.asc --no-color
        """
    )
    
    encode_parser.add_argument(
        "input",
        type=str,
        help="Path to input video or image file"
    )
    
    encode_parser.add_argument(
        "-o", "--output",
        type=str,
        default="ascii_out.mp4",
        help="Output file path (required)"
    )
    
    add_common_args(encode_parser)
    
    encode_parser.set_defaults(func=cmd_encode)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Call the appropriate command function
    args.func(args)

if __name__ == "__main__":
    main()