#!/usr/bin/env python3
"""
Text-to-Speech Converter
========================
Converts text to speech using multiple backend options:
- gTTS (Google Translate TTS) - Online, high quality
- pyttsx3 - Offline, uses system voices

Usage:
    python tts.py "Hello, world!"
    python tts.py "Hello, world!" --engine gtts --lang es
    python tts.py "Hello, world!" --engine pyttsx3 --voice 1
    python tts.py --file input.txt --output speech.mp3
"""

import argparse
import sys
import os


def gtts_speak(text: str, lang: str = "en", output_file: str = "output.mp3", slow: bool = False):
    """Convert text to speech using gTTS (Google Translate TTS)."""
    try:
        from gtts import gTTS
    except ImportError:
        print("Error: gTTS not installed. Install with: pip install gTTS")
        sys.exit(1)

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(output_file)
    print(f"✓ Audio saved to: {output_file}")
    return output_file


def pyttsx3_speak(text: str, voice_index: int = 0, rate: int = 200, volume: float = 1.0, output_file: str = None):
    """Convert text to speech using pyttsx3 (offline, system voices)."""
    try:
        import pyttsx3
    except ImportError:
        print("Error: pyttsx3 not installed. Install with: pip install pyttsx3")
        sys.exit(1)

    engine = pyttsx3.init()
    
    # List available voices if requested
    voices = engine.getProperty('voices')
    if voice_index == -1:
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.id} (Language: {voice.language})")
        return None
    
    # Set voice, rate, and volume
    if voices and voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    
    if output_file:
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        print(f"✓ Audio saved to: {output_file}")
    else:
        engine.say(text)
        engine.runAndWait()
        print("✓ Speech played.")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert text to speech using various TTS engines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello, world!"                          # Default (gTTS, English)
  %(prog)s "Bonjour!" --lang fr                     # French with gTTS
  %(prog)s "Hello" --engine pyttsx3                 # Offline mode
  %(prog)s "Hello" --engine pyttsx3 --voice -1      # List available voices
  %(prog)s --file story.txt --output story.mp3      # From file
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("text", nargs="?", help="Text to convert to speech")
    input_group.add_argument("--file", "-f", help="Read text from a file")
    
    # Engine selection
    parser.add_argument(
        "--engine", "-e", 
        choices=["gtts", "pyttsx3"], 
        default="gtts",
        help="TTS engine to use (default: gtts)"
    )
    
    # gTTS options
    parser.add_argument("--lang", "-l", default="en", help="Language code for gTTS (default: en)")
    parser.add_argument("--slow", action="store_true", help="Slow down speech (gTTS only)")
    
    # pyttsx3 options
    parser.add_argument("--voice", "-v", type=int, default=0, help="Voice index (default: 0, use -1 to list)")
    parser.add_argument("--rate", "-r", type=int, default=200, help="Speech rate in WPM (pyttsx3 only, default: 200)")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume 0.0-1.0 (pyttsx3 only, default: 1.0)")
    
    # Output
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: output.mp3 for gTTS, play for pyttsx3)")
    
    args = parser.parse_args()
    
    # Get text input
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        print(f"Loaded {len(text)} characters from '{args.file}'")
    else:
        text = args.text
    
    if not text:
        print("Error: No text provided.")
        sys.exit(1)
    
    # Determine output file
    output_file = args.output
    if output_file is None:
        output_file = "output.mp3" if args.engine == "gtts" else None
    
    # Run TTS
    print(f"Engine: {args.engine.upper()}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    if args.engine == "gtts":
        gtts_speak(text, lang=args.lang, output_file=output_file, slow=args.slow)
    elif args.engine == "pyttsx3":
        pyttsx3_speak(text, voice_index=args.voice, rate=args.rate, 
                     volume=args.volume, output_file=output_file)


if __name__ == "__main__":
    main()
