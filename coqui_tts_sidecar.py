import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output WAV file path")
    parser.add_argument(
        "--model",
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Coqui TTS model name",
    )
    parser.add_argument("--check", action="store_true", help="Verify TTS import works")
    args = parser.parse_args()

    from TTS.api import TTS

    if args.check:
        print("ok")
        return 0

    if not args.out:
        raise SystemExit("--out is required unless --check is used")

    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("No input text provided on stdin")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = TTS(model_name=args.model, progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
