#!/usr/bin/env python3
"""
Create an SRT subtitle file from a video (or audio) using OpenAI Whisper.

Usage:
    python transcribe.py /path/to/video.mp4
    python transcribe.py /path/to/video.mp4 --model small --language en
    python transcribe.py /path/to/video.mp4 --translate
    python transcribe.py /path/to/video.mp4 -o subtitles.srt
"""

import argparse
import os
import sys
import datetime
import whisper


def srt_timestamp(seconds: float) -> str:
    """
    Convert float seconds to SRT timestamp 'HH:MM:SS,mmm'
    """
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    # Total seconds -> hours, minutes, seconds, milliseconds
    total_ms = int(round(td.total_seconds() * 1000))
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments, out_path: str):
    """
    Write Whisper segments to an SRT file.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = srt_timestamp(seg.get("start", 0.0))
            end = srt_timestamp(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            # Whisper sometimes emits leading/trailing spaces
            text = text.replace("  ", " ")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create an SRT from a video using Whisper."
    )
    parser.add_argument("input", help="Path to video/audio file.")
    parser.add_argument(
        "-o", "--output", help="Output SRT path. Defaults to input name with .srt"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny | base | small | medium | large (default: base). Larger = better but slower.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code (e.g., en, fr, es). If omitted, Whisper will auto-detect.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate speech to English (useful for non-English audio).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda'. Default chooses automatically.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 on GPU (default behavior on CUDA). Ignored on CPU.",
    )
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model, device=args.device)

    transcribe_kwargs = {
        "task": "translate" if args.translate else "transcribe",
        "language": args.language,  # None = auto-detect
        "fp16": args.fp16 if (args.device == "cuda" or args.device is None) else False,
        # You can tweak these for speed/accuracy tradeoffs:
        "temperature": 0.0,  # more deterministic
        "no_speech_threshold": 0.6,  # filter silence
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": False,  # to avoid repeats
    }

    print(f"Transcribing '{in_path}'...")
    result = model.transcribe(in_path, **transcribe_kwargs)

    segments = result.get("segments", [])
    if not segments:
        print("No segments returned. Nothing to write.", file=sys.stderr)
        sys.exit(2)

    base = os.path.splitext(in_path)[0]
    detected = result.get("language")
    lang = "en" if args.translate else (args.language or detected)
    out_path = args.output or (f"{base}.{lang}.srt" if lang else f"{base}.srt")

    write_srt(segments, out_path)
    print(f"Done. Wrote subtitles to: {out_path}")


if __name__ == "__main__":
    main()
