# Create SRT from Video or Audio

A simple command‑line tool to generate `.srt` subtitles from any video (or audio) file using the open‑source **Whisper** speech‑to‑text model.

> ✅ Works offline.  
> ✅ Auto language detection or force a language.  
> ✅ Optional translation to English.  
> ✅ CPU, NVIDIA GPU (`cuda`), and Apple Silicon (`mps`) supported.

## Requirements

- **Python** 3.8+
- **FFmpeg** (required by Whisper to read media)

### Install Python packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai-whisper torch
```

## Usage

Basic (auto‑detect language):

```bash
python transcribe.py "my video.mp4"
```

Force language to English:

```bash
python transcribe.py "my video.mp4" --language en
```

Translate non‑English speech **to English** subtitles:

```bash
python transcribe.py "cours_francais.mp4" --translate
```

Choose a bigger model for accuracy:

```bash
python transcribe.py "interview.mkv" --model small
```

Specify output path:

```bash
python transcribe.py "clip.mov" -o "clip_subs.srt"
```

Run on a specific device:

```bash
# NVIDIA GPU
python transcribe.py "talk.mp4" --device cuda --fp16

# Apple Silicon (M1/M2/M3)
python transcribe.py "talk.mp4" --device mps

# CPU only
python transcribe.py "talk.mp4" --device cpu
```

## Options

- `input` (positional): Path to a video or audio file (any format FFmpeg can read).
- `-o, --output`: Destination `.srt` path. Default is the input name with `.srt`.
- `--model`: `tiny` | `base` | `small` | `medium` | `large` (default: `base`). Larger ≈ better but slower.
- `--language`: ISO code like `en`, `fr`, `es`. If omitted, the model tries to auto‑detect.
- `--translate`: Force English output from non‑English audio.
- `--device`: `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). If omitted, Whisper picks automatically.
- `--fp16`: Use 16‑bit floats on GPU for speed (default behavior on CUDA). Ignored on CPU/MPS.
- `--vad`: Enables a simple, VAD‑like heuristic (sets `beam_size=5`, `best_of=5`) that can improve segmenting in some files.
