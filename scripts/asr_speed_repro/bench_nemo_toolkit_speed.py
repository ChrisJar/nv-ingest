#!/usr/bin/env python3
"""Minimal NeMo Toolkit ASR speed benchmark on one audio file."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn):
    _sync_cuda()
    start = time.perf_counter()
    result = fn()
    _sync_cuda()
    return result, time.perf_counter() - start


def audio_duration_seconds(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return float(json.loads(out)["format"]["duration"])


def convert_to_wav_16k(audio_path: Path, work_dir: Path) -> Path:
    wav_path = work_dir / f"{audio_path.stem}.16k.wav"
    subprocess.check_call(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(wav_path),
        ]
    )
    return wav_path


def _load_model():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    if DEVICE == "cuda":
        model = model.to("cuda")
    return model


def _coerce_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return _coerce_text(result[0]) if result else ""
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return str(result)


def transcribe_nemo(model, wav_path: Path) -> str:
    with torch.inference_mode():
        try:
            result = model.transcribe([str(wav_path)], batch_size=1, verbose=False)
        except TypeError:
            result = model.transcribe([str(wav_path)], batch_size=1)
    return _coerce_text(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    audio_seconds = audio_duration_seconds(args.audio)
    print(f"Audio: {args.audio}")
    print(f"Duration: {audio_seconds:.2f}s ({audio_seconds / 60:.2f} min)")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")

    with tempfile.TemporaryDirectory(prefix="nemo_asr_speed_") as temp_name:
        temp_dir = Path(temp_name)
        wav_path = convert_to_wav_16k(args.audio, temp_dir)

        model, load_seconds = _time_call(_load_model)
        text, transcribe_seconds = _time_call(lambda: transcribe_nemo(model, wav_path))

    print("\nResults")
    print(f"NeMo load: {load_seconds:.2f}s")
    print(f"NeMo transcription: {transcribe_seconds:.2f}s ({audio_seconds / transcribe_seconds:.2f}x realtime)")
    print(f"NeMo transcript chars: {len(text)}")


if __name__ == "__main__":
    main()
