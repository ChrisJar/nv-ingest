#!/usr/bin/env python3
"""Minimal NeMo Toolkit benchmark for Nemotron streaming ASR.

This is intentionally separate from bench_asr_speed.py and its environment.
It tests whether NeMo Toolkit's direct ASRModel path is faster than the
Transformers path for one audio file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16_000


@dataclass
class NemoBenchResult:
    model: str
    audio_seconds: float
    load_seconds: float
    transcribe_seconds: float
    realtime_factor: float
    transcript_chars: int
    transcript_preview: str


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


def _load_model(model_id: str, device: str):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _coerce_text(result: Any) -> str:
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
    parser.add_argument("audio", type=Path, help="Audio file to benchmark.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the converted 16 kHz WAV beside the JSON output or in the temp directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    audio_seconds = audio_duration_seconds(args.audio)
    print(f"audio_file={args.audio}")
    print(f"audio_seconds={audio_seconds:.2f} ({audio_seconds / 60:.2f} min)")
    print(f"device={args.device}")
    print(f"model={args.model}")

    with tempfile.TemporaryDirectory(prefix="nemo_asr_speed_") as temp_name:
        temp_dir = Path(temp_name)
        wav_path = convert_to_wav_16k(args.audio, temp_dir)

        model, load_seconds = _time_call(lambda: _load_model(args.model, args.device))
        text, transcribe_seconds = _time_call(lambda: transcribe_nemo(model, wav_path))

        if args.keep_wav:
            target = (args.json_out.parent if args.json_out else Path.cwd()) / wav_path.name
            target.write_bytes(wav_path.read_bytes())
            print(f"kept wav={target}")

    result = NemoBenchResult(
        model=args.model,
        audio_seconds=audio_seconds,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        realtime_factor=audio_seconds / transcribe_seconds if transcribe_seconds > 0 else 0.0,
        transcript_chars=len(text),
        transcript_preview=text.replace("\n", " ")[:180],
    )

    print("\nResults")
    print("model,load_s,transcribe_s,realtime_factor,chars,preview")
    print(
        f"{result.model},"
        f"{result.load_seconds:.2f},"
        f"{result.transcribe_seconds:.2f},"
        f"{result.realtime_factor:.2f}x,"
        f"{result.transcript_chars},"
        f"{json.dumps(result.transcript_preview)}"
    )

    if args.json_out:
        args.json_out.write_text(json.dumps(asdict(result), indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
