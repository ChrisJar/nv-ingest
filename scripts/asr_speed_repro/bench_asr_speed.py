#!/usr/bin/env python3
"""Minimal speed comparison for Parakeet CTC vs Nemotron streaming ASR.

This intentionally avoids the NeMo-Retriever pipeline, Ray, LanceDB, and
embedding. It measures only model load time and transcription time for one
audio file using Hugging Face Transformers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


SAMPLE_RATE = 16_000
CTC_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
NEMOTRON_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


@dataclass
class BenchResult:
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


def _time_call(fn: Callable[[], object]) -> tuple[object, float]:
    _sync_cuda()
    start = time.perf_counter()
    result = fn()
    _sync_cuda()
    return result, time.perf_counter() - start


def load_audio_16k(path: str | Path) -> np.ndarray:
    """Decode any ffmpeg-supported audio file to mono 16 kHz float32."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    pcm = subprocess.check_output(cmd)
    if not pcm:
        raise RuntimeError(f"ffmpeg decoded no audio from {path}")
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_inputs(inputs, *, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key in {"input_features", "input_values"}:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _dtype_for_device(device: str) -> torch.dtype | str:
    return torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else "auto"


def _load_ctc(model_id: str, device: str):
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(
        model_id,
        dtype=_dtype_for_device(device),
        device_map=device,
    )
    model.eval()
    return processor, model


def transcribe_ctc(
    audio: np.ndarray,
    *,
    model_id: str = CTC_MODEL_ID,
    device: str = "cuda",
    chunk_seconds: float = 240.0,
    batch_chunks: int = 4,
) -> tuple[str, float, float]:
    """Transcribe with Parakeet CTC using large offline chunks."""
    (processor, model), load_seconds = _time_call(lambda: _load_ctc(model_id, device))

    chunk_samples = max(1, int(chunk_seconds * SAMPLE_RATE))
    chunks = [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples)]
    chunks = [chunk for chunk in chunks if chunk.size]

    def _run() -> str:
        texts: list[str] = []
        model_device = _model_device(model)
        model_dtype = getattr(model, "dtype", torch.float32)
        for start in range(0, len(chunks), batch_chunks):
            batch = chunks[start : start + batch_chunks]
            inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = _move_inputs(inputs, device=model_device, dtype=model_dtype)
            with torch.inference_mode():
                logits = model(**inputs).logits
            token_ids = logits.argmax(dim=-1)
            texts.extend(processor.batch_decode(token_ids, skip_special_tokens=True))
        return " ".join(text.strip() for text in texts if text.strip())

    transcript, transcribe_seconds = _time_call(_run)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcript, load_seconds, transcribe_seconds


def _load_nemotron(model_id: str, device: str, lookahead: int):
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.set_num_lookahead_tokens(lookahead)
    model = AutoModelForRNNT.from_pretrained(
        model_id,
        dtype=_dtype_for_device(device),
        device_map=device,
    )
    model.eval()
    return processor, model


def _nemotron_feature_stream(audio: np.ndarray, processor, model) -> Iterable[torch.Tensor]:
    first_samples = processor.num_samples_first_audio_chunk
    first_audio = audio[:first_samples]
    if first_audio.shape[0] < first_samples:
        first_audio = np.pad(first_audio, (0, first_samples - first_audio.shape[0]))

    model_device = _model_device(model)
    model_dtype = getattr(model, "dtype", torch.float32)
    first_inputs = processor(
        first_audio,
        sampling_rate=SAMPLE_RATE,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )
    first_inputs = _move_inputs(first_inputs, device=model_device, dtype=model_dtype)
    yield first_inputs["input_features"][:, : processor.num_mel_frames_first_audio_chunk, :]

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft
    start_idx = mel_frame_idx * hop_length - n_fft // 2
    while start_idx < audio.shape[0]:
        end_idx = start_idx + processor.num_samples_per_audio_chunk
        chunk = audio[start_idx:end_idx]
        if chunk.shape[0] < processor.num_samples_per_audio_chunk:
            chunk = np.pad(chunk, (0, processor.num_samples_per_audio_chunk - chunk.shape[0]))
        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=False,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, device=model_device, dtype=model_dtype)
        yield inputs["input_features"]

        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2


def transcribe_nemotron(
    audio: np.ndarray,
    *,
    model_id: str = NEMOTRON_MODEL_ID,
    device: str = "cuda",
    lookahead: int = 13,
) -> tuple[str, float, float]:
    """Transcribe with Nemotron cache-aware streaming ASR."""
    (processor, model), load_seconds = _time_call(lambda: _load_nemotron(model_id, device, lookahead))

    def _run() -> str:
        model_device = _model_device(model)
        model_dtype = getattr(model, "dtype", torch.float32)
        first_samples = processor.num_samples_first_audio_chunk
        first_audio = audio[:first_samples]
        if first_audio.shape[0] < first_samples:
            first_audio = np.pad(first_audio, (0, first_samples - first_audio.shape[0]))
        first_inputs = processor(
            first_audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
        )
        first_inputs = _move_inputs(first_inputs, device=model_device, dtype=model_dtype)
        generate_kwargs = dict(first_inputs)
        generate_kwargs["input_features"] = _nemotron_feature_stream(audio, processor, model)
        generate_kwargs["num_lookahead_tokens"] = lookahead
        generate_kwargs["return_dict_in_generate"] = True
        with torch.inference_mode():
            output = model.generate(**generate_kwargs)
        decoded = processor.decode(output.sequences, skip_special_tokens=True)
        if isinstance(decoded, list):
            return decoded[0] if decoded else ""
        return str(decoded)

    transcript, transcribe_seconds = _time_call(_run)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcript, load_seconds, transcribe_seconds


def _make_result(model: str, audio_seconds: float, load_seconds: float, transcribe_seconds: float, text: str) -> BenchResult:
    return BenchResult(
        model=model,
        audio_seconds=audio_seconds,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        realtime_factor=audio_seconds / transcribe_seconds if transcribe_seconds > 0 else 0.0,
        transcript_chars=len(text),
        transcript_preview=text.replace("\n", " ")[:180],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio file to benchmark, e.g. an MP3 from data/audio_recall.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--ctc-model", default=CTC_MODEL_ID)
    parser.add_argument("--nemotron-model", default=NEMOTRON_MODEL_ID)
    parser.add_argument("--ctc-chunk-seconds", type=float, default=240.0)
    parser.add_argument("--ctc-batch-chunks", type=int, default=4)
    parser.add_argument("--nemotron-lookahead", type=int, default=13, help="13 corresponds to [70, 13] / 1.12s.")
    parser.add_argument(
        "--only",
        choices=["both", "ctc", "nemotron"],
        default="both",
        help="Run one model or both models.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    audio = load_audio_16k(args.audio)
    audio_seconds = len(audio) / SAMPLE_RATE
    print(f"audio_file={args.audio}")
    print(f"audio_seconds={audio_seconds:.2f} ({audio_seconds / 60:.2f} min)")
    print(f"device={args.device}")

    results: list[BenchResult] = []
    if args.only in {"both", "ctc"}:
        print(f"\nRunning {args.ctc_model} ...", flush=True)
        text, load_s, transcribe_s = transcribe_ctc(
            audio,
            model_id=args.ctc_model,
            device=args.device,
            chunk_seconds=args.ctc_chunk_seconds,
            batch_chunks=args.ctc_batch_chunks,
        )
        results.append(_make_result(args.ctc_model, audio_seconds, load_s, transcribe_s, text))

    if args.only in {"both", "nemotron"}:
        print(f"\nRunning {args.nemotron_model} with lookahead={args.nemotron_lookahead} ...", flush=True)
        text, load_s, transcribe_s = transcribe_nemotron(
            audio,
            model_id=args.nemotron_model,
            device=args.device,
            lookahead=args.nemotron_lookahead,
        )
        results.append(_make_result(args.nemotron_model, audio_seconds, load_s, transcribe_s, text))

    print("\nResults")
    print("model,load_s,transcribe_s,realtime_factor,chars,preview")
    for result in results:
        print(
            f"{result.model},"
            f"{result.load_seconds:.2f},"
            f"{result.transcribe_seconds:.2f},"
            f"{result.realtime_factor:.2f}x,"
            f"{result.transcript_chars},"
            f"{json.dumps(result.transcript_preview)}"
        )

    if len(results) == 2:
        faster = results[0].transcribe_seconds / results[1].transcribe_seconds
        if faster < 1:
            print(f"\n{results[0].model} was {1 / faster:.2f}x faster than {results[1].model} for transcription.")
        else:
            print(f"\n{results[1].model} was {faster:.2f}x faster than {results[0].model} for transcription.")

    if args.json_out:
        args.json_out.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
