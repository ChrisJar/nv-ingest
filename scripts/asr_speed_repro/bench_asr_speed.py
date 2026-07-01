#!/usr/bin/env python3
"""Minimal Transformers ASR speed comparison on one audio file."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


SAMPLE_RATE = 16_000
CTC_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
NEMOTRON_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTC_CHUNK_SECONDS = 240.0
CTC_BATCH_CHUNKS = 4
NEMOTRON_LOOKAHEAD = 13


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


def _model_dtype() -> torch.dtype | str:
    return torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_available() else "auto"


def _load_ctc():
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(CTC_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(
        CTC_MODEL_ID,
        dtype=_model_dtype(),
        device_map=DEVICE,
    )
    model.eval()
    return processor, model


def transcribe_ctc(audio: np.ndarray) -> tuple[str, float, float]:
    """Transcribe with Parakeet CTC using large offline chunks."""
    (processor, model), load_seconds = _time_call(_load_ctc)

    chunk_samples = max(1, int(CTC_CHUNK_SECONDS * SAMPLE_RATE))
    chunks = [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples)]
    chunks = [chunk for chunk in chunks if chunk.size]

    def _run() -> str:
        texts: list[str] = []
        model_device = _model_device(model)
        model_dtype = getattr(model, "dtype", torch.float32)
        for start in range(0, len(chunks), CTC_BATCH_CHUNKS):
            batch = chunks[start : start + CTC_BATCH_CHUNKS]
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


def _load_nemotron():
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(NEMOTRON_MODEL_ID)
    processor.set_num_lookahead_tokens(NEMOTRON_LOOKAHEAD)
    model = AutoModelForRNNT.from_pretrained(
        NEMOTRON_MODEL_ID,
        dtype=_model_dtype(),
        device_map=DEVICE,
    )
    model.eval()
    return processor, model


def _nemotron_feature_stream(audio: np.ndarray, processor, model):
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


def transcribe_nemotron(audio: np.ndarray) -> tuple[str, float, float]:
    """Transcribe with Nemotron cache-aware streaming ASR."""
    (processor, model), load_seconds = _time_call(_load_nemotron)

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
        generate_kwargs["num_lookahead_tokens"] = NEMOTRON_LOOKAHEAD
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    audio = load_audio_16k(args.audio)
    audio_seconds = len(audio) / SAMPLE_RATE
    print(f"Audio: {args.audio}")
    print(f"Duration: {audio_seconds:.2f}s ({audio_seconds / 60:.2f} min)")
    print(f"Device: {DEVICE}")

    print(f"\nRunning {CTC_MODEL_ID} with Transformers ...", flush=True)
    ctc_text, ctc_load, ctc_time = transcribe_ctc(audio)

    print(f"\nRunning {NEMOTRON_MODEL_ID} with Transformers ...", flush=True)
    nemotron_text, nemotron_load, nemotron_time = transcribe_nemotron(audio)

    print("\nResults")
    print(f"CTC load: {ctc_load:.2f}s")
    print(f"CTC transcription: {ctc_time:.2f}s ({audio_seconds / ctc_time:.2f}x realtime)")
    print(f"CTC transcript chars: {len(ctc_text)}")
    print(f"Nemotron load: {nemotron_load:.2f}s")
    print(f"Nemotron transcription: {nemotron_time:.2f}s ({audio_seconds / nemotron_time:.2f}x realtime)")
    print(f"Nemotron transcript chars: {len(nemotron_text)}")
    print(f"CTC speedup vs Nemotron Transformers: {nemotron_time / ctc_time:.2f}x")


if __name__ == "__main__":
    main()
