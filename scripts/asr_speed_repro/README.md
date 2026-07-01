# Minimal ASR Speed Reproducer

This is a standalone comparison of:

- `nvidia/parakeet-ctc-1.1b`
- `nvidia/nemotron-speech-streaming-en-0.6b`

It does not use the NeMo-Retriever pipeline, Ray, LanceDB, retrieval, or embedding. It only measures model load time and transcription time for one audio file.

## Setup

From the repo root:

```bash
bash scripts/asr_speed_repro/setup_env.sh
source .venv-asr-speed/bin/activate
```

The setup requires `ffmpeg` on the system path for MP3 decoding.

## Run

```bash
python scripts/asr_speed_repro/bench_asr_speed.py \
  /localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3
```

Optional JSON output:

```bash
python scripts/asr_speed_repro/bench_asr_speed.py \
  /localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3 \
  --json-out /tmp/asr_speed_results.json
```

## Notes

- The CTC model uses large offline chunks (`--ctc-chunk-seconds 240`, batched by `--ctc-batch-chunks 4`).
- The Nemotron model uses cache-aware streaming with `--nemotron-lookahead 13`, corresponding to `[70, 13]` / `1.12s` chunks.
- First runs may include Hugging Face download/cache overhead in `load_s`. Compare `transcribe_s` and `realtime_factor` for the model speed difference.

## Optional: NeMo Toolkit Test

Use a separate environment so the Transformers repro remains untouched:

```bash
bash scripts/asr_speed_repro/setup_nemo_env.sh
source .venv-nemo-asr-speed/bin/activate
python scripts/asr_speed_repro/bench_nemo_toolkit_speed.py \
  /localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3 \
  --json-out scripts/asr_speed_repro/nemo_toolkit_example_results.json
```

The NeMo setup uses Python 3.10 by default when `uv` is available because some NeMo ASR dependencies are not compatible with Python 3.12. Override with `NEMO_ASR_PYTHON_VERSION=...` if needed.

This tests the direct NeMo Toolkit path:

```python
nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
).transcribe([...])
```

Observed on `An_Introduction_to_DBRX.mp3`:

- Transformers CTC: `1.26s` transcription, `851x` realtime.
- Transformers Nemotron: `45.77s` transcription, `23x` realtime.
- NeMo Toolkit Nemotron: `4.32s` transcription, `247x` realtime.

So the NeMo Toolkit path was about `10.6x` faster than Transformers for Nemotron on this file, but still about `3.4x` slower than CTC.
