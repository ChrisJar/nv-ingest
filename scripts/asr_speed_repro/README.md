# Minimal ASR Speed Reproducer

Standalone timing scripts for one audio file. They do not use NeMo-Retriever,
Ray, LanceDB, retrieval, or embedding.

Example file used for the numbers below:

`/localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3`

That file is about 17 MB and 17.8 minutes long.

## Transformers: CTC vs Nemotron

```bash
bash scripts/asr_speed_repro/setup_env.sh
source .venv-asr-speed/bin/activate
python scripts/asr_speed_repro/bench_asr_speed.py \
  /localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3
```

## NeMo Toolkit: Nemotron

```bash
bash scripts/asr_speed_repro/setup_nemo_env.sh
source .venv-nemo-asr-speed/bin/activate
python scripts/asr_speed_repro/bench_nemo_toolkit_speed.py \
  /localhome/local-cjarrett/data/audio_recall/An_Introduction_to_DBRX.mp3
```

The NeMo setup uses Python 3.10 by default when `uv` is available because some
NeMo ASR dependencies did not install cleanly on Python 3.12.

## Observed Output

Observed on `An_Introduction_to_DBRX.mp3`:

- Transformers CTC: `1.26s` transcription, `851x` realtime.
- Transformers Nemotron: `45.77s` transcription, `23x` realtime.
- NeMo Toolkit Nemotron: `4.32s` transcription, `247x` realtime.

So the NeMo Toolkit path was about `10.6x` faster than Transformers for Nemotron on this file, but still about `3.4x` slower than CTC.

First runs may include Hugging Face download/cache overhead in the load time.
Compare transcription time for the model runtime difference.
