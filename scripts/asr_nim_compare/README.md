# Parakeet CTC vs TDT v2 Retriever Recall Reproducer

This reproducer runs the same full NeMo-Retriever audio recall pipeline against
two local Riva-compatible ASR NIMs:

- Parakeet CTC 1.1B NIM: `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0`
- Parakeet TDT 0.6B v2 NIM: `nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.2.0`

The only intended difference between the two runs is the ASR NIM behind
`AUDIO_GRPC_ENDPOINT`. Both runs use:

- `retriever pipeline run`
- `--run-mode batch`
- full audio recall dataset
- `--evaluation-mode audio_recall`
- `--recall-match-mode audio_segment`
- `--audio-match-tolerance-secs 2.0`
- `--segment-audio`
- VL embedding model with the default local `vllm` backend

## 0. Prerequisites

You need:

- Linux host with Docker and NVIDIA container runtime.
- One NVIDIA GPU available for the ASR NIM.
- `NGC_API_KEY` for pulling/running NIM containers.
- `ffmpeg` and `ffprobe` on the host.
- `uv` for Python environment creation.
- A NeMo-Retriever checkout.
- The full audio recall dataset and ground-truth CSV.

Set paths for the dataset and ground truth:

```bash
export AUDIO_RECALL_DIR=/path/to/audio_recall
export AUDIO_RECALL_GT=/path/to/video_retrieval_eval_gt_audio_only.csv
```

Confirm the inputs exist:

```bash
ls -la "${AUDIO_RECALL_DIR}"
ls -la "${AUDIO_RECALL_GT}"
ffmpeg -version
ffprobe -version
```

## 1. Install NeMo Retriever

From the root of the `NeMo-Retriever` checkout:

```bash
uv python install 3.12
uv venv .venv-audio-recall --python 3.12
source .venv-audio-recall/bin/activate

# Editable install from this checkout. This includes local embedding and audio
# media dependencies needed for the recall command below.
uv pip install -e "./nemo_retriever[local,multimedia]"

retriever --help
retriever pipeline run --help
```

If you are installing a published wheel instead of a local checkout, use:

```bash
uv pip install "nemo-retriever[local,multimedia]"
```

## 2. Authenticate to NGC

```bash
export NGC_API_KEY=<your-ngc-api-key>
docker login nvcr.io -u '$oauthtoken' -p "${NGC_API_KEY}"
```

Keep `NGC_API_KEY` available for `docker run`. The Retriever pipeline commands
below intentionally unset `NVIDIA_API_KEY` and `AUDIO_FUNCTION_ID` so stale
hosted-endpoint credentials do not affect the local-NIM run.

## 3. Start the CTC 1.1B NIM

Use one terminal for the NIM container:

```bash
docker run --rm --name parakeet-ctc-1-1b \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -e NIM_TAGS_SELECTOR="name=parakeet-1-1b-ctc-en-us,mode=ofl,vad=default,diarizer=disabled" \
  -p 9000:9000 \
  -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0
```

In another terminal, wait for the NIM readiness endpoint:

```bash
until curl -fsS http://localhost:9000/v1/health/ready; do
  echo "waiting for CTC NIM..."
  sleep 15
done
```

If startup fails with a cuDNN library error on your host, retry the same
`docker run` command with this additional environment variable:

```bash
-e LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/opt/tritonserver/backends/pytorch:/opt/tritonserver/lib:/opt/riva/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu
```

## 4. Run CTC Full Audio Recall

Run this from the `NeMo-Retriever` checkout root:

```bash
source .venv-audio-recall/bin/activate
mkdir -p runtime_metrics intermediate_segments

env -u NVIDIA_API_KEY -u AUDIO_FUNCTION_ID \
  AUDIO_GRPC_ENDPOINT=localhost:50051 \
  retriever pipeline run "${AUDIO_RECALL_DIR}" \
    --run-mode batch \
    --input-type audio \
    --evaluation-mode audio_recall \
    --query-csv "${AUDIO_RECALL_GT}" \
    --recall-match-mode audio_segment \
    --audio-match-tolerance-secs 2.0 \
    --segment-audio \
    --audio-split-type time \
    --audio-split-interval 36000 \
    --embed-model-name "nvidia/llama-nemotron-embed-vl-1b-v2" \
    --embed-modality text \
    --embed-granularity element \
    --vdb-kwargs-json '{"uri":"lancedb_ctc","table_name":"nv-ingest","overwrite":true}' \
    --save-intermediate intermediate_segments/ctc_1_1b_nim_full_vllm \
    --runtime-metrics-dir runtime_metrics \
    --runtime-metrics-prefix ctc_1_1b_nim_full_vllm
```

This writes CTC rows to `lancedb_ctc/nv-ingest`, so they remain available after
the TDT run.

Stop the CTC NIM before starting TDT, since both examples use the same GPU and
host ports:

```bash
docker stop parakeet-ctc-1-1b
ray stop --force || true
```

## 5. Start the TDT 0.6B v2 NIM

Use the NIM terminal again:

```bash
docker run --rm --name parakeet-tdt-v2 \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.2.0
```

Wait for readiness:

```bash
until curl -fsS http://localhost:9000/v1/health/ready; do
  echo "waiting for TDT v2 NIM..."
  sleep 15
done
```

The TDT NIM can take a long time to become ready on first launch because it may
download or build an optimized engine.

## 6. Run TDT v2 Full Audio Recall

Run the same Retriever pipeline command with the TDT LanceDB URI and metrics
prefix:

```bash
source .venv-audio-recall/bin/activate
mkdir -p runtime_metrics intermediate_segments

env -u NVIDIA_API_KEY -u AUDIO_FUNCTION_ID \
  AUDIO_GRPC_ENDPOINT=localhost:50051 \
  retriever pipeline run "${AUDIO_RECALL_DIR}" \
    --run-mode batch \
    --input-type audio \
    --evaluation-mode audio_recall \
    --query-csv "${AUDIO_RECALL_GT}" \
    --recall-match-mode audio_segment \
    --audio-match-tolerance-secs 2.0 \
    --segment-audio \
    --audio-split-type time \
    --audio-split-interval 36000 \
    --embed-model-name "nvidia/llama-nemotron-embed-vl-1b-v2" \
    --embed-modality text \
    --embed-granularity element \
    --vdb-kwargs-json '{"uri":"lancedb_tdt","table_name":"nv-ingest","overwrite":true}' \
    --save-intermediate intermediate_segments/tdt_0_6b_v2_nim_full_vllm \
    --runtime-metrics-dir runtime_metrics \
    --runtime-metrics-prefix tdt_0_6b_v2_nim_full_vllm
```

Stop the TDT NIM when done:

```bash
docker stop parakeet-tdt-v2
ray stop --force || true
```

## 7. Compare Runtime Summaries

Each pipeline run writes a JSON summary in `runtime_metrics`:

- `runtime_metrics/ctc_1_1b_nim_full_vllm.runtime.summary.json`
- `runtime_metrics/tdt_0_6b_v2_nim_full_vllm.runtime.summary.json`

Print the key recall and runtime metrics:

```bash
python - <<'PY'
import json
from pathlib import Path

for label, path in [
    ("CTC 1.1B", Path("runtime_metrics/ctc_1_1b_nim_full_vllm.runtime.summary.json")),
    ("TDT 0.6B v2", Path("runtime_metrics/tdt_0_6b_v2_nim_full_vllm.runtime.summary.json")),
]:
    data = json.loads(path.read_text())
    metrics = data.get("evaluation_metrics", {})
    print(label)
    print("  recall@1:", metrics.get("recall@1"))
    print("  recall@5:", metrics.get("recall@5"))
    print("  recall@10:", metrics.get("recall@10"))
    print("  ingestion_only_secs:", data.get("ingestion_only_secs"))
    print("  total_secs:", data.get("total_secs"))
    print("  num_rows:", data.get("num_rows"))
PY
```

## 8. Raw Segment Outputs

Each audio segment becomes one pipeline row when `--segment-audio` is enabled.
The commands above use `--save-intermediate`, so Retriever writes those rows to
`extraction.parquet` before/alongside VDB upload. The transcript is stored in
the `text` column, and the segment timing is stored in
`metadata.segment_start_seconds` and `metadata.segment_end_seconds`.

The raw segment Parquet files are:

- `intermediate_segments/ctc_1_1b_nim_full_vllm/extraction.parquet`
- `intermediate_segments/tdt_0_6b_v2_nim_full_vllm/extraction.parquet`

## 9. Observed Single-H100 Results

These results were measured on one H100 running both the ASR NIM and the local
`vllm` embedding backend.

- CTC 1.1B NIM produced 31,399 rows. Recall was `0.5548` at 1, `0.7720` at 5,
  and `0.8258` at 10. Ingestion-only time was `903.04s`; total time was
  `924.42s`.
- TDT 0.6B v2 NIM produced 33,815 rows. Recall was `0.4839` at 1, `0.6989` at
  5, and `0.7376` at 10. Ingestion-only time was `1486.69s`; total time was
  `1508.24s`.
- In this setup, TDT produced more segment rows but took about `1.65x` as long
  for ingestion and had lower audio-segment recall at the same `2.0s` midpoint
  tolerance.

## Notes

- These commands compare the full Retriever audio recall workflow, not just ASR
  request latency.
- The `AUDIO_GRPC_ENDPOINT=localhost:50051` variable is what points Retriever at
  the currently running local NIM.
- The command omits `--local-ingest-embed-backend` because the default local
  ingest backend is `vllm`; both CTC and TDT runs use that same setting.
- `--audio-split-interval 36000` keeps Retriever from pre-splitting each audio
  file into many short time chunks. The ASR NIM still returns timestamped
  utterance segments, and `--segment-audio` turns those into Retriever rows.
- If the pipeline fails with gRPC `UNAVAILABLE`, the NIM is usually not fully
  ready yet even if the port is open. Wait longer and rerun the same pipeline
  command.
