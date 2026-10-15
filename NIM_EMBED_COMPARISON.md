# Compare two embedding NIM images

This guide runs a full-pipeline embedding comparison with NeMo Retriever
Library (NRL). You supply both NVIDIA NIM images. The script does not include
private image references, model identifiers, or benchmark results.

One invocation runs BO767 and all eight ViDoRe v3 subsets. It performs 26
dataset runs and reports five aggregate groups:

| Dataset | Baseline image | Candidate image |
|---|---|---|
| BO767 | Text-only | Text-only |
| All eight ViDoRe v3 subsets | Text-only | Text-only and text + full-page image |

The baseline must support text embeddings. The candidate must support text
and text + image embeddings through the NIM embeddings API.

## Set up the environment

Use Linux, Python 3.12, `uv`, `curl`, Docker with NVIDIA GPU support, and an NVIDIA GPU
compatible with both images. Leave enough GPU memory for the embedding
container and local extraction models together. Use an idle GPU for meaningful
memory and throughput measurements.

Clone the latest `main` from the official NVIDIA repository and install NRL
with local extraction and benchmark dependencies:

```bash
git clone --branch main --single-branch \
  https://github.com/NVIDIA/NeMo-Retriever.git
cd NeMo-Retriever
uv python install 3.12
uv venv retriever --python 3.12
source retriever/bin/activate
uv sync --project nemo_retriever --active --extra local --extra benchmarks --no-dev
git rev-parse HEAD
```

Record the printed commit when sharing results because `main` changes over
time. Current `main` includes the tokenizer counting fix; no compatibility
branch or local patch is required.

Use `uv sync` so the official checkout's dependency overrides and CUDA package sources
apply. A plain `uv pip install -e` does not apply all of those project settings.
Your NVIDIA driver must support the CUDA runtime selected by that checkout.

Download only the standalone comparison helper from the share repository:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/ChrisJar/nv-ingest/share/nim-embedding-reproduction/scripts/compare_embedding_nims.py \
  --output scripts/compare_embedding_nims.py
```

The helper is separate from NRL. It uses the official installation's Python
API, not library code from the share branch, `retriever-harness`, or local
embedding checkpoints. Review the downloaded script before running it.

Set the credentials and image references in your shell:

```bash
export HF_TOKEN='<your-hugging-face-token>'
export NGC_API_KEY='<your-ngc-api-key>'
export BASELINE_NIM_IMAGE='<baseline-image>'
export CANDIDATE_NIM_IMAGE='<candidate-image>'
```

Your Hugging Face token needs access to the extraction models, public Nemotron
3 tokenizer, and ViDoRe query data. Configure Docker registry authentication
and confirm that you can pull both images before running the comparison.
Do not put credentials or private image references in tracked files.

## Provide the datasets

The script does not download the corpora or convert them into the expected
PDF files. It renders pages during ingestion. Prepare this directory layout:

```text
/datasets/nim-comparison/
|-- bo767/
|-- bo767_query_gt.csv
`-- vidore_v3_corpus_pdf/
    |-- vidore_v3_computer_science/
    |-- vidore_v3_energy/
    |-- vidore_v3_finance_en/
    |-- vidore_v3_finance_fr/
    |-- vidore_v3_hr/
    |-- vidore_v3_industrial/
    |-- vidore_v3_pharmaceuticals/
    `-- vidore_v3_physics/
```

Put the 767 BO767 PDFs directly in `bo767/`. Keep their original filenames so
they match the BO767 query CSV. Each ViDoRe PDF must contain one corpus page
and use its Hugging Face `corpus_id` as the filename, such as `157.pdf`.
Place PDFs directly in each subset directory, not in nested directories.

## Validate the configuration

Run the configuration-only check:

```bash
python scripts/compare_embedding_nims.py \
  --data-root /datasets/nim-comparison \
  --output-dir artifacts/nim_comparison_check \
  --dry-run
```

The dry run validates the data layout and records the comparison plan. It does
not contact Docker or Hugging Face, load NRL models, or run ingestion. It does
not verify GPU compatibility, image access, or service support for paired
text + image input.

You can pass `--baseline-image IMAGE` and `--candidate-image IMAGE` instead of
the environment variables. Neither image has a built-in default.

## Run the comparison

Start the full comparison with a new output directory:

```bash
python scripts/compare_embedding_nims.py \
  --data-root /datasets/nim-comparison \
  --output-dir artifacts/nim_comparison
```

The default output directory is `artifacts/nim_comparison`. The script refuses
to overwrite existing runtime artifacts. Use a new output directory when
repeating the comparison. The example dry run uses a separate directory.

The script pulls and starts one container at a time. By default, it uses GPU
0 and binds local port 18016 to the container's port 8000. Keep that port free.
Use `--gpu` and `--port` to change these values. It discovers the served model
through `/v1/models`, validates the embedding responses, and uses that same
model for both ingestion and queries.

The default worker counts are `--pdf-extract-workers 4` and
`--embed-workers 1`. Both images use the same counts. Change them explicitly
if needed, and keep them fixed across comparisons. These settings control
the NRL pipeline, not the NIM engine's internal batching.

The script stops only the containers it creates. Stopped containers and their
cache volumes remain available for inspection. It does not remove them or
stop unrelated services.

## Keep the benchmark settings aligned

Each arm starts fresh extraction, embedding, indexing, and retrieval
evaluation. All arms use `pdfium_hybrid` extraction at 200 DPI, OCR v2,
structured tables, and Markdown table output.

Dataset-specific settings are fixed as follows:

| Setting | BO767 | ViDoRe |
|---|---|---|
| Extract text, images, tables, and charts | Enabled | Enabled |
| Extract infographics | Disabled | Enabled |
| Embedding granularity | Element | Page |
| Extract page as image | Disabled | Enabled only for candidate text + image |

The multimodal arm embeds text with the full page image, not individual image
crops. Text-only arms do not submit page images to the embedding service.
No reranker is used.

Both containers receive these explicit comparison settings:

```text
NIM_HTTP_API_PORT=8000
NIM_TRITON_LOG_VERBOSE=1
OMP_NUM_THREADS=1
NIM_ENGINE_COUNT=1
NIM_PERFORMANCE_MODE=1
```

`NIM_PERFORMANCE_MODE=1` requires an image that supports it. It is an optional
setting in the checked-in Helm values, not an enabled Helm default. The script
also passes `NGC_API_KEY` without writing its value into the comparison plan.

The script does not override the NIM engine's precision, model profile,
batch size, token budget, or maximum sequence length. Each image selects its
own defaults. The script does not force vLLM; the selected NIM image determines
its inference backend.

NRL still applies its default document input admission limit of 8,192 tokens.
The HTTP client allows 32 concurrent ingestion requests per Ray batch, and
query evaluation submits batches of up to 32 queries. These client-side
settings do not select the NIM engine's internal batch size.

Queries and relevance judgments are frozen once and reused across arms.
Generated queries are bounded to 128 tokens with the pinned public Nemotron
3 BF16 tokenizer. To reuse existing query text exactly, pass
`--frozen-queries-dir DIR`. Provide nine `DATASET_frozen.json` files, one for
`bo767` and each named ViDoRe subset. Each file must contain `queries` as a
list of `{"query_id": "...", "query": "..."}` objects and `qrels` as the
query-to-document relevance mapping.

Retrieval uses dense L2 search, top 10 results, `nprobes=64`, and
`refine_factor=10`. BO767 relevance identifiers include both the source
document and page; ViDoRe identifiers use `corpus_id`.

## Read and protect the results

Open `comparison.md` for the comparison tables and `summary.csv` for the
aggregate metrics. The output also preserves frozen queries, runtime
settings, image and model provenance, service logs, per-dataset metrics,
retrieval rankings, and GPU samples.

The CSV includes nDCG@10, Recall@5, Recall@10, pages per second, and peak GPU
memory in MiB. ViDoRe quality metrics are unweighted averages across the eight
subsets. Its throughput is total pages divided by total ingestion time.

Ingestion throughput measures the full NRL extraction, embedding, and
indexing pipeline. It excludes container startup, image downloads, and query
evaluation. It is not an isolated NIM saturation benchmark. Different worker
counts, extraction models, and hardware can affect the measured rate.

GPU samples cover whole-device memory and utilization during ingestion,
including the embedding container and extraction models. They do not measure
only embedding model weights. Sampling failures are recorded in each
dataset's `gpu_sampling.json`, not silently replaced with zero usage. Missing
memory samples remain blank in the reports.

The corrected workflow does not guarantee historical scores or bitwise
reproduction of results produced with older tokenizer or document-ID
handling. Independent extraction can also produce different text across
arms. To test encoder parity, use identical extracted text and query inputs
in a separate controlled test.

Treat generated artifacts as private when using private images. They can
contain image references, served model identifiers, dataset content, and
benchmark results. Review and redact them before sharing. Do not commit the
output directory, service logs, or credentials.
