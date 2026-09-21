# Reproduce the Nemotron 3.5 comparison

This guide runs the end-to-end comparison between the deployed Nemotron 3 1B
NVFP4 model and the local Nemotron 3.5 1B and 8B checkpoints. The launcher
configures all ViDoRe v3 and BO767 runs, uses vLLM, captures GPU memory, and
writes a combined CSV report.

## Before you begin

Use a Linux system with an NVIDIA GPU, CUDA 13.x, Python 3.12, and `uv`.
The original 8B run peaked at approximately 73 GiB of GPU memory. Use an idle
GPU so unrelated processes do not distort the peak memory measurement.

You also need:

- Access to the Nemotron 3 model on Hugging Face.
- An `HF_TOKEN` that can download Hugging Face model and ViDoRe data.
- The local `nemotron-3.5-1b` and `nemotron-3.5-8b` checkpoint directories.
- The BO767 PDF corpus and query CSV.
- The eight ViDoRe v3 PDF directories shown below.

## Set up the environment

Clone the share branch and enter the repository:

```bash
git clone --branch share/nemotron-35-reproduction \
  https://github.com/ChrisJar/nv-ingest.git
cd nv-ingest
```

Create a Python 3.12 environment and install the local GPU and benchmark
dependencies:

```bash
uv python install 3.12
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install -e "./nemo_retriever[local,benchmarks]"
```

If the installed PyTorch build does not support your CUDA 13 environment,
install the CUDA 13 wheels used by this checkout:

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Set your Hugging Face token:

```bash
export HF_TOKEN=<your-token>
```

## Arrange the data and checkpoints

The launcher expects this data layout. Each ViDoRe PDF filename must be its
Hugging Face `corpus_id`, such as `157.pdf`.

```text
/datasets/nemotron-comparison/
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

Place the checkpoints under one directory:

```text
/models/
|-- nemotron-3.5-1b/
`-- nemotron-3.5-8b/
```

## Validate the configuration

Run the configuration-only check first. It does not load models or perform
ingestion:

```bash
scripts/run_nemotron_35_comparison.sh \
  --data-root /datasets/nemotron-comparison \
  --checkpoint-root /models \
  --dry-run
```

The command checks the required directories, creates the harness dataset YAML,
finds `retriever-harness`, and resolves all eight benchmark arms.

## Run the comparison

Remove `--dry-run` to start the full comparison:

```bash
scripts/run_nemotron_35_comparison.sh \
  --data-root /datasets/nemotron-comparison \
  --checkpoint-root /models
```

By default, results are written to
`artifacts/nemotron_35_reproduction/`. Use `--output-dir` to select another
location. The launcher refuses to overwrite a previously created arm. Choose a
new output directory when repeating the benchmark.

After the run finishes, open `summary.csv` for the aggregate nDCG@10,
Recall@5, Recall@10, throughput, and peak GPU memory measurements. Each arm
also contains its exact command, resolved configuration, logs, rankings, and
per-GPU samples.

## Understand the comparison

The benchmark compares the deployed Nemotron 3 1B NVFP4 model with Nemotron
3.5 1B and 8B. There is no Nemotron 3 8B checkpoint in this script.

The text-only arms rerun extraction independently for each model. OCR output
can therefore differ between arms. Treat the results as end-to-end system
measurements, not as proof that frozen text encoder weights produce different
vectors. A controlled encoder comparison must reuse identical extracted text,
prefixes, and queries.
