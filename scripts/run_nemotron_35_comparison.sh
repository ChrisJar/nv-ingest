#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CORE_SCRIPT="${SCRIPT_DIR}/reproduce_nemotron_35_comparison.sh"
DATA_ROOT=""
CHECKPOINT_ROOT="${REPO_ROOT}/../model_checkpoints"
OUTPUT_DIR="${REPO_ROOT}/artifacts/nemotron_35_reproduction"
HARNESS_BIN=""
DRY_RUN=false

usage() {
  cat <<'HELP'
Run the complete Nemotron 3 versus Nemotron 3.5 comparison.

Usage:
  scripts/run_nemotron_35_comparison.sh --data-root DIR [options]

Required:
  --data-root DIR          Directory containing BO767 and ViDoRe data.

Options:
  --checkpoint-root DIR    Directory containing nemotron-3.5-1b and
                           nemotron-3.5-8b. Default: ../model_checkpoints
  --output-dir DIR         Results directory. Default: artifacts/nemotron_35_reproduction
  --harness-bin FILE       Explicit retriever-harness executable.
  --dry-run                Validate configuration without loading models.
  -h, --help               Show this help.

Full setup guide: NEMOTRON_35_COMPARISON.md

Expected data layout:
  DATA_ROOT/
  |-- bo767/                         # 767 PDFs
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

Example:
  scripts/run_nemotron_35_comparison.sh \
    --data-root /datasets/nemotron-comparison \
    --checkpoint-root /models

Start with --dry-run. Full execution is long-running and the 8B arms require
approximately 73 GiB of available GPU memory based on the original run.
HELP
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --data-root)
      (($# >= 2)) || die "--data-root requires a value"
      DATA_ROOT="$2"
      shift 2
      ;;
    --checkpoint-root)
      (($# >= 2)) || die "--checkpoint-root requires a value"
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --harness-bin)
      (($# >= 2)) || die "--harness-bin requires a value"
      HARNESS_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${DATA_ROOT}" ]] || die "--data-root is required; run with --help for the expected layout"
[[ -x "${CORE_SCRIPT}" ]] || die "core reproduction script is missing: ${CORE_SCRIPT}"
[[ -d "${DATA_ROOT}" ]] || die "data root not found: ${DATA_ROOT}"
[[ -d "${CHECKPOINT_ROOT}" ]] || die "checkpoint root not found: ${CHECKPOINT_ROOT}"

DATA_ROOT="$(cd -- "${DATA_ROOT}" && pwd)"
CHECKPOINT_ROOT="$(cd -- "${CHECKPOINT_ROOT}" && pwd)"
N35_1B_MODEL="${CHECKPOINT_ROOT}/nemotron-3.5-1b"
N35_8B_MODEL="${CHECKPOINT_ROOT}/nemotron-3.5-8b"

[[ -d "${N35_1B_MODEL}" ]] || die "1B checkpoint not found: ${N35_1B_MODEL}"
[[ -d "${N35_8B_MODEL}" ]] || die "8B checkpoint not found: ${N35_8B_MODEL}"
[[ -d "${DATA_ROOT}/bo767" ]] || die "BO767 directory not found: ${DATA_ROOT}/bo767"
[[ -f "${DATA_ROOT}/bo767_query_gt.csv" ]] || die "BO767 query file not found: ${DATA_ROOT}/bo767_query_gt.csv"

VIDORE_DATASETS=(
  vidore_v3_computer_science
  vidore_v3_energy
  vidore_v3_finance_en
  vidore_v3_finance_fr
  vidore_v3_hr
  vidore_v3_industrial
  vidore_v3_pharmaceuticals
  vidore_v3_physics
)
for dataset in "${VIDORE_DATASETS[@]}"; do
  path="${DATA_ROOT}/vidore_v3_corpus_pdf/${dataset}"
  [[ -d "${path}" ]] || die "ViDoRe directory not found: ${path}"
done

if [[ -z "${HARNESS_BIN}" ]]; then
  candidates=(
    "${REPO_ROOT}/retriever/bin/retriever-harness"
    "${REPO_ROOT}/.venv/bin/retriever-harness"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      HARNESS_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${HARNESS_BIN}" ]] && command -v retriever-harness >/dev/null; then
  HARNESS_BIN="$(command -v retriever-harness)"
fi
[[ -n "${HARNESS_BIN}" && -x "${HARNESS_BIN}" ]] || die \
  "retriever-harness was not found; activate the environment or pass --harness-bin"

if [[ "${DRY_RUN}" == false ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi was not found"
  nvidia-smi >/dev/null || die "nvidia-smi could not communicate with the GPU driver"
fi

mkdir -p "${OUTPUT_DIR}/config"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"
DATASET_PATHS="${OUTPUT_DIR}/config/dataset_paths.yaml"

{
  printf '%s\n' 'schema_version: 1' 'datasets:'
  printf '  bo767:\n    path: %s\n    query_file: %s\n' \
    "${DATA_ROOT}/bo767" "${DATA_ROOT}/bo767_query_gt.csv"
  for dataset in "${VIDORE_DATASETS[@]}"; do
    printf '  %s:\n    path: %s\n' \
      "${dataset}" "${DATA_ROOT}/vidore_v3_corpus_pdf/${dataset}"
  done
} >"${DATASET_PATHS}"

printf '%s\n' \
  'Nemotron comparison preflight passed.' \
  "  Data root:       ${DATA_ROOT}" \
  "  1B checkpoint:   ${N35_1B_MODEL}" \
  "  8B checkpoint:   ${N35_8B_MODEL}" \
  "  Harness:         ${HARNESS_BIN}" \
  "  Output:          ${OUTPUT_DIR}" \
  "  Dry run:         ${DRY_RUN}"

command=(
  "${CORE_SCRIPT}"
  --dataset-paths "${DATASET_PATHS}"
  --n35-1b "${N35_1B_MODEL}"
  --n35-8b "${N35_8B_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --harness-bin "${HARNESS_BIN}"
)
if [[ "${DRY_RUN}" == true ]]; then
  command+=(--dry-run)
fi

exec "${command[@]}"
