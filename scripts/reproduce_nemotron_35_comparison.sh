#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

N3_1B_MODEL="nvidia/Nemotron-3-Embed-1B-NVFP4"
N3_8B_MODEL="nvidia/Nemotron-3-Embed-8B-BF16"
N35_1B_MODEL="${REPO_ROOT}/../model_checkpoints/nemotron-3.5-1b"
N35_8B_MODEL="${REPO_ROOT}/../model_checkpoints/nemotron-3.5-8b"
DATASET_PATHS=""
OUTPUT_DIR="${REPO_ROOT}/artifacts/nemotron_35_reproduction"
HARNESS_BIN="${REPO_ROOT}/retriever/bin/retriever-harness"
DRY_RUN=false
INCLUDE_MATCHED_8B=false

usage() {
  cat <<'EOF'
Reproduce the Nemotron 3 versus Nemotron 3.5 retrieval comparison.

Usage:
  scripts/reproduce_nemotron_35_comparison.sh --dataset-paths FILE [options]

Required:
  --dataset-paths FILE       Harness dataset-path mapping YAML.

Model options:
  --n3-1b MODEL              Deployed baseline model.
                              Default: nvidia/Nemotron-3-Embed-1B-NVFP4
  --n35-1b MODEL             Nemotron 3.5 1B checkpoint or Hub ID.
  --n3-8b MODEL              Matched Nemotron 3 8B checkpoint or Hub ID.
                              Default: nvidia/Nemotron-3-Embed-8B-BF16
  --n35-8b MODEL             Nemotron 3.5 8B checkpoint or Hub ID.
  --include-matched-8b       Also run N3 8B text-only as an encoder-family control.

Execution options:
  --output-dir DIR           Output root. Must not already contain an arm directory.
  --harness-bin FILE         retriever-harness executable.
  --dry-run                  Resolve every configuration without ingesting or querying.
  -h, --help                 Show this help.

The default arms reproduce the reported product comparison:
  * N3 1B NVFP4: ViDoRe text-only and BO767
  * N3.5 1B: ViDoRe text-only, ViDoRe text+image, and BO767
  * N3.5 8B: ViDoRe text-only, ViDoRe text+image, and BO767

BO767 extracts text, images, tables, and charts; excludes infographics; uses
table structure; and embeds at element granularity. ViDoRe extracts all of
those plus infographics, uses table structure, and embeds at page granularity.
Its text+image arms additionally extract each page as an image.

Important: this is an end-to-end system reproduction. Each arm performs its
own extraction, so text may differ between arms. It is not, by itself, a
controlled proof that two frozen text encoders produce identical vectors.

Example dataset-paths YAML:
  schema_version: 1
  datasets:
    bo767:
      path: /datasets/bo767
      query_file: /datasets/bo767_query_gt.csv
    vidore_v3_computer_science:
      path: /datasets/vidore_v3/vidore_v3_computer_science
    vidore_v3_energy:
      path: /datasets/vidore_v3/vidore_v3_energy
    vidore_v3_finance_en:
      path: /datasets/vidore_v3/vidore_v3_finance_en
    vidore_v3_finance_fr:
      path: /datasets/vidore_v3/vidore_v3_finance_fr
    vidore_v3_hr:
      path: /datasets/vidore_v3/vidore_v3_hr
    vidore_v3_industrial:
      path: /datasets/vidore_v3/vidore_v3_industrial
    vidore_v3_pharmaceuticals:
      path: /datasets/vidore_v3/vidore_v3_pharmaceuticals
    vidore_v3_physics:
      path: /datasets/vidore_v3/vidore_v3_physics
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --dataset-paths)
      (($# >= 2)) || die "--dataset-paths requires a value"
      DATASET_PATHS="$2"
      shift 2
      ;;
    --n3-1b)
      (($# >= 2)) || die "--n3-1b requires a value"
      N3_1B_MODEL="$2"
      shift 2
      ;;
    --n3-8b)
      (($# >= 2)) || die "--n3-8b requires a value"
      N3_8B_MODEL="$2"
      shift 2
      ;;
    --n35-1b)
      (($# >= 2)) || die "--n35-1b requires a value"
      N35_1B_MODEL="$2"
      shift 2
      ;;
    --n35-8b)
      (($# >= 2)) || die "--n35-8b requires a value"
      N35_8B_MODEL="$2"
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
    --include-matched-8b)
      INCLUDE_MATCHED_8B=true
      shift
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

[[ -n "${DATASET_PATHS}" ]] || die "--dataset-paths is required"
[[ -f "${DATASET_PATHS}" ]] || die "dataset paths file not found: ${DATASET_PATHS}"
[[ -x "${HARNESS_BIN}" ]] || die "harness executable not found or not executable: ${HARNESS_BIN}"
command -v python3 >/dev/null || die "python3 is required"
if [[ "${DRY_RUN}" == false ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required for GPU memory sampling"
fi

validate_local_model() {
  local model="$1"
  if [[ "${model}" == /* || "${model}" == ./* || "${model}" == ../* ]]; then
    [[ -d "${model}" ]] || die "local model directory not found: ${model}"
  fi
}

validate_local_model "${N3_1B_MODEL}"
validate_local_model "${N35_1B_MODEL}"
validate_local_model "${N35_8B_MODEL}"
if [[ "${INCLUDE_MATCHED_8B}" == true ]]; then
  validate_local_model "${N3_8B_MODEL}"
fi

mkdir -p "${OUTPUT_DIR}/config/runfiles"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"
DATASET_PATHS="$(cd -- "$(dirname -- "${DATASET_PATHS}")" && pwd)/$(basename -- "${DATASET_PATHS}")"

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
VIDORE_RUNFILES=()
for dataset in "${VIDORE_DATASETS[@]}"; do
  runfile="${OUTPUT_DIR}/config/runfiles/${dataset}_beir.json"
  printf '{"schema_version":1,"name":"%s_beir","benchmark":"%s_beir","mode":"batch","set":{}}\n' \
    "${dataset}" "${dataset}" >"${runfile}"
  VIDORE_RUNFILES+=("${runfile}")
done
BO_RUNFILE="${OUTPUT_DIR}/config/runfiles/bo767_beir.json"
printf '%s\n' '{"schema_version":1,"name":"bo767_beir","benchmark":"bo767_beir","mode":"batch","set":{}}' \
  >"${BO_RUNFILE}"

cat >"${OUTPUT_DIR}/manifest.txt" <<EOF
repository=${REPO_ROOT}
git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)
dataset_paths=${DATASET_PATHS}
n3_1b=${N3_1B_MODEL}
n35_1b=${N35_1B_MODEL}
n3_8b=${N3_8B_MODEL}
n35_8b=${N35_8B_MODEL}
include_matched_8b=${INCLUDE_MATCHED_8B}
dry_run=${DRY_RUN}
EOF

MONITOR_PID=""
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap stop_monitor EXIT INT TERM

start_monitor() {
  local output="$1"
  local session="$2"
  (
    printf '%s\n' 'timestamp_utc,session,gpu_index,memory_used_mib,memory_total_mib,gpu_util_percent,power_watts'
    while true; do
      timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
      while IFS= read -r sample; do
        printf '%s,%s,%s\n' "${timestamp}" "${session}" "${sample// /}"
      done < <(nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits)
      sleep 0.25
    done
  ) >"${output}" &
  MONITOR_PID=$!
}

run_session() {
  local session="$1"
  local model="$2"
  local modality="$3"
  local granularity="$4"
  local page_as_image="$5"
  local infographics="$6"
  local dataset_kind="$7"
  shift 7
  local -a runfiles=("$@")
  local session_dir="${OUTPUT_DIR}/${session}"

  [[ ! -e "${session_dir}" ]] || die "refusing to overwrite existing arm: ${session_dir}"
  mkdir -p "${session_dir}"

  local -a command=(
    "${HARNESS_BIN}" run-files
    --output-dir "${session_dir}"
    --session-name "${session}"
    --dataset-paths "${DATASET_PATHS}"
    --mode batch
    --set ingest.extract.method=pdfium_hybrid
    --set ingest.extract.dpi=200
    --set ingest.extract.extract_text=true
    --set ingest.extract.extract_images=true
    --set ingest.extract.extract_tables=true
    --set ingest.extract.extract_charts=true
    --set ingest.extract.extract_infographics="${infographics}"
    --set ingest.extract.extract_page_as_image="${page_as_image}"
    --set ingest.extract.use_page_elements=true
    --set ingest.extract.use_table_structure=true
    --set ingest.extract.ocr_version=v2
    --set ingest.extract.table_output_format=markdown
    --set ingest.caption.enabled=false
    --set ingest.dedup.enabled=false
    --set ingest.chunk.enabled=false
    --set ingest.storage.index_mode=dense
    --set ingest.embed.local_ingest_embed_backend=vllm
    --set ingest.embed.embed_model_name="${model}"
    --set ingest.embed.embed_modality="${modality}"
    --set ingest.embed.embed_granularity="${granularity}"
    --set query.embed_model_name="${model}"
    --set query.top_k=10
    --set query.candidate_k=null
    --set query.page_dedup=false
    --set query.retrieval_mode=dense
    --set query.rerank=false
  )
  if [[ "${dataset_kind}" == vidore ]]; then
    # ViDoRe qrels identify individual pages by corpus_id. Using pdf_page or
    # pdf_basename silently compares different identifier namespaces.
    command+=(--set evaluation.doc_id_field=corpus_id)
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    command+=(--dry-run)
  fi
  command+=("${runfiles[@]}")

  printf '\n[%s] model=%s modality=%s granularity=%s dataset=%s\n' \
    "${session}" "${model}" "${modality}" "${granularity}" "${dataset_kind}"
  printf '%q ' "${command[@]}" >"${session_dir}/command.txt"
  printf '\n' >>"${session_dir}/command.txt"

  if [[ "${DRY_RUN}" == false ]]; then
    start_monitor "${session_dir}/gpu_memory.csv" "${session}"
  fi
  set +e
  "${command[@]}" 2>&1 | tee "${session_dir}/harness.log"
  status=${PIPESTATUS[0]}
  set -e
  stop_monitor
  printf '%s\n' "${status}" >"${session_dir}/exit_code.txt"
  ((status == 0)) || die "session ${session} failed with exit code ${status}"
}

run_vidore_text() {
  run_session "$1" "$2" text page false true vidore "${VIDORE_RUNFILES[@]}"
}

run_vidore_text_image() {
  run_session "$1" "$2" text_image page true true vidore "${VIDORE_RUNFILES[@]}"
}

run_bo() {
  run_session "$1" "$2" text element false false bo767 "${BO_RUNFILE}"
}

# Deployed product baseline.
run_vidore_text n3_1b_vidore_text "${N3_1B_MODEL}"
run_bo n3_1b_bo767 "${N3_1B_MODEL}"

# Nemotron 3.5 1B product candidates.
run_vidore_text n35_1b_vidore_text "${N35_1B_MODEL}"
run_vidore_text_image n35_1b_vidore_text_image "${N35_1B_MODEL}"
run_bo n35_1b_bo767 "${N35_1B_MODEL}"

# Nemotron 3.5 8B product candidates, compared operationally with deployed 1B.
run_vidore_text n35_8b_vidore_text "${N35_8B_MODEL}"
run_vidore_text_image n35_8b_vidore_text_image "${N35_8B_MODEL}"
run_bo n35_8b_bo767 "${N35_8B_MODEL}"

# Optional architecture-matched 8B control for the frozen-text-encoder question.
if [[ "${INCLUDE_MATCHED_8B}" == true ]]; then
  run_vidore_text n3_8b_vidore_text "${N3_8B_MODEL}"
  run_bo n3_8b_bo767 "${N3_8B_MODEL}"
fi

if [[ "${DRY_RUN}" == false ]]; then
  python3 - "${OUTPUT_DIR}" <<'PY'
import csv
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/session_summary.json")):
    session = summary_path.parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = [r for r in summary.get("runs", []) if r.get("success")]
    if not runs:
        continue
    metrics = [r["summary_metrics"] for r in runs]
    pages = sum(float(m.get("pages") or 0) for m in metrics)
    ingest_secs = sum(float(m.get("ingest_secs") or 0) for m in metrics)
    def macro(key):
        values = [float(m[key]) for m in metrics if m.get(key) is not None]
        return sum(values) / len(values) if values else None
    peaks = []
    for gpu_csv in glob.glob(str(summary_path.parent / "gpu_memory.csv")):
        with open(gpu_csv, newline="", encoding="utf-8") as stream:
            peaks.extend(float(r["memory_used_mib"]) for r in csv.DictReader(stream))
    rows.append({
        "arm": session,
        "pages": int(pages),
        "ingest_secs": ingest_secs,
        "pages_per_sec": pages / ingest_secs if ingest_secs else None,
        "ndcg_10_macro": macro("ndcg_10"),
        "recall_5_macro": macro("recall_5"),
        "recall_10_macro": macro("recall_10"),
        "gpu_peak_mib": max(peaks) if peaks else None,
    })

columns = [
    "arm", "pages", "ingest_secs", "pages_per_sec", "ndcg_10_macro",
    "recall_5_macro", "recall_10_macro", "gpu_peak_mib",
]
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {root / 'summary.csv'}")
PY
fi

printf '\nReproduction complete: %s\n' "${OUTPUT_DIR}"
