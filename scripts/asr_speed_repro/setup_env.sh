#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv-asr-speed}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required for MP3 decoding. Install it first, e.g. sudo apt-get install ffmpeg." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
  if [ ! -x "${PYTHON_BIN}" ]; then
    uv venv "${VENV_DIR}"
  fi
  uv pip install --python "${PYTHON_BIN}" --upgrade \
    accelerate \
    librosa \
    numpy \
    torch \
    "git+https://github.com/huggingface/transformers.git"
else
  python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip
  python -m pip install --upgrade \
    accelerate \
    librosa \
    numpy \
    torch \
    "git+https://github.com/huggingface/transformers.git"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import transformers

print("Environment ready")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
