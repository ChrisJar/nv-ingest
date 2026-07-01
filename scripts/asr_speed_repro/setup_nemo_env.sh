#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv-nemo-asr-speed}"
PYTHON_VERSION="${NEMO_ASR_PYTHON_VERSION:-3.10}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required for audio conversion. Install it first, e.g. sudo apt-get install ffmpeg." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
  if [ ! -x "${PYTHON_BIN}" ]; then
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
  fi
  uv pip install --python "${PYTHON_BIN}" --upgrade Cython packaging
  uv pip install --python "${PYTHON_BIN}" --upgrade "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
else
  python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install --upgrade Cython packaging
  python -m pip install --upgrade "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import nemo.collections.asr as nemo_asr

print("NeMo environment ready")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("nemo_asr", nemo_asr.__name__)
PY
