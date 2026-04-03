#!/bin/bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/root/code/codevllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REMOTE_ROOT/.venv-vllm-gpu}"

mkdir -p "${REMOTE_ROOT}"
cd "${REMOTE_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [ -d "${REMOTE_ROOT}/vllm" ]; then
  python -m pip install -e "${REMOTE_ROOT}/vllm"
else
  echo "ERROR: ${REMOTE_ROOT}/vllm not found. Sync or clone the vllm repo first." >&2
  exit 1
fi

python -m pip install requests

echo "Bootstrap complete."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"

