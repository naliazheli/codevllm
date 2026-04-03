#!/bin/bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-qwen3.5-dense}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CONCURRENCY="${CONCURRENCY:-8}"
TOTAL="${TOTAL:-60}"

python /root/code/codevllm/bench_vllm.py \
  --url "${BASE_URL}" \
  --model "${MODEL}" \
  --prompt "Write a short summary of dual batch overlap." \
  --max-tokens "${MAX_TOKENS}" \
  --total "${TOTAL}" \
  --concurrency "${CONCURRENCY}"

