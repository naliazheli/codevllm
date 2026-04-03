#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-8B}"
PORT="${PORT:-8002}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-dense}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name "${SERVED_NAME}" \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching

