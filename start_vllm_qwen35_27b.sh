#!/usr/bin/env bash
set -eo pipefail
pkill -f "vllm serve" || true
sleep 2
set +e
set +u
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -u
set -e
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=1
export PYTHONPATH=/vllm-workspace/vllm:${PYTHONPATH:-}
nohup vllm serve /data/model/Qwen3.5-27B-w8a8-mtp/ \
    --served-model-name qwen3.5 \
    --host 0.0.0.0 \
    --port 8010 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --max-model-len 50000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.94 \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    > /tmp/vllm_qwen35_27b.log 2>&1 &
echo "started"
