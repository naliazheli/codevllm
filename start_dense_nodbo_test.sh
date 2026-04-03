#!/bin/bash
# Dense model (Qwen3.5-27B) + DP=2, DBO disabled — baseline for comparison
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /root/code/Qwen3.5-27B \
--host 0.0.0.0 \
--port 8001 \
--data-parallel-size 2 \
--tensor-parallel-size 1 \
--seed 1024 \
--served-model-name qwen3.5 \
--max-num-seqs 16 \
--max-model-len 4096 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching
