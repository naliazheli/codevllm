#!/bin/bash
# Dense model (Qwen3.5-27B) + DP=2 + DBO + threshold=1
# Purpose: force should_ubatch=True on every decode step to exercise the
# sequential dual-execution loop added to execute_model and _dummy_run.
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
--no-enable-prefix-caching \
--enable-dbo \
--dbo-decode-token-threshold 1
