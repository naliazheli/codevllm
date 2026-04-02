#!/bin/bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /root/code/Qwen3.5-27B \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 1 \
--tensor-parallel-size 2 \
--seed 1024 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 133000 \
--max-num-batched-tokens 8096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling \
--enable-dbo \
--dbo-decode-token-threshold 128
