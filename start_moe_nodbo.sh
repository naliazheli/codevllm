#!/bin/bash
# Qwen3.5-35B-A3B (MoE) + DP=2, TP=2 + EP + DBO off (baseline)
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
# vllm-a3-orig has /dev/davinci4-7; without this, torch_npu defaults to
# device indices 0-3 (which are occupied by vllm-a3). Restrict to chips 4-7.
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /root/code/Qwen3.5-35B-A3B \
--host 0.0.0.0 \
--port 8004 \
--data-parallel-size 2 \
--tensor-parallel-size 2 \
--seed 1024 \
--served-model-name qwen3.5moe \
--max-num-seqs 32 \
--max-model-len 32768 \
--max-num-batched-tokens 8096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
