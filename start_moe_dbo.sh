#!/bin/bash
# Qwen3.5-35B-A3B (MoE) + DP=2, TP=2 + DBO on (without EP)
#
# NOTE: --enable-expert-parallel is intentionally omitted.
# When EP is enabled with DBO on Ascend, vLLM tries to instantiate
# DeepEPLLPrepareAndFinalize (a DeepEP library class) at runtime, but
# DeepEP is not installed on Ascend. The all2all_utils.py import is
# guarded by `if current_platform.is_cuda_alike()`, so the class
# is never loaded on NPU -> NameError at runtime.
#
# Current validation scope (mechanism layer):
#   - MoE model: correct model type for DBO intent
#   - DP=2: activates ubatching path (should_ubatch=True)
#   - DBO on: verifies ubatch splitting works on MoE model
#   - platform.py will log WARNING: "DBO without EP, no all-to-all to overlap"
#
# Future work: add Ascend-native fallback in all2all_utils.py so that
# MoE + EP + DBO can be used without requiring DeepEP.
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /root/code/Qwen3.5-35B-A3B \
--host 0.0.0.0 \
--port 8003 \
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
--async-scheduling \
--enable-dbo \
--dbo-decode-token-threshold 32
