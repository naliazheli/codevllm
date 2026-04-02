# Sync modified vllm-ascend files to a3 server
# Usage: .\sync_to_a3.ps1

$SERVER = "root@199.98.2.2"
# Files are scp'd to host, then accessible inside container via -v mount or
# directly at the pip-installed location. We overwrite the installed files
# inside the container path mapped from host.
# Container layout: /vllm-workspace/vllm-ascend/vllm_ascend/...
# But scp goes to HOST first. The host mounts /root/code -> /root/code.
# The container's installed vllm-ascend is at /vllm-workspace/vllm-ascend.
# Strategy: scp to host /tmp, then docker cp into container.

$LOCAL_BASE = "e:\codevllm\vllm-ascend"
$CONTAINER = "vllm-a3"
$REMOTE_BASE = "/vllm-workspace/vllm-ascend"

$files = @(
    "vllm_ascend/platform.py",
    "vllm_ascend/worker/worker.py",
    "vllm_ascend/worker/model_runner_v1.py"
)

Write-Host "=== Syncing modified files to a3 server ===" -ForegroundColor Cyan

# Step 1: scp files to host /tmp/vllm-ascend-sync/
Write-Host "`n[Step 1] SCP files to host /tmp/vllm-ascend-sync/" -ForegroundColor Yellow
ssh $SERVER "mkdir -p /tmp/vllm-ascend-sync/vllm_ascend/worker"
foreach ($file in $files) {
    $local = "$LOCAL_BASE\$($file -replace '/', '\')"
    $remote = "${SERVER}:/tmp/vllm-ascend-sync/${file}"
    Write-Host "  $file" -ForegroundColor Gray
    scp $local $remote
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: $file" -ForegroundColor Red
        exit 1
    }
}

# Step 2: docker cp into container
Write-Host "`n[Step 2] docker cp into container $CONTAINER" -ForegroundColor Yellow
foreach ($file in $files) {
    $cmd = "docker cp /tmp/vllm-ascend-sync/${file} ${CONTAINER}:${REMOTE_BASE}/${file}"
    Write-Host "  $cmd" -ForegroundColor Gray
    ssh $SERVER $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED docker cp: $file" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n=== All files synced successfully ===" -ForegroundColor Green
Write-Host @"

Next steps inside the container:
  docker exec -it vllm-a3 bash

  # Baseline (your working command, no DBO):
  vllm serve /root/code/Qwen3.5-27B \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 1 --tensor-parallel-size 2 \
    --seed 1024 --served-model-name qwen3.5 \
    --max-num-seqs 32 --max-model-len 133000 --max-num-batched-tokens 8096 \
    --trust-remote-code --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
    --speculative_config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' --async-scheduling

  # DBO enabled (add --enable-dbo + threshold):
  vllm serve /root/code/Qwen3.5-27B \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 1 --tensor-parallel-size 2 \
    --seed 1024 --served-model-name qwen3.5 \
    --max-num-seqs 32 --max-model-len 133000 --max-num-batched-tokens 8096 \
    --trust-remote-code --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
    --speculative_config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' --async-scheduling \
    --enable-dbo --dbo-decode-token-threshold 32

  # Test:
  curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5","prompt":"Hello","max_tokens":32}'
"@
