param(
    [string]$Context = "a2-169",
    [string]$Container = "vllm-qwen35-27b",
    [string]$RepoRoot = "e:\codevllm",
    [string]$Manifest = "e:\codevllm\scripts\sync_manifest.txt",
    [string]$ModelPath = "/data/model/qwen3.5-27b/",
    [int]$Port = 8010,
    [int]$OmpNumThreads = 1,
    [int]$MaxNumBatchedTokens = 16384,
    [int]$MaxNumSeqs = 128,
    [switch]$SkipReinstall
)

$ErrorActionPreference = "Stop"
$env:DOCKER_API_VERSION = "1.43"

function Invoke-DockerExec([string]$Cmd) {
    docker --context $Context exec $Container bash -lc $Cmd
}

Write-Host "==> Sync files to container: $Container"
if (!(Test-Path $Manifest)) {
    throw "Manifest not found: $Manifest"
}

$lines = Get-Content $Manifest | Where-Object { $_ -and -not $_.StartsWith("#") }
foreach ($rel in $lines) {
    $src = Join-Path $RepoRoot $rel
    if (!(Test-Path $src)) {
        throw "Source file not found: $src"
    }
    $dst = "/vllm-workspace/" + ($rel -replace "\\","/")
    Write-Host "  - $rel"
    docker --context $Context cp $src "${Container}:$dst" | Out-Null
}

Write-Host "==> Validate python syntax for patched files"
Invoke-DockerExec "python -m py_compile /vllm-workspace/vllm-ascend/vllm_ascend/utils.py"

if ($SkipReinstall) {
    Write-Host "==> Skip reinstall editable packages"
} else {
    Write-Host "==> Reinstall editable packages"
    Invoke-DockerExec "cd /vllm-workspace/vllm && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps"
    Invoke-DockerExec "cd /vllm-workspace/vllm-ascend && pip install -e . --no-build-isolation"
}

Write-Host "==> Write and run start script"
$startScript = @'
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
export OMP_NUM_THREADS=__OMP_NUM_THREADS__
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=1
export PYTHONPATH=/vllm-workspace/vllm:${PYTHONPATH:-}
nohup vllm serve __MODEL_PATH__ \
  --served-model-name qwen3.5 \
  --host 0.0.0.0 \
  --port __PORT__ \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 50000 \
  --max-num-batched-tokens __MAX_NUM_BATCHED_TOKENS__ \
  --max-num-seqs __MAX_NUM_SEQS__ \
  --gpu-memory-utilization 0.94 \
  --trust-remote-code \
  --async-scheduling \
  --allowed-local-media-path / \
  --mm-processor-cache-gb 0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  > /tmp/vllm_qwen35_27b.log 2>&1 &
echo "started"
'@
$startScript = $startScript.Replace("__MODEL_PATH__", $ModelPath).Replace("__PORT__", "$Port").Replace("__OMP_NUM_THREADS__", "$OmpNumThreads").Replace("__MAX_NUM_BATCHED_TOKENS__", "$MaxNumBatchedTokens").Replace("__MAX_NUM_SEQS__", "$MaxNumSeqs")
$tmpLocal = Join-Path $env:TEMP "start_vllm_qwen35_27b.sh"
$startScript | Set-Content -Path $tmpLocal -Encoding ascii
docker --context $Context cp $tmpLocal "${Container}:/tmp/start_vllm_qwen35_27b.sh" | Out-Null
Invoke-DockerExec "sed -i 's/\r$//' /tmp/start_vllm_qwen35_27b.sh && chmod +x /tmp/start_vllm_qwen35_27b.sh && /tmp/start_vllm_qwen35_27b.sh"

Write-Host "==> Wait for /health"
$ok = $false
for ($i = 1; $i -le 36; $i++) {
    $code = docker --context $Context exec $Container bash -lc "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$Port/health 2>/dev/null || true"
    Write-Host "  try=$i code=$code"
    if ($code -eq "200") { $ok = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ok) {
    Write-Host "==> Last logs"
    Invoke-DockerExec "tail -n 200 /tmp/vllm_qwen35_27b.log || true"
    throw "Service failed to become healthy."
}

Write-Host "==> Success. Service is healthy on port $Port."
