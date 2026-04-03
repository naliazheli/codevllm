param(
    [string]$HostName = "root@YOUR_VAST_IP",
    [int]$Port = 22,
    [string]$RemoteRoot = "/root/code/codevllm",
    [switch]$SyncVllm
)

$ErrorActionPreference = "Stop"

$Workspace = "E:\codevllm"
$GpuDir = Join-Path $Workspace "vllm-gpu"
$VllmDir = Join-Path $Workspace "vllm"

Write-Host "Preparing remote directories on $HostName ..."
ssh -p $Port $HostName "mkdir -p $RemoteRoot"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare remote directory: $RemoteRoot"
}

Write-Host "Syncing vllm-gpu/ ..."
scp -P $Port -r $GpuDir "${HostName}:$RemoteRoot/"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to sync vllm-gpu directory."
}

if ($SyncVllm) {
    Write-Host "Syncing vllm/ (this may take a while) ..."
    scp -P $Port -r $VllmDir "${HostName}:$RemoteRoot/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to sync vllm directory."
    }
}

Write-Host "Sync complete."
Write-Host "Remote workspace:"
Write-Host "  $RemoteRoot/vllm-gpu"
if ($SyncVllm) {
    Write-Host "  $RemoteRoot/vllm"
}

