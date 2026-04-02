param(
    [string]$BaseUrl = "http://10.28.158.169:8010",
    [string]$Model = "qwen3.5",
    [string]$Concurrency = "16,32,64",
    [int]$RequestsPerLevel = 128,
    [int]$MaxTokens = 128,
    [int]$WarmupRounds = 1,
    [int]$Repeats = 3,
    [string]$Label = "run",
    [string]$OutputJson = ""
)

$ErrorActionPreference = "Stop"
if (-not $OutputJson) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputJson = "e:\\codevllm\\bench_results\\${ts}_${Label}.json"
}

python e:\codevllm\scripts\bench_openai_completions.py `
    --base-url $BaseUrl `
    --model $Model `
    --concurrency $Concurrency `
    --requests-per-level $RequestsPerLevel `
    --max-tokens $MaxTokens `
    --warmup-rounds $WarmupRounds `
    --repeats $Repeats `
    --label $Label `
    --output-json $OutputJson
