# codevllm

`codevllm` is a local workspace for Ascend-focused `vllm-ascend` operator adaptation, validation, and performance tuning.

The current primary task is to adapt and validate these vLLM flags on Ascend:

- `--enable-dbo`
- `--dbo-decode-token-threshold`

This repository is organized as a top-level workflow repo plus two submodules:

- `vllm`: upstream official `vllm` source, kept as the official codebase
- `vllm-ascend`: development target for Ascend adaptation, tracked against a personal fork for easier PR preparation

## Repository Layout

- `vllm/`: official upstream submodule from `https://github.com/vllm-project/vllm.git`
- `vllm-ascend/`: fork-backed submodule for local development and PR preparation
- `sync_to_a3.ps1`: sync selected modified `vllm-ascend` files to the A3 server container
- `start_dbo.sh`: baseline DBO-enabled startup script for local reference
- `start_orig_nodbo.sh`: original no-DBO reference startup script
- `start_dbo_t1.sh`, `start_dbo_t128.sh`: threshold-specific startup variants
- `start_dp2_tp2_*.sh`: DP=2 / TP=2 experiments
- `bench_vllm.py`: lightweight completion benchmark for quick checks
- `scripts/bench_openai_completions.py`: more general benchmark runner for repeated concurrency testing
- `bench_results/`: benchmark outputs and comparison notes
- `a3.txt`: working notes, validation records, and performance observations

## Current Goal

The current project goal is not just to make DBO parse on Ascend, but to complete the full path:

1. Make `vllm-ascend` correctly accept and run DBO-related options on Ascend.
2. Validate the functional path end to end on the A3 environment.
3. Tune performance, especially around threshold selection and data-parallel execution.
4. Improve reusable benchmark methodology so the performance story is convincing.
5. Summarize the final design and submit a clean upstream PR to `vllm-ascend`.

## Working Model

The normal development loop is:

1. Modify `vllm-ascend` locally, usually under `vllm_ascend/platform.py`, `vllm_ascend/worker/worker.py`, and `vllm_ascend/worker/model_runner_v1.py`.
2. Use `sync_to_a3.ps1` to copy the changed files to the A3 host and then into the running `vllm-a3` container.
3. Restart or relaunch the service inside the container using the fixed startup scripts in this repo.
4. Run functional validation first, then performance checks.
5. Iterate on code, sync again, and repeat.
6. Once stable, commit to the local `vllm-ascend` branch and push to the fork for PR preparation.

This workflow is optimized for fast iteration on a remote Ascend environment without rebuilding the whole image for each change.

## Submodule Strategy

This repo intentionally keeps the two codebases separate:

- `vllm` stays on the official upstream source.
- `vllm-ascend` points to the personal fork as `origin`, with official upstream retained as `upstream`.

That means the typical `vllm-ascend` remote setup is:

```bash
origin   = https://github.com/naliazheli/vllm-ascend.git
upstream = https://github.com/vllm-project/vllm-ascend.git
```

This makes it easy to:

- keep local work isolated on your own branch
- sync from upstream when needed
- open PRs from fork branches back to `vllm-project/vllm-ascend`

## A3 Environment Assumptions

The current workflow assumes:

- an A3 server is available
- passwordless SSH should be configured before using sync scripts
- the model already exists at `/root/code/Qwen3.5-27B`
- the container image already contains a working `vllm` + `vllm-ascend` runtime
- the active container name is typically `vllm-a3`

`sync_to_a3.ps1` currently syncs these files:

- `vllm_ascend/platform.py`
- `vllm_ascend/worker/worker.py`
- `vllm_ascend/worker/model_runner_v1.py`

The script first copies files to the host, then uses `docker cp` to overwrite the installed files inside the container.

## Service Startup Profiles

The main startup patterns currently used are:

- original baseline, no DBO
- DBO enabled with `--dbo-decode-token-threshold`
- DP=1 / TP=2 for early bring-up and narrow functional checks
- DP=2 / TP=2 for more realistic DBO evaluation

The startup scripts also standardize several environment variables:

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
```

## Benchmarking Workflow

There are two benchmark layers in this repo.

### 1. Quick verification

`bench_vllm.py` is a small concurrent benchmark for quick sanity checks:

- sends OpenAI-compatible `/v1/completions` requests
- records wall time, request throughput, token throughput, and latency percentiles
- useful for fast threshold comparisons during iteration

Example:

```bash
python bench_vllm.py --url http://127.0.0.1:8000 --model qwen3.5 --prompt Hello --max-tokens 32 --total 20 --concurrency 4
```

### 2. More reusable performance testing

`scripts/bench_openai_completions.py` is the more general performance tool for repeated runs:

- supports multiple concurrency levels
- supports warmup rounds
- supports repeated measurement rounds
- writes structured JSON output
- summarizes median results across repeats

This is the better base for turning the current tuning workflow into a more convincing PR-quality benchmark story.

Example:

```bash
python scripts/bench_openai_completions.py \
  --base-url http://127.0.0.1:8002 \
  --model qwen3.5 \
  --max-tokens 128 \
  --concurrency 8,16 \
  --requests-per-level 100 \
  --warmup-rounds 1 \
  --repeats 3 \
  --label dp2-tp2-dbo \
  --output-json bench_results/dp2_tp2_dbo.json
```

## What Has Been Verified So Far

Based on the current notes and scripts:

- original upstream-style `vllm-ascend` behavior could not directly start with `--enable-dbo`
- the local Ascend adaptation has already enabled DBO startup successfully
- the modified path has already returned valid inference results
- the issue is no longer just "can it start", but "how to tune it well and prove it"

The current understanding is:

- DP=1 is useful for bring-up and functional verification
- DP=1 is not the best place to claim real DBO overlap gains
- DP=2 / TP=2 is a better validation path for meaningful DBO tuning
- threshold choice matters, and `32` is not obviously optimal in the current small-sample runs

## Practical Recommendations

The most useful next steps are:

1. Continue tuning under DP=2 / TP=2 instead of over-interpreting DP=1 results.
2. Expand threshold search, especially around `1`, `64`, and `128`.
3. Increase benchmark scale with larger request counts, higher concurrency, and longer decode lengths.
4. Record warmup-separated repeated runs to reduce one-off graph capture and cache noise.
5. Turn the benchmark outputs into a concise PR-ready comparison between:
   - original upstream behavior
   - adapted DBO-enabled behavior
   - different threshold settings

## Typical Daily Flow

For day-to-day work, the path is usually:

```bash
# 1. edit local vllm-ascend code

# 2. from Windows workspace
.\sync_to_a3.ps1

# 3. on A3 host
docker exec -it vllm-a3 bash

# 4. inside container
bash /root/code/codevllm/start_dbo.sh

# 5. benchmark from the appropriate environment
python /root/code/codevllm/bench_vllm.py ...
```

Adjust the exact startup script depending on whether the test target is:

- DBO on vs off
- threshold comparison
- DP=1 / TP=2 vs DP=2 / TP=2

## Notes

- `Qwen3.5-27B/`, `transformers/`, and `bench_results/` are intentionally not tracked in the top-level repo.
- The authoritative implementation work happens inside the `vllm-ascend` submodule.
- The top-level repo exists to preserve the full engineering workflow: scripts, benchmarks, notes, and reproducible experiment entry points.
