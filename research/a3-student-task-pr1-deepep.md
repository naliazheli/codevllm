# A3 Student Task: PR1 DeepEP Validation

## Task Goal

Validate the PR1 DeepEP backend bring-up on A3 for `vllm-ascend`.

This task is about correctness-first bring-up, not performance tuning.

## Branches to Use

- workspace branch: `codex/pr1-deepep-backend`
- `vllm-ascend` branch: `codex/pr1-vllm-ascend-deepep`

## Code Locations

- `vllm-ascend` repo: `E:/codevllm/vllm-ascend`
- `sgl-kernel-npu` repo: `E:/codevllm/sgl-kernel-npu`
- PR1 plan doc: `E:/codevllm/research/pr1-deepep-backend-plan-and-a3-validation.md`

## What PR1 Is Supposed to Validate

The target of this validation is:

- `MoECommType.DEEPEP`
- `DeepEPCommImpl`
- `TokenDispatcherWithDeepEP`
- DeepEP-based `dispatch/combine`

PR1 does not include:

- DBO overlap
- fused DeepEP path
- performance optimization

## Environment Checklist

Before running anything, record:

- machine model
- number of NPUs
- CANN version
- Python version
- PyTorch version
- `torch_npu` version
- `vllm` commit
- `vllm-ascend` commit

## Step 1: Sync Code

In the workspace:

```powershell
git branch --show-current
git -C E:\codevllm\vllm-ascend branch --show-current
git submodule status
```

Expected:

- outer repo on `codex/pr1-deepep-backend`
- `vllm-ascend` on `codex/pr1-vllm-ascend-deepep`
- `sgl-kernel-npu` present

## Step 2: Build and Install DeepEP

Use the Python environment that will also run `vllm-ascend`.

Go to:

- `E:/codevllm/sgl-kernel-npu`

Recommended build/install flow:

```powershell
cd E:\codevllm\sgl-kernel-npu
bash build.sh -a deepep
pip install output\deep_ep*.whl
```

If the environment requires a symlink for `deep_ep_cpp`, follow the instructions
in:

- `E:/codevllm/sgl-kernel-npu/python/deep_ep/README.md`

## Step 3: Run Install Smoke Check

Run:

```powershell
python E:\codevllm\vllm-ascend\tools\check_deepep_install.py
```

Expected:

- `deep_ep` import passes
- `deep_ep_cpp` import passes

If this step fails, stop and report the error before trying model execution.

## Step 4: Enable the Backend

Set:

```powershell
$env:VLLM_ASCEND_ENABLE_DEEPEP="1"
```

For PR1 validation, keep the setup conservative:

- A3 only
- eager mode preferred first
- MoE model only
- avoid DBO
- avoid graph mode for the first pass

## Step 5: Basic Import / Startup Validation

Run a Python sanity check inside the same environment:

```powershell
@'
import os
os.environ["VLLM_ASCEND_ENABLE_DEEPEP"] = "1"

import deep_ep
import deep_ep_cpp
import vllm_ascend.envs as envs

print("deep_ep ok:", deep_ep.__path__)
print("deep_ep_cpp ok:", deep_ep_cpp.__file__)
print("VLLM_ASCEND_ENABLE_DEEPEP:", envs.VLLM_ASCEND_ENABLE_DEEPEP)
'@ | python -
```

## Step 6: Minimal Model Validation

Use one MoE model that is already known to run on A3 in your lab environment.

Recommended validation order:

1. run baseline with current backend
2. run same model with `VLLM_ASCEND_ENABLE_DEEPEP=1`
3. compare behavior

What to verify:

- process starts successfully
- model initializes
- EP group initializes
- one prompt completes
- no deadlock in MoE dispatch/combine
- no obvious numerical corruption

## Step 7: Capture Runtime Evidence

Please save:

- full startup log
- model launch command
- environment variables you changed
- first successful output, if any
- first failing stack trace, if any

## Step 8: Report Back in This Format

Please send back a short report with these sections:

1. Environment
2. DeepEP install result
3. Backend enable result
4. Model startup result
5. One-prompt inference result
6. Error category
7. Raw logs attachment path

## Error Categories

If validation fails, classify the failure as one of:

- install failure
- import failure
- wheel / extension mismatch
- HCCL / process-group failure
- DeepEP dispatch failure
- DeepEP combine failure
- model startup failure
- numerical mismatch

## Important Notes

- Do not start with DBO enabled.
- Do not start with graph mode enabled.
- Do not start with quantized MoE unless BF16 MoE is unavailable.
- If BF16 MoE works, that is already a useful PR1 validation result.

## Success Criteria

PR1 validation is considered successful if:

- `deep_ep` and `deep_ep_cpp` import correctly
- `vllm-ascend` starts with `VLLM_ASCEND_ENABLE_DEEPEP=1`
- one MoE model completes at least one prompt on A3

That is enough for the first student validation round.
