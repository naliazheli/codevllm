# PR1 Plan: DeepEP Backend for vLLM-Ascend

## Branch

- top-level workspace branch: `codex/pr1-deepep-backend`
- `vllm-ascend` implementation branch: `codex/pr1-vllm-ascend-deepep`

## PR1 Goal

PR1 only targets one thing:

- add a minimal DeepEP-based MoE communication backend to `vllm-ascend`

PR1 is not the DBO optimization PR.
PR1 should make DeepEP usable as a standalone backend first.

## PR1 Scope

Included in PR1:

- add `MoECommType.DEEPEP`
- add backend selection logic for DeepEP
- add `DeepEPCommImpl`
- add `TokenDispatcherWithDeepEP`
- add DeepEP-specific combine metadata
- add optional import / graceful fallback when `deep_ep` is not installed
- wire DeepEP `dispatch/combine` into the existing vLLM-Ascend MoE flow
- add basic tests or at least one minimal validation path
- add installation and runtime notes for A3 verification

Explicitly excluded from PR1:

- no DBO overlap logic
- no DeepEP fused path
- no `fused_deep_moe`
- no `dispatch_ffn_combine`
- no performance tuning
- no aggressive graph-capture optimization

## Design Summary

The intended integration point is the existing MoE communication abstraction in
`vllm-ascend`, not a new MoE stack.

High-level flow after PR1:

1. vLLM router selects `topk_ids` and `topk_weights`
2. DeepEP backend computes dispatch layout
3. DeepEP backend dispatches tokens to local experts
4. existing vLLM-Ascend MLP path computes expert outputs
5. DeepEP backend combines outputs back to original token order

This means PR1 changes communication backend only.
Routing, MLP compute, shared experts, and quant framework stay under existing
`vllm-ascend` control.

## Files Expected to Change

Primary code changes:

- `E:/codevllm/vllm-ascend/vllm_ascend/ascend_forward_context.py`
- `E:/codevllm/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `E:/codevllm/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `E:/codevllm/vllm-ascend/vllm_ascend/ops/fused_moe/moe_runtime_args.py`
- `E:/codevllm/vllm-ascend/vllm_ascend/envs.py`

Possible docs / tests:

- `E:/codevllm/vllm-ascend/README.md`
- `E:/codevllm/vllm-ascend/docs/...`
- `E:/codevllm/vllm-ascend/tests/...`

Dependency source:

- `E:/codevllm/sgl-kernel-npu/python/deep_ep`

## Proposed Implementation Details

### 1. Add a new communication type

Add:

- `MoECommType.DEEPEP`

Selection policy for PR1 should stay conservative:

- only enable when `VLLM_ASCEND_ENABLE_DEEPEP=1`
- only enable when `enable_expert_parallel=True`
- only enable on A3 first
- prefer non-draft path only

Fallback behavior:

- if `deep_ep` is missing or backend constraints are not satisfied, fall back to
  existing comm type instead of crashing during config selection

### 2. Add a new MoE communication method

Add:

- `DeepEPCommImpl(MoECommMethod)`

Suggested behavior:

- reuse existing `PrepareAndFinalizeWithMC2` initially
- use new token dispatcher `TokenDispatcherWithDeepEP`
- reuse existing `_apply_mlp(...)` from `MoECommMethod`

### 3. Add a DeepEP token dispatcher

The dispatcher should own a `deep_ep.Buffer` instance built from the EP group.

Suggested mapping:

- `get_ep_group().device_group` -> `deep_ep.Buffer(group=...)`
- `token_dispatch(...)` -> `buffer.get_dispatch_layout(...)` then `buffer.dispatch(...)`
- `token_combine(...)` -> `buffer.combine(...)`

### 4. Add DeepEP combine metadata

Need a dedicated metadata type, because DeepEP returns its own `handle`.

Suggested fields:

- `handle`
- `recv_topk_idx`
- `recv_topk_weights`
- `num_recv_tokens_per_expert`

### 5. Keep MLP path unchanged

PR1 should keep:

- `unified_apply_mlp(...)`
- current weight layout
- current quant framework

This keeps PR1 focused and easier to verify.

## Constraints for PR1

These should be documented and enforced where reasonable:

- target hardware: A3 first
- target precision: BF16 first
- target MoE routing: standard routed experts path
- do not promise all quant modes in PR1
- do not promise DBO compatibility in PR1

## Risks to Watch

- `deep_ep` import path or wheel install may differ across A3 environments
- EP group naming / HCCL group extraction may differ from current MC2 path
- DeepEP may have hidden shape constraints on `topk`, `hidden_size`, or token counts
- `dynamic_eplb` and `log2phy` may need careful handling if routed IDs are remapped
- graph capture may fail even if eager mode works

## Deliverables for PR1

PR1 should be considered done when all of the following are true:

- code builds / imports on the target environment
- DeepEP backend can be selected by config or env flag
- one MoE model can run in eager mode with DeepEP backend on A3
- outputs are numerically sane against an existing backend baseline
- the deployment student has a clear A3 setup and verification checklist

## A3 Deployment Handoff

This section is for the student who will validate on A3.

### Repository Preparation

Expected repo state:

- top-level branch: `codex/pr1-deepep-backend`
- `vllm-ascend` branch: `codex/pr1-vllm-ascend-deepep` or the final PR1 branch
- submodule initialized:
  - `E:/codevllm/sgl-kernel-npu`

### Install Target

Need to install DeepEP from:

- `E:/codevllm/sgl-kernel-npu/python/deep_ep`

Expected rough process on A3:

1. enter the correct Python environment used by `vllm-ascend`
2. build `deep_ep` wheel from `sgl-kernel-npu`
3. install the wheel
4. verify `python -c "import deep_ep; print(deep_ep.__path__)"`

If needed, also verify the backend extension import path:

5. verify `python -c "import deep_ep_cpp"`

### Runtime Preconditions

The student should record these before testing:

- exact A3 machine type
- CANN version
- torch / torch_npu version
- vllm commit
- vllm-ascend commit
- whether EP group is initialized correctly
- whether HCCL-related env vars are customized

### Minimal Functional Validation

Run a minimal MoE case first.

Recommended validation order:

1. baseline with current backend
2. same model with DeepEP enabled
3. compare:
   - startup success
   - one-step forward success
   - logits or hidden state sanity
   - no deadlock in dispatch/combine

### What the Student Should Report Back

The student should return:

- install log summary
- whether `deep_ep` import works
- whether model starts
- whether one prompt completes
- backend selected at runtime
- any crash stack traces
- if successful, latency / throughput notes

### If Validation Fails

Please categorize failure as one of:

- install failure
- import failure
- process group / HCCL failure
- dispatch failure
- combine failure
- numerical mismatch
- graph mode failure

This categorization matters more than a long raw log.

## PR2 Dependency Note

PR2 will use PR1 as the base for:

- DBO + DeepEP overlap
- microbatch scheduling over DeepEP communication
- later fused-path evaluation if worthwhile

That means PR1 should stay clean, minimal, and correctness-focused.

## Suggested PR1 Title

- `Add DeepEP-based MoE communication backend for Ascend`

## Suggested PR1 Description

This PR adds a minimal DeepEP-based MoE communication backend to
`vllm-ascend`. It integrates DeepEP at the existing `MoECommMethod` /
`TokenDispatcher` layer and reuses the current vLLM-Ascend routing and expert
MLP compute path. The scope is intentionally limited to DeepEP
`dispatch/combine` for correctness-first bring-up on A3. DBO overlap and fused
DeepEP paths are intentionally deferred to follow-up work.
