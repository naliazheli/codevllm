# PR1 Description: Add DeepEP-based MoE Communication Backend for Ascend

## Title

Add DeepEP-based MoE communication backend for Ascend

## What This PR Does

This PR adds a minimal DeepEP-based MoE communication backend to
`vllm-ascend`.

The integration is intentionally scoped to the existing MoE communication
abstraction layer, instead of introducing a parallel MoE implementation stack.
The new backend plugs into the current:

- `MoECommType`
- `MoECommMethod`
- `TokenDispatcher`

pipeline and reuses the existing vLLM-Ascend routing and expert MLP compute
path.

The initial implementation targets correctness-first bring-up on A3 and only
covers DeepEP `dispatch/combine`.

## Why We Need This

Current `vllm-ascend` MoE execution already supports several Ascend-native
communication modes such as:

- `ALLGATHER`
- `ALLTOALL`
- `MC2`
- `FUSED_MC2`

However, we also want to evaluate the DeepEP implementation from
`sgl-kernel-npu` as an alternative MoE communication backend for Ascend.

This PR is the first step toward that goal:

- it makes DeepEP selectable as a backend
- it keeps the scope small and reviewable
- it gives us a clean base for later DBO + DeepEP overlap work

## Scope of This PR

Included:

- add `MoECommType.DEEPEP`
- add `VLLM_ASCEND_ENABLE_DEEPEP`
- add `DeepEPCommImpl`
- add `TokenDispatcherWithDeepEP`
- add DeepEP-specific combine metadata
- wire DeepEP `dispatch/combine` into the current MoE flow
- add lightweight unit-test coverage for dispatcher and comm-method wiring
- add a small install smoke-check utility for A3 bring-up

Not included:

- no DBO overlap integration
- no `fused_deep_moe`
- no `dispatch_ffn_combine`
- no performance tuning
- no graph-mode optimization

## Design Notes

The intended execution flow is:

1. existing vLLM routing produces `topk_ids` and `topk_weights`
2. DeepEP computes dispatch layout
3. DeepEP dispatches tokens to local experts
4. existing vLLM-Ascend MLP path computes expert outputs
5. DeepEP combines outputs back to the original token layout

This keeps PR1 focused on communication backend substitution only.

## Current Constraints

This first version should be treated as experimental and conservative:

- intended for A3 first
- intended for eager-mode bring-up first
- intended for MoE models with expert parallel enabled
- currently only supports the non-quantized dispatch output path in the new
  dispatcher
- DBO compatibility is deferred to a follow-up PR

## Testing

What has been done in the current development environment:

- Python syntax checks for modified implementation files
- Python syntax checks for newly added unit-test files

What still needs real environment validation:

- `deep_ep` wheel build and install on A3
- `deep_ep` / `deep_ep_cpp` import verification
- one MoE model eager-mode bring-up on A3 with
  `VLLM_ASCEND_ENABLE_DEEPEP=1`

## Follow-up Work

Planned next step after this PR:

- DBO + DeepEP overlap integration as a separate PR

Potential later follow-ups:

- DeepEP fused path evaluation
- graph-mode compatibility
- broader quantization coverage
- performance benchmarking and tuning
