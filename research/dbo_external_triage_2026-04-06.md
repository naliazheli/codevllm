# DBO external triage (2026-04-06)

This note pulls three external signals into the local workspace and evaluates
whether they help the current Ascend DBO goal:

1. make `vllm-ascend` accept and run DBO options on Ascend
2. validate the full path on A3
3. tune performance
4. improve benchmark methodology
5. upstream a clean PR to `vllm-ascend`

## Local baseline confirmed

- Superproject gitlink for `vllm` is `bcf2be96120005e9aea171927f85055a6a5c0cf6`,
  which matches upstream tag `v0.18.0`.
- Superproject gitlink for `vllm-ascend` is
  `ddf8bc3fbd73425efc5a924c62b6d8d1d0e312b6`, which is the current
  `origin/dbo-dev` HEAD in `https://github.com/naliazheli/vllm-ascend.git`.
- Current local `dbo-dev` already documents that Ascend DBO is only doing
  sequential dual-execution today. See
  `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`:
  - the main execute path says true overlap still needs `dbo_yield` hooks in
    FusedMoE
  - `_dummy_run` uses the same sequential pattern for warmup/profile

## 1. Ascend DeepEP implementation in `sgl-kernel-npu`

Source:

- <https://github.com/sgl-project/sgl-kernel-npu/tree/main/python/deep_ep>

Verified facts:

- The repo explicitly describes itself as "Ascend Implementation of DeepEP".
- It supports A2 and A3.
- It exposes low-latency dispatch/combine operators and test scripts.

Assessment:

- Value to current goal: medium-high.
- Why it matters: your current `dbo-dev` branch has already reached the same
  conclusion as the conversation note: functional microbatch execution exists,
  but real overlap is still missing. An Ascend-native DeepEP implementation is
  relevant if you want to move from "sequential ubatch execution" to
  communication/compute overlap.
- Why it is not an immediate drop-in: current local code does not yet have the
  full hook-based overlap architecture from the community branch, and the hard
  problem is not only a backend library but also where to insert stream/event /
  `dbo_yield` control in the Ascend MoE execution path.

Recommendation:

- Treat this as a future implementation reference for stage 3 and stage 5.
- Do not assume it can be transplanted directly into `dbo-dev` without first
  defining Ascend-native overlap hooks in `vllm-ascend`.

## 2. `vllm-ascend` issue #5591 and PR #4894

Sources:

- <https://github.com/vllm-project/vllm-ascend/issues/5591>
- <https://github.com/vllm-project/vllm-ascend/pull/4894>

Verified facts:

- Issue #5591 was opened on 2026-01-04.
- The issue states that DBO on GPU depends on DeepEP and DP+EP deployment and
  therefore cannot be used directly on `vllm-ascend`.
- The issue points to PR #4894 as an adaptation attempt.
- PR #4894 is an older `v0.13.0`-based implementation that:
  - moves DBO logic to `model_runner`
  - adds hooks around collective communication
  - supports token/request microbatch splitting
  - claims 10%-20% prefill gains in some Ascend scenarios

Assessment:

- Value to current goal: high as design reference, medium as code source.
- Why it matters:
  - it proves that other contributors are already trying to solve the same
    Ascend DBO problem
  - it offers a more ambitious overlap architecture than current `dbo-dev`
  - it is directly relevant to the final upstream PR story
- Why it is not a direct cherry-pick:
  - the PR is based on `vLLM v0.13.0`, while this workspace is pinned to
    `v0.18.0`
  - it carries its own abstractions and assumptions around overlap templates,
    collective hooks, and communication policy
  - merging it blindly would likely create a large rebase and review burden

Recommendation:

- Track issue #5591 and PR #4894 closely.
- Borrow architecture and benchmark ideas, not code wholesale.
- If this project moves toward true overlap, compare your `dbo-dev` branch
  against PR #4894 specifically around:
  - `model_runner` control flow
  - comm hook insertion points
  - stream/event ownership
  - benchmark setup and reported counters

## 3. MindIE Micro Batch support

Source:

- <https://www.hiascend.com/document/detail/zh/mindie/230/mindiellm/llmdev/mindie_llm0510.html>

Verified facts:

- The verified public page is a MindIE 2.3.0 documentation page, not a 2.1 RC1
  page.
- The page says Micro Batch splits one batch into two batches and runs them on
  two streams.
- It uses Event synchronization between streams.
- It states the feature is mainly for Prefill and claims 70%+ communication /
  compute masking in that implementation.
- It also lists important limitations, including incompatibility with some
  other acceleration features and extra memory cost.

Assessment:

- Value to current goal: high for design justification, medium for tuning,
  low for direct implementation.
- Why it matters:
  - it is strong proof that the Ascend platform can benefit from the exact
    "dual-stream microbatch overlap" direction you are pursuing
  - it strengthens the performance story for a future upstream PR
  - its constraints are useful when designing fair A3 benchmarks
- Why it does not solve the code problem directly:
  - this is product documentation, not reusable source code
  - MindIE integration constraints do not map 1:1 to `vllm-ascend`

Recommendation:

- Use it as external validation in notes / PR motivation.
- Reuse its design language when explaining why sequential execution is not the
  end state and why stream/event-based overlap is the real target.

## Bottom line

All three external signals are valuable, but they help at different layers:

- `sgl-kernel-npu/deep_ep`: best future reference for Ascend-native overlap
  backend work
- issue #5591 / PR #4894: best direct reference for `vllm-ascend` community
  direction and possible upstream alignment
- MindIE Micro Batch docs: best external evidence that the Ascend hardware /
  software stack can benefit from the same overlap idea

Practical next step:

- keep `dbo-dev` as the functional baseline
- treat "true overlap in Ascend FusedMoE / comm path" as the next technical
  milestone
- use PR #4894 and MindIE as design references when planning that step
