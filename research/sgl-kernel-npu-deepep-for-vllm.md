# sgl-kernel-npu DeepEP 对接 vLLM-Ascend 的实现思路

## 当前结论

- `sgl-kernel-npu` 里的 DeepEP 不是写死给 SGLang 业务层的逻辑，它本质上是一个独立的 NPU EP 通信/融合库。
- 最自然的接入点不是重做一套 `FusedMoE`，而是在 `vllm-ascend` 现有的 `MoECommMethod` 抽象里新增一个 `DEEPEP` 后端。
- 对接可以分两层：
  - 最小可行版本：只接 `deep_ep.Buffer.dispatch/combine`，复用 vLLM 现有的 routing 和 MLP 计算。
  - 高性能版本：再接 `deep_ep.Buffer.fused_deep_moe` / `dispatch_ffn_combine`，直接替换当前 `FUSED_MC2` 的融合路径。

## 已拉入代码

- submodule: `sgl-kernel-npu`
- URL: `https://github.com/sgl-project/sgl-kernel-npu.git`
- branch: `main`
- 当前提交: `d16fb136873b6fa01df51b8bb7773642058c3e12`

## DeepEP 自身的边界

DeepEP 的 Python 入口在：

- `sgl-kernel-npu/python/deep_ep/deep_ep/buffer.py`

核心对象是 `Buffer`，它暴露了几类能力：

- `dispatch` / `combine`
- `low_latency_dispatch` / `low_latency_combine`
- `fused_deep_moe`
- `dispatch_ffn_combine`

对应的 C++/pybind 边界在：

- `sgl-kernel-npu/csrc/deepep/deep_ep.hpp`
- `sgl-kernel-npu/csrc/deepep/pybind_extension.cpp`

这说明 DeepEP 已经把“路由后的 token 跨 EP rank 分发、专家计算后的结果回收、以及部分融合算子”都做成了独立库，不依赖 SGLang runtime。

## vLLM-Ascend 里真正该接的位置

vLLM-Ascend 已经把 MoE 通信策略抽象成一层独立策略对象：

- `vllm-ascend/vllm_ascend/ascend_forward_context.py`
- `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`

当前已经有四种通信模式：

- `ALLGATHER`
- `MC2`
- `ALLTOALL`
- `FUSED_MC2`

它们的调用链是：

1. `select_moe_comm_method(...)` 选择通信类型。
2. `setup_moe_comm_method(...)` 注册具体实现。
3. `AscendFusedMoE.forward_impl(...)` 调 `moe_comm_method.prepare(...)`。
4. `quant_method.apply(...)` 内部再调 `moe_comm_method.fused_experts(...)`。
5. `fused_experts(...)` 里分成两段：
   - token dispatch/combine 由 `token_dispatcher`
   - 专家 MLP 由 `unified_apply_mlp(...)`
6. `moe_comm_method.finalize(...)` 做收尾。

所以，对接 DeepEP 最合适的方式是：

- 新增 `MoECommType.DEEPEP`
- 新增 `DeepEPCommImpl`
- 新增 `TokenDispatcherWithDeepEP`
- 视情况复用 `PrepareAndFinalizeWithMC2` 或新增 `PrepareAndFinalizeWithDeepEP`

## 为什么不是直接照搬 SGLang

SGLang 的 DeepEP 接法通常是“路由、调度、通信、部分融合算子”一起围绕它自己的 execution path 来写。

而 vLLM-Ascend 现在已经有成熟的：

- router 选择逻辑
- expert map / eplb
- quant method
- shared experts overlap
- prepare/finalize
- MoE runner

如果直接把 SGLang 的上层逻辑搬过来，会和 vLLM 当前抽象重叠，维护成本很高。

更合理的做法是：

- 只把 DeepEP 当成新的 EP backend
- 让 vLLM 继续掌握 routing、weight layout、shared experts、量化策略

## 最小可行版本怎么做

目标：先跑通，不追求一步做到 fused。

### 1. 新增通信类型

修改：

- `vllm-ascend/vllm_ascend/ascend_forward_context.py`

动作：

- 给 `MoECommType` 增加 `DEEPEP`
- 在 `select_moe_comm_method(...)` 中增加开关判断

建议先走环境变量开关，例如：

- `VLLM_ASCEND_ENABLE_DEEPEP=1`

首版策略建议：

- 只在 `enable_expert_parallel=True`
- 只在 A3
- 只在非 draft
- 先限制 `top_k <= 2`
- 先限制 `bf16` / 非量化或只支持当前 DeepEP 已验证的量化形式

### 2. 注册新的 comm method

修改：

- `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py`

动作：

- 在 `setup_moe_comm_method(...)` 注册 `MoECommType.DEEPEP`
- 新增 `DeepEPCommImpl(MoECommMethod)`

### 3. 新增 DeepEP token dispatcher

新文件建议：

- `vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher_deepep.py`

或者先直接放进：

- `vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py`

实现要点：

- 初始化时创建 `deep_ep.Buffer(process_group, ...)`
- `token_dispatch(...)`：
  - 输入仍然使用 vLLM 已经选好的 `topk_ids` / `topk_weights`
  - 先调用 `buffer.get_dispatch_layout(...)`
  - 再调用 `buffer.dispatch(...)`
  - 返回：
    - `hidden_states=recv_x`
    - `group_list=num_recv_tokens_per_expert_list`
    - `combine_metadata=DeepEPCombineMetadata(...)`
- `token_combine(...)`：
  - 调 `buffer.combine(...)`

这里需要新增一个 metadata dataclass，内容至少包括：

- `handle`
- `topk_ids`
- `topk_weights`
- `recv_topk_idx`
- `recv_topk_weights`
- 可能还要保存 dispatch 后的 expert token count

### 4. MLP 先复用 vLLM 现有实现

这是 MVP 的关键。

不要一开始就碰 `fused_deep_moe`。先让流程变成：

- vLLM router 选专家
- DeepEP dispatch 到本地专家
- vLLM 现有 `unified_apply_mlp(...)` 算专家前向
- DeepEP combine 回原 token 顺序

这样改动面最小，也最容易验证正确性。

### 5. prepare/finalize 先复用 MC2

首版大概率可以直接复用：

- `PrepareAndFinalizeWithMC2`

原因：

- DeepEP 本身也要求 token 数、EP group、部分 shape 有规则
- vLLM 现有 MC2 路径已经处理了 TP slicing、padding、`padded_num_tokens`

如果后面发现 DeepEP 对输入 shape 或 graph capture 有额外要求，再单独抽 `PrepareAndFinalizeWithDeepEP`。

## 真正高性能版本怎么做

目标：替换当前 `FUSED_MC2` 的融合路径。

### 路线 A：接 `dispatch_ffn_combine`

对应：

- `deep_ep.Buffer.fused_deep_moe(..., fuse_mode=DISPATCH_FFN_COMBINE)`
- 或 `deep_ep.Buffer.dispatch_ffn_combine(...)`

这个路径最像 vLLM-Ascend 现在的：

- `torch.ops._C_ascend.dispatch_ffn_combine`

适合 decode 或固定上界较明确的场景。

### 路线 B：接 `fused_deep_moe`

对应：

- `deep_ep.Buffer.fused_deep_moe(...)`

这会把：

- dispatch
- expert GMM/FFN
- combine

一把做掉，潜力更大，但约束也最多：

- 权重布局必须符合 DeepEP 预期
- scale tensor 形状必须匹配
- `num_max_dispatch_tokens_per_rank` 必须在 vLLM 侧稳定可计算
- 量化模式要和 vLLM 当前 `QuantType` 对齐

### 建议顺序

1. 先做 `dispatch/combine` 路径。
2. 再做 `dispatch_ffn_combine` 替换 `FUSED_MC2`。
3. 最后再评估 `fused_deep_moe` 是否值得完全切换。

## 与当前 vLLM-Ascend 的主要差异点

### 1. DeepEP 用的是独立 Python wheel

它不是 `torch_npu.*` 内建 op，而是：

- `import deep_ep`
- `deep_ep_cpp.Buffer`

所以工程上需要增加一层可选依赖处理：

- 没安装 `deep_ep` 时优雅回退
- 安装时从 `sgl-kernel-npu/python/deep_ep` 构建 wheel

### 2. Group 传递方式不同

当前 MC2 主要依赖：

- HCCL group name
- `torch_npu.npu_moe_distribute_dispatch(_v2)`

DeepEP 则是：

- Python 层传 `dist.ProcessGroup`
- 初始化时内部提取 `hccl_comm_name`

这点反而更适合 vLLM 抽象，因为 `get_ep_group()` 已经现成。

### 3. 权重布局和融合算子接口不同

`FUSED_MC2` 现在直接调用：

- `torch.ops._C_ascend.dispatch_ffn_combine`
- `torch.ops._C_ascend.dispatch_gmm_combine_decode`

DeepEP 的融合接口要求的是：

- `gmm1_permuted_weight`
- `gmm1_permuted_weight_scale`
- `gmm2_weight`
- `gmm2_weight_scale`

所以如果要做 fused，不只是通信替换，还涉及：

- `process_weights_after_loading(...)`
- 量化权重存储格式
- scale tensor 组织方式

### 4. combine handle 不兼容

DeepEP 的 `dispatch` 返回自己的 `handle` tuple。

所以不能直接复用 `TokenDispatcherWithMC2` 的 combine metadata，必须新建。

## 推荐的代码改造清单

### 第一阶段：先跑通

- `vllm-ascend/vllm_ascend/ascend_forward_context.py`
  - 新增 `MoECommType.DEEPEP`
  - 在 `select_moe_comm_method(...)` 增加开关逻辑

- `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py`
  - 注册 `DeepEPCommImpl`
  - 新增 `DeepEPCommImpl`

- `vllm-ascend/vllm_ascend/ops/fused_moe/moe_runtime_args.py`
  - 新增 `DeepEPCombineMetadata`

- `vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py`
  - 新增 `TokenDispatcherWithDeepEP`

- `vllm-ascend/vllm_ascend/envs.py`
  - 新增 `VLLM_ASCEND_ENABLE_DEEPEP`

- `vllm-ascend/requirements` 或安装文档
  - 增加可选 `deep_ep` 安装说明

### 第二阶段：做融合

- `vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`
  - 在 `AscendUnquantizedFusedMoEMethod.apply(...)` 或量化方法里接入 `deep_ep.Buffer.fused_deep_moe`

- 相关 quant method
  - 调整权重布局以兼容 DeepEP

## 伪代码骨架

```python
class DeepEPCommImpl(MoECommMethod):
    def _get_token_dispatcher(self):
        return TokenDispatcherWithDeepEP(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)
```

```python
class TokenDispatcherWithDeepEP(MoETokenDispatcher[DeepEPCombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import deep_ep
        self.buffer = deep_ep.Buffer(get_ep_group().device_group, ...)

    def token_dispatch(self, token_dispatch_input):
        topk_ids = token_dispatch_input.topk_ids
        topk_weights = token_dispatch_input.topk_weights
        x = token_dispatch_input.hidden_states

        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
            self.buffer.get_dispatch_layout(topk_ids, self.num_experts)

        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
            self.buffer.dispatch(
                x=x,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
            )

        return MoETokenDispatchOutput(
            hidden_states=recv_x,
            dynamic_scale=None,
            group_list=torch.tensor(num_recv_tokens_per_expert_list, device=x.device),
            group_list_type=1,
            combine_metadata=DeepEPCombineMetadata(
                handle=handle,
                recv_topk_idx=recv_topk_idx,
                recv_topk_weights=recv_topk_weights,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        out, _, _ = self.buffer.combine(
            x=hidden_states,
            handle=combine_metadata.handle,
            topk_weights=combine_metadata.recv_topk_weights,
        )
        return out
```

## 风险点

- DeepEP 当前接口对 dtype、topk、hidden size、量化格式都有隐含约束，首版不要假设它和 vLLM 全量模型都兼容。
- `num_max_dispatch_tokens_per_rank` 这类参数和 vLLM 的 scheduler/cudagraph capture 强相关，做 fused 前必须理清。
- `dynamic_eplb` / `log2phy` / redundant experts 需要确认 DeepEP 路由 id 是否接受 vLLM 当前的物理 expert 编号方案。
- shared experts overlap 当前依赖 `before_dispatch_evt` / `before_combine_evt`，如果 DeepEP 路径想保留并行重叠，最好也补上事件语义。
- graph capture 兼容性要单独验证，尤其是 low-latency 和 fused 路径。

## 建议执行顺序

1. 先做 `DEEPEP + dispatch/combine`，验证 correctness。
2. 跑一个简单的 Qwen3-MoE / DeepSeek-MoE EP case。
3. 再接 `dispatch_ffn_combine`。
4. 最后再考虑替换成 `fused_deep_moe`。

## 结论

这件事可做，而且和 vLLM-Ascend 现有架构是对得上的。

最小成本方案不是“把 SGLang 的 DeepEP 上层代码搬过来”，而是：

- 把 `sgl-kernel-npu` 的 `deep_ep.Buffer` 当成新的 token dispatch/combine backend
- 接到 `vllm-ascend` 的 `MoECommMethod` / `TokenDispatcher` 体系里
- 先复用 vLLM 现有 MLP
- 再逐步升级到 fused 路径
