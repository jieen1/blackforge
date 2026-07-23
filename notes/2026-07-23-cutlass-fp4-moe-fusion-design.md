# CUTLASS FP4 MoE Grouped GEMM & 融合 Expert Kernel 设计输入（2026-07-23）

> 研究范围：CUTLASS 4.6.1 Blackwell MoE GEMM 示例 + vLLM fused_moe 路由实现
> 目标：为自研融合 expert kernel（SM120/RTX 5090）提供设计输入

---

## 1. CUTLASS 92_blackwell_moe_gemm_fp4_grouped.cu 核心设计

### 1.1 Grouped GEMM 处理（每个 expert 不同 M）

**关键洞察：MoE 中 N（tokens）是变量，M/K 是常量。**

CUTLASS 使用 `MoEProblemShape<Shape<int,int,int>>`（`group_array_problem_shape.hpp:85`）：

```cpp
struct MoEProblemShape {
  int32_t max_m, max_n, max_k;     // 全局最大维度
  int32_t num_groups;               // expert 数量
  int32_t* tokens_per_expert;       // device 指针：每个 expert 的 token 数
  int32_t* tokens_per_expert_host;  // host 镜像

  // get_problem_shape(group_idx) 返回 {max_m, tokens_per_expert[group_idx], max_k}
  // → M 和 K 跨 expert 共享，只有 N（tokens）按 expert 变化
};
```

**调度链**：
- `GemmUniversal<MoEProblemShape, CollectiveMainloop, CollectiveEpilogue>`
- `IsGroupedGemmKernel = true` → `TileSchedulerTag = GroupScheduler`
- → `PersistentTileSchedulerSm100Group`（`sm100_tile_scheduler_group.hpp`）
- 按最大 problem shape 发射 grid，每个 CTA 在 tile dispatch 时查询 `get_problem_shape(L_idx)` 获取实际 N

**混合加载策略**（核心创新）：
- **A 矩阵（权重）= TMA 加载**：权重形状固定（max_m × max_k），TMA descriptor 只需设置一次
- **B 矩阵（activation）= CPASYNC 加载**：token 数按 expert 变化，避免频繁更新 TMA descriptor 的开销
- 这是 SM100 特有的 `KernelMixedTmaCpAsyncWarpSpecialized*` 调度

### 1.2 FP4 权重布局和解码

**数据类型**：
```cpp
ElementA = nv_float4_t<float_e2m1_t>   // NVFP4 权重
ElementB = nv_float4_t<float_e2m1_t>   // NVFP4 activation
ElementSF = float_ue4m3_t              // 8-bit 无符号 E4M3 scale factor
```

**Block Scaling 结构**（`sm100_blockscaled_layout.hpp`）：
- `Blk_MN = 128`：每 128 个元素共享一组 scale factor
- `Blk_SF = 4`：每个 chunk 4 个 SF 值
- SF 向量大小 `SFVecSize = 16`（输出 SFD 的向量宽度）
- K-major SF atom：`Shape<Shape<32,4>, Shape<16,4>>`，stride `<Stride<16,4>, Stride<0,1>>`
- 布局由 `Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA/SFB` 从 problem shape 推导

**对齐**：`AlignmentA = AlignmentB = 32`（32 个 4-bit 元素 = 16 字节）

**操作类**：`OpClassBlockScaledTensorOp`（Blackwell 原生 block-scaled tensor core 指令）

### 1.3 Prologue/Epilogue 融合

**当前 FP4 grouped 示例的融合**：
```cpp
// FuseQuantization = false 时：
FusionOperation = LinearCombination<ElementC, ElementAccumulator>  // D = α·acc + β·C

// FuseQuantization = true 时：
FusionOperation = LinCombBlockScaleFactor<16, ElementD, ElementCompute, ElementSFD, LayoutSFD, ElementC>
// → D = α·acc + β·C + 输出 block scale factor 生成（量化输出）
```

**CUTLASS 可用的完整融合操作表**（`epilogue/fusion/operations.hpp`）：

| 操作 | 公式 | MoE 用途 |
|---|---|---|
| `LinearCombination` | D = α·acc + β·C | 基础 GEMM |
| `LinCombEltAct` | D = act(α·acc + β·C) | SiLU 融合 |
| `LinCombBlockScaleFactor` | D = α·acc + β·C + SF生成 | 量化输出 |
| **`LinCombEltActBlockScaleFactor`** | **D = act(α·acc + β·C) + SF生成** | **gate_up → SiLU → 量化** |
| `LinCombPerRowBiasBlockScaleFactor` | D = α·acc + β·C + bias + SF | 带 bias |
| `LinCombPerRowBiasEltActBlockScaleFactor` | D = act(α·acc + β·C + bias) + SF | 完整融合 |

**关键发现**：`92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` 已演示了完整的 MoE 融合：
```cpp
LinCombEltActBlockScaleFactor<SiLu, 16, ElementD, ElementAccumulator, ElementSFD, LayoutC, ElementC>
```
→ **SiLU 激活 + block scale factor 输出（量化）**，正是 gate_up 投影后需要的操作。

### 1.4 SM120 上的 Tile Shape 和 Warp 分配

**FP4 grouped 示例（SM100）**：
- 1SM：`MmaTileMNK = Shape<128, 64, 256>`
- 2SM：`MmaTileMNK = Shape<256, 64, 256>`
- N=64 选择理由：匹配 decode 阶段小 token 数
- Cluster shapes：1×1×1, 2×2×1（1SM）；2×1×1, 2×4×1（2SM）

**SM120 关键差异**：
- **SM120 没有 MixedTmaCpAsync 变体**——这是 SM100 独有的
- SM120 使用纯 TMA 路径：`MainloopSm120ArrayTmaWarpSpecializedBlockScaled`（`dispatch_policy.hpp:1502`）
- SM120 grouped/array 集合体：`sm120_blockscaled_mma_array_tma.hpp`
- SM120 调度：`KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`（line 617）或 `Pingpong` 变体（line 622）
- SM120 不支持 2SM 模式（无 `KernelSchedule2Sm` 继承）

**SM120 可用调度**：
```
KernelTmaWarpSpecializedNvf4Sm120          — Cooperative, NVFP4
KernelTmaWarpSpecializedPingpongNvf4Sm120  — Pingpong, NVFP4
KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120 — Array/Grouped
KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120   — Array/Grouped
```

---

## 2. vLLM MoE 路由实现

### 2.1 fused_experts_impl 完整数据流

`fused_moe.py:1592` 的 `fused_experts_impl` 是 Triton 路径的核心：

```
hidden_states [M, K]
    │
    ├─① moe_kernel_quantize_input()     → qhidden_states, a1q_scale
    │     BF16 → FP8/FP4 量化
    │
    ├─② _prepare_expert_assignment()    → sorted_token_ids, expert_ids, num_tokens_post_padded
    │     topk_ids → moe_align_block_size → 排序 + 填充
    │
    ├─③ dispatch_fused_moe_kernel()     → intermediate_cache1 [M*topk, N]
    │     GEMM1: qhidden × w1 (gate_up)
    │     Triton kernel: fused_moe_kernel
    │
    ├─④ apply_moe_activation()          → intermediate_cache2 [M*topk, N/2]
    │     SiLU(gate) × up
    │
    ├─⑤ moe_kernel_quantize_input()     → qintermediate_cache2, a2q_scale
    │     中间结果再量化
    │
    ├─⑥ dispatch_fused_moe_kernel()     → intermediate_cache3 [M*topk, K]
    │     GEMM2: qintermediate × w2 (down)
    │
    └─⑦ ops.moe_sum()                   → out_hidden_states [M, K]
          topk 加权求和
```

**6 个 kernel launch + 2 个量化 + 1 个 activation + 1 个 sum = 10+ 个独立操作**

### 2.2 FLASHINFER_CUTLASS Backend 调用链

`FlashInferExperts`（`experts/flashinfer_cutlass_moe.py`）是**单调用单 kernel**：

```python
flashinfer_cutlass_fused_moe(
    input=hidden_states,           # BF16 [M, K]
    token_selected_experts=topk_ids,
    token_final_scales=topk_weights,
    fc1_expert_weights=w1,         # NVFP4 packed
    fc2_expert_weights=w2,
    quant_scales=[a1_gscale, w1_scale, g1_alphas, a2_gscale, w2_scale, g2_alphas],
    activation_type="silu",
    ...
)
```

**内部融合**：routing → permute → GEMM1 → SiLU → GEMM2 → finalize 全在一个 C++ 调用内。
但底层仍是多个 CUDA kernel（doActivation, computeStrides, ExpertPrefixSum, grouped GEMM ×2）。

**NVFP4 路径的量化参数**：
- `a1_gscale`：activation 全局 scale
- `w1_scale`：权重 block scale（view 为 int32）
- `g1_alphas`：per-expert 权重全局 scale（= 1/w_gs）
- FC2 同理

### 2.3 FlashInferB12xExperts（SM120 专用！）

`experts/flashinfer_b12x_moe.py` — **这是 SM120 的原生 MoE 路径**：

```python
class FlashInferB12xExperts:
    """Uses b12x_fused_moe from FlashInfer PR #3080.
    Fuses token dispatch, two GEMMs, SwiGLU activation, and topk-weight
    reduction into a single kernel call.
    Input quantization (BF16→FP4) is performed inside the kernel."""
```

**关键特性**：
- `expects_unquantized_inputs = True`：接收 BF16，kernel 内部做 FP4 量化
- `B12xMoEWrapper`：封装了完整的 MoE 流程
- 权重 SF 预转换为 MMA layout：`convert_sf_to_mma_layout`
- 支持 CUDA Graph：`use_cuda_graph=True`
- 仅支持 NVFP4（kNvfp4Static/kNvfp4Dynamic）
- 不支持 EP（expert parallelism）

**wrapper.run() 调用**：
```python
wrapper.run(
    x=hidden_states,                    # BF16 [M, hidden_dim]
    w1_weight=w1, w1_weight_sf=w1_sf_mma, w1_alpha=g1_alphas,
    fc2_input_scale=self._fc2_input_scale,
    w2_weight=w2, w2_weight_sf=w2_sf_mma, w2_alpha=g2_alphas,
    token_selected_experts=topk_ids,
    token_final_scales=topk_weights,
)
```

### 2.4 Marlin MoE 设计对比

Marlin（`experts/marlin_moe.py`）是 **W4A16** 路径，设计哲学不同：

```
hidden_states [M, K]
    │
    ├─① moe_align_block_size()          → sorted_token_ids, expert_ids
    │
    ├─② ops.moe_wna16_marlin_gemm()     → intermediate_cache1 [M*topk, 2N]
    │     GEMM1: BF16 × W4 (gate_up)
    │     Marlin kernel: SM80 优化，block_size_m 对齐
    │
    ├─③ apply_moe_activation()          → intermediate_cache2 [M*topk, N]
    │     SiLU + clamp
    │
    ├─④ ops.moe_wna16_marlin_gemm()     → output [M*topk, K]
    │     GEMM2: BF16 × W4 (down)
    │
    └─⑤ ops.moe_sum()                   → 加权求和
```

**Marlin 设计特点**：
- 3 个 kernel launch（GEMM1 + activation + GEMM2），比 Triton 路径少
- `moe_block_size` 对齐：token 按 block 排序，每个 block 属于同一 expert
- SM80 架构优化（无 TMA，无 tensor core FP4）
- 支持 `fe2m1f`（FP4 E2M1）权重 + BF16/FP8 activation
- `use_fp32_reduce=True`：FP32 归约保证精度
- 不支持 in-kernel 量化（activation 保持 BF16/FP8）

### 2.5 Routing 开销分析（来自项目 profiling）

来自 `notes/2026-07-23-laguna-moe-node-trace.md`：

| 组件 | ms/step (b1) | 占比 |
|---|---:|---:|
| **routed MoE grouped GEMM** (cutlass ×94) | **2.63** | 20% |
| cutlass splitKreduce (×238) | 0.33 | 2.5% |
| **MoE 路由开销** (doActivation×47 + computeStrides×47 + ExpertPrefixSum×47) | **~0.5** | 3.8% |
| MoE routing/perm/fin 总计 | 1.02 | 7.8% |
| **MoE 路径总计** | **~3.6** | **27%** |

---

## 3. 模型参数（Laguna-S-2.1）

```
hidden_size = 3072
num_experts = 256
top_k = 10
moe_intermediate_size = 1024
shared_expert_intermediate = 1024
num_layers = 48 (12 全局 attn + 36 SWA)
MoE 层数 = 47
vocab_size = 100352
```

**每 expert 权重（NVFP4）**：
- gate_up: 3072 × 2048 × 0.5 bytes = 3.15 MB（gate + up 合并）
- down: 1024 × 3072 × 0.5 bytes = 1.57 MB
- 合计 ~4.7 MB/expert（加 SF ~5.3 MB）

**M=1 decode 带宽账**：
- top-10 × 47 层 = 470 个 expert 被激活
- 权重流量：470 × 5.3 MB ≈ 2.49 GB
- Peak BW 1338.8 GB/s → 理论下限 1.86ms
- 实测 2.63ms → **63% peak（843 GB/s）**
- 提到 85% peak → 1.56ms（省 ~1.1ms）

---

## 4. 融合机会分析

### 4.1 当前 FLASHINFER_CUTLASS 的 kernel 链（每 MoE 层）

```
① doActivation        — topk gating 后处理
② computeStrides      — 计算每个 expert 的 stride
③ ExpertPrefixSum     — expert token 前缀和
④ grouped GEMM (gate_up) — cutlass GroupProblemShape
⑤ splitKreduce        — split-K 归约（如果启用）
⑥ SiLU + quantize     — activation + FP4 量化
⑦ grouped GEMM (down) — cutlass GroupProblemShape
⑧ splitKreduce        — split-K 归约
⑨ finalize/sum        — topk 加权求和
```

**× 47 层 = ~423 个 kernel launch/step**

### 4.2 可融合操作

**Level 1：GEMM + Epilogue 融合（CUTLASS 已支持）**

```
gate_up GEMM + SiLU + FP4 量化输出
→ LinCombEltActBlockScaleFactor<SiLu, 16, FP4, float, ue4m3_t, ...>
```

CUTLASS 的 `blockscaled_rcgrouped` 示例已验证此路径。这消除了 ⑥ 的独立 kernel。

**Level 2：Routing + GEMM 融合**

将 ①②③ 的 routing 开销融入 GEMM kernel 的 tile scheduler：
- `MoEProblemShape` 已支持 device-side `tokens_per_expert`
- 可在 tile scheduler 中直接消费 topk_ids，省去独立的 permute/prefix-sum

**Level 3：完全融合 Expert Kernel（终极目标）**

```
单 kernel 完成：
  BF16 input → FP4 量化 → gate_up GEMM → SiLU → FP4 量化 → down GEMM → topk 加权 → BF16 output
```

这正是 FlashInfer `b12x_fused_moe` 的设计（SM120 专用）。

### 4.3 M=1..16 Autotune 空间

| 参数 | M=1 | M=4 | M=16 | 说明 |
|---|---|---|---|---|
| TileM | 128 | 128 | 128 | SM120 1SM 固定 |
| TileN | 16~64 | 32~64 | 64~128 | decode 小 N |
| TileK | 128~256 | 128~256 | 128~256 | FP4 K 方向深 |
| Cluster | 1×1 | 1×1 | 1×1~2×1 | SM120 无 2SM |
| Stages | 4~8 | 4~8 | 3~6 | smem 受限 |
| SplitK | 1~4 | 1~2 | 1 | M 小需 splitK |

**关键约束**：
- M=1 时每个 expert 最多 1 个 token → N=1 → 纯 GEMV
- 256 experts 中只有 10 个被激活 → 97% 的 expert 权重不读
- 带宽利用率取决于能否将多个 expert 的 GEMV 合并为 batched GEMV

### 4.4 与 Marlin 的设计对比

| 维度 | Marlin | CUTLASS MoE | 自研目标 |
|---|---|---|---|
| 架构 | SM80（Ampere） | SM100（Blackwell） | SM120 |
| 权重格式 | W4A16（INT4/NF4） | NVFP4 W4A4 | NVFP4 W4A4 |
| 加载方式 | 全局内存 + shared | TMA + CPASYNC | TMA（SM120 无 CPASYNC MoE） |
| 量化 | 外部（activation 保持 BF16） | 外部或 epilogue 融合 | **In-kernel BF16→FP4** |
| 融合度 | GEMM only（3 launch） | GEMM + SF（2 launch） | **全融合（1 launch）** |
| 小 M 优化 | block_size_m 对齐 | MoEProblemShape | 专用 GEMV/batched GEMV |
| Tensor Core | WMMA（16×16） | UMMA（block-scaled） | SM120 UMMA |

### 4.5 设计建议

**推荐路径：基于 FlashInfer B12xMoEWrapper 的自研适配**

理由：
1. `FlashInferB12xExperts` 已是 SM120 原生路径，设计完全匹配
2. 单 kernel 调用，内部融合 routing + 2×GEMM + SiLU + 量化 + finalize
3. 支持 CUDA Graph（`use_cuda_graph=True`）
4. 接受 BF16 输入，kernel 内部做 FP4 量化
5. 权重 SF 预转换为 MMA layout，避免运行时开销

**自研 kernel 的差异化方向**：

1. **M=1 GEMV 特化**：
   - 当前 grouped GEMM 对 M=1 效率低（tile 利用率 1/128）
   - 自研：每个 CTA 处理 1 个 expert 的完整 GEMV
   - 10 个 active expert → 10 个 CTA，每个读 ~5.3 MB 权重
   - 目标：85%+ peak BW（vs 当前 63%）

2. **Routing 零开销**：
   - 将 topk gating 结果直接编码为 CTA 分配
   - 省去 doActivation/computeStrides/ExpertPrefixSum 三个 kernel
   - 节省 ~0.5ms/step

3. **两级 GEMM 流水线**：
   - gate_up GEMM 的 SiLU 输出直接 feed 到 down GEMM
   - 中间结果留在 shared memory / register，不写 HBM
   - 对 M=1：中间向量仅 1024 × 2 bytes = 2 KB，完全可驻留

4. **Split-K 消除**：
   - 当前 238 个 splitKreduce kernel（0.33ms）
   - M=1 时 K=3072/1024 足够小，不需要 split-K
   - 自研 kernel 直接做完整 K 归约

**性能目标**：
- 当前 MoE 路径：~3.6ms/step（GEMM 2.63 + routing 0.5 + splitK 0.33 + 其他 0.14）
- 目标：~2.0ms/step（85% peak BW GEMM + 零 routing 开销 + 无 splitK）
- 节省：~1.6ms/step（44% 减少）

**实现优先级**：
1. 先验证 FlashInfer B12xMoEWrapper 在 SM120 上的实际性能（env 切换测试）
2. 如果 B12x 已接近目标，直接采用 + 微调
3. 如果 B12x 不够优化，基于 CUTLASS SM120 array TMA 集合体自研
4. 最终目标：完全融合的 single-launch expert kernel

---

## 附录：关键文件索引

### CUTLASS 4.6.1
- `examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu` — FP4 MoE grouped GEMM 示例
- `examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` — SiLU+SF 融合示例
- `include/cutlass/gemm/group_array_problem_shape.hpp:85` — MoEProblemShape 定义
- `include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp` — SM100 MoE kernel
- `include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp` — **SM120 array/grouped 集合体**
- `include/cutlass/gemm/dispatch_policy.hpp:1502` — SM120 array block-scaled 调度策略
- `include/cutlass/detail/sm100_blockscaled_layout.hpp` — Block scaling factor 布局
- `include/cutlass/epilogue/fusion/operations.hpp` — 融合操作表

### vLLM
- `vllm/model_executor/layers/fused_moe/fused_moe.py:1592` — fused_experts_impl 数据流
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` — FLASHINFER_CUTLASS backend
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py` — **SM120 B12x backend**
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` — Marlin W4A16 backend
- `vllm/model_executor/layers/fused_moe/modular_kernel.py` — 模块化 kernel 抽象
- `vllm/utils/flashinfer.py:328` — has_flashinfer_b12x_moe() 检测

### 项目
- `notes/2026-07-23-laguna-moe-node-trace.md` — MoE kernel 级 profiling 取证
- `runtime/backends/laguna.py` — Laguna 模型后端
