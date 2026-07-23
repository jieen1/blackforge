# Laguna MoE Node Trace — 生产残留量取证（2026-07-23）

> 用户裁定：profiling 驱动，别按 eager 数字立项。本笔记 = graph/compile 生产
> 路径 MoE 残留量的 kernel 级取证 + GEMM 家族归属，作为 L3 性能攻坚的立项依据。
> 脚本：`benchmarks/laguna_moe_node_trace.py`（单次加载跑全矩阵）
> 数据：`benchmarks/fixtures/laguna_moe_node_trace_eager.json`

## 方法

- torch.profiler 抓 decode step kernel 分解，eager 模式（kernel 逐个可见）；
- 物理论据：**MoE GEMM kernel 时间是 eager/compile 模式不变量**——同一批
  cutlass/nvjet kernel 读同样权重，compile 只融合 elementwise + 消 launch gap，
  不碰 GEMM。故 eager 测得的 GEMM 即生产残留量；compile 只压缩 norm/quant + gap。
- 单次加载（~65s）遍历 (batch 1/4 × ctx 1017/16380) 四组合，避免重复读 71GB 盘。
- 验证：kernel/step 13.15ms 复现用户 eager ledger 14.88ms（方法学对齐）。

## 修正后 ledger（ctx=1017, batch=1, eager）

| 类别 | ms/step | 占比 | 对账用户 ledger |
|---|---:|---:|---:|
| GEMM (MoE+dense) | 7.51 | 57.1% | 8.73 (58.7%) |
| Norm/quant/elem | 3.08 | 23.5% | 2.77 (18.6%) |
| MoE routing/perm/fin | 1.02 | 7.8% | 1.20 (8.1%) |
| Attention(+RoPE+KV) | 0.69 | 5.2% | 0.96 (6.4%) |
| lm_head (gemvx) | 0.79 | 6.0% | 0.42 (2.8%) |

> lm_head 偏高疑点：gemvx 桶可能连带吃了解码 QKV gemvx，需 shape 标注确认。
> 不影响主结论（GEMM 主导）。

## GEMM 家族归属（关键发现）

| 家族 | ms/step (b1) | b4 | 归属 |
|---|---:|---:|---|
| **nvjet** (SM120 JIT) | 4.35 | 4.95 | **dense GEMM**：QKV/out/gate_up（×96/×36/×94/×12 对应层结构）；shared expert 疑似在此（待 shape 确认） |
| **cutlass** GroupProblemShape | 2.63 (×94) | 2.54 | **routed MoE grouped GEMM**（GroupProblemShape=分组；×94=47 MoE 层×2 gate_up+down） |
| cutlass splitKreduce | 0.33 (×238) | 0.36 | grouped GEMM 的 split-K 归约 |
| MoE 路由开销 (cutlass trtllm) | ~0.5 | ~0.5 | doActivation×47 + computeStrides×47 + ExpertPrefixSum×47 |

**routed MoE grouped GEMM (cutlass 2.63ms) + 路由开销 (~1ms) ≈ 3.6ms 是自研融合
expert kernel 的可吸收目标；nvjet dense 4.35ms 是 vLLM 调优路径，非主战场。**

## batch 扩展性（生产 4 并发）

| ctx | batch | kernel | GEMM | attention |
|---|---|---:|---:|---:|
| 1017 | 1 | 13.15 | 7.51 | 0.69 |
| 1017 | 4 | 14.27 | 8.79 | 0.83 |
| 16380 | 1 | 13.50 | 7.58 | 0.93 |
| 16380 | 4 | 15.31 | 8.91 | 1.71 |

- batch 1→4 GEMM 仅 +17%（7.5→8.8ms）→ **M=1..4 带宽受限**（读专家权重主导，
  非算力），印证用户「HBM 流式单遍读」自研设计；M=16（DFlash verify）同甜区。
- attention 随 ctx 增长（SWA 36 层窗口 512 有界 + 12 全局层），长上下文 batch=4
  升到 1.71ms，仍是边角。

## 结论与下一步

1. **战场收窄确认**：生产路径 MoE GEMM + 路由 ≈ 65%，唯一主战场。
2. **先打谁**：routed cutlass grouped GEMM (2.63ms, BW 利用率 ~24%) + 路由开销。
3. **下一步取证**（~30min，复用加载）：torch.profiler `record_shapes=True` 按
   M/N/K 分离 routed vs shared vs dense GEMM，确认 shared expert 归属（nvjet?），
   定死自研 kernel 的吸收范围。
4. 随后走用户三级火箭①：vLLM MoE backend 扫描（marlin/triton/trtllm）+ marlin
   悬案（csrc/moe/marlin_moe_wna16/、csrc/quantization/marlin/ 两个未跟踪目录）。

## 带宽账（config 静态归属，M=1 decode 纯权重读，peak 1338.8 GB/s）

模型维度：hidden=3072 · 48 层（12 全局 attn + 36 SWA）· 256 专家 top-10 ·
moe_intermediate=1024 · shared_expert=1024 · vocab=100352。

| GEMM | 权重流量 | 带宽下限 | 实测 | 利用率 | 归属 kernel |
|---|---:|---:|---:|---:|---|
| **routed experts** (top10×47层) | 2.218 GB | 1.66ms | **2.63ms** | **63% peak** | cutlass GroupProblemShape ×94 |
| shared expert (47层) | 0.222 GB | 0.17ms | (估在 nvjet) | — | nvjet 候选 |
| attn QKV+out (48层) | 1.057 GB | 0.79ms | ~1.0ms | ~79% | nvjet ×96/×36/×12 |
| lm_head **bf16** | 0.617 GB | 0.46ms | 0.42-0.79 | — | gemvx（**未量化**） |

**修正判断**：routed MoE grouped GEMM 是 **63% peak（843 GB/s），非接近上限**——
提到 85% peak 可到 1.23ms（省 ~1.4ms）。加上路由开销 ~1ms 可融合，MoE 路径
（现 ~3.6ms）现实目标 ~2ms。

## 优化杠杆排序（证据定）

1. **routed MoE grouped GEMM 2.63ms @ 63% peak** → 自研融合 kernel 或更优 backend
   提到 85% peak 省 ~1.4ms；融合 routing 开销再省 ~1ms。
2. **lm_head bf16 0.42ms 未量化** → 量化 NVFP4 省 ~0.3ms（白捡，用户 ③ 边角）。
3. nvjet dense 4.35ms（QKV/out/shared）≈79% peak，vLLM 调优路径，非主战场。

## MoE backend 扫描清单（vLLM 已暴露，当前 FLASHINFER_CUTLASS）

可选：`FLASHINFER_TRTLLM` · `FLASHINFER_CUTEDSL` · `FLASHINFER_CUTEDSL_BATCHED` ·
`VLLM_CUTLASS` · **`MARLIN`** · `HUMMING` · `EMULATION`。
→ 三级火箭①：env 切换测吞吐，MARLIN 专测 M≤16 带宽利用率。
