# A1a: GDN 逐层 Profiling — Decode Step 时间占比账本（2026-07-22）

> 路线图 A1a 门票：nsys/torch.profiler 分解 decode step 的完整 kernel 序列。

## 测试条件

- prompt_len=4096, concurrency=1, K=3 (MTP), eager mode (no CUDA Graph)
- GPU: RTX PRO 6000 Blackwell (SM120, 96 GB)
- 10 MTP verify rounds, torch.profiler CUDA activity capture

## 修正后占比（手工修正 kernel 分类）

| 类别 | GPU time (ms/10 rounds) | 占比 | 关键 kernel |
|---|---:|---:|---|
| **NVFP4 GEMM** | ~192 | **71.1%** | cutlass sm120 NVFP4 GEMM (2 个 kernel 合计 71%) |
| **Other (quant/act/memcpy)** | ~22 | 8.2% | fp8_quant, act_and_mul, DtoD memcpy |
| **GDN 全栈** | ~14 | **5.1%** | fused_sigmoid_gating (2.0%), rms_norm (2.0%), causal_conv1d (0.6%) |
| **Attention (SM120 kernel)** | ~10 | **3.5%** | decode_v2_nativefp8 (1.8%), prefill_fp8 (1.2%), partial (0.5%) |
| **WMA GEMM (bf16)** | ~23 | 8.5% | cutlass_80_wmma (5.5%), gemvx (3.0%) |

Per-round GPU time: **27.01 ms** (eager, c=1, 4K context)

## 关键发现

1. **NVFP4 GEMM 是绝对主导**（71%），远超 attention（3.5%）和 GDN（5.1%）。
   这与 128K 长上下文场景不同——128K 时 attention 占比会大幅上升。

2. **GDN 48 层合计仅占 5.1%**（每层 ~0.1ms），在 4K 上下文下融合收益有限。
   但 GDN 的占比随上下文长度变化不大（线性注意力），而 attention 随上下文
   线性增长——128K 时 attention 可能占 40%+，GDN 占比相对缩小。

3. **Attention kernel 已经很快**（3.5%），自研 SM120 kernel 的 1.56× 加速
   在 4K 上下文下对端到端贡献有限（~2% 端到端提速）。

4. **M2 优先级建议**：
   - 4K 短上下文：A2（NVFP4 GEMM autotune）收益最大（71% 占比）
   - 128K 长上下文：需要重跑 profiling 确认 attention vs GDN 占比
   - GDN 融合（A1）在 4K 下收益有限，但在 128K 下可能有意义

## 下一步

- [ ] 重跑 128K context profiling（需要更多 GPU 内存和时间）
- [ ] 对比 CUDA Graph 模式下的占比（eager 有额外 Python 开销）
- [ ] 按占比排序更新 M2 工作计划
