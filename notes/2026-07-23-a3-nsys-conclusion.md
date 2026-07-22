# A3 · MTP 融合 nsys 验证结论（2026-07-23）

## 核心问题

A3 前期分析（2026-07-22）提出 Draft K-step batching 可减少 ~30% kernel launch overhead。
本验证用 nsys + 生产路径数据确认：**CUDA Graphs 已经消除了 launch overhead，A3 收益远低于门禁。**

## 证据

### 1. 生产路径已使用 CUDA Graph

`CapturedMTPDraftStepGraph.replay_incremental()` 将每个 draft step 的所有 kernel
打包为单次 `graph.replay()` 调用。K=3 时只有 3 次 graph replay + 3 次 `_fill_buffers_incremental`。

### 2. Eager vs CUDA Graph 对比（128K/c=4）

| 模式 | accepted tok/s | step time | 差异 |
|------|---------------|-----------|------|
| Eager (no graph) | 203.45 | 70.3ms | — |
| CUDA Graph (生产) | 212.2 | ~67.4ms | **+4.3%** |

CUDA Graph 仅节省 ~3ms/step — GPU 计算是瓶颈，不是 launch overhead。

### 3. nsys kernel 分析（4K/c=4, CUDA Graph 路径）

- `cudaLaunchKernel`: 4027 calls, avg 51.9μs/call（含 prefill + warmup）
- 生产 decode 路径中，每个 draft step 是 1 次 `graph.replay()`（~5-10μs CPU 开销）
- GPU kernel 占 step time 的 94%+（63.3ms CUDA / 67.4ms wall）

### 4. A3 理论最大收益

Draft K-step batching 会省去：
- 2 次 `graph.replay()` 调用（~10-20μs）
- 2 次 `_fill_buffers_incremental` CPU 工作（~0.3-0.5ms）
- 合计 ~0.5-1ms/step ≈ **0.7-1.5% 改善**

## 裁决

**A3 Draft K-step batching 不值得实施。**

- 预期收益 0.7-1.5% << 门禁 5%
- 原因：CUDA Graphs 已经将 K 个 draft step 的 kernel launch 压缩为 K 次 graph replay
- GPU 计算（NVFP4 GEMM 54-77% + Attention 3.5-28%）是绝对瓶颈
- 剩余 CPU 开销（buffer fill + accept/reject）仅占 ~4-6%

### 其他 A3 融合点评估

| 融合点 | 预期收益 | 裁决 |
|--------|---------|------|
| Draft K-step batching | 0.7-1.5% | ❌ 远低于门禁 |
| GPU-side accept | ~0.5% (省 1 次 sync ~10-50μs) | ❌ 不值得 |
| Draft logits 融合 | 0% (MTP 已用 hidden state) | ❌ 已实现 |
| GDN state scatter | ~0.3% | ❌ 不值得 |

## 后续

A3 关闭。性能优化重心转向：
- A6 adaptive split（128K 下 19% 加速，需 kernel 改动）
- 接受率提升（算法层面，非 kernel）
- CPU 开销进一步优化（Python 循环、metadata 构建）

---

*数据来源：nsys profile /tmp/a3_decode_profile.nsys-rep, benchmarks/fixtures/speed_baseline.json, /tmp/fresh_profile_128k.log*
