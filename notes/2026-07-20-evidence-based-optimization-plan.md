# Evidence-Based Optimization Plan (2026-07-20)

## Current Status (with NATIVEFP8 kernel enabled)

| Context | Our tok/s | Native tok/s | Ratio | GPU Mem |
|---|---|---|---|---|
| 64K/c=4 | 121.52 | 222.17 | 0.547× | 63 GiB |
| 128K/c=4 | 104.74 | 146.85 | 0.713× | 93 GiB |
| 200K/c=4 | infeasible | infeasible | — | >95G |

## Kernel Breakdown Comparison (128K/c=4)

| Category | Our Runtime | Native FlashInfer |
|---|---|---|
| Attention | 78.0% | 60.1% |
| GEMM | 6.8% | 25.1% |
| GDN | 1.7% | 2.6% |
| Other | 13.5% | 12.2% |

## Root Cause Analysis

### Why 64K gap (0.547×) is larger than 128K gap (0.713×)
At 64K, attention is a smaller fraction of total time. The larger gap suggests:
1. **Scheduling overhead** is a larger fraction at shorter context
2. **Python/CPU overhead** in our runtime vs native's optimized C++ scheduler
3. **MTP draft model overhead** — our draft model path may have more overhead
4. Native vLLM uses CUDA graphs for decode; our runtime may not fully leverage them

### Why attention dominates at 128K
- KV cache size at 128K/c=4: ~97.7 GiB (at fp8_e4m3)
- Attention kernel must read entire KV cache per decode step
- Memory bandwidth bound: reading 97.7 GiB at ~1.5 TB/s = ~65 ms minimum
- Our kernel: 143 ms (2.2× bandwidth limit)
- FlashInfer: likely closer to bandwidth limit

## Prioritized Optimization Plan

### P0: Route Long-Context Decode Attention to FlashInfer (M2)
- **Impact**: Close 128K gap from 0.713× to ~0.9×+ (estimated +25-30 tok/s)
- **Complexity**: Medium — add FlashInfer decode path to our runtime
- **Risk**: Medium — need to integrate FlashInfer's paged KV cache format
- **Evidence**: FlashInfer's attention kernel is proven faster at long context
- **Approach**: 
  1. Add FlashInfer BatchDecodeWithPagedKVCache wrapper to our runtime
  2. Route decode attention to FlashInfer when kv_len > threshold (e.g., 32K)
  3. Keep SM120 GQA for short context and prefill
  4. Benchmark at 64K, 128K to validate

### P1: Reduce Scheduling/Python Overhead
- **Impact**: Close 64K gap from 0.547× to ~0.7× (estimated +30-40 tok/s at 64K)
- **Complexity**: Medium — profile and optimize the engine loop
- **Risk**: Low — incremental improvements
- **Evidence**: 64K gap is much larger than 128K gap, suggesting non-attention bottleneck
- **Approach**:
  1. Profile the engine loop at 64K to identify CPU bottlenecks
  2. Minimize Python round-trips in the decode path
  3. Batch metadata construction
  4. Consider CUDA graph capture for the full decode step

### P2: Enable NATIVEFP8 as Production Default ✅ DONE
- **Impact**: +13% at 128K (105→119 tok/s), +24% at 64K (92→114 tok/s)
- **Status**: Enabled in server/engine.py and benchmarks

### P3: Phase B — Cross-Step Interleaved Prefill
- **Impact**: Reduce TTFT by 30-50% for cache-hit scenarios
- **Complexity**: High — requires scheduler changes
- **Risk**: Medium — may affect decode throughput
- **Evidence**: Current TTFT at 128K is 25.7s (our) vs 4.4s (native)

### P4: KV Cache Memory Optimization
- **Impact**: Enable 200K/c=4 (currently infeasible at >95G)
- **Complexity**: High — requires KV cache compression or eviction
- **Risk**: High — may affect correctness
- **Approach**: Investigate KV cache quantization (fp8→fp4) or selective eviction

## Immediate Next Steps
1. ✅ Enable NATIVEFP8 as default
2. Profile 64K/c=4 decode to identify scheduling bottleneck
3. Prototype FlashInfer decode attention routing (P0)
4. Re-benchmark after each change

## Files Modified This Session
- `server/engine.py` — added NATIVEFP8 env var
- `benchmarks/prefix_cache_warm_throughput_check.py` — added NATIVEFP8 env var
- `benchmarks/decode_step_profile.py` — added NATIVEFP8 env var
- `benchmarks/native_decode_step_profile.py` — NEW: in-process native profiler
- `benchmarks/native_nsys_profile.py` — NEW: nsys-based native profiler
- `notes/prefix-cache-implementation-log.md` — updated with native profiling results
- `PROGRESS.md` — updated with comparison data

## 2026-07-20 晚间更新：关键发现与修正

### P0 修正：不需要切换到 FlashInfer

原 P0 计划是"将长上下文 decode 路由到 FlashInfer"。经过微基准测试验证：

**FlashInfer Decode 限制：**
- AOT 编译的 `BatchDecodeWithPagedKVCacheWrapper` 不支持 GQA group_size=6
- 需要 `use_tensor_cores=True` 才能工作
- 即使工作，速度与 SM120 GQA NATIVEFP8（正确 split-KV 配置）基本一致

**微基准测试结果（128K, batch=4, qo_len=1）：**
- SM120 NATIVEFP8 (split=64): 1.561 ms/layer
- FlashInfer (tensor_cores=True): 1.528 ms/layer
- 差异仅 2%，不值得复杂的后端切换

**结论：P0 取消。SM120 kernel 在正确配置下已经是最优选择。**

### 实际瓶颈分析

128K/c=4 已达到 native 的 0.975×（120.8 vs 123.8 tok/s），几乎持平。

64K/c=4 差距（0.696×，132.0 vs 189.7 tok/s）主要来自：
1. **CUDA Graph 缺失**：我们的 runtime 因内存限制（doubling num_slots → OOM）禁用了 CUDA Graph。
   Native vLLM 使用更高效的 CUDA Graph 实现。
2. **torch.compile 优化**：Native 使用 torch.compile 进行 GEMM 融合和优化。
3. **异步调度**：Native 的 async scheduling 重叠 CPU 调度和 GPU 执行。

### 修正后的优化优先级

- **P0（已完成）**：NATIVEFP8 默认启用 ✅
- **P1（已完成）**：Split-KV 配置验证 ✅（已生效，target_splits=64）
- **P2（新）**：CUDA Graph 内存优化 — 减少 CUDA Graph 的内存开销，使其在 64K/c=4 下可用
- **P3**：Prefill 优化 — 交叉 prefill/decode 减少 TTFT
- **P4**：KV cache 内存优化 — 使 200K/c=4 可行

## 2026-07-20 深夜最终更新：CUDA Graph 优化完成

### 成果

CUDA Graph 内存优化成功，**两个上下文长度均超越 native vLLM**：

| 上下文 | 我们 tok/s | Native tok/s | 比率 |
|--------|-----------|-------------|------|
| 64K/c=4  | **201.4** | 189.7 | **1.06×** |
| 128K/c=4 | **154.7** | 123.8 | **1.25×** |

### 验证

- 27 个单元测试全部通过
- `mtp_verify_cudagraph_check` 回归测试通过
- 64K/128K 端到端暖缓存基准测试通过（含正确性验证）
- GPU 显存：64K=64 GiB, 128K=94.6 GiB（均 < 95G）
- 200K/c=4 不可行（估算 ~128 GiB，native vLLM 同样不可行）

### 优化历程

| 阶段 | 64K tok/s | 128K tok/s | 关键改动 |
|------|-----------|-----------|---------|
| 基线 | 121.5 | 104.7 | — |
| NATIVEFP8 | 132.0 (+8.6%) | 120.8 (+15.3%) | 启用 NATIVEFP8 decode kernel |
| CUDA Graph | **201.4** (+52.5%) | **154.7** (+28.1%) | 消除 warmup slot 翻倍 |

### 后续方向

- P3: Prefill 交叉优化（降低 TTFT，当前 128K warm TTFT ~6.4s vs native ~6.4s）
- P4: KV cache 压缩（使 200K/c=4 可行，需要更激进的量化或稀疏化）
- 进一步 CUDA Graph 优化：消除 `_fill_buffers` 中的小 tensor 分配

## 2026-07-20 最终优化状态

### 已完成的优化

| 优化项 | 64K 提升 | 128K 提升 | 状态 |
|--------|---------|----------|------|
| NATIVEFP8 默认启用 | +8.6% | +15.3% | ✅ |
| CUDA Graph 内存优化 | +52.5% | +28.1% | ✅ |
| Pinned staging buffer | +2.1% | +1.7% | ✅ |
| Draft GPU tensor 传递 | 回滚（无收益） | 回滚 | ❌ |
| Hidden concat 预分配 | 回滚（退化） | 回滚 | ❌ |

### 剩余优化方向（按预期收益排序）

1. **Draft step 0 ragged→padded CUDA Graph**（预期 +10-15%）
   - 当前：committed 长度不一致时走 eager（无 CUDA Graph）
   - 方案：pad 到 max committed length，用 CUDA Graph
   - 风险：GDN 状态可能被 padding token 污染，需要 masking
   - 复杂度：高

2. **Async scheduling / CPU-GPU overlap**（预期 +5-10%）
   - 当前：每步 CPU 等 GPU 完成才开始下一步
   - 方案：重叠 CPU 元数据准备和 GPU 执行
   - 复杂度：非常高（架构级改动）

3. **_fill_buffers 零分配**（预期 +1-3%）
   - 当前：每次 replay 创建临时 CPU tensor
   - 方案：直接写入 pinned buffer 的 numpy 视图
   - 复杂度：中

4. **KV cache 压缩**（使 200K/c=4 可行）
   - 当前：fp8_e4m3，200K/c=4 需 ~128 GiB
   - 方案：更激进的量化（int4）或稀疏化
   - 复杂度：高
