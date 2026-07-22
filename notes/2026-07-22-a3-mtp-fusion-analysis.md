# A3 · MTP 链路融合前期分析（2026-07-22）

## 当前 MTP 实现概况

Runner 中 18 个 MTP 相关方法，核心数据流：

```
mtp_prefill_batch → _mtp_forward_batch → _mtp_sync_and_propose_batch
                                              ↓
                              draft_tokens[K] per slot
                                              ↓
mtp_verify_and_commit_batch → verify_batch_spec → determine_accept_reject_batch
                                              ↓
                              committed tokens + GDN state update
```

### 每轮 MTP 的 kernel 调用序列（K=3, batch=4）

1. **Draft forward ×K**（3 次独立 forward）:
   - NVFP4 GEMM (gate_up + down) × 48 layers × 3 steps
   - GDN (conv1d + SSM update) × 48 layers × 3 steps
   - Attention (SM120 kernel) × 16 layers × 3 steps
   - RMSNorm × 48 layers × 3 steps

2. **Verify forward** (1 次, qo_len=K+1=4):
   - 同上但 qo_len=4（MMA kernel path）

3. **Accept/reject** (CPU, 已提取为 mtp_accept.py):
   - argmax + compare（向量化，1 次 GPU→CPU）

4. **GDN state commit**:
   - 根据 accept 结果选择正确的 SSM state row

### 融合机会分析

| 融合点 | 当前开销 | 融合方案 | 预期收益 | 难度 |
|--------|---------|---------|---------|------|
| Draft K 步合并 | K 次独立 kernel launch | 单次 batched forward (batch×K) | 减少 launch overhead ~30% | M |
| Draft logits 计算 | 每步独立 compute_logits | 融合到最后一步 | 省 K-1 次 lm_head GEMM | S |
| Verify + accept | verify forward + CPU roundtrip | GPU-side accept (custom kernel) | 省 1 次 sync | M |
| GDN state scatter | accept 后 CPU 索引 + copy | Fused accept+scatter kernel | 省 1 次 sync + copy | L |
| Draft attention | K 次独立 attention call | Persistent KV + single multi-step attention | 减少 KV read ×K | L |

### 优先级排序（按 ROI）

1. **Draft K 步合并**（M）：最大收益，减少 ~30% kernel launch overhead
   - 当前：3 次独立 forward（每次 48 层 × 4 kernels）
   - 目标：1 次 batched forward（batch=4×3=12，共享权重）
   - 门禁：golden fixtures bit-exact + accepted tok/s 不降

2. **Draft logits 融合**（S）：省 K-1=2 次 lm_head GEMM
   - 当前：每步都算 logits（用于下一步 input）
   - 目标：只在最后一步算 logits（中间步用 hidden states）
   - 注意：MTP 的 draft input 是上一步的 hidden state，不是 token
   - 实际上 MTP 已经是这样做的！需要确认

3. **GPU-side accept**（M）：消除 verify→accept 的 CPU roundtrip
   - 当前：verify_logits → CPU argmax → compare → CPU decision
   - 目标：custom kernel 在 GPU 上做 argmax + compare + scatter
   - 收益：省 1 次 cudaDeviceSynchronize (~10-50μs)

### 门禁（合入标准）

- accepted tok/s 净收益 ≥ 5%
- 接受率不降 > 1pp
- golden fixtures bit-exact（greedy 路径）
- 关闭投机时 fallback 稳定（纯 decode 不受影响）

### 依赖

- 需要 sm120-flash-attention 的 MTP-verify kernel 支持（已有：`flash_attn_decode_partial_kernel` 的 qo_len>1 路径）
- 需要 FLA GDN kernel 的 batched multi-step 支持（待确认）
- CUDA Graph 兼容性（所有融合后的 kernel 必须可捕获）

### 结论

A3 的最大杠杆是 **Draft K 步合并**（减少 kernel launch overhead）。
但需要确认 FLA GDN kernel 是否支持 batched multi-step（当前是逐步调用）。
建议：先用 nsys profile 确认 launch overhead 占比，再决定是否值得做。
