# Decode Attention Kernel 优化规划

## 当前状态 (ncu profiling)
- Kernel: `flash_attn_decode_v2_fp8kv_paged_split_nativefp8<256, 6, 0>`
- Registers: **255/thread** (max), 256 threads/CTA (8 warps)
- SMEM: 56KB dynamic (SM has 228KB, 余量 172KB)
- Occupancy: **16.67%** (1 CTA/SM, register-limited)
- DRAM: **40.81%** of peak (~735 GB/s actual vs 1800 GB/s peak)
- Warps active: 7.98/8 (all active)
- Inst/cycle: 0.24 (极度 memory-bound, warps 大部分时间等数据)
- Grid: (32 splits, 4 KV heads, 4 batch) = 512 CTAs on 96 SMs

## 寄存器分布估算 (255 regs/thread)
| 消费者 | 估算 regs | 说明 |
|--------|----------|------|
| O_acc (output accumulator) | ~128 | D=256, 32 N-tiles × 4 elements/lane |
| MMA fragments (A/B operands) | ~40 | QK^T 和 PV 的 MMA 操作数 |
| Online softmax (m, l) | ~12 | 6 GQA heads × 2 floats |
| Q staging / addresses | ~30 | Q 量化 + 地址计算 |
| Loop state + misc | ~45 | 循环变量、page table 索引等 |
| **Total** | **~255** | |

## 已尝试（失败）
- `__launch_bounds__(256, 2)`: 编译器 spill 到 local memory → 0.77× 退化

## 优化方案（按优先级）

### P0: D-dimension PV tiling（预期 -64 regs → ~191 regs）
**原理**: PV 计算分 2 pass（D/2=128 each），O_acc 峰值从 128 降到 64 floats/lane。
第一 pass 的 O_acc 存 SMEM（172KB 余量足够）。

**实施**:
1. KV loop 内部：QK^T + softmax 不变（full D=256）
2. P 矩阵存 SMEM（[6, 32] × 4B = 768B/tile，极小）
3. PV 分 2 pass：先 V[0:128] → O_acc_lo，存 SMEM；再 V[128:256] → O_acc_hi
4. 最终 merge：从 SMEM 读回 O_acc_lo + O_acc_hi

**预期效果**: 255 → ~191 regs。仍不够 2 CTAs（需 ≤128），但可能让编译器更好优化。
**风险**: 额外 SMEM 读写可能抵消收益。需要实测。
**难度**: 高（kernel 核心循环重构）

### P1: 减少 warps/CTA（256→128 threads）
**原理**: 128 threads × 255 regs = 32,640 regs/CTA → 2 CTAs/SM (65,280 < 65,536)
**实施**: 将 8 warps 的工作重新分配给 4 warps（每个 warp 处理 2× KV tiles）
**预期效果**: occupancy 16.67% → 33.3%，DRAM 40% → 60-70%
**风险**: 4 warps 的 latency hiding 能力弱于 8 warps × 2 CTAs
**难度**: 高（warp 分配逻辑重写）

### P2: Persistent kernel
**原理**: 96 CTAs（1/SM），每个 CTA 内部循环处理多个 (split, head, batch)
**实施**: 外层循环遍历 work queue，内层保持当前 kernel 逻辑
**预期效果**: 消除 launch overhead + L2 跨 split 复用
**风险**: 不改变 occupancy，收益可能有限（launch overhead 本身很小）
**难度**: 中

### P3: GQA_GROUP 减半（6→3）
**原理**: 每个 CTA 处理 3 Q heads（而非 6），减半 Q-related 寄存器
**预期效果**: ~128 regs → 可能 fit 2 CTAs
**风险**: KV 读取量翻倍（3 heads 共享 1 KV read → 6 heads 需要 2 次 KV read）
**难度**: 中（模板参数改动 + host wrapper）

### P4: Warp specialization
**原理**: 2 warps 做 memory loading (cp.async)，6 warps 做 compute
**预期效果**: 更好的 memory/compute overlap
**风险**: SM120 没有 TMA，cp.async 的 benefit 有限
**难度**: 高

## 推荐执行顺序
1. **P3 (GQA=3)** — 最简单，改模板参数即可验证。如果 regs 降到 ≤128 且 2 CTAs 有效，直接 1.5-2× 加速
2. **P1 (4 warps)** — 如果 P3 不够，减少 warps 是第二选择
3. **P0 (D-tiling)** — 最复杂但最优雅，不增加 KV 读取
4. **P2 (Persistent)** — 补充优化，不解决 occupancy 问题

## 验证方法
每个方案：
1. 微基准：单次 kernel 调用时间 + ncu occupancy/DRAM
2. 端到端：128K/c=4 warm cache throughput
3. 正确性：cosine similarity vs 原始 kernel ≥ 0.999
