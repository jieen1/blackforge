# SWA 环形 KV 调研与设计方案（阶段二 Step 1）

日期：2026-07-23
状态：**调研完成，待审查**
前置：阶段一基线已固化（commit `a860391`），Lane 1 server 接线已并入 main（`a2128ef`）

---

## 1. 问题陈述

Laguna-S-2.1 架构：48 层 = **12 层 full_attention**（48 QO heads）+ **36 层 sliding_attention**（72 QO heads，window=512）。全部 48 层共享 8 KV heads / head_dim=128 / FP8 KV。

当前 `LagunaBackend`（`runtime/backends/laguna.py:156-171`）对全部 48 层统一按 `blocks_per_slot`（= max_context / block_size）分配 KV cache。SWA 层的 `window_left=511` 只影响 FlashInfer 注意力计算范围，**不影响存储分配**。

### 显存账（128K 上下文，单槽，block_size=16，FP8 KV）

| 层类型 | 层数 | 当前分配 | 实际需要 | 浪费 |
|---|---:|---:|---:|---:|
| full_attention | 12 | 3.00 GiB | 3.00 GiB | 0 |
| sliding_attention | 36 | **9.00 GiB** | **0.036 GiB** | **8.96 GiB** |
| 合计 | 48 | 12.00 GiB | 3.04 GiB | 8.96 GiB |

**SWA 层过度分配 248×**。128K 单槽 KV 从 12.0 → 3.04 GiB，释放 ~9 GiB。

这直接回答用户点名的「128K 显存占用太高」问题：当前 128K 需要 gpu_mem_util≥0.92 才能装下 KV（0.85 时 KV 仅 4.49 GiB < 12.0 GiB 需求），环形 KV 后 0.85 即可轻松容纳。

---

## 2. 验证结果（全部通过）

### 2.1 vLLM per-layer KV spec 确认

脚本 `/tmp/verify_spec2.py`，GPU 实测：

```
FullAttentionSpec sliding_window=None: 12 层
SlidingWindowSpec sliding_window=512: 36 层
```

vLLM 的 `static_forward_context` 已正确区分每层的 KV spec。我们 backend 的 `get_kv_cache_spec()` 调用（`laguna.py:147`）已在使用这个信息来分组 FlashInfer builder——但 KV **分配**没有利用这个区分。

### 2.2 环形索引数学（CPU-only）

脚本 `/tmp/verify_ring_kv_math.py`：

- `ring_blocks = cdiv(512-1, 16) + 1 = 33`，`ring_slots = 528`
- 窗口 512 内无 ring slot 碰撞 ✓
- 块对齐窗口下 block_table 正确映射 ✓
- prefill 后环形缓冲区包含正确的最后 512 个位置 ✓

### 2.3 FlashInfer 环形 KV decode（GPU 实测）

脚本 `/tmp/verify_ring_aligned.py`，GQA 64/8，block_size=16：

```
pos=   512: diff=0.00e+00 cos=1.00000000  ✓
pos=  1024: diff=0.00e+00 cos=1.00000000  ✓
pos= 65536: diff=0.00e+00 cos=1.00000000  ✓
Multi-step decode (10 steps): ALL diff=0.00e+00  ✓
```

**Bit-exact**。环形 block_table + 块对齐窗口 → FlashInfer decode 输出与连续 KV 完全一致。

### 2.4 FlashInfer AOT SWA kernel 限制

发现：FlashInfer AOT 编译的 SWA decode kernel（`use_swa_True`）**不支持 GQA group_size=6 或 9**（Laguna 的两种层类型）。vLLM 在 Blackwell（SM120）上实际使用 **TRTLLM decode 路径**（`flashinfer.py:731`：`use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode`），该路径支持任意 group_size 和 window_left。

**影响**：我们的 backend 在 SM120 上走 TRTLLM decode，window_left=511 由 TRTLLM 内核处理。环形 KV 的块对齐窗口会多包含 ≤15 个额外位置（block_size-1），TRTLLM 的 window_left 会正确跳过它们。

### 2.5 Prefill 正确性约束

**关键发现**：prefill 阶段不能使用环形 KV。

原因：prefill 一次性处理全部 prompt 位置。环形写入会覆盖旧位置的 KV（position 0 被 position 528 覆盖）。FlashInfer 先写全部 KV 再计算注意力，导致中间位置（如 position 100）读到被覆盖的 KV（position 628 的数据），产生错误注意力 → 错误 hidden state → 跨层级联 → 最终输出错误。

**结论**：prefill 必须使用临时全量 KV，完成后将最后 window 个位置拷贝到环形缓冲区，释放临时 KV。

---

## 3. 设计方案

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────┐
│                  LagunaBackend                       │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │  Full KV Pool     │  │  SWA Ring KV Pool        │ │
│  │  12 layers        │  │  36 layers               │ │
│  │  blocks_per_slot  │  │  33 blocks/slot          │ │
│  │  per layer        │  │  per layer               │ │
│  │  (128K=8192 blk)  │  │  (固定，不随上下文增长)   │ │
│  └──────────────────┘  └──────────────────────────┘ │
│                                                      │
│  Prefill: temp full KV for SWA → copy last 512 →    │
│           ring → free temp                           │
│  Decode:  ring buffer, block-aligned window,         │
│           window_left=511 via TRTLLM                 │
└─────────────────────────────────────────────────────┘
```

### 3.2 KV 分配（`__init__` 修改）

```python
# 当前：全部 48 层统一分配
num_blocks = (num_slots + RESERVED) * blocks_per_slot
for name in attn_layer_names:
    kv_caches[name] = torch.zeros(shape(num_blocks, ...))

# 改为：按层类型分组分配
for name in attn_layer_names:
    layer = sfc[name]
    spec = layer.get_kv_cache_spec(vllm_config)
    if isinstance(spec, SlidingWindowSpec):
        # 环形：33 blocks/slot
        ring_blocks = cdiv(spec.sliding_window - 1, block_size) + 1
        n = (num_slots + RESERVED) * ring_blocks
    else:
        # 全量：blocks_per_slot
        n = (num_slots + RESERVED) * blocks_per_slot
    kv_caches[name] = torch.zeros(shape(n, ...))
```

### 3.3 Decode 路径修改

**`_fill_decode_buffers`**：为 SWA 组生成独立的 block_table 和 slot_mapping。

```python
# SWA 组的 block_table（块对齐窗口）
window_start = max(0, kv_len - WINDOW + 1)
aligned_start = (window_start // block_size) * block_size
aligned_len = kv_len + 1 - aligned_start  # 包含 decode token
n_ring_blocks = cdiv(aligned_len, block_size)

for j in range(n_ring_blocks):
    actual_pos = aligned_start + j * block_size
    ring_block = (actual_pos % ring_slots) // block_size
    swa_block_table[i, j] = swa_ring_base + ring_block

# SWA 组的 slot_mapping（环形写入）
ring_block = (pos % ring_slots) // block_size
ring_offset = pos % block_size
swa_slot_mapping[i] = (swa_ring_base + ring_block) * block_size + ring_offset
```

**`_build_common_attn_metadata`**：为 SWA 组生成独立的 `CommonAttentionMetadata`，其中 `seq_lens` 和 `block_table_tensor` 使用环形值。

### 3.4 Prefill 路径修改

```python
def prefill(self, slot, prompt_ids):
    # 1. 分配临时全量 KV（SWA 层）
    temp_kv = {name: torch.zeros(full_shape, ...) for name in swa_layers}
    
    # 2. 重绑 SWA 层 KV 到临时缓冲区
    for name in swa_layers:
        sfc[name].kv_cache = temp_kv[name]
    
    # 3. 正常 prefill（全量 block_table + slot_mapping）
    logits = self._forward([slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False)
    
    # 4. 拷贝最后 window 个位置到环形缓冲区
    prompt_len = len(prompt_ids)
    window_start = max(0, prompt_len - WINDOW)
    for name in swa_layers:
        for p in range(window_start, prompt_len):
            src_block = p // block_size
            src_offset = p % block_size
            dst_block = (p % ring_slots) // block_size
            dst_offset = p % block_size
            ring_kv[name][dst_block, :, dst_offset] = temp_kv[name][src_block, :, src_offset]
    
    # 5. 重绑回环形 KV + 释放临时
    for name in swa_layers:
        sfc[name].kv_cache = self.kv_caches[name]
    del temp_kv
    
    return first_token
```

**临时 KV 显存**：128K prompt → 36 层 × 8192 块 × 32 KiB/块 ≈ 9.0 GiB（峰值，prefill 期间）。prefill 是一次性操作，峰值可接受。后续实现 chunked prefill 后可降到 ~150 MB。

### 3.5 `reset_slot` 修改

```python
def reset_slot(self, slot):
    # Full 层：清零 blocks_per_slot 个块（不变）
    # SWA 层：只清零 33 个环形块（而非 blocks_per_slot 个）
    for name in swa_layers:
        ring_base = phys * ring_blocks_per_slot
        self.kv_caches[name][ring_base:ring_base + ring_blocks_per_slot].zero_()
```

### 3.6 CUDA Graph 兼容性

环形 KV 对 CUDA Graph 完全兼容：
- 预分配的 buffer 地址不变（`_decode_block_table`、`_decode_slot_mapping`）
- 每步只更新 buffer 内容（环形 block_table 值、ring slot_mapping 值）
- FlashInfer/TRTLLM 的 plan 参数（indptr、indices、last_page_len）通过 fast_decode_plan 更新
- 环形 block_table 长度固定（≤33），比当前（≤8192）更短，plan 更快

### 3.7 与 vLLM SlidingWindowManager 的关系

vLLM 的 `SlidingWindowManager`（`single_type_kv_cache_manager.py:675`）实现了完整的有界块管理：
- `max_admission_blocks_per_request = cdiv(window-1+in_flight, block_size) + 1`
- `remove_skipped_blocks` 释放超窗块
- 与 BlockPool 的引用计数、前缀缓存集成

**我们不复用它**。原因：
1. 我们的 backend 绕过了 vLLM 的 scheduler/block_pool，直接管理 KV
2. vLLM 的 manager 与 BlockPool、prefix cache、chunked prefill 深度耦合
3. 我们的场景更简单（≤4 槽，无前缀缓存），自实现环形更直接

但设计参考了 vLLM 的核心思想：`cdiv(window-1, block_size) + 1` 的有界块数公式。

---

## 4. 正确性风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---|---|
| Prefill 环形覆盖导致错误 | **高** | 临时全量 KV（§3.4），已验证 |
| 块对齐窗口多 ≤15 个额外位置 | 低 | TRTLLM window_left=511 跳过，已验证 bit-exact |
| 环形 block_table wrap-around | 中 | 每步重建 block_table，已验证 10 步 decode bit-exact |
| reset_slot 未清零环形块 | 中 | 修改 reset_slot 只清环形块（§3.5） |
| CUDA Graph capture 时环形参数 | 低 | 环形 block_table 更短（33 vs 8192），capture 更快 |
| 多槽并发时环形基址计算 | 中 | 每槽独立 ring_base = phys * ring_blocks_per_slot |
| prefill 临时 KV 峰值显存 | 低 | 128K 峰值 ~9 GiB，96 GiB 卡可承受；后续 chunked prefill 消除 |

---

## 5. 验证方案

### 5.1 单元测试（CPU-only）

- 环形索引数学：block_table、slot_mapping、wrap-around（已有 `/tmp/verify_ring_kv_math.py`）
- per-group KV 分配形状
- reset_slot 只清环形块

### 5.2 GPU 正确性测试

- **Golden token 对比**：同一 prompt，full KV vs ring KV，greedy decode 200 tokens，逐 token 对比
  - 预期：bit-exact（环形不改变注意力计算，只改变存储布局）
- **多步 decode**：1000 步 decode，验证环形 wrap-around 不累积误差
- **多槽并发**：2 槽同时 decode，验证环形基址隔离
- **prefill→decode 过渡**：验证临时 KV 拷贝到环形后 decode 正确

### 5.3 性能测试

- 128K 上下文显存占用：预期从 ~12 GiB → ~3 GiB（单槽 KV）
- decode ITL：预期不变或略降（更短的 block_table → 更快的 plan）
- prefill 延迟：预期略增（临时 KV 分配 + 拷贝），但 prefill 是一次性的

### 5.4 门禁

- 全部 216+ CPU 测试通过（零回归）
- GPU golden token bit-exact
- 128K 显存占用实测 ≤ 4 GiB（单槽 KV）
- decode ITL 不退化（±5%）

---

## 6. 工作量评估

| 子任务 | 估计 | 依赖 |
|---|---|---|
| per-group KV 分配 + 层分类 | 2h | 无 |
| decode 路径：per-group block_table/slot_mapping | 3h | KV 分配 |
| prefill 路径：临时 KV + 拷贝 + 重绑 | 3h | KV 分配 |
| reset_slot 修改 | 0.5h | KV 分配 |
| CommonAttentionMetadata per-group | 2h | decode 路径 |
| CPU 测试 | 2h | 全部 |
| GPU 正确性验证 | 2h（含模型加载） | 全部 |
| 性能验证 | 1h（含模型加载） | 正确性 |
| **合计** | **~15h** | |

---

## 7. 后续步骤（Step 2-4 概览，不在本次范围）

- **Step 2**：自研 NVFP4 MoE kernel（对标 autotuned cutlass，融合 routing + 提 BW 利用率）
- **Step 3**：DFlash 投机集成到自研 runtime
- **Step 4**：CUDA Graph 全形态（decode M=1 + verify M=16）+ 收尾对标

Step 1（本文档）是 Step 2-4 的前置：释放的 ~9 GiB 显存用于 DFlash draft 模型权重 + verify buffer + 多槽并发。

---

## 附录 A：验证脚本清单

| 脚本 | 类型 | 结果 |
|---|---|---|
| `/tmp/verify_spec2.py` | GPU | 12 Full + 36 SWA(512) 确认 |
| `/tmp/verify_ring_kv_math.py` | CPU | 索引数学全部通过 |
| `/tmp/verify_ring_aligned.py` | GPU | FlashInfer decode bit-exact |
| `/tmp/verify_ring_flashinfer_gpu.py` | GPU | AOT SWA 不支持 gs=9（记录） |

## 附录 B：关键代码引用

- 当前 KV 分配：`runtime/backends/laguna.py:156-171`
- 层分组：`runtime/backends/laguna.py:103-137`（按 window_left, nqh, nkvh）
- FlashInfer builder：`runtime/backends/laguna.py:139-154`
- decode buffer 填充：`runtime/backends/laguna.py:226-252`
- CommonAttentionMetadata：`runtime/backends/laguna.py:254-326`
- vLLM SlidingWindowSpec：`vllm/v1/kv_cache_interface.py:529`
- vLLM SlidingWindowManager：`vllm/v1/core/single_type_kv_cache_manager.py:675`
- vLLM Laguna 模型 per-layer SWA：`vllm/model_executor/models/laguna.py:301-306`

---

## 8. 审查修正（2026-07-23 审查后追加）

### 阻断①：ring_blocks 33 → 34（DFlash qo=16 前向兼容）

原公式 `cdiv(window-1, block_size)+1 = 33` 只对 qo_len=1 成立。DFlash verify（Step 3）
的 qo=16 在同一次 forward 内最早 query 读 [p−511, p]、最晚写到 p+15，跨度 542 > 528（33 块），
环形会覆盖仍要被读的位置——§2.5 prefill 覆盖问题的缩小版。

**修正公式**（参数化）：

```python
ring_blocks = cdiv(window - 1 + qo_max, block_size) + 1
# qo_max=1  → 33（当前 decode）
# qo_max=16 → 34（DFlash verify）
```

采用 **qo_max=16 → ring_blocks=34**（544 槽，max_span=542，margin=2）。
代价：+1 块 × 36 层 × 槽数 ≈ 1.1 MiB/槽，可忽略。

### 阻断②：生产形态 golden 验证

§2.3 的 GPU 验证用 GQA 64/8 合成 wrapper，§2.4 错误断言 TRTLLM decode。
实际生产路径（启动日志 `decode_backend=flashinfer-native, arch=sm120`）：

- cudagraph wrapper：`use_tensor_cores=True` + `fast_plan_decode(window_left=wl)` + `fixed_split_size=2048`
- 不是 AOT SWA kernel（不支持 gs6/9），不是 TRTLLM（SM120 未启用）
- window_left 由 FlashInfer tensor-core decode kernel 处理

**合入门禁追加**：ring golden 对比必须在真实生产形态跑——
group_size 6/9、cudagraph wrapper、fast_plan、fixed_split_size=2048，
挂进现有 graph parity harness。

### 非阻断条件（实现时顺手做）

3. Prefill 临时 KV → 持久共享 scratch（分配一次、各槽复用、不必 zeros）
4. §3.4 拷贝循环向量化（按 wrap 切分，每层 ≤3 段连续 slab）
5. 验证脚本从 /tmp 迁入 `benchmarks/` 或 `tests/`
6. E1 抽象层文档加「缓存物种」接口占位（前缀缓存前向兼容）
