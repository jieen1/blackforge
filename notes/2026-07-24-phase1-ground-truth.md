# Phase 1: bf16 Ground Truth 实验报告 (2026-07-24)

## 实验设计

用 NVFP4 权重精确反量化到 bf16 后做 fp32 矩阵乘，作为唯一合法裁判，
对比 sparkinfer（两种 scale 配置）和 vLLM CUTLASS 的输出。

- 模型层: layer 1 (MoE, 256 experts, top_k=10)
- 输入: M=4, hidden=3072, 随机 bf16 × 0.1
- 指标: cosine similarity + relative norm + relative error

## 实验 1: bf16 真值 vs sparkinfer

### Layer 1 (gate_gs: 2624–13504)

| 配置 | vs truth cos | vs truth rel_norm | vs truth rel_err |
|------|-------------|-------------------|------------------|
| FOLDING (bs/gs→fp8, unit alpha) | 0.9614 | 0.9984 | 0.2775 |
| ALPHA (a1_gscale=1.0, w1_gs=1/gs) | **0.9645** | 0.9978 | 0.2663 |
| ALPHA vs FOLDING | 0.9913 | 0.9994 | 0.1322 |

### Layer 11 (gate_gs: 3488–17280, 16 experts > 16384)

| 配置 | vs truth cos | vs truth rel_norm | vs truth rel_err |
|------|-------------|-------------------|------------------|
| FOLDING | 0.9624 | 0.9920 | 0.2733 |
| ALPHA | **0.9662** | 0.9888 | 0.2586 |

### 高 gs 专家逐一对比 (layer 11, gs > 16K)

| Expert | gs | truth norm | fold norm | alpha norm | fold/truth | alpha/truth |
|--------|------|-----------|-----------|------------|-----------|------------|
| 6 | 16512 | 0.8672 | 0.8067 | 0.8605 | 0.930 | **0.992** |
| 17 | 16896 | 0.8086 | 0.7622 | 0.7777 | 0.943 | **0.962** |
| 98 | 16512 | 0.7921 | 0.7466 | 0.7955 | 0.943 | **1.004** |

**结论**: Alpha 路径在高 gs 专家上显著优于 folding（0.99 vs 0.93）。

## 实验 2: 零化定位

### 原始假设推翻

之前认为 runtime-alpha 路径「对小值产生零」——**根因是 a1_gscale 用错了**。

| a1_gscale 设置 | 结果 | 原因 |
|---------------|------|------|
| a1_gscale = checkpoint_gs (2624) | 输出 ~100,000× 过大 | 激活量化用了权重的 gs，不是激活的 |
| a1_gscale = 1.0 (unit) | **正确** (cos=0.964) | 与 folding 路径一致的动态量化 |

### 机制

- `a1_gscale` 控制**激活量化**（应设为 1.0 让 kernel 动态计算）
- `w1_global_scale` 控制**权重反量化**（应设为 1/checkpoint_gs）
- 两者独立，之前错误地把权重的 gs 传给了激活通道

### fp16 边界假设

- Layer 11 最大 gs=17280, 1/17280=5.79e-5 < fp16 min normal 6.1e-5
- 但 runtime alpha 全程 fp32（`_prepare_expert_scale_vector` 强制 `.to(torch.float32)`）
- **fp16 中转假设不成立**——alpha 值在 fp32 通路中无零化
- 实际精度损失来自 folding 路径的 fp8 block scale 压缩（mean 4.4%, max 32%）

## 实验 3: FusedMoE combine 代数

### vLLM 源码实锤 (moe_runner.py:390-406)

```python
# bf16 路径:
fused_output *= self.routed_scaling_factor  # routed *= 2.5
# shared_output 不变
# 最终: routed * 2.5 + shared
```

Laguna 使用 `apply_routed_scale_to_output=True` (laguna.py:231):
- Router: `routed_scaling_factor=1.0` (topk_weights 不缩放)
- Runner: `fused_output *= 2.5`
- 最终: **output = routed × 2.5 + shared**

### 当前 adapter bug (laguna.py:568-570)

```python
routed_out = routed_out * _scaling      # routed *= 2.5 ✓
shared_out = shared_out / _scaling      # shared /= 2.5 ✗ 错误!
```

产生: `routed * 2.5 + shared / 2.5` — 与 vLLM 不一致。
B12x 路径 (laguna.py:437-440) 是正确的: `routed * 2.5 + shared`。

### 影响量化

| combine 方式 | vs truth cos | vs truth rel_norm |
|-------------|-------------|-------------------|
| routed*2.5 + shared (正确) | **0.9714** | 0.9712 |
| routed*2.5 + shared/2.5 (当前 bug) | 0.9448 | 0.8735 |

÷2.5 bug 导致 cosine 下降 2.7%，rel_norm 偏差从 2.9% 扩大到 12.7%。

## 实验 4: vLLM CUTLASS 对比

### 意外发现

| 配置 | vs truth cos | vs truth rel_norm | output norm |
|------|-------------|-------------------|-------------|
| bf16 truth | 1.0000 | 1.0000 | 2.363 |
| sparkinfer (correct) | **0.9714** | 0.9712 | 2.295 |
| vLLM CUTLASS | 0.4546 | 0.6748 | 1.594 |

vLLM CUTLASS 输出与 bf16 真值的 cosine 仅 0.455，远低于 sparkinfer 的 0.971。

### 可能原因（待验证）

1. **激活量化 scale 差异**: vLLM 使用 `a1_gscale = input_scale = 4.96e-4`（校准值），
   sparkinfer 使用 `a1_gscale = 1.0`（动态）。两者的激活量化精度不同。
2. **Forward context 不完整**: 测试中 `set_forward_context(attn_metadata={})` 可能
   导致 MoE runner 走了非标准路径。
3. **Weight reformat 差异**: vLLM 的 CUTLASS reformat 可能改变了权重布局/scale 约定。

### 关键观察

- vLLM 输出 norm=1.594 远小于 truth=2.363（67.5%），说明存在系统性 scale 偏差
- 但 vLLM E2E 输出正确文本 → layer norm 补偿了 scale 偏差
- 需要进一步隔离：直接调用 expert kernel（绕过 runner）对比

## 全局 scale a1_gscale 约定总结

| 参数 | 含义 | 正确值 | 来源 |
|------|------|--------|------|
| a1_gscale | 激活量化全局 scale | 1.0 (动态) 或 input_scale (校准) | 取决于 kernel 约定 |
| w1_global_scale | 权重全局 scale | 1/checkpoint_gs | checkpoint |
| runtime_alpha | w1_global_scale / a1_gscale | 1/checkpoint_gs (当 a1=1.0) | 计算得出 |

## 下一步 (Phase 2)

1. **修复 ÷2.5 bug**: laguna.py:570 删除 `shared_out / _scaling`
2. **更新 adapter**: 使用 checkpoint-direct 加载 + 正确 alpha 配置
3. **调查 vLLM cos=0.455**: 隔离 expert kernel 对比，排除 runner 干扰
4. **E2E 验证**: 修复后跑 50-token greedy，与 CUTLASS 基线对齐
5. **CG 集成**: standalone 38μs/layer × 47 = 1.8ms MoE 段

## 脚本归档

- `/tmp/phase1_exp.py` → 基础 3-way 对比 (layer 1)
- `/tmp/phase1_fix.py` → 修正后的 alpha 配置对比
- `/tmp/phase1_layer11.py` → 高 gs 层对比 (layer 11)
- `/tmp/phase1_vllm_simple.py` → vLLM CUTLASS 对比 (需模型加载)
