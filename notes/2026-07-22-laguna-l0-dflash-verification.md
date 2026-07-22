# Laguna-S-2.1-DFlash-NVFP4 · L0 DFlash 校验（roadmap §6.1）

日期：2026-07-22
方法：本地 safetensors header 解析 + config.json/config.py 静态分析，零 GPU 操作。

## 1. 权重校验

| 项 | 值 |
|---|---|
| 本地快照 | `~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/723794750422b3efbf3a7b3af76dffb4ba035943` |
| 文件 | 单文件 `model.safetensors`（symlink → blob） |
| 文件大小 | **2,229,955,584 B = 2.230 GB = 2.077 GiB** |
| 参数量 | **1,114,977,792 (1.115B)** |
| dtype | 全部 BF16（69 tensors） |
| 与用户报告对照 | 「2.23 GB」✅ 一致 |

## 2. 架构形态

| 参数 | 值 |
|---|---|
| 架构类 | `DFlashLagunaForCausalLM`（config.json）/ `DFlashSpeculator`（speculators 库） |
| 层数 | 6（全部 `sliding_attention`，窗口 512） |
| hidden_size | 3072 |
| num_attention_heads | 72（GQA 72:8） |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 12288 |
| MoE | **无**（`num_experts=0`，纯 dense） |
| vocab_size | 100352（与主模型共享） |
| max_position_embeddings | 262144 |
| 共享 embedding/lm_head | **否**（无 embed_tokens / lm_head 权重） |

### DFlash 特有结构

| 组件 | 形状 | 说明 |
|---|---|---|
| `fc.weight` | [3072, 18432] | 6 层 aux hidden states 拼接投影（6×3072=18432 → 3072） |
| `aux_hidden_norms.{0-5}.weight` | [3072] × 6 | 各 aux 层 RMSNorm |
| `hidden_norm.weight` | [3072] | 投影后 RMSNorm |
| `norm.weight` | [3072] | 最终 RMSNorm |

### dflash_config（config.json）

```json
{
  "block_size": 16,
  "mask_token_id": 12,
  "num_target_layers": 48,
  "target_layer_ids": [1, 10, 19, 29, 38, 47],
  "causal": true
}
```

- `eagle_aux_hidden_state_layer_ids`: [2, 11, 20, 30, 39, 48]（主模型中间层）
- 机制：**并行 masked prediction**（非自回归），一次 forward 生成 block_size-1=15 个 draft token

## 3. K=15 Verify 形状分析

| 项 | 值 |
|---|---|
| block_size | 16 |
| verify qo_len | 16（15 draft + 1 target bonus） |
| draft KV per token | 24 KiB（6 层 × 2(K+V) × 8 头 × 128 dim × 2B） |
| draft KV per block | 384 KiB |
| sliding KV per slot（固定） | **12.0 MiB**（6 层 × 512 window × 24 KiB/token） |
| 4 slots sliding KV | 48.0 MiB |
| verify activation（粗估） | ~2.2 MiB |

### CUDA Graph 影响

- verify pass qo_len=16 是固定形状 → 可捕获为 CUDA Graph
- 主模型 verify 同样 qo_len=16（与当前 MTP K=3 的 qo_len=4 不同）
- 需要新的 CUDA Graph bucket：qo_len=16 × batch={1,2,4}

## 4. 显存预算更新

| 组件 | 占用 |
|---|---:|
| 主模型权重 | 66.96 GiB |
| DFlash 权重 | 2.08 GiB |
| DFlash 4-slot sliding KV | 0.047 GiB |
| DFlash activation | ~0.002 GiB |
| **DFlash 总计** | **~2.13 GiB** |
| 2×200K 主模型 KV | 9.4 GiB |
| workspace + CUDA Graph | 4 GiB |
| **总计（2×200K + DFlash）** | **~82.5 GiB** |
| 可用 | 95.59 GiB |
| **剩余** | **~13.1 GiB** ✅ |

结论：DFlash 加入后 2×200K 形态仍宽裕（余 13 GiB），4×128K 同样可行。

## 5. 上游支持状态

| 框架 | PR/Issue | 状态 |
|---|---|---|
| vLLM | #46853 | in progress |
| SGLang | #29446 | in progress |
| TRT-LLM | #15666 | in progress |
| speculators 库 | poolside 官方 | 已发布（config.py 依赖） |

README 推荐 `num_speculative_tokens=7`；benchmark 用 15。
接受长度：4.0–6.5 tokens（任务/并发相关）。
吞吐加速：1.7×–3.7×（并发 1–16，任务相关）。

## 6. L0 §6.1 关账判定

- [x] 本地校验：index/size 与报告一致
- [x] config 分析：架构形态、KV 头、draft 机制已明确
- [x] K=15 verify 形状：qo_len=16，CUDA Graph 可捕获
- [x] 显存预算：DFlash 总计 ~2.13 GiB，2×200K 形态余量充足
- [x] 上游支持：vLLM #46853 in progress，speculators 库已可用

**§6.1 DFlash 校验：✅ 关账**
