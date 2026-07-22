# Laguna-S-2.1-NVFP4 · L0 显存预算合同（roadmap Track E / E3-L0）

日期：2026-07-22
方法：仅本地文件元数据核验（config.json / model.safetensors.index.json / 磁盘尺寸），零 GPU 操作、零 tensor payload 读取——沿用 `hy3-sm120-research/04-memory-budget` 的合同格式与纪律。

## 1. 权重来源与实测校验

| 项 | 值 |
|---|---|
| 本地快照 | `~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/b482b5d57fda6e4e562a652869bde24ba2a57c92` |
| 分片 | 14 × safetensors，全部就位 |
| index `total_size` | **71,898,733,760 B = 66.96 GiB** |
| 磁盘合计（含 tokenizer/config） | 71,938,634,828 B ≈ 67.0 GiB |
| 与模型卡对照 | 「roughly 71 GB」为十进制 GB，与实测 66.96 GiB 一致，无缺片 |

## 2. 结构参数（config.json 实测，非发布稿转述）

| 参数 | 值 |
|---|---|
| 层数 | 48 = **12 层 `full_attention`（位置 0,4,8,…,44，每 4 层 1 层）+ 36 层 `sliding_attention`（窗口 512）** |
| KV 头 | `num_key_value_heads = 8`，`head_dim = 128`（GQA 48:8） |
| 上下文 | `max_position_embeddings = 262144` |
| MoE | 256 experts，**每 token 激活 10 个**（注意：非发布稿转述的 top-8） |
| KV 量化 | 8-bit（quantization_config，FP8 规划） |

## 3. KV 增长模型

- 全局层每 token：12 × 2(K+V) × 8 头 × 128 dim × 1 B = **24 KiB/token**（随上下文线性增长）
- 滑窗层每槽固定：36 × 512 × 2 KiB = **36 MiB/槽**（有界，不随上下文增长）
- 对照：HY3 为 160 KiB/token（80 层全 GQA），Laguna 增长率仅其 **1/6.7**

## 4. 场景判定（可用显存 95.59 GiB；workspace + CUDA Graph 预留 4 GiB）

| 场景 | KV 合计 | 权重+KV+预留 | 剩余 | 判定 |
|---|---:|---:|---:|---|
| 2 槽 × 200K | 9.4 GiB | 80.4 GiB | **15.2 GiB** | ✅ 宽裕，**无需 expert offload** |
| 2 槽 × 256K | 12.1 GiB | 83.0 GiB | 12.6 GiB | ✅ 无需 offload |
| 4 槽 × 128K | 12.1 GiB | 83.1 GiB | 12.5 GiB | ✅ 无需 offload |
| 4 槽 × 200K | 18.9 GiB | 89.9 GiB | 5.7 GiB | ⚠️ 可行但余量收窄，prefix cache 空间受限 |
| 4 槽 × 256K | 24.1 GiB | 95.1 GiB | ~0.5 GiB | ❌ 不可行——需 offload / 动态 KV 预算 / 降并发 |

**核心结论：目标形态（2 并发 × 200K，乃至 2×256K / 4×128K）下，256 个专家全部常驻显存，expert offload 不需要。** offload 议题仅在 4×256K 或未来 1M 上下文时重开，届时按件取用 `hy3-sm120-research` 的 expert cache 设计与路由 trace 方法论。

## 5. 剩余显存的用途（按 roadmap A4 纪律 A/B）

2×200K 形态下的 ~15 GiB 剩余分配（更新 2026-07-22）：

| 用途 | 预算 |
|---|---:|
| DFlash draft 模型权重（2.23 GB ≈ 2.08 GiB，下载中，待本地校验） | ~2.1 GiB |
| draft 模型 KV + verify 持久 buffer（待其 config 落地后精算） | ~1–2 GiB（规划） |
| 前缀缓存块池 / CUDA Graph 多 bucket / autotune 权重副本 | 其余 ~11 GiB |

计入 draft 后 2×200K 总占用 ≈ 83–84 GiB，仍余 ~12 GiB；**4×200K + DFlash** 余量收窄至 ~3 GiB，投机与 4 并发同开时需精算，届时以实测为准。

## 6. 未决项（L0 关账前必须补齐）

1. **DFlash 本地校验**：`poolside/Laguna-S-2.1-DFlash-NVFP4` 用户报告 2.23 GB，下载完成后核 index `total_size` 与 config（架构形态：自回归 draft 还是并行多 token 草稿、KV 头配置、与主模型共享 embedding 与否）；K=15 的 verify 形状（qo_len=16）对 CUDA Graph 持久 buffer 与 verify 激活的放大分析随 config 一并出；
2. **pinned vLLM 0.25 加载冒烟未做**（需 GPU 窗口）：确认架构类支持、NVFP4 路径在 SM120 上的实际行为（模型卡提示 FlashInfer nightly，我方 A2 自研 GEMM 路线不受此限但过渡期受影响）；
3. **workspace 实测**：4 GiB 预留是规划值，冒烟后以实测峰值替换。

## 7. 门禁状态

L0 = 账本（本文档，✅ 已关）+ 冒烟（§6.2，⬜ 待 GPU 窗口）+ DFlash 分析（§6.1，⬜）。全部完成后 L0 关账，进入 L1。
