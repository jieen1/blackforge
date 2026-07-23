# vLLM 生产基线（阶段一收官，自研对标尺，2026-07-23）

> 用户裁定两阶段：① stock vLLM 跑 DFlash + CUDA Graph + 最优 kernel 测基准（自研
> 对标尺 + 发布门禁②对照尺）；② 参考 vLLM kernel/DFlash 逻辑针对 SM120 自研优化。
> 脚本 `benchmarks/laguna_vllm_dflash_baseline.py`；数据 `benchmarks/fixtures/laguna_*.json`。
> 全部在 RTX PRO 6000 Blackwell（96GB）实测，greedy，gpu_mem_util 0.92。

## 最优 kernel 确认（64K, seqs=1, DFlash K=15, CUDA Graph ON）

| backend | accepted tok/s | SM120 状态 |
|---|---:|---|
| **cutlass (VLLM_CUTLASS)** | 360.9 | ✅ 可用 |
| **flashinfer_cutlass**（auto 选） | 359.6 | ✅ 可用（并列最优）|
| marlin | — | ❌ 加载失败 |
| flashinfer_cutedsl / flashinfer_trtllm | — | ❌ kernel 不支持 SM120 |

**结论：SM120 上 NVFP4 MoE 仅 cutlass 系两个 backend 可用且性能并列最优
（~360 tok/s，差 0.4% 在噪声内）。auto 选 flashinfer_cutlass 已是最优，无更好
backend 可换 → 阶段二只能走自研 kernel（对标 autotuned cutlass）。**

## DFlash 开关对比（flashinfer_cutlass, CUDA Graph ON, seqs=1）

| 上下文 | DFlash ON | DFlash OFF | DFlash 加速 |
|---|---:|---:|---:|
| 64K | **367.3 tok/s** (ITL 2.72ms) | 54.6 tok/s (ITL 18.32ms) | **6.7×** |
| 128K | **311.0 tok/s** (ITL 3.22ms) | 54.0 tok/s (ITL 18.52ms) | **5.8×** |

- DFlash 是主导杠杆（5.8-6.7×），北极星 = accepted tokens/s。
- 128K 比 64K 慢 ~15%（367→311）：attention 随上下文增长（12 全局层；36 SWA 窗口
  512 有界）；DFlash OFF 时 64K≈128K（54.6≈54.0）→ MoE 权重带宽受限、与 ctx 无关。
- 此前用 GB10 的 13-14 tok/s 对比是错的（不同卡），已纠正。

## 阶段二输入（自研对标）

- 对标数字：**64K 367 / 128K 311 accepted tok/s**（autotuned flashinfer_cutlass 全栈）。
- MoE GEMM 占生产路径 ~65%（见 laguna-moe-node-trace 笔记），routed cutlass grouped
  GEMM 2.63ms @ 63% peak → 自研融合 kernel 的空间在「提 BW 利用率 + 融合 routing 开销」。
- DFlash verify 是 M=16 形状 → 自研 kernel autotune 覆盖 M=1..16（decode + verify）。
- 参考实现：`~/project/{cutlass-4.6.1, flashinfer, TensorRT-LLM, tilelang}`。
