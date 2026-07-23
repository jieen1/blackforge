# vLLM DFlash 生产基线（阶段一：自研对标尺，2026-07-23）

> 用户裁定两阶段：① stock vLLM 跑 DFlash + CUDA Graph + 最优 kernel 测基准速度
> （后续自研的对标尺 + 发布门禁②「vs stock vLLM 显著优势」的对照尺）；
> ② 参考 vLLM 的 kernel 与 DFlash 逻辑，针对 Laguna 模型 + SM120 显卡针对性优化。
> 脚本：`benchmarks/laguna_vllm_dflash_baseline.py`
> 数据：`benchmarks/fixtures/laguna_vllm_dflash_baseline.json`

## 配置（model card 推荐）

- target `poolside/Laguna-S-2.1-NVFP4` + draft `poolside/Laguna-S-2.1-DFlash-NVFP4`
- `speculative_config={method:dflash, num_speculative_tokens:15}`
- `enforce_eager=False`（CUDA Graph 开）· greedy · gpu_mem_util 0.85

## 首个数据点（auto kernel）

| 配置 | accepted tok/s | ITL | 备注 |
|---|---:|---:|---|
| seqs=1, ctx=4096 | **352.2** | 2.84ms | DFlash K=15 + CUDA Graph |

- ⚠️ 官方 13-14 tok/s 是 **GB10（另一块卡）无投机**数字，**不能**直接对比。
  必须在同一块 RTX PRO 6000 上做 2×2 消融才能拆解各分量加速（见下）。
- 加载 446s。**auto 实测选 FLASHINFER_CUTLASS**（日志 `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`）；
  加载期 AutoTuner 调 `trtllm::fused_moe::gemm1/gemm2` 是 vLLM 的 kernel 调优步骤，
  与最终 backend 选择是两回事（此前误推断为 TRTLLM，已纠正）。

## 关键含义（对阶段二）

1. **北极星 = accepted tokens/s（含投机）**，不是单步 decode tok/s。
2. 自研 MoE kernel 的对手是 **autotuned trtllm fused_moe**，不是裸 cutlass。
3. DFlash verify 是 M=16 形状 → 自研 kernel autotune 覆盖 M=1..16（decode+verify）。
4. vLLM auto 与我们的 runtime 都用 FLASHINFER_CUTLASS（同一 backend）→ 自研 kernel
   的对标 = vLLM autotuned FLASHINFER_CUTLASS 全栈（含 DFlash + CUDA Graph）。

## 待补全（矩阵在跑）

- seqs 1/4/8 并发扩展（agent 多请求）× ctx 4K/32K（前缀缓存场景）。
- 确认 auto 选的 backend（捕获 vLLM 选择日志）+ 对比 flashinfer_cutlass/trtllm。

## 加速分量消融（2×2，同一块 RTX PRO 6000，ctx=4096 seqs=1）

> 此前用 GB10 的 13-14 tok/s 做对比是错的。正确做法：同卡测 4 个配置拆解。

| 配置 | DFlash | CUDA Graph | accepted tok/s | 增量归因 |
|---|---|---|---:|---|
| eager 裸 decode | ✗ | ✗ | 待测 | 基线 |
| + CUDA Graph | ✗ | ✓ | 待测 | graph 消 launch overhead |
| + DFlash | ✓ | ✗ | 待测 | 投机解码 |
| + DFlash + Graph | ✓ | ✓ | 352.2 | 全栈 |
