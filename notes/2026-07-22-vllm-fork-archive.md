# B7-V0: vLLM Fork 状态存档（2026-07-22）

> 路线图 B7「去 vLLM 化」的硬前置：盘点本地 vLLM 树的全部偏差，存档为可复现参考。
> 此后任何替换工作都以本文档为基准，确保隐性行为不丢失。

## 1. 基线

- **路径**: `/home/bot/vllm`
- **上游版本**: v0.25.0（commit `e12b91b03`）
- **性质**: 上游检出 + 5 个文件未提交补丁 + 1 个未跟踪文件
- **补丁总量**: +135/−52 行（不含 sm120_gqa.py）

## 2. 已修改文件清单

### 2.1 `tests/v1/core/test_prefix_caching.py` (+41 行)

新增两个测试：
- `test_hybrid_mamba_eagle_does_not_reuse_lookahead_state`: 验证 EAGLE 模式下
  hybrid mamba 模型不会错误复用 lookahead state
- `test_mamba_align_prefill_split_keeps_intermediate_chunks_aligned`: 验证
  chunked prefill 的 mamba block 对齐逻辑

**分类**: 上游 bug 修复的测试（stock-vLLM 基线路径用）

### 2.2 `vllm/utils/jit_monitor.py` (+38/−13 行)

将 `_setup_cutedsl_jit_hook` 中的 `@functools.wraps` 包装替换为
`_MonitoredCuteCompile` 代理类，支持 `cute.compile[options](fn, ...)` 下标形式。

**分类**: 上游 bug 修复（FlashInfer SM120 GDN prefill kernel 的 CuTeDSL 编译监控）

### 2.3 `vllm/v1/core/kv_cache_coordinator.py` (+4/−3 行)

- 使用 `spec.supports_eagle_cache_peek` 替代硬编码的 `use_eagle` 判断
- `curr_hit_length` 更新改为 `min(curr_hit_length, _new_hit_length)` 防止长度膨胀

**分类**: 上游 bug 修复（hybrid cache + EAGLE 交互）

### 2.4 `vllm/v1/core/sched/scheduler.py` (+30/−22 行)

重写 `_mamba_block_aligned_split` 逻辑：
- 修复 chunked prefill 中间 chunk 的 mamba block 对齐
- EAGLE 模式下保留一个 block 的余量
- Marconi cache admission 优化改为 end-position cap

**分类**: 上游 bug 修复（hybrid model chunked prefill 调度）

### 2.5 `vllm/v1/kv_cache_interface.py` (+16 行)

为 `FullAttentionSpec`、`MLAAttentionSpec`、`SlidingWindowSpec`、`MambaSpec`
添加 `supports_eagle_cache_peek` 属性。Mamba/GDN 返回 False（循环状态不可回退）。

**分类**: 上游 bug 修复（EAGLE cache peek 的层型感知）

## 3. 未跟踪文件

### 3.1 `vllm/v1/attention/backends/sm120_gqa.py` (1115 行)

SM120 GQA attention backend 的 vLLM 集成层。本应属于 `sm120-flash-attention`
仓库，散落在 vLLM 树内。纯增量文件，不影响 vLLM 默认行为。

**处置**: 应迁移到 `sm120-flash-attention` 仓库或本项目的 `runtime/` 下。
B7-V1 的 compat 收口将把 `SM120GQAMetadata` 定义搬进本仓库。

### 3.2 其他未跟踪

- `csrc/moe/marlin_moe_wna16/`, `csrc/quantization/marlin/`: 构建产物
- `nohup.out`, `vllm/nohup.out`: 日志文件

## 4. BlackForge 生产路径的 vLLM 依赖面

`runtime/direct_model_runner.py` 的全部 vLLM import：

| 符号 | 来源 | 分级 | 替换方式 |
|---|---|---|---|
| `VllmConfig`, `set_current_vllm_config` | `vllm.config` | 中 | 自有 config dataclass |
| `EngineArgs` | `vllm.engine.arg_utils` | 中 | 自有 config 构建 |
| `set_forward_context` | `vllm.forward_context` | 薄 | 自有 context manager |
| `prepare_chunk_indices`, `prepare_chunk_offsets` | `vllm...fla.ops.index` | 中 | 直接用 flash-linear-attention 上游包 |
| `FLA_CHUNK_SIZE` | `vllm...fla.ops.utils` | 薄 | 常量，直接定义 |
| `get_model` | `vllm...model_loader` | **厚** | 自有模型图（E1/A1 拉动） |
| `get_distributed_init_method`, `get_open_port` | `vllm.utils.network_utils` | 薄 | 自写（几行代码） |
| `GDNAttentionMetadata` | `vllm...gdn_attn` | 薄 | 自写 dataclass |
| `AttentionBackendEnum`, `register_backend` | `vllm...registry` | 薄 | 删除（改为直调 kernel） |
| `SM120GQAMetadata` | `vllm...sm120_gqa` | 薄 | 自写 dataclass（搬进本仓库） |
| `compute_causal_conv1d_metadata` | `vllm...backends.utils` | 薄 | 自写（纯计算） |
| `init_worker_distributed_environment` | `vllm...gpu_worker` | 薄 | 自写初始化 |
| `bind_kv_cache` | `vllm...worker.utils` | 薄 | 自写（字典绑定） |
| `load_eagle_model` | `vllm...eagle.utils` | **厚** | 自有 MTP 加载（A3 拉动） |

其他文件：
- `runtime/gemma_norm_patch.py`: `from vllm import ir` + `GemmaRMSNorm`（中）
- `runtime/triton_norm_ops.py`: `from vllm.triton_utils import tl, triton`（薄→直接 import triton）

## 5. 补丁性质判定

5 个已修改文件的补丁**全部服务于 stock-vLLM 基线路径**（hybrid model 的
EAGLE/chunked-prefill 正确性），不是 BlackForge 生产路径的组成部分。
BlackForge 生产路径（`direct_model_runner.py`）绕过了 vLLM 的调度器和
cache coordinator，直接使用底层原语。

**结论**: 这些补丁对 BlackForge 生产无影响，但 stock-vLLM A/B 基线对比
（`runtime/vllm_*_baseline.py`）需要它们。存档后，A/B 基线脚本应标注
所需的 vLLM 补丁版本。

## 6. 完整 diff

见附件: `notes/2026-07-22-vllm-fork-diff.patch`
