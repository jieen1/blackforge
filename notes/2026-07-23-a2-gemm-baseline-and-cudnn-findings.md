# A2: NVFP4 GEMM 优化 — 基线 + 多后端评估 + CutlassDirect 突破（2026-07-23）

> 路线图 A2 实测：在 RTX PRO 6000 Blackwell (188 SMs, 1338.8 GB/s peak) 上
> 录制完整原生 NVFP4 GEMM 基线，评估 5 种后端，发现 CutlassDirect 路径
> 实现 **29.7% E2E 加速**（bit-exact）。

## 测试环境

- GPU: RTX PRO 6000 Blackwell Max-Q (188 SMs, 96 GB, SM 12.0)
- torch: 2.13.0a0+gitcf30153, CUDA 13.3
- vLLM: 0.25.1.dev0 (local), FlashInfer: 0.6.15rc2
- Peak memcpy BW: 1338.8 GB/s
- 模型: unsloth/Qwen3.6-27B-NVFP4

## 原生 CUTLASS 微基准（decode c=1, M=4）

| Shape | P50 ms | BW GB/s | % peak | vs bf16 | 加权 ms |
|---|---:|---:|---:|---:|---:|
| gate_up_proj (4×34816×5120) ×56 | 0.1090 | 817 | 61.0% | 1.94× | 6.111 |
| down_proj (4×5120×17408) ×56 | 0.0710 | 628 | 46.9% | 1.59× | 3.978 |
| down_proj_attn (4×17408×17408) ×8 | 0.1847 | 820 | 61.3% | 1.98× | 1.478 |
| out_proj (4×6144×6144) ×64 | 0.0301 | 628 | 46.9% | 0.66× | 1.926 |
| in_proj_qkvz (4×5120×5120) ×72 | 0.0254 | 516 | 38.6% | 0.62× | 1.829 |
| in_proj_ba (4×96×5120) ×48 | 0.0251 | 10 | 0.8% | 0.92× | 1.205 |
| **Total** | | | | | **16.527** |

## 多后端微基准对比（FlashInfer mm_fp4 API）

| 后端 | 加权 ms/round | vs cutlass | bit-exact | SM120 支持 |
|---|---:|---:|---|---|
| cutlass (FlashInfer) | 92.278 | REF | ✅ | ✅ |
| **b12x** (SM120 专用) | **86.535** | **−6.2%** | ✅ 全部 | ✅ |
| cudnn | 105.226 | +14.0% | ✅ 全部 | ✅ |
| cute-dsl | — | — | — | ❌ (需 SM10x) |
| trtllm | — | — | — | ❌ (需 SM10x) |

注：微基准使用 Python 级 time.perf_counter + cuda.synchronize，
包含 ~0.05ms/call Python 开销。304 calls/round → ~15ms 纯开销。

## 端到端 A/B 测试（10 reps, prompt=2048, max_tokens=128）

| 配置 | 内核 | AVG tok/s | 范围 | accept rate | bit-exact |
|---|---|---:|---|---:|---|
| Stock | FlashInferCutlassNvFp4 | **40.8** | 34.9–45.8 | 76.8% | REF |
| B12x | FlashInferB12xNvFp4 | 40.6 | 37.5–43.5 | 76.8% | ✅ |
| **CutlassDirect** | **CutlassNvFp4Linear** | **52.9** | **46.4–60.0** | **76.8%** | **✅** |

### 🏆 CutlassDirect: +29.7% E2E 加速，token 级 bit-exact

## 根因分析

**为什么 CutlassDirect 快 30%？**

vLLM 默认选择 `FlashInferCutlassNvFp4LinearKernel`（优先级 #2），
它通过 FlashInfer Python wrapper 调用 CUTLASS：

```
apply_weights → flashinfer_scaled_fp4_mm → mm_fp4 → FlashInfer C++ → CUTLASS
```

而 `CutlassNvFp4LinearKernel`（优先级 #4）直接调用 vLLM 的 C++ custom op：

```
apply_weights → cutlass_scaled_fp4_mm → CUTLASS
```

每轮 decode 有 **304 次 GEMM 调用**（56+56+8+64+72+48）。
FlashInfer wrapper 每次调用增加 ~0.05ms Python 开销，
304 × 0.05ms = **~15ms/round 纯 Python 开销**。
总 E2E ~75ms/round 中，这 15ms 占 **20%**。

**为什么 B12x 微基准快 6% 但 E2E 无提升？**

B12x 优化了 GPU kernel 本身（更好的 SM120 tile 配置），
但它仍然通过 FlashInfer Python wrapper 调用，
所以 Python 开销没有减少。GPU kernel 节省的 ~5ms
被 Python 开销淹没。

## cuDNN 评估（已否决）

- 微基准（随机张量）：快 12.6%，bit-exact ✅
- **真实模型权重 E2E：MTP accept rate 76.8% → 60.4%，吞吐量反降 27%**
- 裁决：❌ 不可用（违反 greedy bit-exact 门禁）

## 最终方案

**`CutlassNvFp4LinearKernel`（直接 C++ CUTLASS 路径）**

- 实现：`runtime/nvfp4_cutlass_direct_patch.py`
- 默认启用（`QSR_A2_CUTLASS_DIRECT=1`）
- 在 `get_model()` 前将 CutlassNvFp4LinearKernel 提升至优先级 #1
- 门禁通过：token 级 bit-exact + 326 tests pass + ruff clean

## 文件清单

| 文件 | 用途 |
|---|---|
| `runtime/nvfp4_cutlass_direct_patch.py` | CutlassDirect patch（**默认启用**） |
| `runtime/nvfp4_b12x_patch.py` | B12x patch（备用，默认启用但被 CutlassDirect 覆盖） |
| `runtime/nvfp4_cudnn_patch.py` | cuDNN patch（默认关闭，仅供实验） |
| `benchmarks/a2_native_baseline.py` | 原生微基准录制 |
| `benchmarks/a2_backend_sweep.py` | 多后端对比 |
| `benchmarks/a2_e2e_ab_test.py` | E2E A/B 测试 |
| `benchmarks/fixtures/a2_native_baseline.json` | 原生基线数据 |
| `benchmarks/fixtures/a2_backend_sweep.json` | 多后端对比数据 |
| `benchmarks/fixtures/a2_e2e_b12x_10.json` | E2E 10-rep 对比数据 |

---

## 自研 SM120 NVFP4 GEMM Kernel（A2 核心交付）

### 编译与架构

- 源码：`runtime/kernels/nvfp4_gemm_sm120.cu`
- 编译产物：`runtime/kernels/nvfp4_gemm_sm120.so`（SM120a, nvcc 13.2）
- 基于 CUTLASS 4.5.1 `GemmUniversal` + `OpClassBlockScaledTensorOp`
- 4 种 tile 配置，通过 C API `qsr_nvfp4_gemm(config_id, ...)` 选择

### Tile 配置

| Config | Tile | Scheduler | 适用场景 |
|---|---|---|---|
| A | 128×128×128 | 非持久 | 中等 N（5120-6144），通用 |
| B | 128×128×128 | Persistent | 小 N（<5120），SM 利用率敏感 |
| C | 256×128×128 | Persistent | 大 M（>256），非 decode 路径 |
| D | 128×256×128 | Persistent | 大 N（≥17408），宽 N tile |

### 微基准（per-shape 最优选择，加权 ms/round）

| 配置 | 加权 ms | vs vLLM baseline |
|---|---:|---:|
| vLLM baseline | 76.079 | REF |
| **Hybrid（per-shape）** | **62.855** | **−17.4%** |
| Config A only | 66.669 | −12.4% |
| Config B only | 66.851 | −12.1% |

**全部 4 配置 × 6 形状 = 24 组合均 bit-exact ✅**

### Per-shape 最优加速

| Shape | 最优 Config | 加速 |
|---|---|---:|
| gate_up_proj (N=34816) | D (128×256) | 1.184× |
| down_proj (N=5120) | A (128×128) | 1.129× |
| down_proj_attn (N=17408) | D (128×256) | 1.256× |
| out_proj (N=6144) | A (128×128) | 1.048× |
| in_proj_qkvz (N=5120) | B (persist) | **1.538×** |
| in_proj_ba (N=96) | B (persist) | 1.255× |

### E2E 验证

| 配置 | AVG tok/s | accept rate | vs stock |
|---|---:|---:|---:|
| Stock (FlashInferCutlass) | 40.8 | 76.8% | REF |
| CutlassDirect | 52.9 | 76.8% | +29.7% |
| **自研 Custom GEMM** | **52.1** | **76.8%** | **+27.7%** |

E2E 提升主要来自 CutlassDirect（绕过 FlashInfer Python wrapper）。
自研 kernel 的 tile 优化在微基准显著（17.4%），但 E2E 中 GEMM 仅占
~22% 时间，tile 优化被非 GEMM 开销稀释。

### 集成方式

- `runtime/nvfp4_custom_gemm.py`：Python wrapper + per-shape config 选择
- `QSR_A2_CUSTOM_GEMM=1`（默认启用）
- 自动 fallback 到 vLLM 原生 kernel（任何错误时）
- 与 CutlassDirect 叠加使用（CutlassDirect 选 kernel 路径，Custom GEMM 替换底层实现）

### 质量门禁

- ✅ 24/24 组合 bit-exact（torch.equal）
- ✅ E2E accept rate 76.8%（与 stock 完全一致）
- ✅ 326 tests pass
- ✅ ruff lint + format clean
