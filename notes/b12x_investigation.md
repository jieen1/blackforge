# B12x MoE Kernel Investigation (2026-07-24)

## Status: FlashInfer B12x wrapper broken on SM120 — sparkinfer integration needed

## Root Causes Found

### 1. vLLM B12x alpha computation is WRONG (fixed in laguna_moe_patch.py)

vLLM's `FlashInferB12xExperts.process_weights_after_loading` computes wrong runtime alphas:

- **vLLM passes**: `w1_alpha = g1_alphas = 1/w_gs` (~7.4e-05)
- **Kernel expects**: `w1_alpha = w_gs × input_scale = 1/(g1_alphas × a1_gscale)` (~1.3-6.7)
- **Off by ~235,000×**

Verified against sparkinfer reference (`_prepare_modelopt_nvfp4_runtime_alphas`):
```python
# sparkinfer correct computation:
w1_runtime = w1_global_scale / a1_gscale  # = w_gs × input_scale
```

The old code also baked `1/w_gs` into fp8 block scales (causing 1.8% underflow in worst layer) and set alpha=1.0, compounding the error.

**Fix**: `laguna_moe_patch.py` — monkey-patches `process_weights_after_loading` with correct alpha.

### 2. FlashInfer B12x wrapper incompatible with cutlass-dsl 4.6.0

- cutlass-dsl 4.5.1: kernel compiles but produces numerically wrong output (6.7% acceptance)
- cutlass-dsl 4.6.0: `TypeError: make_kwargs_wrapper() got unexpected keyword 'map_dataclass_to_tuple'`
  - Fixed by upgrading `apache-tvm-ffi` 0.1.9 → 0.1.12
  - After fix: kernel compiles and runs, but STILL produces wrong output (6.7% acceptance)

### 3. vLLM excludes B12x from auto-selection

```python
# FLASHINFER_B12X is intentionally excluded from auto-selection until
# the upstream CUTLASS SM121 MMA op guard is resolved
```

This confirms the kernel has known issues on SM120/SM121.

## Evidence

| Configuration | Acceptance | tok/s | Notes |
|---|---|---|---|
| MARLIN (baseline) | 30-47% | 40-192 | Works correctly |
| B12x original bake-in (cutlass 4.5.1) | 9.2% | 0.8 | fp8 underflow + wrong alpha |
| B12x no-bake + g1_alphas (4.5.1) | 6.7% | 9.5 | Wrong alpha direction |
| B12x correct alpha (4.5.1) | 6.7% | 9.5 | Alpha fixed, kernel still wrong |
| B12x correct alpha (4.6.0) | 6.7% | 9.9 | cutlass upgrade didn't help |
| Standalone random weights | N/A | N/A | Non-zero output (kernel runs) |

## Scale Factor Data (Layer 1)

- `w13_weight_scale` (fp8): [0.002, 448], mean=93.2
- `w13_weight_scale_2` (= g1_alphas = 1/w_gs): [7.4e-05, 3.8e-04]
- Bake-in product: 1.8% underflow to zero in fp8
- `a2_gscale`: 776.0
- Correct w1_alpha: [1.3, 6.7]
- Correct w2_alpha: [2.5, 23.1]

## Path Forward: sparkinfer Integration

The sparkinfer library (cloned at `/home/bot/project/sparkinfer`, formerly b12x) is the
actual SM120 MoE kernel implementation. It requires cutlass-dsl >= 4.6.0 (now installed)
and reports `is_supported() = True` on this GPU.

### Integration Plan

1. Write `SparkinferMoEExperts` class implementing vLLM's `FusedMoEExpertsModular` interface
2. Use sparkinfer's `plan_weights` → `prepare_weights` → `plan` → `bind` → `run` lifecycle
3. Weight preparation: use `prepare_sparkinfer_fp4_moe_weights` with correct source_format='modelopt_nvfp4'
4. Replace FlashInfer B12x wrapper with sparkinfer kernel
5. Parity testing against MARLIN baseline
6. E2E benchmark with DFlash + CUDA Graph

### Key API Mapping

| vLLM B12x | sparkinfer |
|---|---|
| `B12xMoEWrapper.run()` | `sparkinfer_moe_fp4()` |
| `process_weights_after_loading` | `plan_weights` + `prepare_weights` |
| `w1_sf_mma` (MMA layout) | `ExpertWeights.w1_blockscale` (source-native) |
| `g1_alphas` | `ExpertWeights.w1_alphas` (computed correctly) |
| `fc2_input_scale` | `ExpertWeights.a2_gscale` |

### Estimated Effort: 1-2 days

## Environment Changes Made

- `nvidia-cutlass-dsl`: 4.5.1 → 4.6.0
- `nvidia-cutlass-dsl-libs-{base,core,cu12,cu13}`: → 4.6.0
- `apache-tvm-ffi`: 0.1.9 → 0.1.12
- MARLIN verified working after upgrade (no regression)

## Files

- `runtime/backends/laguna_moe_patch.py` — correct alpha monkey-patch (ready for when kernel is fixed)
- `notes/b12x_investigation.md` — this document
- `/home/bot/project/sparkinfer/` — sparkinfer library (cloned)
