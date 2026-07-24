# sparkinfer Dynamic Kernel FC2 Quantization Underflow Fix

## Date: 2026-07-24
## Commit: 2f66e5a

## Root Cause

The sparkinfer dynamic kernel (used for M≥7, routed_rows≥64) has a FC2
intermediate quantization step that computes:

```
scale_float = block_max × a2_gscale / 6.0
scale_byte = cvt_f32_to_e4m3(scale_float)
```

When w1_alpha (= 1/checkpoint_global_scale) is small (~5.6e-5 to 5e-4 for
Laguna), the SiLU output between FC1 and FC2 is proportional to alpha²
(~1e-8 to 1e-7). This makes `block_max` very small, and `scale_float`
underflows below fp8-e4m3's minimum positive subnormal (2^-9 ≈ 0.00195).

When the scale rounds to zero, `quantize_block_fp4` returns packed64=0
(all FP4 values zero), effectively zeroing the FC2 input. The MoE output
loses ~28% magnitude (rel_norm≈0.72, cos≈0.95 vs reference).

## Evidence Chain

1. M-sweep: M=1..6 (micro kernel) correct, M≥7 (dynamic kernel) broken
2. Forced micro for M=7,8: correct → bug 100% in dynamic kernel
3. top_k=1: still broken → bug in single-expert GEMM, not multi-expert accumulation
4. Uniform alpha (all experts same): correct at ALL magnitudes (0.001 to 1.0)
5. Per-expert varying alpha with same-expert routing: correct
6. **Uniform alpha magnitude sweep**: boundary at alpha≈0.0005
   - alpha≥0.0005: rel_norm≈1.0 ✓
   - alpha=0.0001: rel_norm=0.76 ✗
   - alpha=0.00001: rel_norm=0.0 ✗ (complete zeroing)
7. `quantize_block_fp4` code: scale underflows fp8 → packed64=0

## Fix

Set `SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE=1` before sparkinfer import.
This enables tile-level adaptive scale computation:

```python
tile_gs_value = 2688.0 / tile_amax  # adapts to data range
fc2_down_alpha_value = down_alpha_value * (gs_value / tile_gs_value)
```

The dynamic scale prevents fp8 underflow regardless of alpha magnitude.

## Verification

### M-sweep (with fix)
| M   | cos      | rel_norm | status |
|-----|----------|----------|--------|
| 1   | 0.992597 | 1.004436 | ✓      |
| 8   | 0.993736 | 1.007582 | ✓      |
| 16  | 0.994287 | 1.006546 | ✓      |
| 128 | 0.994104 | 1.004151 | ✓      |

### E2E correctness
- "The capital of France is" → "Paris..." ✓
- fibonacci → correct ✓
- meaning of life → coherent ✓

### CUDA Graph integration
- CG capture: ✓ (0.5s)
- Token match vs eager: EXACT (50 tokens)
- Eager: 70.4 ms/tok (14.2 tok/s)
- CG: 13.9 ms/tok (72.1 tok/s)
- Speedup: 5.07×

## Impact

- Fixes DFlash verify (M=16) and prefill (M≥2048) which require dynamic kernel
- Decode (M=1) was unaffected (uses micro kernel)
- No performance regression observed

## Upstream

This should be filed as an issue/PR to sparkinfer upstream:
- `dynamic_down_scale` should default to True for nvfp4 quant_mode
- Or: the quantization should handle small scales gracefully (fallback to
  float32 scale when fp8 underflows)
