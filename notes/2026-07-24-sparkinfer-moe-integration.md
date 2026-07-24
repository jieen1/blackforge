# sparkinfer MoE Integration — Investigation & Results

Date: 2026-07-24
sparkinfer version: cc9b476 (jieen1/sparkinfer, branch blackforge-main)
GPU: RTX PRO 6000 Blackwell 96GB (SM120)

## Scale Convention (Critical Finding)

Laguna NVFP4 checkpoint stores:
- `weight_global_scale` (wgs): ~2624-17920 per expert
- `input_global_scale` (igs): ~776-22016 per expert
- `weight_scale` (block scale, fp8_e4m3fn): ~104-192
- `weight_packed` (FP4 E2M1, uint8): packed nibbles, lo-first

Dequantization: `w_approx = fp4_value × block_scale / weight_global_scale`

sparkinfer's `_prepare_modelopt_nvfp4_runtime_alphas` computes:
`w1_runtime = w1_global_scale / a1_gscale`

**Problem**: passing raw checkpoint values causes activation quantization
underflow (a1_gscale = 1/igs ≈ 5e-4 → activations quantize to zero).

**Solution**: fold weight_global_scale into block scales, use unit alphas:
```python
block_scale_new = (block_scale.float() / weight_global_scale).to(fp8)
w1_global_scale = 1.0  # unit
a1_gscale = 1.0        # unit
```
Verified: cosine=0.954 vs fp32 reference (FP4 quantization error expected).

## Performance Results

### Single-layer (E=256, K=3072, I=1024, top_k=10)

| Mode | M=1 | M=16 | 47L projection |
|------|-----|------|----------------|
| Eager (median) | 794μs | 1057μs | 37-50ms |
| Eager (min, 42% of runs) | 44μs | 465μs | 2.1-21.8ms |
| **CUDA graph** | **38μs** | — | **1.8ms** |
| CUTLASS eager (CUPTI) | 186μs | — | 8.73ms |

**CUDA graph is essential** — eager has bimodal timing (workspace realloc).
With CUDA graph: **4.8× faster than CUTLASS** (1.8ms vs 8.73ms for 47 layers).

### 47-layer chain (eager, M=1)
- Median: 29.3ms, Min: 24.0ms
- Per-layer avg: 0.6ms

### Weight loading
- Per-layer: ~2.0s (safetensors → GPU → swizzle → prepare)
- All 47 layers: ~96s

## Architecture

```
laguna_sparkinfer_moe.py (zero vLLM dependency)
├── load_moe_layer_weights()  — direct safetensors loading
├── prepare_sparkinfer_layer() — scale fold + swizzle + plan/prepare
├── SparkinferMoELayer        — single layer forward()
└── SparkinferMoEModel        — all 47 layers, load_all() + forward_layer()
```

Dependency: `sparkinfer` via editable install (jieen1/sparkinfer fork).
Fallback: `BF_SPARKINFER_PATH` env var.
Version stamp: `sparkinfer_version()` returns git sha.

## Checkpoint Format (Laguna NVFP4)

Per expert per projection (gate_proj, up_proj, down_proj):
- `weight_packed`: [rows, cols//2] uint8 (FP4 E2M1, lo nibble first)
- `weight_scale`: [rows, cols//16] fp8_e4m3fn (block scale)
- `weight_global_scale`: scalar f32
- `input_global_scale`: scalar f32 (not used in folded convention)

Layers 1-47 are MoE; layer 0 is dense MLP.

## Next Steps

1. **CUDA graph integration**: capture sparkinfer kernel in the runtime's
   existing CG infrastructure for consistent 38μs/layer
2. **Router integration**: the runtime's router (gate linear + top-10)
   feeds topk_ids/topk_weights to SparkinferMoELayer.forward()
3. **E2E benchmark**: DFlash + CUDA graph + sparkinfer MoE vs MARLIN baseline
4. **Kernel optimization**: investigate eager bimodal timing (workspace prealloc)

## B12x vs sparkinfer (Clarification)

B12x (FlashInfer CuTe-DSL wrapper): 0.057ms/layer at M=1, **but outputs
all zeros on SM120** (CUTLASS SM121 MMA op guard). Dead end.

sparkinfer: separate kernel, works correctly on SM120. With CUDA graph,
38μs/layer — faster than B12x's broken 57μs.
