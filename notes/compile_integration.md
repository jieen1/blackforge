# Compile Integration Investigation (2026-07-24)

## Problem
vLLM compile pipeline (`enforce_eager=False`, `cudagraph_mode=NONE`) + our custom
CUDA Graph capture failed with shape mismatch: `expected size 16==1`.

## Root Cause (two layers)

### Layer 1: AOT compile auto-enabled
`VLLM_USE_AOT_COMPILE` defaults to `"1"` when torch ≥ 2.10.0 (vllm/envs.py:338-344).
AOT compile bakes fixed input shapes from the first call (M=1 warmup). The inductor
generated code has `assert_size_stride(arg0_1, (1,), (1,), 'input')` — hard assertion
that input batch dim is 1. Calling at M=16 (verify) triggers AssertionError.

**Fix**: `os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")` in compat_vllm.py.
JIT torch.compile with dropped guards handles dynamic shapes correctly.

### Layer 2: Argument count mismatch between decode (M=1) and verify (M=16)
Even with JIT compile, the compiled graph has a fixed number of internal tensor
arguments determined during tracing. Decode metadata (M=1) and prefill/verify
metadata (M=16) produce different argument counts because the FlashInfer attention
metadata structure differs between decode and prefill modes.

If decode CG capture triggers compilation first (M=1), verify CG capture at M=16
fails with `ValueError: too many values to unpack (expected 14)`.

**Fix**: `skip_compiled=True` on decode CG capture (laguna_cuda_graph.py).
Compilation triggers during verify CG capture at M=16. Decode CG captures eager
operations (acceptable: decode is 4.4ms vs verify 28.8ms).

## Architecture
- Decode CG (M=1): eager (skip_compiled=True) — captures eager ops
- Verify CG (M=16): compiled (JIT, guards dropped) — captures fused ops
- Draft CG (M=16): compiled (JIT, separate model) — captures fused ops
- Prefill: eager (skip_compiled=True) — variable length, no CG

## Results

| Context | Baseline CG | Compile+CG | Δ step | Δ tok/s |
|---------|-------------|------------|--------|---------|
| 64K     | 36.7ms, 192 tok/s | 36.0ms, 253 tok/s | -0.7ms (2%) | +31%* |
| 128K    | 39.2ms, 191 tok/s | 39.2ms, 181 tok/s | 0ms | -5%* |

*tok/s delta driven by acceptance rate variance, not step time.

## Key Finding
Compile benefit is mostly redundant with CG. CG already eliminates kernel launch
overhead; compile's elementwise fusion saves <1ms on top. The expected 35→27-29ms
did not materialize because that estimate was for eager→compile without CG.

## Compile Timing
- First compile: ~14s (backbone) + ~2s (draft head) = ~16s total
- Cached load: ~1s (backbone) + ~0.1s (draft head) = ~1.1s total
- Cache location: `~/.cache/vllm/torch_compile_cache/`

## Prefill Slowdown
128K prefill: 30.2s (compile) vs 26.3s (baseline) — 15% slower.
Cause: likely memory pressure from compiled graph cache (~2GB extra GPU memory).
Investigation deferred to prefill optimization lane.

## Next Steps
1. Step latency: L1 host-side thinning (precompute, accept, sync consolidation)
2. MoE kernel: MARLIN optimization / B12x microbench (post-compile)
3. Prefill: metadata reuse, chunk size tuning
4. Full-step CG: beyond vLLM (超越 move, fixed slots enable it)

---

# B12x MoE Kernel Microbench (2026-07-24)

## Setup
- Model loaded WITHOUT `moe_backend="marlin"` (original NVFP4 weights)
- B12x kernel: `LagunaMoEB12x` from `runtime/backends/laguna_moe_kernel.py`
- Fixed `intermediate_size` bug: was `w13.shape[1]*2=4096`, correct is `w13.shape[1]//2=1024`
- Weight loading succeeded with original NVFP4 format

## Results (single MoE layer, eager, no CG)

| M | B12x per layer | B12x ×47 layers | MARLIN ×47 (eager ledger) | Speedup |
|---|---|---|---|---|
| 1 | 0.057ms | 2.7ms | 8.73ms + 1.2ms routing | 3.7× |
| 16 | 0.381ms | 17.9ms | ~25ms (est. from verify CG) | ~1.4× |

## Key Findings
1. B12x fuses routing+FC1+SiLU+FC2+scatter into ONE kernel (vs MARLIN's 8-9 launches)
2. At M=1: 3.7× faster (2.7ms vs 9.9ms) — massive launch overhead elimination
3. At M=16: ~1.4× faster (17.9ms vs ~25ms) — fusion benefit persists but compute grows
4. Weight format: B12x needs original NVFP4, NOT MARLIN-repacked weights

## Integration Blocker
MARLIN repacks weights during `process_weights_after_loading`. B12x needs original
NVFP4 format. Options:
1. Load without MARLIN → B12x for MoE, default backend for linears (~0.3ms penalty)
2. Keep original weight copies alongside MARLIN (~5GB extra memory)
3. Convert MARLIN→NVFP4 at runtime (complex, potentially lossy)

**Recommended**: Option 1. Non-MoE linears are ~1ms total; MARLIN→CUTLASS penalty
is ~0.3ms. MoE savings from B12x are 7+ms at M=1, 7+ms at M=16. Net win: 6+ms.

## Projected Impact
- Verify CG (M=16): 28.8ms → ~21ms (MoE 25→17.9ms)
- Step time: 36ms → ~29ms
- 64K tok/s: 253 → ~310 (at same acceptance)
- Combined with compile: ~28ms step → ~330 tok/s
