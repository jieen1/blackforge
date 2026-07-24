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
