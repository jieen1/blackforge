#!/usr/bin/env python3
"""A2: Benchmark ALL available NVFP4 GEMM backends on decode shapes.

Tests: CUTLASS, FlashInfer, Humming, Marlin, FBGEMM (whichever are available).
"""
from __future__ import annotations

import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

SHAPES = [
    (4, 34816, 5120, 56, "gate_up_proj"),
    (4, 5120, 17408, 56, "down_proj"),
    (4, 17408, 17408, 8, "down_proj_attn"),
    (4, 6144, 6144, 64, "out_proj"),
    (4, 5120, 5120, 72, "in_proj_qkvz"),
    (4, 96, 5120, 48, "in_proj_ba"),
]


def setup_nvfp4(m, n, k):
    """Create properly quantized NVFP4 tensors."""
    a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    b_amax = b_bf16.abs().max().to(torch.float32)
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    b_fp4, scale_b = ops.scaled_fp4_quant(b_bf16, b_global_scale)

    a_amax = a_bf16.abs().max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    alpha = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

    a_fp4, scale_a = ops.scaled_fp4_quant(a_bf16, a_global_scale)

    return a_bf16, b_bf16, a_fp4, b_fp4, scale_a, scale_b, alpha, a_global_scale, b_global_scale


def bench_cutlass(a_fp4, b_fp4, scale_a, scale_b, alpha, warmup=100, iters=500):
    """CUTLASS backend."""
    for _ in range(warmup):
        ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def bench_flashinfer(a_fp4, b_fp4, scale_a, scale_b, alpha, warmup=100, iters=500):
    """FlashInfer backend."""
    try:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
    except ImportError:
        return None
    for _ in range(warmup):
        flashinfer_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        flashinfer_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def bench_bf16(b_bf16, a_bf16, warmup=100, iters=500):
    """bf16 reference."""
    for _ in range(warmup):
        torch.mm(a_bf16, b_bf16.t())
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        torch.mm(a_bf16, b_bf16.t())
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def bench_with_quant(a_bf16, b_fp4, scale_b, alpha, a_global_scale, warmup=100, iters=500):
    """CUTLASS with on-the-fly activation quantization (realistic)."""
    for _ in range(warmup):
        aq, saq = ops.scaled_fp4_quant(a_bf16, a_global_scale)
        ops.cutlass_scaled_fp4_mm(aq, b_fp4, saq, scale_b, alpha, torch.bfloat16)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        aq, saq = ops.scaled_fp4_quant(a_bf16, a_global_scale)
        ops.cutlass_scaled_fp4_mm(aq, b_fp4, saq, scale_b, alpha, torch.bfloat16)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def main():
    gpu = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU: {gpu} ({sm} SMs)")

    # Check available backends
    backends = ["cutlass", "bf16_ref", "cutlass+quant"]
    try:
        from vllm.utils.flashinfer import has_flashinfer
        if has_flashinfer():
            backends.insert(1, "flashinfer")
            print("FlashInfer: available")
        else:
            print("FlashInfer: not available")
    except Exception:
        print("FlashInfer: import failed")

    print(f"Backends: {backends}\n")

    results = []
    for m, n, k, count, name in SHAPES:
        print(f"{name} ({m}x{n}x{k}):")
        a_bf16, b_bf16, a_fp4, b_fp4, scale_a, scale_b, alpha, a_gs, b_gs = setup_nvfp4(m, n, k)

        entry = {"name": name, "m": m, "n": n, "k": k, "count": count}

        cutlass_ms = bench_cutlass(a_fp4, b_fp4, scale_a, scale_b, alpha)
        entry["cutlass_ms"] = round(cutlass_ms, 4)
        print(f"  CUTLASS:      {cutlass_ms:.4f} ms")

        if "flashinfer" in backends:
            try:
                fi_ms = bench_flashinfer(a_fp4, b_fp4, scale_a, scale_b, alpha)
                if fi_ms is not None:
                    entry["flashinfer_ms"] = round(fi_ms, 4)
                    speedup = cutlass_ms / fi_ms
                    print(f"  FlashInfer:   {fi_ms:.4f} ms  ({speedup:.2f}x vs CUTLASS)")
            except Exception as exc:
                print(f"  FlashInfer:   FAILED ({exc})")

        bf16_ms = bench_bf16(b_bf16, a_bf16)
        entry["bf16_ms"] = round(bf16_ms, 4)
        print(f"  bf16 ref:     {bf16_ms:.4f} ms  (NVFP4 {bf16_ms/cutlass_ms:.2f}x)")

        quant_ms = bench_with_quant(a_bf16, b_fp4, scale_b, alpha, a_gs)
        entry["cutlass_quant_ms"] = round(quant_ms, 4)
        print(f"  CUTLASS+Q:    {quant_ms:.4f} ms  (quant overhead {(quant_ms/cutlass_ms-1)*100:.1f}%)")

        weight_mb = n * k / 2 / 1e6
        entry["weight_mb"] = round(weight_mb, 1)
        entry["cutlass_bw_gbps"] = round(weight_mb / cutlass_ms * 1000, 1)
        entry["weighted_ms"] = round(cutlass_ms * count, 3)

        results.append(entry)
        print()

    total = sum(r["weighted_ms"] for r in results)
    print(f"Total weighted GEMM time: {total:.3f} ms/round")

    out = "benchmarks/fixtures/a2_multi_backend.json"
    with open(out, "w") as f:
        json.dump({"gpu": gpu, "sm_count": sm, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "results": results, "total_weighted_ms": round(total, 3)}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
