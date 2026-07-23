#!/usr/bin/env python3
"""A2: Complete native NVFP4 GEMM performance baseline.

Records CUTLASS sm120 NVFP4 GEMM latency for ALL decode/prefill/verify
shapes. Output: benchmarks/fixtures/a2_native_baseline.json

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_native_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Qwen3.6-27B NVFP4 decode GEMM shapes (from a2_gemm_shape_profile)
# (M, N, K, count_per_round, name)
SHAPES_C1 = [
    (4, 34816, 5120, 56, "gate_up_proj"),
    (4, 5120, 17408, 56, "down_proj"),
    (4, 17408, 17408, 8, "down_proj_attn"),
    (4, 6144, 6144, 64, "out_proj"),
    (4, 5120, 5120, 72, "in_proj_qkvz"),
    (4, 96, 5120, 48, "in_proj_ba"),
]

SHAPES_M1 = [
    (1, 34816, 5120, 56, "gate_up_proj"),
    (1, 5120, 17408, 56, "down_proj"),
    (1, 17408, 17408, 8, "down_proj_attn"),
    (1, 6144, 6144, 64, "out_proj"),
    (1, 5120, 5120, 72, "in_proj_qkvz"),
    (1, 96, 5120, 48, "in_proj_ba"),
]

SHAPES_C4 = [
    (16, 34816, 5120, 56, "gate_up_proj"),
    (16, 5120, 17408, 56, "down_proj"),
    (16, 17408, 17408, 8, "down_proj_attn"),
    (16, 6144, 6144, 64, "out_proj"),
    (16, 5120, 5120, 72, "in_proj_qkvz"),
    (16, 96, 5120, 48, "in_proj_ba"),
]

SHAPES_PREFILL = [
    (512, 34816, 5120, 1, "gate_up_proj_pf512"),
    (512, 5120, 17408, 1, "down_proj_pf512"),
    (512, 6144, 6144, 1, "out_proj_pf512"),
    (512, 5120, 5120, 1, "in_proj_pf512"),
]


def bench_shape(m: int, n: int, k: int, warmup: int, iters: int) -> dict:
    """Benchmark CUTLASS NVFP4 GEMM for one shape."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    # Quantize weight to NVFP4
    b_amax = b_bf16.abs().max().to(torch.float32)
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    b_fp4, scale_b_fp4 = ops.scaled_fp4_quant(b_bf16, b_global_scale)

    # Quantize activation
    a_amax = a_bf16.abs().max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    alpha = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

    # Pre-quantize activation for "no quant overhead" timing
    a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a_bf16, a_global_scale)

    out_dtype = torch.bfloat16

    # Warmup
    for _ in range(warmup):
        ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, out_dtype)
    torch.cuda.synchronize()

    # Timed: GEMM only (activation pre-quantized)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, out_dtype)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    p50 = times_ms[len(times_ms) // 2]
    p10 = times_ms[max(0, int(len(times_ms) * 0.1))]
    p90 = times_ms[min(len(times_ms) - 1, int(len(times_ms) * 0.9))]

    # Also time WITH activation quantization (realistic end-to-end)
    for _ in range(warmup):
        aq, saq = ops.scaled_fp4_quant(a_bf16, a_global_scale)
        ops.cutlass_scaled_fp4_mm(aq, b_fp4, saq, scale_b_fp4, alpha, out_dtype)
    torch.cuda.synchronize()

    start2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end2 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start2[i].record()
        aq, saq = ops.scaled_fp4_quant(a_bf16, a_global_scale)
        ops.cutlass_scaled_fp4_mm(aq, b_fp4, saq, scale_b_fp4, alpha, out_dtype)
        end2[i].record()
    torch.cuda.synchronize()

    times_with_quant = sorted(s.elapsed_time(e) for s, e in zip(start2, end2))
    p50_with_quant = times_with_quant[len(times_with_quant) // 2]

    # NVFP4 weight: 4 bits per element = k/2 bytes per row
    weight_bytes = n * k // 2
    # FP4 activation: m * k / 2 bytes
    act_bytes = m * k // 2
    total_bytes = weight_bytes + act_bytes
    bw_gbps = total_bytes / (p50 * 1e-3) / 1e9
    flops = 2 * m * n * k
    tflops = flops / (p50 * 1e-3) / 1e12

    return {
        "m": m, "n": n, "k": k,
        "p50_ms": round(p50, 4),
        "p10_ms": round(p10, 4),
        "p90_ms": round(p90, 4),
        "p50_with_quant_ms": round(p50_with_quant, 4),
        "tflops": round(tflops, 2),
        "bw_gbps": round(bw_gbps, 1),
        "weight_mb": round(weight_bytes / 1e6, 1),
    }


def bench_bf16_ref(m: int, n: int, k: int, warmup: int, iters: int) -> float:
    """Benchmark bf16 GEMM as reference."""
    import torch

    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        torch.mm(a, b.t())
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        torch.mm(a, b.t())
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return round(times[len(times) // 2], 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--out", default="benchmarks/fixtures/a2_native_baseline.json")
    args = parser.parse_args()

    import torch

    gpu_name = torch.cuda.get_device_name(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU: {gpu_name} ({sm_count} SMs)")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Peak memcpy BW
    print("\n=== Peak memcpy bandwidth ===")
    buf = 256 * 1024 * 1024
    src = torch.empty(buf // 4, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        dst.copy_(src)
    torch.cuda.synchronize()
    peak_bw = 2 * buf * 50 / (time.perf_counter() - t0) / 1e9
    print(f"  Peak memcpy BW: {peak_bw:.1f} GB/s")

    results = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": gpu_name,
        "sm_count": sm_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "peak_memcpy_bw_gbps": round(peak_bw, 1),
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": {},
    }

    groups = [
        ("decode_c1_M4", SHAPES_C1),
        ("decode_M1", SHAPES_M1),
        ("decode_c4_M16", SHAPES_C4),
        ("prefill_M512", SHAPES_PREFILL),
    ]

    for group_name, shapes in groups:
        print(f"\n=== {group_name} ===")
        entries = []
        total_ms = 0.0
        for m, n, k, count, name in shapes:
            print(f"  {name} ({m}x{n}x{k})...", end=" ", flush=True)
            try:
                r = bench_shape(m, n, k, args.warmup, args.iters)
                bf16_ms = bench_bf16_ref(m, n, k, args.warmup, args.iters)
                r["bf16_p50_ms"] = bf16_ms
                r["nvfp4_vs_bf16"] = round(bf16_ms / r["p50_ms"], 2) if r["p50_ms"] > 0 else 0
                r["count_per_round"] = count
                r["name"] = name
                r["bw_pct_peak"] = round(r["bw_gbps"] / peak_bw * 100, 1)
                wms = r["p50_ms"] * count
                r["weighted_ms_per_round"] = round(wms, 3)
                total_ms += wms
                entries.append(r)
                print(f"P50={r['p50_ms']:.4f}ms  BW={r['bw_gbps']:.0f}({r['bw_pct_peak']:.1f}%)  "
                      f"vs_bf16={r['nvfp4_vs_bf16']}x  +quant={r['p50_with_quant_ms']:.4f}ms")
            except Exception as exc:
                print(f"FAILED: {exc}")
                entries.append({"name": name, "m": m, "n": n, "k": k, "error": str(exc)})

        results["shapes"][group_name] = {
            "entries": entries,
            "total_weighted_ms_per_round": round(total_ms, 3),
        }
        print(f"  --- Total weighted: {total_ms:.3f} ms/round ---")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
