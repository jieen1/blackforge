#!/usr/bin/env python3
"""A2: NVFP4 GEMM 微基准 — 精确 decode shapes 的 kernel 计时。

对每个 decode GEMM shape 计时当前 CUTLASS NVFP4 kernel，
建立 autotune 基线。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_gemm_microbench \
        [--warmup 50] [--iters 200] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Decode GEMM shapes from a2_gemm_shape_profile (c=1, K=3)
# (M, N, K, count_per_round, name)
DECODE_SHAPES = [
    (4, 34816, 5120, 56, "gate_up_proj"),
    (4, 5120, 17408, 56, "down_proj"),
    (4, 17408, 17408, 8, "down_proj_attn"),
    (4, 6144, 6144, 64, "out_proj"),
    (4, 5120, 5120, 72, "in_proj_qkvz"),
    (4, 96, 5120, 48, "in_proj_ba"),
]

# Also test with c=4 (M=16)
DECODE_SHAPES_C4 = [
    (16, 34816, 5120, 56, "gate_up_proj"),
    (16, 5120, 17408, 56, "down_proj"),
    (16, 17408, 17408, 8, "down_proj_attn"),
    (16, 6144, 6144, 64, "out_proj"),
    (16, 5120, 5120, 72, "in_proj_qkvz"),
    (16, 96, 5120, 48, "in_proj_ba"),
]


def bench_cutlass_nvfp4(m: int, n: int, k: int, warmup: int, iters: int) -> dict:
    """Benchmark CUTLASS NVFP4 GEMM for a single shape."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    # Create random activation (bf16, will be quantized)
    a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    # Create random weight (bf16, will be quantized)
    b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    # Quantize weight to NVFP4
    b_amax = b_bf16.abs().max().to(torch.float32)
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    b_fp4, scale_b = ops.scaled_fp4_quant(b_bf16, b_global_scale)

    # Quantize activation
    a_amax = a_bf16.abs().max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    a_fp4, scale_a = ops.scaled_fp4_quant(a_bf16, a_global_scale)

    alpha = (1.0 / (a_global_scale * b_global_scale)).unsqueeze(0)

    # Warmup
    for _ in range(warmup):
        out = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.bfloat16)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    flops = 2 * m * n * k
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    # Memory: read weight (n*k/2 bytes for FP4) + activation (m*k/2) + scales + write output
    weight_bytes = n * k // 2  # FP4 packed
    act_bytes = m * k // 2
    out_bytes = m * n * 2  # bf16
    total_bytes = weight_bytes + act_bytes + out_bytes
    bw_gbs = total_bytes / (elapsed_ms * 1e-3) / 1e9

    return {
        "m": m, "n": n, "k": k,
        "latency_ms": round(elapsed_ms, 4),
        "tflops": round(tflops, 2),
        "eff_bw_gb_s": round(bw_gbs, 1),
        "flops": flops,
    }


def bench_torch_bf16(m: int, n: int, k: int, warmup: int, iters: int) -> dict:
    """Benchmark torch bf16 GEMM as reference."""
    import torch
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        out = torch.mm(a, b.t())
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = torch.mm(a, b.t())
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    flops = 2 * m * n * k
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    return {
        "m": m, "n": n, "k": k,
        "latency_ms": round(elapsed_ms, 4),
        "tflops": round(tflops, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--c4", action="store_true", help="Also bench M=16 (c=4)")
    args = parser.parse_args()

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")

    shapes = DECODE_SHAPES[:]
    if args.c4:
        shapes += DECODE_SHAPES_C4

    results = []
    print(f"\n{'Shape':>20} {'Name':<16} {'CUTLASS ms':>12} {'TFLOPS':>8} {'BW GB/s':>10} {'bf16 ms':>10} {'speedup':>8}")
    print("-" * 90)

    total_cutlass_ms = 0
    total_bf16_ms = 0
    for m, n, k, count, name in shapes:
        try:
            cutlass = bench_cutlass_nvfp4(m, n, k, args.warmup, args.iters)
        except Exception as e:
            print(f"  [{m}×{n}×{k}] {name}: CUTLASS FAILED ({e})")
            continue
        try:
            bf16 = bench_torch_bf16(m, n, k, args.warmup, args.iters)
        except Exception as e:
            bf16 = {"latency_ms": 0}

        speedup = bf16["latency_ms"] / cutlass["latency_ms"] if cutlass["latency_ms"] > 0 else 0
        shape_str = f"{m}×{n}×{k}"
        print(f"{shape_str:>20} {name:<16} {cutlass['latency_ms']:>12.4f} {cutlass['tflops']:>8.2f} "
              f"{cutlass['eff_bw_gb_s']:>10.1f} {bf16['latency_ms']:>10.4f} {speedup:>7.2f}×")

        weighted_cutlass = cutlass["latency_ms"] * count
        weighted_bf16 = bf16["latency_ms"] * count
        total_cutlass_ms += weighted_cutlass
        total_bf16_ms += weighted_bf16

        results.append({
            **cutlass,
            "name": name,
            "count_per_round": count,
            "bf16_latency_ms": bf16["latency_ms"],
            "speedup_vs_bf16": round(speedup, 3),
            "weighted_ms_per_round": round(weighted_cutlass, 3),
        })

    print(f"\n{'Weighted total/round':>20} {'':16} {total_cutlass_ms:>12.3f} {'':>8} {'':>10} {total_bf16_ms:>10.3f} "
          f"{total_bf16_ms/total_cutlass_ms if total_cutlass_ms > 0 else 0:>7.2f}×")

    if args.json:
        print(json.dumps({"shapes": results, "total_weighted_ms": round(total_cutlass_ms, 3)}))


if __name__ == "__main__":
    main()
