#!/usr/bin/env python3
"""A2: Benchmark custom SM120 NVFP4 GEMM kernel configs vs vLLM baseline.

Tests 4 tile configs (A=baseline, B=persistent, C=256x128, D=128x256)
against vLLM's cutlass_scaled_fp4_mm. Checks bit-exactness + latency.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_custom_kernel_bench
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SO_PATH = os.path.join(_REPO, "runtime/kernels/nvfp4_gemm_sm120.so")
PEAK_BW = 1338.8
WARMUP = 50
ITERS = 300

SHAPES = [
    (4, 34816, 5120, 56, "gate_up_proj"),
    (4, 5120, 17408, 56, "down_proj"),
    (4, 17408, 17408, 8, "down_proj_attn"),
    (4, 6144, 6144, 64, "out_proj"),
    (4, 5120, 5120, 72, "in_proj_qkvz"),
    (4, 96, 5120, 48, "in_proj_ba"),
]

CONFIG_NAMES = {
    0: "A(128x128,nopersist)",
    1: "B(128x128,persist)",
    2: "C(256x128,persist)",
    3: "D(128x256,persist)",
}


def load_custom_lib():
    lib = ctypes.CDLL(SO_PATH)
    lib.qsr_nvfp4_gemm.restype = ctypes.c_int
    lib.qsr_nvfp4_gemm.argtypes = [
        ctypes.c_int,  # config_id
        ctypes.c_void_p,  # D
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # Asf
        ctypes.c_void_p,  # Bsf
        ctypes.c_void_p,  # alpha
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
        ctypes.c_void_p,  # stream
    ]
    return lib


def prepare_inputs(M, N, K, device="cuda"):
    """Create NVFP4-quantized inputs in vLLM's swizzled format."""
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
        pad_nvfp4_weight_for_cutlass,
        pad_nvfp4_activation_for_cutlass,
    )

    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    alpha = torch.ones(1, dtype=torch.float32, device=device)

    # Quantize weight
    w_fp4, w_bs = scaled_fp4_quant(w_bf16, alpha, is_sf_swizzled_layout=True, backend="cutlass")
    w_bs_sw = swizzle_blockscale(w_bs)
    w_fp4_pad, pad_cols = pad_nvfp4_weight_for_cutlass(w_fp4)

    # Quantize activation
    x_fp4, x_bs = scaled_fp4_quant(x_bf16, alpha, is_sf_swizzled_layout=True, backend="cutlass")
    x_fp4_pad = pad_nvfp4_activation_for_cutlass(x_fp4, pad_cols)

    return x_fp4_pad, w_fp4_pad, x_bs, w_bs_sw, alpha, pad_cols


def bench_vllm_baseline(x_fp4, w_fp4, x_bs, w_bs, alpha, M, N, K):
    """Benchmark vLLM's cutlass_scaled_fp4_mm."""
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    def run():
        return cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_bs, w_bs, alpha, torch.bfloat16)

    for _ in range(WARMUP):
        out = run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return out, np.percentile(times, 50)


def bench_custom(lib, config_id, x_fp4, w_fp4, x_bs, w_bs, alpha, M, N, K):
    """Benchmark custom kernel config."""
    N_padded = w_fp4.shape[0]
    out = torch.empty(M, N_padded, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        return lib.qsr_nvfp4_gemm(
            config_id,
            out.data_ptr(),
            x_fp4.data_ptr(),
            w_fp4.data_ptr(),
            x_bs.data_ptr(),
            w_bs.data_ptr(),
            alpha.data_ptr(),
            M,
            N_padded,
            K,
            stream,
        )

    # Warmup
    for _ in range(WARMUP):
        rc = run()
        if rc != 0:
            return None, None, f"CUTLASS error {rc}"
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return out, np.percentile(times, 50), None


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Kernel: {SO_PATH}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print()

    lib = load_custom_lib()
    results = {}

    for M, N, K, count, name in SHAPES:
        print(f"{'=' * 70}")
        print(f"{name}: M={M}, N={N}, K={K} (×{count}/round)")
        print(f"{'=' * 70}")

        x_fp4, w_fp4, x_bs, w_bs, alpha, pad_cols = prepare_inputs(M, N, K)
        N_padded = w_fp4.shape[0]
        weight_mb = N * K * 0.5 / 1e6

        # Baseline
        ref_out, ref_p50 = bench_vllm_baseline(x_fp4, w_fp4, x_bs, w_bs, alpha, M, N, K)
        ref_bw = weight_mb / (ref_p50 / 1000) / 1e3
        print(
            f"  vLLM baseline:  p50={ref_p50:.4f}ms  BW={ref_bw:.0f}GB/s ({ref_bw / PEAK_BW * 100:.1f}%)"
        )

        shape_results = {"baseline_p50_ms": round(ref_p50, 4)}

        # Custom configs
        for cfg_id in range(4):
            cfg_name = CONFIG_NAMES[cfg_id]
            out, p50, err = bench_custom(lib, cfg_id, x_fp4, w_fp4, x_bs, w_bs, alpha, M, N, K)

            if err:
                print(f"  {cfg_name}: FAILED - {err}")
                shape_results[cfg_name] = {"error": err}
                continue

            # Bit-exact check (slice to actual N)
            ref_sliced = ref_out[:, :N]
            out_sliced = out[:, :N]
            exact = torch.equal(ref_sliced, out_sliced)
            max_diff = (ref_sliced.float() - out_sliced.float()).abs().max().item()

            bw = weight_mb / (p50 / 1000) / 1e3
            speedup = ref_p50 / p50 if p50 > 0 else 0
            status = "✅" if exact else f"❌ max_diff={max_diff:.6f}"

            print(f"  {cfg_name}: p50={p50:.4f}ms  BW={bw:.0f}GB/s  {speedup:.3f}x  {status}")
            shape_results[cfg_name] = {
                "p50_ms": round(p50, 4),
                "bw_gbps": round(bw, 1),
                "speedup": round(speedup, 3),
                "bit_exact": exact,
                "max_diff": max_diff,
            }

        results[name] = shape_results
        print()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Weighted ms/round")
    print(f"{'=' * 70}")
    header = f"{'Shape':<20s} {'baseline':>10s}"
    for cfg_id in range(4):
        header += f" {CONFIG_NAMES[cfg_id]:>22s}"
    print(header)

    totals = {"baseline": 0.0}
    for cfg_id in range(4):
        totals[CONFIG_NAMES[cfg_id]] = 0.0

    for M, N, K, count, name in SHAPES:
        row = f"{name:<20s}"
        bl = results[name]["baseline_p50_ms"] * count
        totals["baseline"] += bl
        row += f" {bl:>10.3f}"
        for cfg_id in range(4):
            cn = CONFIG_NAMES[cfg_id]
            r = results[name].get(cn, {})
            if "p50_ms" in r:
                w = r["p50_ms"] * count
                totals[cn] += w
                row += f" {w:>22.3f}"
            else:
                row += f" {'FAIL':>22s}"
        print(row)

    print("-" * 120)
    row = f"{'TOTAL':<20s} {totals['baseline']:>10.3f}"
    for cfg_id in range(4):
        cn = CONFIG_NAMES[cfg_id]
        row += f" {totals[cn]:>22.3f}"
    print(row)

    # Save
    out_path = Path("benchmarks/fixtures/a2_custom_kernel_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "date": datetime.now().isoformat(timespec="seconds"),
                "gpu": torch.cuda.get_device_name(),
                "results": results,
                "totals_weighted_ms": {k: round(v, 3) for k, v in totals.items()},
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
