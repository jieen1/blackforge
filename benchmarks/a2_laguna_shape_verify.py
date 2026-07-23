#!/usr/bin/env python3
"""A2: Verify custom NVFP4 GEMM bit-exactness on Laguna-S-2.1 shapes.

Laguna GEMM shapes (NVFP4 linear, decode M=1..4):
  - QKV proj (full-attn): 3072 → 8192  (48+8+8)*128
  - QKV proj (SWA):       3072 → 11264 (72+8+8)*128
  - O proj (full-attn):   6144 → 3072  48*128
  - O proj (SWA):         9216 → 3072  72*128
  - Dense FFN gate_up:    3072 → 24576 12288*2
  - Dense FFN down:       12288 → 3072
  - MoE expert gate_up:   3072 → 2048  1024*2
  - MoE expert down:      1024 → 3072
  - Shared expert gate_up: 3072 → 2048
  - Shared expert down:   1024 → 3072
  - Router:               3072 → 256

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_laguna_shape_verify
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LAGUNA_SHAPES = [
    # (name, K, N) — M is batch (decode: 1..4)
    ("qkv_proj_full_attn", 3072, 8192),
    ("qkv_proj_swa", 3072, 11264),
    ("o_proj_full_attn", 6144, 3072),
    ("o_proj_swa", 9216, 3072),
    ("dense_ffn_gate_up", 3072, 24576),
    ("dense_ffn_down", 12288, 3072),
    ("moe_expert_gate_up", 3072, 2048),
    ("moe_expert_down", 1024, 3072),
    ("shared_expert_gate_up", 3072, 2048),
    ("shared_expert_down", 1024, 3072),
    ("router", 3072, 256),
]

BATCH_SIZES = [1, 2, 4]


def main():
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    # Load custom GEMM
    from runtime.nvfp4_custom_gemm import _load_lib, custom_scaled_fp4_mm, _select_config

    lib = _load_lib()
    if lib is None:
        print('ERROR: Custom GEMM .so not found, cannot verify')
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Verifying {len(LAGUNA_SHAPES)} shapes × {len(BATCH_SIZES)} batch sizes")
    print()

    total = 0
    passed = 0
    failed_shapes = []

    for name, k, n in LAGUNA_SHAPES:
        for m in BATCH_SIZES:
            total += 1
            # Create random NVFP4 inputs
            torch.manual_seed(42)
            a_fp32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
            b_fp32 = torch.randn(n, k, device="cuda", dtype=torch.float32)

            # Quantize to NVFP4
            a_fp4, a_scale = ops.scaled_fp4_quant(a_fp32, 1.0 / FLOAT4_E2M1_MAX)
            b_fp4, b_scale = ops.scaled_fp4_quant(b_fp32, 1.0 / FLOAT4_E2M1_MAX)

            a_scale_fp32 = a_scale.to(torch.float32)
            b_scale_fp32 = b_scale.to(torch.float32)
            global_scale = 1.0 / (FLOAT4_E2M1_MAX * FLOAT4_E2M1_MAX)

            # vLLM baseline
            ref = ops.cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, a_scale_fp32, b_scale_fp32, global_scale, torch.bfloat16
            )

            # Custom GEMM
            cfg = _select_config(n)
            custom = custom_scaled_fp4_mm(
                a_fp4, b_fp4, a_scale_fp32, b_scale_fp32,
                torch.tensor(global_scale, dtype=torch.float32, device="cuda"),
                torch.bfloat16,
            )

            match = torch.equal(ref, custom)
            if match:
                passed += 1
                status = "✅"
            else:
                max_diff = (ref.float() - custom.float()).abs().max().item()
                status = f"❌ max_diff={max_diff:.6f}"
                failed_shapes.append((name, m, n, max_diff))

            print(f"  {status} {name:25s} M={m} K={k} N={n} cfg={cfg}")

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} bit-exact")
    if failed_shapes:
        print("FAILED:")
        for name, m, n, diff in failed_shapes:
            print(f"  {name} M={m} N={n}: max_diff={diff:.6f}")
    else:
        print("ALL SHAPES BIT-EXACT ✅")
    print(f"{'='*60}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
