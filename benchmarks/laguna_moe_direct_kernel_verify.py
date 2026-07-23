#!/usr/bin/env python3
"""Verify runtime/backends/laguna_moe_kernel.py (direct-FlashInfer, zero
vLLM import) at Laguna's exact MoE shape.

Isolated KERNEL-level test -- synthetic NVFP4 weights only, no model load.
Peak footprint ~1.2 GB (256 experts x ~4.7 MB/expert each for w13+w2),
nowhere near the 73 GB full Laguna checkpoint. Check `nvidia-smi` before
running -- skip if another process has a large model resident.

CUDA Graph is the primary path tested (production's real hot path is
CUDA-Graph-captured, not eager -- see notes/2026-07-23-laguna-moe-b12x-
direct-kernel.md for why eager-only numbers aren't representative). Eager
mode is tested too, as a correctness cross-check and a "graph off" baseline.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_moe_direct_kernel_verify
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Laguna-S-2.1-NVFP4's actual MoE shape (config.json, verified 2026-07-23).
HIDDEN_SIZE = 3072
NUM_EXPERTS = 256
TOP_K = 10
INTERMEDIATE_SIZE = 1024

M_VALUES = [1, 4, 16, 64]
WARMUP_REPS = 5
BENCH_REPS = 20

# Current per-layer baseline: 8.73ms MoE GEMM / 47 layers (eager, no CUDA
# Graph, real model profiling -- see notes/2026-07-23-laguna-moe-b12x-
# direct-kernel.md "更新" section).
BASELINE_PER_LAYER_MS = 8.73 / 47


def _reference_moe(
    a: torch.Tensor,
    w1_gate_up: torch.Tensor,
    w2_down: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Minimal BF16 reference: per-selected-expert SiLU-gated MLP, topk-weighted sum.

    w1_gate_up: [E, 2N, K] in [gate, up] order. w2_down: [E, K, N].
    """
    m, k = a.shape
    topk = topk_ids.shape[1]
    out = torch.zeros(m, k, dtype=torch.float32, device=a.device)
    for i in range(m):
        acc = torch.zeros(k, dtype=torch.float32, device=a.device)
        for j in range(topk):
            e = int(topk_ids[i, j].item())
            gate_up = a[i].float() @ w1_gate_up[e].float().t()
            n = gate_up.shape[0] // 2
            gate, up = gate_up[:n], gate_up[n:]
            hidden = torch.nn.functional.silu(gate) * up
            down = hidden @ w2_down[e].float().t()
            acc += topk_weight[i, j].float() * down
        out[i] = acc
    return out.to(a.dtype)


def _quantize_weights(w1_bf16: torch.Tensor, w2_bf16: torch.Tensor, e: int, k: int, n: int):
    """Quantize BF16 [gate,up]-order weights to the NVFP4 tensors LagunaMoEB12x
    expects (B12x wants [up,gate] order; global_scale=1.0 keeps w1_alpha=
    w2_alpha=1.0, matching vLLM's own proven-correct test convention)."""
    from flashinfer.fp4_quantization import fp4_quantize

    gs = torch.ones(1, device=w1_bf16.device, dtype=torch.float32)
    sf_vec_size = 16

    w1_reordered = torch.cat([w1_bf16[:, n:, :], w1_bf16[:, :n, :]], dim=1)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_reordered.reshape(e * 2 * n, k),
        global_scale=gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(e, 2 * n, k // 2)
    w1_blockscale = w1_sf_flat.view(e, 2 * n, w1_sf_flat.shape[1])

    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_bf16.reshape(e * k, n),
        global_scale=gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True,
    )
    w2_q = w2_q_flat.view(e, k, n // 2)
    w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])

    ones_e = torch.ones(e, device=w1_bf16.device, dtype=torch.float32)
    return w1_q, w1_blockscale, w2_q, w2_blockscale, ones_e


def _bench(fn, reps: int = BENCH_REPS) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Shape: hidden={HIDDEN_SIZE} intermediate={INTERMEDIATE_SIZE} "
        f"experts={NUM_EXPERTS} topk={TOP_K}"
    )

    import torch.nn.functional as F

    from runtime.backends.laguna_moe_kernel import LagunaMoEB12x, LagunaMoEB12xMultiBatch

    def fused_topk(hidden_states, gating_output, topk, renormalize):
        """Plain PyTorch softmax+topk, matching FlashInfer's own proven-
        correct test helper (tests/moe/test_b12x_fused_moe.py::
        create_moe_tensors). NOT vllm.model_executor.layers.fused_moe.
        router.fused_topk_router.fused_topk -- an earlier version of this
        script used that and had its (weights, ids, _) return tuple
        unpacked in the wrong order, which silently produced garbage
        routing data that looked exactly like a CUDA Graph correctness bug
        (see notes/2026-07-23-laguna-moe-b12x-direct-kernel.md). This
        plain version has no such footgun.
        """
        routing_weights = F.softmax(gating_output, dim=-1, dtype=torch.float)
        topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.float(), topk_ids.to(torch.int32)

    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.bfloat16
    e, k, n, topk = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, TOP_K

    w1_bf16 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 15
    w2_bf16 = torch.randn((e, k, n), device=device, dtype=dtype) / 15
    w1_q, w1_blockscale, w2_q, w2_blockscale, ones_e = _quantize_weights(w1_bf16, w2_bf16, e, k, n)
    weight_kwargs = dict(
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w13_weight_scale_2=ones_e,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        w2_weight_scale_2=ones_e,
    )

    mem_after_weights = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU mem after weight quantization: {mem_after_weights:.0f} MiB")

    results: dict = {}

    # ================================================================
    # Part 1: eager mode, single shared instance, M=1/4/16/64.
    # ================================================================
    print("\n=== Eager mode (no CUDA Graph) ===")
    moe_eager = LagunaMoEB12x(num_experts=e, top_k=topk, hidden_size=k, intermediate_size=n, device=device)
    moe_eager.load_weights(**weight_kwargs)

    for m in M_VALUES:
        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        score = torch.randn((m, e), device=device, dtype=dtype)
        topk_weight, topk_ids = fused_topk(a, score, topk, renormalize=False)

        out = moe_eager.forward(a, topk_ids, topk_weight)
        ref_out = _reference_moe(a, w1_bf16, w2_bf16, topk_weight, topk_ids)
        max_diff = (out.float() - ref_out.float()).abs().max().item()
        correct = torch.allclose(out, ref_out, atol=2e-1, rtol=2e-1)

        for _ in range(WARMUP_REPS):
            moe_eager.forward(a, topk_ids, topk_weight)
        elapsed_ms = _bench(lambda: moe_eager.forward(a, topk_ids, topk_weight))

        print(
            f"  M={m:>3}: correct={'PASS' if correct else 'FAIL'} "
            f"(max_diff={max_diff:.4f})  latency={elapsed_ms:.3f} ms/call"
        )
        results[f"eager_m{m}"] = {
            "correct": correct, "max_diff": round(max_diff, 4), "latency_ms": round(elapsed_ms, 4),
        }

    # ================================================================
    # Part 2: CUDA Graph, single instance per batch size, M=1/4/16/64 --
    # this is the PRIMARY, production-representative path.
    # ================================================================
    print("\n=== CUDA Graph mode (LagunaMoEB12x.capture(), one instance per M) ===")
    for m in M_VALUES:
        moe_g = LagunaMoEB12x(
            num_experts=e, top_k=topk, hidden_size=k, intermediate_size=n,
            device=device, use_cuda_graph=True, max_num_tokens=m,
        )
        moe_g.load_weights(**weight_kwargs)
        moe_g.capture()

        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        score = torch.randn((m, e), device=device, dtype=dtype)
        topk_weight, topk_ids = fused_topk(a, score, topk, renormalize=False)
        ref_out = _reference_moe(a, w1_bf16, w2_bf16, topk_weight, topk_ids)

        out = moe_g.forward(a, topk_ids, topk_weight)
        max_diff = (out.float() - ref_out.float()).abs().max().item()
        correct = torch.allclose(out, ref_out, atol=2e-1, rtol=2e-1)

        elapsed_ms = _bench(lambda: moe_g.forward(a, topk_ids, topk_weight), reps=BENCH_REPS * 5)
        speedup = (1 - elapsed_ms / BASELINE_PER_LAYER_MS) * 100

        print(
            f"  M={m:>3}: correct={'PASS' if correct else 'FAIL'} "
            f"(max_diff={max_diff:.4f})  latency={elapsed_ms:.4f} ms/call  "
            f"vs baseline({BASELINE_PER_LAYER_MS:.4f}ms): {speedup:+.1f}%"
        )
        results[f"graph_m{m}"] = {
            "correct": correct, "max_diff": round(max_diff, 4), "latency_ms": round(elapsed_ms, 4),
            "vs_baseline_pct": round(speedup, 1),
        }

    # ================================================================
    # Part 3: LagunaMoEB12xMultiBatch end-to-end -- capture_all() then
    # dispatch by actual batch size, matching how a real decode loop with
    # a varying active-slot count would use this.
    # ================================================================
    print("\n=== LagunaMoEB12xMultiBatch (capture_all + dispatch) ===")
    # Scoped to realistic production decode concurrency (server/app.py's
    # Laguna default is capacity=1; 4 covers real headroom) -- NOT
    # max(M_VALUES)=64, which would capture 64 separate graphs (one per
    # exact batch size 1..64), needlessly slow/heavy for what this test
    # needs to demonstrate. Dispatch is only exercised at bs values within
    # this range (1 and 4 from M_VALUES); M=16/64 aren't valid dispatch
    # targets for this manager instance and are intentionally not tested
    # here -- a production deployment would size max_batch_size to its own
    # real capacity, same as this.
    multibatch_max_bs = 4
    mb = LagunaMoEB12xMultiBatch(
        num_experts=e, top_k=topk, hidden_size=k, intermediate_size=n,
        device=device, max_batch_size=multibatch_max_bs,
    )
    mb.load_weights(**weight_kwargs)
    t0 = time.perf_counter()
    mb.capture_all()
    capture_s = time.perf_counter() - t0
    print(
        f"  capture_all() for batch_size=1..{multibatch_max_bs}: "
        f"{capture_s:.1f}s, captured={mb.captured_sizes}"
    )

    for m in [v for v in M_VALUES if v <= multibatch_max_bs]:
        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        score = torch.randn((m, e), device=device, dtype=dtype)
        topk_weight, topk_ids = fused_topk(a, score, topk, renormalize=False)
        ref_out = _reference_moe(a, w1_bf16, w2_bf16, topk_weight, topk_ids)

        out = mb.forward(a, topk_ids, topk_weight)
        max_diff = (out.float() - ref_out.float()).abs().max().item()
        correct = torch.allclose(out, ref_out, atol=2e-1, rtol=2e-1)
        print(f"  M={m:>3} (via MultiBatch dispatch): correct={'PASS' if correct else 'FAIL'} (max_diff={max_diff:.4f})")
        results[f"multibatch_m{m}"] = {"correct": correct, "max_diff": round(max_diff, 4)}

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nGPU mem peak: {mem_peak:.0f} MiB")

    out_path = Path("benchmarks/fixtures/laguna_moe_direct_kernel_verify.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "gpu": torch.cuda.get_device_name(0),
                "date": datetime.now().isoformat(timespec="seconds"),
                "shape": {"hidden": HIDDEN_SIZE, "intermediate": INTERMEDIATE_SIZE,
                          "experts": NUM_EXPERTS, "topk": TOP_K},
                "gpu_mem_peak_mib": round(mem_peak, 0),
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nSaved: {out_path}")
    all_pass = all(r["correct"] for r in results.values())
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
