#!/usr/bin/env python3
"""Isolated KERNEL-level check for the "集成后端端到端测试" finding:
B12x MoE integration (_patch_moe_b12x in runtime/backends/laguna.py) failed
with "CUDA graph不兼容" + underwhelming performance. Root cause + fix are
documented in notes/2026-07-24-moe-b12x-cudagraph-incompat-fix.md.

No model load -- synthetic NVFP4 weights only, same shape as
laguna_moe_direct_kernel_verify.py (peak footprint ~1.2 GB). Check
`nvidia-smi` before running -- skip if another process has a large model
resident.

1. Reproduces the exact failure: LagunaMoEB12x(use_cuda_graph=False) (the
   pre-fix _patch_moe_b12x config) called inside an outer torch.cuda.graph()
   capture -- mirrors what DFlashVerifyCudaGraph.capture() /
   LagunaCudaGraphDecode.capture() do to backend.model.forward() when
   QSR_MOE_B12X=1.
2. Confirms the fix: LagunaMoEB12x(use_cuda_graph=True, max_num_tokens=16)
   captures cleanly the same way, replays, and matches eager output.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_moe_b12x_cudagraph_compat_check
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from flashinfer.fp4_quantization import fp4_quantize

from runtime.backends.laguna_moe_kernel import LagunaMoEB12x

HIDDEN_SIZE, NUM_EXPERTS, TOP_K, INTERMEDIATE_SIZE = 3072, 256, 10, 1024
M = 16  # DFlash verify/draft num_tokens -- the exact shape that broke


def fused_topk(hidden_states, gating_output, topk, renormalize):
    routing_weights = F.softmax(gating_output, dim=-1, dtype=torch.float)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.float(), topk_ids.to(torch.int32)


def _quantize_weights(device, dtype):
    e, k, n = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    w1_bf16 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 15
    w2_bf16 = torch.randn((e, k, n), device=device, dtype=dtype) / 15
    gs = torch.ones(1, device=device, dtype=torch.float32)
    w1_reordered = torch.cat([w1_bf16[:, n:, :], w1_bf16[:, :n, :]], dim=1)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_reordered.reshape(e * 2 * n, k), global_scale=gs, sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(e, 2 * n, k // 2)
    w1_blockscale = w1_sf_flat.view(e, 2 * n, w1_sf_flat.shape[1])
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_bf16.reshape(e * k, n), global_scale=gs, sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    w2_q = w2_q_flat.view(e, k, n // 2)
    w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])
    ones_e = torch.ones(e, device=device, dtype=torch.float32)
    return w1_q, w1_blockscale, w2_q, w2_blockscale, ones_e


def _build_moe(weights, use_cuda_graph, max_num_tokens=None):
    w1_q, w1_blockscale, w2_q, w2_blockscale, ones_e = weights
    kwargs = dict(
        num_experts=NUM_EXPERTS, top_k=TOP_K, hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE, device="cuda",
        use_cuda_graph=use_cuda_graph,
    )
    if max_num_tokens is not None:
        kwargs["max_num_tokens"] = max_num_tokens
    moe = LagunaMoEB12x(**kwargs)
    moe.load_weights(
        w13_weight=w1_q, w13_weight_scale=w1_blockscale, w13_weight_scale_2=ones_e,
        w2_weight=w2_q, w2_weight_scale=w2_blockscale, w2_weight_scale_2=ones_e,
    )
    return moe


def main() -> int:
    torch.manual_seed(7)
    device, dtype = "cuda", torch.bfloat16
    weights = _quantize_weights(device, dtype)

    a = torch.randn((M, HIDDEN_SIZE), device=device, dtype=dtype) / 10
    score = torch.randn((M, NUM_EXPERTS), device=device, dtype=dtype)
    topk_weight, topk_ids = fused_topk(a, score, TOP_K, renormalize=False)

    moe_ref = _build_moe(weights, use_cuda_graph=False)
    ref_out = moe_ref.forward(a, topk_ids, topk_weight).clone()
    print(f"[ref] eager output sum={ref_out.float().sum().item():.4f}")

    # Repro: pre-fix config (use_cuda_graph=False) inside outer graph capture.
    moe_old = _build_moe(weights, use_cuda_graph=False)
    for _ in range(3):
        moe_old.forward(a, topk_ids, topk_weight)
    torch.cuda.synchronize()
    repro_failed = False
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            moe_old.forward(a, topk_ids, topk_weight)
        print("[repro] UNEXPECTED: old config captured without error")
    except RuntimeError as ex:
        repro_failed = True
        print(f"[repro] OLD config raised RuntimeError during capture (expected): {ex}")

    # Fix: use_cuda_graph=True, max_num_tokens=16, same outer-capture pattern.
    # Never call moe_new.capture() -- that's the standalone-usage inner-graph
    # API; here forward() must take the eager _forward_impl branch so the
    # OUTER capture (simulated below) records its kernels directly.
    moe_new = _build_moe(weights, use_cuda_graph=True, max_num_tokens=16)
    for _ in range(3):
        moe_new.forward(a, topk_ids, topk_weight)
    torch.cuda.synchronize()

    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer):
        captured_out = moe_new.forward(a, topk_ids, topk_weight)
    outer.replay()
    torch.cuda.synchronize()

    max_diff = (captured_out.float() - ref_out.float()).abs().max().item()
    print(
        f"[fix] captured-inside-outer-graph output sum="
        f"{captured_out.float().sum().item():.4f}, max_diff_vs_eager={max_diff:.6f}"
    )

    ok = repro_failed and max_diff < 1e-2
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
