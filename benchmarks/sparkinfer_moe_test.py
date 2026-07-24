#!/usr/bin/env python3
"""sparkinfer MoE kernel correctness + perf test for Laguna NVFP4.

Loads weights directly from the Laguna checkpoint (bypassing vLLM),
prepares them for sparkinfer, runs the kernel, and compares against
a dequantized fp32 reference.

Usage:
    /home/bot/.venvs/vllm/bin/python benchmarks/sparkinfer_moe_test.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bot/project/sparkinfer")

from sparkinfer._lib.intrinsics import swizzle_block_scale
from sparkinfer.moe.fused_moe._impl import (
    plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
    allocate_tp_moe_workspace_pool,
    build_tp_moe_fp4_binding,
    sparkinfer_moe_fp4,

)
from sparkinfer.moe.fused_moe import is_supported

CKPT = pathlib.Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843"
    )
)

E = 256
TOP_K = 10
K = 3072
I = 1024
LAYER = 1

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def load_checkpoint_weights(layer_idx: int, device: str = "cuda"):
    """Load per-expert weights from the Laguna checkpoint for one layer."""
    idx_path = CKPT / "model.safetensors.index.json"
    with open(idx_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = set()
    prefix = f"model.layers.{layer_idx}.mlp.experts"
    for eid in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
                key = f"{prefix}.{eid}.{proj}.{suffix}"
                shard_files.add(weight_map[key])

    tensors = {}
    from safetensors import safe_open
    for shard in sorted(shard_files):
        path = CKPT / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    tensors[key] = f.get_tensor(key)

    gate_w = torch.empty(E, I, K // 2, dtype=torch.uint8)
    up_w = torch.empty(E, I, K // 2, dtype=torch.uint8)
    down_w = torch.empty(E, K, I // 2, dtype=torch.uint8)

    gate_sf = torch.empty(E, I, K // 16, dtype=torch.float8_e4m3fn)
    up_sf = torch.empty(E, I, K // 16, dtype=torch.float8_e4m3fn)
    down_sf = torch.empty(E, K, I // 16, dtype=torch.float8_e4m3fn)

    gate_gs = torch.empty(E, dtype=torch.float32)
    up_gs = torch.empty(E, dtype=torch.float32)
    down_gs = torch.empty(E, dtype=torch.float32)
    gate_is = torch.empty(E, dtype=torch.float32)
    up_is = torch.empty(E, dtype=torch.float32)
    down_is = torch.empty(E, dtype=torch.float32)

    for eid in range(E):
        ep = f"{prefix}.{eid}"
        gate_w[eid] = tensors[f"{ep}.gate_proj.weight_packed"]
        gate_sf[eid] = tensors[f"{ep}.gate_proj.weight_scale"]
        gate_gs[eid] = tensors[f"{ep}.gate_proj.weight_global_scale"]
        gate_is[eid] = tensors[f"{ep}.gate_proj.input_global_scale"]

        up_w[eid] = tensors[f"{ep}.up_proj.weight_packed"]
        up_sf[eid] = tensors[f"{ep}.up_proj.weight_scale"]
        up_gs[eid] = tensors[f"{ep}.up_proj.weight_global_scale"]
        up_is[eid] = tensors[f"{ep}.up_proj.input_global_scale"]

        down_w[eid] = tensors[f"{ep}.down_proj.weight_packed"]
        down_sf[eid] = tensors[f"{ep}.down_proj.weight_scale"]
        down_gs[eid] = tensors[f"{ep}.down_proj.weight_global_scale"]
        down_is[eid] = tensors[f"{ep}.down_proj.input_global_scale"]

    return {
        "gate_w": gate_w, "up_w": up_w, "down_w": down_w,
        "gate_sf": gate_sf, "up_sf": up_sf, "down_sf": down_sf,
        "gate_gs": gate_gs, "up_gs": up_gs, "down_gs": down_gs,
        "gate_is": gate_is, "up_is": up_is, "down_is": down_is,
    }


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 packed weight to fp32. packed: [rows, cols//2] uint8."""
    rows, half_cols = packed.shape
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    lut = FP4_LUT.to(packed.device)
    vals = torch.stack([lut[low], lut[high]], dim=-1).reshape(rows, half_cols * 2)
    scale_expanded = scale.float().repeat_interleave(16, dim=-1)
    return vals * scale_expanded * global_scale


def reference_moe(hidden, gate_w, up_w, down_w, gate_sf, up_sf, down_sf,
                  gate_gs, up_gs, down_gs, gate_is, up_is, down_is,
                  topk_ids, topk_weights):
    """Pure fp32 reference MoE computation."""
    M = hidden.shape[0]
    output = torch.zeros(M, K, dtype=torch.float32, device=hidden.device)

    gate_deq = torch.stack([dequant_fp4(gate_w[e], gate_sf[e], gate_gs[e]) for e in range(E)])
    up_deq = torch.stack([dequant_fp4(up_w[e], up_sf[e], up_gs[e]) for e in range(E)])
    down_deq = torch.stack([dequant_fp4(down_w[e], down_sf[e], down_gs[e]) for e in range(E)])

    hidden_f32 = hidden.float()
    for m in range(M):
        for ki in range(TOP_K):
            eid = topk_ids[m, ki].item()
            w = topk_weights[m, ki].float()
            gate_out = hidden_f32[m] @ gate_deq[eid].T
            up_out = hidden_f32[m] @ up_deq[eid].T
            act = F.silu(gate_out) * up_out
            down_out = act @ down_deq[eid].T
            output[m] += w * down_out
    return output


def main():
    print(f"sparkinfer supported: {is_supported()}")
    device = "cuda"

    print(f"Loading layer {LAYER} weights from checkpoint ({E} experts)...")
    t0 = time.time()
    raw = load_checkpoint_weights(LAYER)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    for name in ("gate_gs", "up_gs", "down_gs", "gate_is", "up_is", "down_is"):
        t = raw[name]
        print(f"  {name}: min={t.min():.6e} max={t.max():.6e} mean={t.mean():.6e}")

    print("\nPreparing w13 (w31 layout: up first, gate second)...")
    w13_fp4 = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).to(device).contiguous()
    w13_sf = torch.cat([raw["up_sf"], raw["gate_sf"]], dim=1).to(device).contiguous()
    w13_sf_swizzled = swizzle_block_scale(w13_sf)

    w2_fp4 = raw["down_w"].to(device).contiguous()
    w2_sf = raw["down_sf"].to(device).contiguous()
    w2_sf_swizzled = swizzle_block_scale(w2_sf)

    w1_global_scale = raw["up_gs"].to(device).contiguous()
    w2_global_scale = raw["down_gs"].to(device).contiguous()

    input_scale_w1 = torch.max(
        raw["up_is"].max(), raw["gate_is"].max()
    ).to(device)
    input_scale_w2 = raw["down_is"].max().to(device)
    a1_gscale = (1.0 / input_scale_w1).to(device)
    a2_gscale = (1.0 / input_scale_w2).to(device)

    print(f"  w13_fp4: {w13_fp4.shape} {w13_fp4.dtype}")
    print(f"  w13_sf_swizzled: {w13_sf_swizzled.shape} {w13_sf_swizzled.dtype}")
    print(f"  w2_fp4: {w2_fp4.shape} {w2_fp4.dtype}")
    print(f"  w1_global_scale: {w1_global_scale.shape} range=[{w1_global_scale.min():.6e}, {w1_global_scale.max():.6e}]")
    print(f"  w2_global_scale: {w2_global_scale.shape} range=[{w2_global_scale.min():.6e}, {w2_global_scale.max():.6e}]")
    print(f"  a1_gscale: {a1_gscale.item():.6e}")
    print(f"  a2_gscale: {a2_gscale.item():.6e}")

    print("\nPlanning + preparing sparkinfer weights...")
    wplan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=K,
        intermediate_size=I,
        w13_layout="w31",
    )
    print(f"  Plan: prepares_runtime_alphas={wplan.prepares_runtime_alphas}")

    experts = prepare_sparkinfer_fp4_moe_weights(
        plan=wplan,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_fp4=w13_fp4,
        w1_blockscale=w13_sf_swizzled,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_sf_swizzled,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )
    print(f"  w1_alphas range: [{experts.w1_alphas.min():.6e}, {experts.w1_alphas.max():.6e}]")
    print(f"  w2_alphas range: [{experts.w2_alphas.min():.6e}, {experts.w2_alphas.max():.6e}]")

    for M in [1, 4, 16]:
        print(f"\n{'='*60}")
        print(f"Testing M={M}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1

        router_logits = torch.randn(M, E, dtype=torch.float32, device=device)
        topk_weights_raw, topk_ids = torch.topk(router_logits, TOP_K, dim=-1)
        topk_weights_val = F.softmax(topk_weights_raw, dim=-1).to(torch.bfloat16)

        print("  Running sparkinfer kernel...")
        workspace = allocate_tp_moe_workspace_pool()
        binding = build_tp_moe_fp4_binding(
            scratch=workspace,
            a=hidden,
            experts=experts,
            topk_weights=topk_weights_val,
            topk_ids=topk_ids.to(torch.int32),
            quant_mode="nvfp4",
            input_scales_static=True,
        )

        torch.cuda.synchronize()
        t0 = time.time()
        result = sparkinfer_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"  Result: {result.shape} {result.dtype}")
        print(f"  Result norm: {result.float().norm():.4f}")
        print(f"  Result nonzero: {(result != 0).sum().item()}/{result.numel()}")
        print(f"  Time: {elapsed*1000:.2f}ms")

        if result.float().norm() < 1e-6:
            print("  *** WARNING: output is all zeros! ***")
            continue

        print("  Computing fp32 reference (slow, subset of experts)...")
        active_experts = topk_ids.unique().cpu()
        print(f"  Active experts: {len(active_experts)}")

        ref_output = torch.zeros(M, K, dtype=torch.float32, device=device)
        gate_deq_cache = {}
        up_deq_cache = {}
        down_deq_cache = {}
        for eid_t in active_experts:
            eid = eid_t.item()
            gate_deq_cache[eid] = dequant_fp4(
                raw["gate_w"][eid].to(device), raw["gate_sf"][eid].to(device),
                raw["gate_gs"][eid].to(device))
            up_deq_cache[eid] = dequant_fp4(
                raw["up_w"][eid].to(device), raw["up_sf"][eid].to(device),
                raw["up_gs"][eid].to(device))
            down_deq_cache[eid] = dequant_fp4(
                raw["down_w"][eid].to(device), raw["down_sf"][eid].to(device),
                raw["down_gs"][eid].to(device))

        hidden_f32 = hidden.float()
        for m in range(M):
            for ki in range(TOP_K):
                eid = topk_ids[m, ki].item()
                w = topk_weights_val[m, ki].float()
                gate_out = hidden_f32[m] @ gate_deq_cache[eid].T
                up_out = hidden_f32[m] @ up_deq_cache[eid].T
                act = F.silu(gate_out) * up_out
                down_out = act @ down_deq_cache[eid].T
                ref_output[m] += w * down_out

        print(f"  Reference norm: {ref_output.norm():.4f}")

        cos = F.cosine_similarity(result.float().flatten(), ref_output.flatten(), dim=0)
        rel_err = (result.float() - ref_output).norm() / ref_output.norm()
        max_abs = (result.float() - ref_output).abs().max()
        print(f"  Cosine similarity: {cos.item():.6f}")
        print(f"  Relative error: {rel_err.item():.6f}")
        print(f"  Max abs error: {max_abs.item():.6f}")

        if cos.item() > 0.99:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL (cosine < 0.99)")

    print("\n\nTiming sweep (M=1,16 with warmup)...")
    for M in [1, 16]:
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
        router_logits = torch.randn(M, E, dtype=torch.float32, device=device)
        topk_weights_raw, topk_ids = torch.topk(router_logits, TOP_K, dim=-1)
        topk_weights_val = F.softmax(topk_weights_raw, dim=-1).to(torch.bfloat16)

        workspace = allocate_tp_moe_workspace_pool()
        binding = build_tp_moe_fp4_binding(
            scratch=workspace, a=hidden, experts=experts,
            topk_weights=topk_weights_val,
            topk_ids=topk_ids.to(torch.int32),
            quant_mode="nvfp4",
        )

        for _ in range(5):
            sparkinfer_moe_fp4(binding=binding)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.time()
            sparkinfer_moe_fp4(binding=binding)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

        import statistics
        print(f"  M={M}: median={statistics.median(times):.3f}ms "
              f"mean={statistics.mean(times):.3f}ms "
              f"min={min(times):.3f}ms")
        print(f"    Projected 47 layers: {statistics.median(times)*47:.1f}ms")


if __name__ == "__main__":
    main()
