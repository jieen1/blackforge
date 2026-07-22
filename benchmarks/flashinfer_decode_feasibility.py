"""Feasibility microbenchmark: FlashInfer vs SM120 GQA NATIVEFP8 decode kernel.

Compares decode attention kernel speed AND correctness at long context on our
EXACT KV layout (fp8_e4m3, NHD, GQA 24:4, head_dim=256, page_size=16).

Answers the critical question: is FlashInfer's decode kernel faster than our
NATIVEFP8 kernel at 128K, and does it produce identical output?

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.flashinfer_decode_feasibility \
        --kv-len 141312 --batch 4 --qo-len 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")

import torch

# Qwen3.6-27B full-attention layer config
NUM_QO_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE = 16
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 6


def _quantize_fp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 -> fp8_e4m3 with per-tensor scale (vLLM convention).
    stored = true/scale; read: true = stored.float()*scale."""
    amax = x_bf16.abs().amax().clamp(min=1e-8)
    scale = (amax / 448.0).float()
    x_fp8 = (x_bf16.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def _build_paged_kv(num_pages_total: int, device, seed=0):
    """Build paged KV cache [num_pages, 2, page_size, KVH, D] uint8 (fp8_e4m3).
    Returns (kv_cache_uint8, k_scale, v_scale, k_bf16_ref, v_bf16_ref)."""
    torch.manual_seed(seed)
    shape = (num_pages_total, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
    v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
    k_fp8, k_scale = _quantize_fp8(k_bf16)
    v_fp8, v_scale = _quantize_fp8(v_bf16)
    # Combined [num_pages, 2, page_size, KVH, D] uint8
    kv = torch.stack([k_fp8.view(torch.uint8), v_fp8.view(torch.uint8)], dim=1)
    return kv, k_scale, v_scale, k_fp8, v_fp8


def _run_sm120(q_decode, k_cache, v_cache, kv_page_indptr, kv_page_indices,
               kv_last_page_len, k_scale, v_scale, kv_split_size, num_warmup=5, num_iters=20):
    import flash_attn_sm120 as kernel
    # Warmup
    for _ in range(num_warmup):
        out = kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8(
            q_decode, k_cache, v_cache, kv_page_indptr, kv_page_indices,
            kv_last_page_len, PAGE_SIZE, k_scale, v_scale, kv_split_size,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8(
            q_decode, k_cache, v_cache, kv_page_indptr, kv_page_indices,
            kv_last_page_len, PAGE_SIZE, k_scale, v_scale, kv_split_size,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / num_iters * 1000
    return out, elapsed


def _run_flashinfer(q, kv_cache_uint8, paged_kv_indptr, paged_kv_indices,
                    paged_kv_last_page_len, k_scale, v_scale, batch, qo_len,
                    num_warmup=5, num_iters=20):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    kv_fp8 = kv_cache_uint8.view(torch.float8_e4m3fn)

    def _plan():
        wrapper.plan(
            paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
            NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
            data_type=torch.bfloat16, q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )

    run_kwargs = dict(k_scale=k_scale.item(), v_scale=v_scale.item())
    if qo_len > 1:
        run_kwargs["q_len_per_req"] = qo_len

    for _ in range(num_warmup):
        _plan()
        out = wrapper.run(q, kv_fp8, **run_kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _plan()
        out = wrapper.run(q, kv_fp8, **run_kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / num_iters * 1000
    return out, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=141312)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--qo-len", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    batch = args.batch
    qo_len = args.qo_len
    kv_len = args.kv_len
    num_pages_per_req = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages_total = num_pages_per_req * batch
    last_page_len = kv_len - (num_pages_per_req - 1) * PAGE_SIZE

    print("=== FlashInfer vs SM120 GQA NATIVEFP8 decode ===")
    print(f"kv_len={kv_len}, batch={batch}, qo_len={qo_len}, page_size={PAGE_SIZE}")
    print(f"num_pages_per_req={num_pages_per_req}, total_pages={num_pages_total}")
    print(f"GQA {NUM_QO_HEADS}:{NUM_KV_HEADS}, head_dim={HEAD_DIM}")

    # Build KV cache
    print("\nBuilding paged KV cache...")
    kv_cache, k_scale, v_scale, k_fp8, v_fp8 = _build_paged_kv(num_pages_total, device)
    print(f"  k_scale={k_scale.item():.6f}, v_scale={v_scale.item():.6f}")
    print(f"  KV cache: {kv_cache.shape} ({kv_cache.numel()/2**30:.2f} GiB)")

    # Query: [batch, qo_len, QH, D] bf16
    torch.manual_seed(42)
    q_decode = torch.randn(batch, qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1

    # Metadata for SM120 GQA
    kv_page_indptr = torch.arange(0, (batch + 1) * num_pages_per_req, num_pages_per_req,
                                   dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(num_pages_total, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((batch,), last_page_len, dtype=torch.int32, device=device)
    kv_split_size = kv_len

    # k_cache/v_cache for SM120: [num_pages, page_size, KVH, D]
    k_cache_sm120 = k_fp8.view(torch.uint8)
    v_cache_sm120 = v_fp8.view(torch.uint8)

    # --- SM120 GQA NATIVEFP8 ---
    print("\n--- SM120 GQA NATIVEFP8 ---")
    try:
        out_sm120, t_sm120 = _run_sm120(
            q_decode, k_cache_sm120, v_cache_sm120, kv_page_indptr, kv_page_indices,
            kv_last_page_len, k_scale, v_scale, kv_split_size, num_iters=args.num_iters,
        )
        out_sm120 = out_sm120.reshape(batch * qo_len, NUM_QO_HEADS, HEAD_DIM)
        print(f"  Time: {t_sm120:.3f} ms")
    except Exception as e:
        print(f"  FAILED: {e}")
        out_sm120, t_sm120 = None, float("inf")

    # --- FlashInfer ---
    print("\n--- FlashInfer BatchDecode ---")
    q_flat = q_decode.reshape(batch * qo_len, NUM_QO_HEADS, HEAD_DIM)
    try:
        out_fi, t_fi = _run_flashinfer(
            q_flat, kv_cache, kv_page_indptr, kv_page_indices,
            kv_last_page_len, k_scale, v_scale, batch, qo_len, num_iters=args.num_iters,
        )
        print(f"  Time: {t_fi:.3f} ms")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {e}")
        out_fi, t_fi = None, float("inf")

    # --- Correctness comparison ---
    if out_sm120 is not None and out_fi is not None:
        print("\n--- Correctness ---")
        a = out_sm120.float()
        b = out_fi.float()
        if a.shape != b.shape:
            print(f"  Shape mismatch: sm120={a.shape}, fi={b.shape}")
            b = b.reshape(a.shape)
        diff = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  max abs diff: {diff.max().item():.6f}")
        print(f"  mean abs diff: {diff.mean().item():.6f}")
        print(f"  cosine sim: {cos.item():.8f}")
        rel = diff.max().item() / (a.abs().max().item() + 1e-8)
        print(f"  max rel diff: {rel:.6f}")

    # --- Speedup ---
    if t_sm120 < float("inf") and t_fi < float("inf"):
        print("\n=== RESULT ===")
        print(f"SM120 GQA NATIVEFP8: {t_sm120:.3f} ms")
        print(f"FlashInfer:          {t_fi:.3f} ms")
        print(f"Speedup (SM120/FI):  {t_sm120/t_fi:.3f}x")
        if t_fi < t_sm120:
            print(f"FlashInfer is FASTER by {(t_sm120-t_fi):.3f} ms ({(1-t_fi/t_sm120)*100:.1f}%)")
        else:
            print(f"SM120 GQA is FASTER by {(t_fi-t_sm120):.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
