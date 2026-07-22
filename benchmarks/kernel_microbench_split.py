"""Production-matching decode-attention microbench (split=4096, max_num_splits=64).

Mirrors the EXACT production launch config (sm120_gqa.py: fixed_kv_split_size=
ceil(262144/64)=4096, fixed_max_num_splits=64) so per-call timings reflect real
serving, unlike flashinfer_decode_feasibility.py (single-split). Measures the
native-FP8 v2 decode kernel at qo_len in {1,4}, kv_len in {64K,128K}, c=4, and
checks correctness vs a dequantized bf16 SDPA reference.

Usage:
  /home/bot/.venvs/vllm/bin/python -m benchmarks.kernel_microbench_split \
      --kv-len 141312 --batch 4 --qo-len 4 --split 4096 --max-splits 64
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("USE_LIBUV", "0")
import torch
import torch.nn.functional as F

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 24, 4, 256, 16

def _quantize_fp8(x_bf16):
    amax = x_bf16.abs().amax().clamp(min=1e-8)
    scale = (amax / 448.0).float()
    return (x_bf16.float() / scale).to(torch.float8_e4m3fn), scale

def _build_paged_kv(num_pages_total, device, seed=0):
    torch.manual_seed(seed)
    shape = (num_pages_total, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
    v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
    k_fp8, k_scale = _quantize_fp8(k_bf16)
    v_fp8, v_scale = _quantize_fp8(v_bf16)
    return k_fp8.view(torch.uint8), v_fp8.view(torch.uint8), k_scale, v_scale, k_bf16, v_bf16

def _sdpa_ref(q, k_bf16_pages, v_bf16_pages, num_pages_per_req, batch, qo_len):
    # q: [batch, qo_len, QH, D]; pages: [num_pages_total, page, KVH, D]
    gqa = NUM_QO_HEADS // NUM_KV_HEADS
    outs = []
    for b in range(batch):
        p0 = b * num_pages_per_req
        k = k_bf16_pages[p0:p0+num_pages_per_req].reshape(-1, NUM_KV_HEADS, HEAD_DIM)  # [L, KVH, D]
        v = v_bf16_pages[p0:p0+num_pages_per_req].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        k = k.repeat_interleave(gqa, dim=1).transpose(0,1)  # [QH, L, D]
        v = v.repeat_interleave(gqa, dim=1).transpose(0,1)
        qq = q[b].transpose(0,1).float()  # [QH, qo, D]
        # causal: query position i attends to kv <= L-qo+i
        L = k.shape[1]
        scores = torch.matmul(qq, k.float().transpose(-1,-2)) / (HEAD_DIM**0.5)  # [QH, qo, L]
        qi = torch.arange(qo_len, device=q.device).view(-1,1)
        ki = torch.arange(L, device=q.device).view(1,-1)
        mask = ki > (L - qo_len + qi)
        scores = scores.masked_fill(mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, v.float())  # [QH, qo, D]
        outs.append(o.transpose(0,1))  # [qo, QH, D]
    return torch.stack(outs, dim=0).bfloat16()  # [batch, qo, QH, D]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len", type=int, default=141312)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--qo-len", type=int, default=4)
    ap.add_argument("--split", type=int, default=4096)
    ap.add_argument("--max-splits", type=int, default=64)
    ap.add_argument("--num-iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    import flash_attn_sm120 as kernel

    nppr = (args.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    npt = nppr * args.batch
    last = args.kv_len - (nppr - 1) * PAGE_SIZE
    print(f"kv_len={args.kv_len} batch={args.batch} qo={args.qo_len} split={args.split} max_splits={args.max_splits}")
    print(f"pages/req={nppr} total_pages={npt} last_page_len={last} num_splits_actual={-(-args.kv_len//args.split)}")

    k_u8, v_u8, k_scale, v_scale, k_bf16, v_bf16 = _build_paged_kv(npt, device)
    torch.manual_seed(42)
    q = torch.randn(args.batch, args.qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1
    indptr = torch.arange(0, (args.batch+1)*nppr, nppr, dtype=torch.int32, device=device)
    indices = torch.arange(npt, dtype=torch.int32, device=device)
    lastpl = torch.full((args.batch,), last, dtype=torch.int32, device=device)
    kv_split_t = torch.tensor([args.split], dtype=torch.int32, device=device)

    def run():
        return kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8_adaptive(
            q, k_u8, v_u8, indptr, indices, lastpl, PAGE_SIZE, k_scale, v_scale, kv_split_t, args.max_splits)

    for _ in range(args.warmup): run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.num_iters): out = run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.num_iters * 1000
    # bytes read: K+V over kv_len per req, KVH heads, D bytes (fp8=1B), batch
    kv_bytes = 2 * args.kv_len * NUM_KV_HEADS * HEAD_DIM * 1 * args.batch
    eff_bw = kv_bytes / (ms/1000) / 1e9
    print(f"\nTIME: {ms:.3f} ms/call   eff_KV_BW: {eff_bw:.0f} GB/s")

    try:
        fi_ms, fi_out = run_flashinfer(args, q, k_u8, v_u8, indptr, indices, lastpl, k_scale, v_scale)
        kv_bytes = 2 * args.kv_len * NUM_KV_HEADS * HEAD_DIM * 1 * args.batch
        print(f"FLASHINFER: {fi_ms:.3f} ms/call   eff_KV_BW: {kv_bytes/(fi_ms/1000)/1e9:.0f} GB/s   ratio(sm120/fi)={ms/fi_ms:.3f}x")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FLASHINFER FAILED: {e}")

    if not args.no_ref:
        ref = _sdpa_ref(q, k_bf16, v_bf16, nppr, args.batch, args.qo_len)
        a = out.reshape(args.batch, args.qo_len, NUM_QO_HEADS, HEAD_DIM).float()
        b = ref.float()
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = (a-b).abs().max().item() / (b.abs().max().item()+1e-8)
        print(f"CORRECTNESS: cos={cos:.8f} max_rel={rel:.6f}")


def run_flashinfer(args, q, k_u8, v_u8, indptr, indices, lastpl, k_scale, v_scale):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    nppr = (args.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    workspace = torch.empty(256*1024*1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    kv = torch.stack([k_u8.view(torch.float8_e4m3fn), v_u8.view(torch.float8_e4m3fn)], dim=1)
    q_flat = q.reshape(args.batch*args.qo_len, NUM_QO_HEADS, HEAD_DIM)
    def go():
        wrapper.plan(indptr, indices, lastpl, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
                     data_type=torch.bfloat16, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
        kw = dict(k_scale=k_scale.item(), v_scale=v_scale.item())
        if args.qo_len > 1: kw["q_len_per_req"] = args.qo_len
        return wrapper.run(q_flat, kv, **kw)
    for _ in range(args.warmup): go()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.num_iters): out = go()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/args.num_iters*1000, out


if __name__ == "__main__":
    main()
