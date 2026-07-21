"""NVFP4-KV vs FP8-KV decode attention microbench (128K/c=4, production split).

Decisive feasibility test for the FP4-KV-cache lever: NVFP4 KV stores ~0.56
byte/elem (0.5 data + 1/16 scale) vs FP8's 1 byte/elem, so a memory-bound
decode kernel should read ~1.8x less. Measures the NVFP4-KV decode kernel's
speed AND accuracy (cos vs dequantized-bf16 SDPA) at the production shape
(128K, c=4, qo in {1,4}, split=4096), against the FP8 nativefp8 baseline.

Usage:
  /home/bot/.venvs/vllm/bin/python -m benchmarks.kernel_microbench_nvfp4kv --kv-len 141312 --qo-len 4
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/bot/project/sm120-flash-attention/kernel")
sys.path.insert(0, "/home/bot/project/sm120-flash-attention/kernel/tests")

QH, KVH, D, PAGE = 24, 4, 256, 16

def _sdpa_ref(q, k_list, v_list, qo_len):
    gqa = QH // KVH
    outs = []
    for b in range(q.shape[0]):
        k = k_list[b].repeat_interleave(gqa, dim=1).transpose(0,1).float()
        v = v_list[b].repeat_interleave(gqa, dim=1).transpose(0,1).float()
        qq = q[b].transpose(0,1).float()
        L = k.shape[1]
        s = torch.matmul(qq, k.transpose(-1,-2)) / (D**0.5)
        qi = torch.arange(qo_len, device=q.device).view(-1,1); ki = torch.arange(L, device=q.device).view(1,-1)
        s = s.masked_fill(ki > (L - qo_len + qi), float("-inf"))
        outs.append(torch.matmul(torch.softmax(s,-1), v).transpose(0,1))
    return torch.stack(outs,0).bfloat16()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len", type=int, default=141312)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--qo-len", type=int, default=4)
    ap.add_argument("--split", type=int, default=4096)
    ap.add_argument("--num-iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    device = "cuda"
    from flash_attn_sm120 import flash_attn_sm120_nvfp4_kv_decode_paged
    from nvfp4_paged_test_utils import build_nvfp4_paged_cache, simulated_quantized_kv

    torch.manual_seed(2025*2026+27); torch.cuda.manual_seed(2025*2026+27)
    k_list = [torch.randn(args.kv_len, KVH, D, device=device).add(0.5).bfloat16() for _ in range(args.batch)]
    v_list = [torch.randn(args.kv_len, KVH, D, device=device).add(0.5).bfloat16() for _ in range(args.batch)]
    q = torch.stack([torch.randn(args.qo_len, QH, D, device=device).add(0.5).bfloat16() for _ in range(args.batch)],0)

    print(f"Building NVFP4 paged cache: kv_len={args.kv_len} batch={args.batch} qo={args.qo_len} ...")
    cache = build_nvfp4_paged_cache(k_list, v_list, PAGE, device, shuffle=True, seed=7)
    print(f"  k_nib {tuple(cache['k_nib'].shape)} k_scale {tuple(cache['k_scale'].shape)}")

    def run():
        return flash_attn_sm120_nvfp4_kv_decode_paged(
            q, cache["k_nib"], cache["k_scale"], cache["v_nib"], cache["v_scale"],
            cache["kv_page_indptr"], cache["kv_page_indices"], cache["kv_last_page_len"],
            PAGE, kv_split_size=args.split, max_num_splits_override=64)

    for _ in range(args.warmup): run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.num_iters): out = run()
    torch.cuda.synchronize()
    ms = (time.perf_counter()-t0)/args.num_iters*1000
    # NVFP4 bytes: data 0.5B + scale 1/16 B per elem, K+V, kv_len*KVH*D, batch
    kv_bytes = 2 * args.kv_len * KVH * D * (0.5 + 1/16) * args.batch
    print(f"\nNVFP4-KV TIME: {ms:.3f} ms/call   eff_KV_BW: {kv_bytes/(ms/1000)/1e9:.0f} GB/s")
    print(f"  (FP8 nativefp8 baseline @ same shape ~1.69 ms; FP8 reads {1/(0.5+1/16):.2f}x more bytes)")

    ref = _sdpa_ref(q, *simulated_quantized_kv(k_list, v_list), args.qo_len)
    a = out.reshape(args.batch, args.qo_len, QH, D).float(); b = ref.float()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = (a-b).abs().max().item()/(b.abs().max().item()+1e-8)
    print(f"ACCURACY (vs dequant-bf16 SDPA): cos={cos:.6f} max_rel={rel:.4f}  [project NVFP4 bar: cos>0.99]")

if __name__ == "__main__":
    main()
