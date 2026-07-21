"""NVFP4-KV v2 tensor-core decode kernel: correctness + speed (128K/c=4).
Compares v2 (mxf4nvf4 MMA) vs scalar NVFP4 decode kernel (proven reference) and
dequant-bf16 SDPA. Speed vs FP8 nativefp8 baseline (~1.69ms)."""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/bot/project/sm120-flash-attention/kernel")
sys.path.insert(0, "/home/bot/project/sm120-flash-attention/kernel/tests")
QH, KVH, D, PAGE = 24, 4, 256, 16

def _sdpa_ref(q, k_list, v_list, qo_len):
    gqa = QH // KVH; outs = []
    for b in range(q.shape[0]):
        k = k_list[b].repeat_interleave(gqa,1).transpose(0,1).float()
        v = v_list[b].repeat_interleave(gqa,1).transpose(0,1).float()
        qq = q[b].transpose(0,1).float(); L = k.shape[1]
        s = torch.matmul(qq, k.transpose(-1,-2))/(D**0.5)
        qi=torch.arange(qo_len,device=q.device).view(-1,1); ki=torch.arange(L,device=q.device).view(1,-1)
        s = s.masked_fill(ki>(L-qo_len+qi), float("-inf"))
        outs.append(torch.matmul(torch.softmax(s,-1), v).transpose(0,1))
    return torch.stack(outs,0).bfloat16()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--kv-len",type=int,default=141312)
    ap.add_argument("--batch",type=int,default=4); ap.add_argument("--qo-len",type=int,default=4)
    ap.add_argument("--split",type=int,default=4096); ap.add_argument("--num-iters",type=int,default=30)
    ap.add_argument("--warmup",type=int,default=10)
    args=ap.parse_args(); device="cuda"
    from flash_attn_sm120 import (flash_attn_sm120_nvfp4_kv_decode_paged,
                                   flash_attn_sm120_fwd_v2_decode_nvfp4kv_paged)
    from nvfp4_paged_test_utils import build_nvfp4_paged_cache, simulated_quantized_kv
    torch.manual_seed(2025*2026+27); torch.cuda.manual_seed(2025*2026+27)
    k_list=[torch.randn(args.kv_len,KVH,D,device=device).add(0.5).bfloat16() for _ in range(args.batch)]
    v_list=[torch.randn(args.kv_len,KVH,D,device=device).add(0.5).bfloat16() for _ in range(args.batch)]
    q=torch.stack([torch.randn(args.qo_len,QH,D,device=device).add(0.5).bfloat16() for _ in range(args.batch)],0)
    print(f"Building NVFP4 cache kv_len={args.kv_len} batch={args.batch} qo={args.qo_len}...")
    cache=build_nvfp4_paged_cache(k_list,v_list,PAGE,device,shuffle=True,seed=7)
    kv_split_t=torch.tensor([args.split],dtype=torch.int32,device=device)
    def run_v2():
        return flash_attn_sm120_fwd_v2_decode_nvfp4kv_paged(
            q, cache["k_nib"], cache["k_scale"], cache["v_nib"], cache["v_scale"],
            cache["kv_page_indptr"], cache["kv_page_indices"], cache["kv_last_page_len"],
            PAGE, kv_split_t, 64)
    def run_scalar():
        return flash_attn_sm120_nvfp4_kv_decode_paged(
            q, cache["k_nib"], cache["k_scale"], cache["v_nib"], cache["v_scale"],
            cache["kv_page_indptr"], cache["kv_page_indices"], cache["kv_last_page_len"],
            PAGE, kv_split_size=args.split, max_num_splits_override=64)
    out_v2=run_v2(); out_sc=run_scalar()
    ref=_sdpa_ref(q, *simulated_quantized_kv(k_list,v_list), args.qo_len)
    a=out_v2.reshape(args.batch,args.qo_len,QH,D).float()
    sc=out_sc.reshape(args.batch,args.qo_len,QH,D).float()
    b=ref.float()
    cos_sdpa=F.cosine_similarity(a.flatten(),b.flatten(),dim=0).item()
    cos_sc=F.cosine_similarity(a.flatten(),sc.flatten(),dim=0).item()
    rel=(a-b).abs().max().item()/(b.abs().max().item()+1e-8)
    print(f"\nV2 CORRECTNESS: cos(vs SDPA)={cos_sdpa:.6f}  cos(vs scalar NVFP4)={cos_sc:.6f}  max_rel={rel:.4f}")
    print(f"  scalar cos(vs SDPA)={F.cosine_similarity(sc.flatten(),b.flatten(),dim=0).item():.6f}  [bar cos>0.99]")
    for _ in range(args.warmup): run_v2()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(args.num_iters): run_v2()
    torch.cuda.synchronize(); v2_ms=(time.perf_counter()-t0)/args.num_iters*1000
    for _ in range(args.warmup): run_scalar()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(args.num_iters): run_scalar()
    torch.cuda.synchronize(); sc_ms=(time.perf_counter()-t0)/args.num_iters*1000
    kv_bytes=2*args.kv_len*KVH*D*(0.5+1/16)*args.batch
    print(f"\nSPEED: v2={v2_ms:.3f}ms ({kv_bytes/(v2_ms/1000)/1e9:.0f} GB/s)  scalar={sc_ms:.3f}ms  v2/scalar={v2_ms/sc_ms:.3f}x")
    print(f"  FP8 nativefp8 baseline ~1.69ms; v2/FP8={v2_ms/1.69:.3f}x")

if __name__=="__main__": main()
