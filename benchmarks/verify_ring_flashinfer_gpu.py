"""Ring buffer verification with block-aligned windows.
Eliminates starting-offset mismatch between ring and FlashInfer.
"""
import math, torch, flashinfer

BS = 16; WINDOW = 512
RING_BLOCKS = math.ceil((WINDOW - 1) / BS) + 1  # 33
RING_SLOTS = RING_BLOCKS * BS  # 528
NQO, NKV, HD = 64, 8, 128
DT = torch.bfloat16
dev = torch.device("cuda:0")
ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)

def make_kv(n):
    return torch.zeros(n, 2, BS, NKV, HD, dtype=DT, device=dev)

def run_decode(kv, bt, seq_len, q):
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(
        indptr=torch.tensor([0, len(bt)], dtype=torch.int32, device=dev),
        indices=torch.tensor(bt, dtype=torch.int32, device=dev),
        last_page_len=torch.tensor([(seq_len-1)%BS+1], dtype=torch.int32, device=dev),
        num_qo_heads=NQO, num_kv_heads=NKV, head_dim=HD, page_size=BS,
        data_type=DT, q_data_type=DT,
    )
    return w.run(q, kv)

torch.manual_seed(42)
ok_all = True
print("=== Block-aligned ring KV verification ===")

# Use block-aligned positions to eliminate offset issues
for prompt_len in [512, 528, 1024, 4096, 65536]:
    # Align prompt_len to block boundary
    pos = prompt_len
    aligned_start = ((pos - WINDOW + 1) // BS) * BS
    if aligned_start < 0: aligned_start = 0
    aligned_len = pos - aligned_start + 1
    n_blocks = math.ceil(aligned_len / BS)
    
    all_k = torch.randn(pos+1, NKV, HD, dtype=DT, device=dev) * 0.01
    all_v = torch.randn(pos+1, NKV, HD, dtype=DT, device=dev) * 0.01
    q = torch.randn(1, NQO, HD, dtype=DT, device=dev) * 0.01
    
    # Reference: contiguous KV for [aligned_start, pos]
    ref_kv = make_kv(n_blocks + 1)
    for i, p in enumerate(range(aligned_start, pos+1)):
        ref_kv[i//BS, 0, i%BS] = all_k[p]
        ref_kv[i//BS, 1, i%BS] = all_v[p]
    out_ref = run_decode(ref_kv, list(range(n_blocks)), aligned_len, q)
    
    # Ring KV
    ring_kv = make_kv(RING_BLOCKS + 1)
    for p in range(aligned_start, pos+1):
        rb = (p % RING_SLOTS) // BS; ro = p % BS
        ring_kv[rb, 0, ro] = all_k[p]
        ring_kv[rb, 1, ro] = all_v[p]
    
    # Block table: map logical block j to ring block
    rbt = []
    for j in range(n_blocks):
        actual = aligned_start + j * BS
        rbt.append((actual % RING_SLOTS) // BS)
    out_ring = run_decode(ring_kv, rbt, aligned_len, q)
    
    diff = (out_ref - out_ring).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten().float(), out_ring.flatten().float(), dim=0).item()
    ok = diff < 1e-5
    if not ok: ok_all = False
    print(f"  {'✓' if ok else '✗'} pos={pos:>6d} aligned_start={aligned_start:>6d} "
          f"len={aligned_len:>4d} n_blk={n_blocks:>2d}: diff={diff:.2e} cos={cos:.8f}")

# Multi-step with block-aligned windows
print("\n=== Multi-step decode (block-aligned) ===")
prompt_len = 1024
all_k = torch.randn(prompt_len+20, NKV, HD, dtype=DT, device=dev)*0.01
all_v = torch.randn(prompt_len+20, NKV, HD, dtype=DT, device=dev)*0.01
ring_kv = make_kv(RING_BLOCKS + 1)
for p in range(prompt_len):
    rb=(p%RING_SLOTS)//BS; ro=p%BS
    ring_kv[rb,0,ro]=all_k[p]; ring_kv[rb,1,ro]=all_v[p]

for step in range(10):
    pos = prompt_len + step
    rb=(pos%RING_SLOTS)//BS; ro=pos%BS
    ring_kv[rb,0,ro]=all_k[pos]; ring_kv[rb,1,ro]=all_v[pos]
    
    aligned_start = ((pos - WINDOW + 1) // BS) * BS
    if aligned_start < 0: aligned_start = 0
    aligned_len = pos - aligned_start + 1
    n_blocks = math.ceil(aligned_len / BS)
    
    ref_kv = make_kv(n_blocks+1)
    for i, p in enumerate(range(aligned_start, pos+1)):
        ref_kv[i//BS,0,i%BS]=all_k[p]; ref_kv[i//BS,1,i%BS]=all_v[p]
    
    q = torch.randn(1,NQO,HD,dtype=DT,device=dev)*0.01
    out_ref = run_decode(ref_kv, list(range(n_blocks)), aligned_len, q)
    
    rbt = []
    for j in range(n_blocks):
        actual = aligned_start + j * BS
        rbt.append((actual % RING_SLOTS) // BS)
    out_ring = run_decode(ring_kv, rbt, aligned_len, q)
    
    diff = (out_ref-out_ring).abs().max().item()
    ok = diff < 1e-5
    if not ok: ok_all = False
    print(f"  {'✓' if ok else '✗'} step={step} pos={pos}: diff={diff:.2e}")

print(f"\n{'ALL PASSED ✓' if ok_all else 'SOME FAILED ✗'}")
print("\nNote: block-aligned windows add up to BS-1 extra positions.")
print("In production, FlashInfer window_left=511 (via TRTLLM decode) skips them.")
