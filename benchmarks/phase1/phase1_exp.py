"""Phase 1: bf16 ground truth + sparkinfer comparison + zero-ification check."""
import os, sys, json, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)

HIDDEN = 3072; INTER = 1024; NUM_EXP = 256; TOP_K = 10; BLOCK = 16
FP4_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)

def unpack_fp4(packed):
    p = packed.cpu()
    low = (p & 0x0F).long()
    high = ((p >> 4) & 0x0F).long()
    return torch.stack([FP4_LUT[low], FP4_LUT[high]], dim=-1).reshape(p.shape[0], -1)

def dequant_weight(w_packed, w_sf, w_gs):
    fp4 = unpack_fp4(w_packed).to(w_packed.device)
    sf = w_sf.float()
    sf_exp = sf.repeat_interleave(BLOCK, dim=1)
    return fp4 * sf_exp / w_gs

from runtime.backends.laguna_sparkinfer_moe import _find_checkpoint, load_moe_layer_weights
ckpt = _find_checkpoint()

# First: scan ALL layers for global scale range
print("Scanning global scales across all MoE layers...")
from safetensors import safe_open
with open(ckpt / "model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]

max_gs_layer = -1; max_gs_val = 0
for layer_idx in range(1, 48):
    key = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight_global_scale"
    shard = weight_map.get(key)
    if shard is None:
        continue
    with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
        gs0 = f.get_tensor(key).item()
    # Also check a few more experts
    for eid in [128, 255]:
        key2 = f"model.layers.{layer_idx}.mlp.experts.{eid}.gate_proj.weight_global_scale"
        shard2 = weight_map.get(key2)
        if shard2:
            with safe_open(str(ckpt / shard2), framework="pt", device="cpu") as f2:
                gs_e = f2.get_tensor(key2).item()
                gs0 = max(gs0, gs_e)
    if gs0 > max_gs_val:
        max_gs_val = gs0
        max_gs_layer = layer_idx
    if layer_idx % 10 == 0:
        print(f"  layer {layer_idx}: max_gs_so_far={gs0:.0f}")

print(f"  Max global scale: layer {max_gs_layer} = {max_gs_val:.0f}")
print(f"  1/{max_gs_val:.0f} = {1/max_gs_val:.2e} (fp16 min normal = 6.10e-5)")

# Use layer 1 for main comparison (fast), note max-gs layer for zero-ification
LAYER = 1
print(f"\nLoading layer {LAYER} weights...")
t0 = time.time()
raw = load_moe_layer_weights(ckpt, LAYER, "cuda")
print(f"  loaded in {time.time()-t0:.1f}s")

gate_gs = raw["gate_gs"]
print(f"  gate_gs: min={gate_gs.min().item():.0f}, max={gate_gs.max().item():.0f}")

# Select 10 experts spanning the global scale range
sorted_idx = gate_gs.argsort()
step = NUM_EXP // TOP_K
selected = sorted_idx[::step][:TOP_K]
print(f"  Selected experts: {selected.tolist()}")
print(f"  Their gate_gs: {[f'{gate_gs[e].item():.0f}' for e in selected]}")

M = 4
torch.manual_seed(42)
hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
topk_ids = selected.unsqueeze(0).expand(M, -1).contiguous()
topk_weights = torch.ones(M, TOP_K, device="cuda") / TOP_K

# ══ PART A: bf16 ground truth ══
print("\n" + "="*60)
print("PART A: bf16 ground truth (exact dequant + fp32 matmul)")
print("="*60)
truth_out = torch.zeros(M, HIDDEN, dtype=torch.float32, device="cuda")
t0 = time.time()
for t in range(M):
    x = hidden[t].float()
    for k in range(TOP_K):
        eid = topk_ids[t, k].item()
        w = topk_weights[t, k].item()
        gw = dequant_weight(raw["gate_w"][eid], raw["gate_sf"][eid], raw["gate_gs"][eid].item())
        uw = dequant_weight(raw["up_w"][eid], raw["up_sf"][eid], raw["up_gs"][eid].item())
        dw = dequant_weight(raw["down_w"][eid], raw["down_sf"][eid], raw["down_gs"][eid].item())
        g = gw @ x; u = uw @ x
        inter = F.silu(g) * u
        d = dw @ inter
        truth_out[t] += w * d
truth_bf16 = truth_out.to(torch.bfloat16)
print(f"  Computed in {time.time()-t0:.1f}s, norm={truth_bf16.float().norm().item():.6f}")

# ══ PART B: sparkinfer FOLDING path ══
print("\n" + "="*60)
print("PART B: sparkinfer FOLDING path (bs/gs → fp8)")
print("="*60)
from sparkinfer._lib.intrinsics import swizzle_block_scale
from sparkinfer.moe.fused_moe._impl import (
    allocate_tp_moe_workspace_pool,
    plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
)
from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer, prepare_sparkinfer_layer

# Diagnose folding damage
sample_eid = selected[-1].item()  # highest gs
gs_val = raw["gate_gs"][sample_eid].item()
bs_orig = raw["gate_sf"][sample_eid].float()
bs_folded = bs_orig / gs_val
bs_fp8 = bs_folded.to(torch.float8_e4m3fn).float()
n_zero = (bs_fp8 == 0).sum().item()
print(f"  Expert {sample_eid} (gs={gs_val:.0f}): folded range [{bs_folded.min():.2e}, {bs_folded.max():.2e}]")
print(f"    fp8 zeros: {n_zero}/{bs_fp8.numel()} ({100*n_zero/bs_fp8.numel():.1f}%)")

experts_fold = prepare_sparkinfer_layer(raw, "cuda")
workspace = allocate_tp_moe_workspace_pool()
si_fold = SparkinferMoELayer(experts_fold, workspace, "cuda")
fold_out = si_fold.forward(hidden, topk_ids.to(torch.int32), topk_weights)
print(f"  fold output norm = {fold_out.float().norm().item():.6f}")

# ══ PART C: sparkinfer RUNTIME-ALPHA path ══
print("\n" + "="*60)
print("PART C: sparkinfer RUNTIME-ALPHA path")
print("="*60)
gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())
w13_fp4 = torch.cat([raw["gate_w"], raw["up_w"]], dim=1).contiguous()
w13_sf = torch.cat([gate_sf_sw, up_sf_sw], dim=1).contiguous()

w1_gs_alpha = torch.ones(NUM_EXP, dtype=torch.float32, device="cuda")
w2_gs_alpha = torch.ones(NUM_EXP, dtype=torch.float32, device="cuda")
a1_gs = raw["gate_gs"].clone().float()
a2_gs = raw["down_gs"].clone().float()

wplan = plan_sparkinfer_fp4_moe_weights(
    quant_modes="nvfp4", source_format="modelopt_nvfp4",
    activation="silu", params_dtype=torch.bfloat16,
    num_experts=NUM_EXP, hidden_size=HIDDEN,
    intermediate_size=INTER, w13_layout="w13",
)
experts_alpha = prepare_sparkinfer_fp4_moe_weights(
    plan=wplan,
    w1_global_scale=w1_gs_alpha, w2_global_scale=w2_gs_alpha,
    w1_fp4=w13_fp4, w1_blockscale=w13_sf,
    w2_fp4=raw["down_w"].clone().contiguous(), w2_blockscale=down_sf_sw,
    a1_gscale=a1_gs, a2_gscale=a2_gs,
    params_dtype=torch.bfloat16,
)
print(f"  w1_alphas: [{experts_alpha.w1_alphas.min():.2e}, {experts_alpha.w1_alphas.max():.2e}], "
      f"dtype={experts_alpha.w1_alphas.dtype}")
print(f"  w1_alphas zeros: {(experts_alpha.w1_alphas == 0).sum().item()}/{NUM_EXP}")

si_alpha = SparkinferMoELayer(experts_alpha, workspace, "cuda")
alpha_out = si_alpha.forward(hidden, topk_ids.to(torch.int32), topk_weights)
print(f"  alpha output norm = {alpha_out.float().norm().item():.6f}")

# ══ PART D: Comparison ══
print("\n" + "="*60)
print("PART D: Comparison vs bf16 ground truth")
print("="*60)
def compare(label, out, ref):
    o = out.float().view(-1); r = ref.float().view(-1)
    cos = F.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    rel_norm = (o.norm() / r.norm()).item()
    rel_err = ((o - r).norm() / r.norm()).item()
    per_tok = F.cosine_similarity(out.float(), ref.float(), dim=1)
    n_zero = (out == 0).sum().item()
    n_nan = torch.isnan(out.float()).sum().item()
    print(f"  {label}: cos={cos:.6f}, rel_norm={rel_norm:.6f}, rel_err={rel_err:.6f}")
    print(f"    per-token cos: {[f'{v:.4f}' for v in per_tok.tolist()]}")
    if n_zero > 0 or n_nan > 0:
        print(f"    ⚠ zeros={n_zero}, nans={n_nan}")

compare("FOLDING vs truth", fold_out, truth_bf16)
compare("ALPHA vs truth", alpha_out, truth_bf16)
compare("ALPHA vs FOLDING", alpha_out, fold_out)

# ══ PART E: Per-expert zero-ification ══
print("\n" + "="*60)
print("PART E: Per-expert zero-ification check")
print("="*60)
print("  Alpha path per-expert:")
for eid in selected.tolist():
    gs = raw["gate_gs"][eid].item()
    ids1 = torch.tensor([[eid]], dtype=torch.int32, device="cuda")
    w1 = torch.ones(1, 1, device="cuda")
    out = si_alpha.forward(hidden[:1], ids1, w1)
    norm = out.float().norm().item()
    print(f"    E{eid:3d}: gs={gs:8.0f}, 1/gs={1/gs:.2e}, norm={norm:.6f} {'⚠ ZERO' if norm < 1e-10 else '✓'}")

print("\n  Folding path per-expert:")
for eid in selected.tolist():
    gs = raw["gate_gs"][eid].item()
    ids1 = torch.tensor([[eid]], dtype=torch.int32, device="cuda")
    w1 = torch.ones(1, 1, device="cuda")
    out = si_fold.forward(hidden[:1], ids1, w1)
    norm = out.float().norm().item()
    print(f"    E{eid:3d}: gs={gs:8.0f}, 1/gs={1/gs:.2e}, norm={norm:.6f} {'⚠ ZERO' if norm < 1e-10 else '✓'}")

print("\n  Ground truth per-expert:")
for eid in selected.tolist():
    gs = raw["gate_gs"][eid].item()
    x = hidden[0].float()
    gw = dequant_weight(raw["gate_w"][eid], raw["gate_sf"][eid], gs)
    uw = dequant_weight(raw["up_w"][eid], raw["up_sf"][eid], raw["up_gs"][eid].item())
    dw = dequant_weight(raw["down_w"][eid], raw["down_sf"][eid], raw["down_gs"][eid].item())
    d = dw @ (F.silu(gw @ x) * (uw @ x))
    print(f"    E{eid:3d}: gs={gs:8.0f}, truth_norm={d.norm().item():.6f}")

print("\n✓ Phase 1 complete")
