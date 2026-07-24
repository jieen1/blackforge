# Step Latency Analysis & Optimization Roadmap (2026-07-24)

## vLLM Referent (64K context, DFlash K=15, auto MoE backend)
- **376.9 tok/s**, ITL=2.65ms, step_time≈10.6ms
- Acceptance ≈20% (same as ours — ceiling confirmed for synthetic text)
- Configuration: FLASHINFER_TRTLLM MoE + FlashInfer attention + CUDA Graph

## Our Runtime (64K context, DFlash K=15, MARLIN MoE)
- **52.8 tok/s**, ITL=18.8ms, step_time≈75ms
- Acceptance ≈20% (at ceiling)
- Gap: **7× in step latency**, 0× in acceptance

## Per-Stage Breakdown (64K, 50 steps, all CG enabled)

| Stage | Mean (ms) | P50 (ms) | P95 (ms) | % Total |
|-------|-----------|----------|----------|---------|
| decode_cg (M=1, REDUNDANT) | 35.92 | 34.38 | 47.17 | 51.5% |
| verify_cg (M=16) | 25.94 | 25.43 | 29.34 | 37.2% |
| draft_cg (M=16, 6 layers) | 4.75 | 4.65 | 5.71 | 6.8% |
| combine+precompute | 2.28 | 2.19 | 3.29 | 3.3% |
| accept_reject | 0.77 | 0.73 | 1.19 | 1.1% |
| **TOTAL** | **69.81** | **68.63** | **80.87** | **100%** |

## Physical Budget (reviewer's analysis)

| Segment | Weight/KV Read | Physical Min | Current | Gap |
|---------|---------------|-------------|---------|-----|
| verify (48 layers, M=16) | MoE 2.7GB + attn KV 1.5GB | ~5-8ms | 25.9ms | 3-5× |
| draft (6 layers SWA, M=16) | ~1.3GB bf16 | ~1.5ms | 4.8ms | 3× |
| accept/bookkeeping | — | ~1ms | 0.8ms | ✓ |
| **Total (verify-only)** | | **~8-11ms** | **69.8ms** | **~7×** |

## Optimization Roadmap

### Phase 1: Eliminate Redundant Decode (69.8→~34ms, +105% tok/s)
The decode_cg (M=1 full model forward) exists only to produce:
1. bonus_token (argmax of decode logits)
2. aux_hidden_states (for draft context precompute)

Both can be obtained from the VERIFY forward:
- bonus_token = argmax of verify logits at position 0 (the bonus position)
- aux = verify's aux at accepted positions (aux hooks already on main model)

**Verify-only design** (like Qwen MTP):
1. Verify(M=16) → logits + aux → accept/reject → extract bonus + aux at accepted pos
2. Draft(M=16) → next 15 draft tokens using bonus + precomputed context from verify aux

Eliminates 35.9ms/step. Target: ~34ms → ~117 tok/s.

### Phase 2: MoE Kernel in Verify (34→~15ms)
Replace MARLIN with TRTLLM/nvjet MoE kernel in verify path.
verify_cg: 25.9ms → ~8ms (physical BW limit for M=16).
Target: ~15ms/step → ~265 tok/s.

### Phase 3: Single Sync Point (−3-5ms)
Merge per-stage cuda.synchronize() into one end-of-step sync.
Target: ~12ms/step → ~330 tok/s (approaching vLLM).

## Key Findings

1. **CG correctness confirmed**: FlashInfer CG produces cos>0.9999 vs eager at all page counts.
   Earlier "aux drift" was a diagnostic artifact (order-dependent KV cache interaction).
2. **Acceptance at ceiling**: vLLM 64K = 376.9 tok/s implies ~20% acceptance = same as ours.
3. **Draft position encoding**: Uses absolute positions (same as vLLM reference). Not a bug.
4. **Memory**: 66.96 GiB params + 8.4 GiB load residue (auditable) + 7 GiB KV/draft/CG = 82 GiB total.

## Data Files
- `benchmarks/fixtures/diag_step_latency.json` — per-stage timing
- `benchmarks/fixtures/diag_cg_aux_drift.json` — CG correctness evidence
- `benchmarks/fixtures/laguna_vllm_dflash_baseline.json` — vLLM referent
