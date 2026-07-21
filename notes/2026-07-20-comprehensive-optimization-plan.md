All analysis complete. Here is the comprehensive, evidence-backed optimization plan.

---

# Performance Gap Analysis & Optimization Plan — Qwen3.6-27B SM120 Runtime

## 0. Architecture facts (verified from source)

- **Model** (`unsloth/Qwen3.6-27B-NVFP4` config): hidden=5120, **64 layers = 16 full-attention + 48 GDN (linear_attention)**, 24 q-heads / **4 KV-heads (GQA 6:1)**, head_dim=256, intermediate=17408, vocab=**248320**, **MTP = 1 full-attention layer**.
- **Everything lives in one file**: `runtime/direct_model_runner.py` (5915 lines). The "SM120 kernel" is **not** in `kernels/` (that dir is an empty stub) — it's the custom vLLM fork backend `/home/bot/vllm/vllm/v1/attention/backends/sm120_gqa.py` (1060 lines) calling CUDA kernels from `/home/bot/project/sm120-flash-attention/`.
- **KV per token** = 2 (K+V) × 4 heads × 256 × 1B(fp8) = **2 KB/token/layer** → at 131K, one attention layer = 262 MB; **16 layers × batch-4 ≈ 8.4 GB read per verify step**.

**The single most important reframe** (changes how you read the whole table): the reported `tok/s` is `total_accepted / (TTFT + decode)`. At 128K/c=4 our TTFT is 25.7 s vs native's 4.4 s — that gap alone dominates the 0.718× ratio. Backing it out (concurrency-4 wall clock): our steady-state decode ≈ **262 agg tok/s ≈ 15 ms/step**, native ≈ **197 agg tok/s ≈ 20 ms/step**. **Our steady-state decode step is already competitive with / faster than native; the headline gap is TTFT (prefill scheduling) + acceptance length, not the decode kernel loop.** The "30 vs 26 steps/s" framing is an artifact of amortizing a 25.7 s TTFT over only 256 tokens.

---

## 1. Bottleneck ranking

### A. Reported warm tok/s (TTFT-dominated) — 128K/c=4
| Component | Share of the *gap to native* | Evidence |
|---|---|---|
| **TTFT / prefill scheduling** (no cross-step interleaved chunked prefill) | **~60-70%** | Our 25.7 s vs 4.4 s TTFT; `notes/2026-07-20-inv8` confirms Phase A chunking only gave −10.7% because it doesn't interleave with decode |
| **Acceptance length** (3.3 vs 4.85 tokens/step) | **~20-25%** | `PROGRESS.md:104`; kernel-numerics divergence (SM120 draft+target vs FlashInfer draft+target) |
| Steady-state decode step time | **~10-15%** | Backed-out 15 vs 20 ms/step (we're roughly at parity here) |

### B. Steady-state decode step (per `mtp_verify_and_commit_batch`, c=4, 128K)
| Component | Est. share | Notes |
|---|---|---|
| **GEMM** (target 64 layers + draft, NVFP4/FP8) | ~40-50% | `PROGRESS.md:3261` nsys: GEMM=76% of GPU time at batch=1; still dominant |
| **Attention (SM120 kernel)** | ~20-30% | 16 layers × 8.4 GB KV/step; **kernel is 5.8× slower than FlashInfer** at 131K — the biggest single optimizable item |
| **Draft model overhead** (3 extra forwards/step, each attn over 128K) | ~12-18% | `_mtp_sync_and_propose_batch` step0 + `_mtp_run_continuation_steps` ×2; draft uses the **slow qo_len=1 CUDA-core kernel** |
| **GDN** (48 layers) | ~8-10% | nsys 8% at batch=1; per-step = in_proj GEMM + conv1d_update + delta-rule + sigmoid-gating + out_proj × 48 |
| **Logits/sampling** (vocab 248320) | ~6-9% | nsys 3.7%; `compute_logits` over 16 rows × 248K |
| **Python/eager overhead** | **0% (cudagraph) → 15-25% (eager)** | Warm bench is `enable_cudagraph=False` (`prefix_cache_warm_throughput_check.py:552`) |

---

## 2. Decode-path analysis (focus #1)

**Per step** (`mtp_verify_and_commit_batch`, `direct_model_runner.py:4909`):
1. **1× target verify forward**, qo_len=4 — `graph.replay()` (cudagraph) or `verify_batch_spec` (eager) `:4998-5014`
2. `determine_accept_reject_batch` `:1220` — already well-vectorized (1 GPU argmax + 1 `.tolist()`); **not a bottleneck**
3. Python bookkeeping + `publish_committed_decode_blocks` (block hashing, prefix cache) `:5028`
4. `torch.cat` of ragged hidden slices `:5043`
5. **Draft propose** `_mtp_sync_and_propose_batch` `:3381`: **step0 resync** (qo_len=committed_len) + **K−1=2 continuation steps** (qo_len=1)

**= 4 model forwards per step** (1 target + 3 draft). This is structurally more than native, which proposes K drafts with fewer tree-attention forwards.

**Two concrete per-step leaks found:**
- **Step0 resync is almost always EAGER, even with cudagraph ON.** Gate at `:3470-3475` requires `len(set(num_new_tokens_list))==1` (uniform committed length across slots) AND `≤ _MAX_DECODE_QO_LEN(16)`. At 70% acceptance, committed lengths (1–4) differ across the 4 slots → **ragged → eager fallback** → full metadata rebuild (`build_attention_metadata_batch` + `build_gdn_metadata_spec_batch` + `torch.tensor(...)` H2D copies, `_forward_batch:2208`) every step. Native pads ragged verify batches to keep them graph-captured.
- **Eager metadata churn** (`_forward_batch:2390-2440`): every eager forward builds ~10 fresh GPU tensors (`qo_indptr`, `kv_page_indices`, `slot_mapping`, `input_ids`, `positions`, …) via `torch.tensor(..., device=...)`. In the warm bench (cudagraph OFF) this runs on **every forward of every step** — 4× per step × ~10 tensors of Python→CUDA dispatch.

---

## 3. Attention kernel analysis (focus #2)

Dispatch (`sm120_gqa.py:822+`, fp8 KV, GQA 6:1, head_dim 256):
- **Verify qo_len=4** → `flash_attn_sm120_fwd_v2_decode_fp8kv_paged` (tensor-core, fast) — **only because `server/engine.py:65` and the benchmarks set `SM120_GQA_USE_V2_DECODE_KERNEL=1`** (default is OFF, `sm120_gqa.py:123`). ✓ Already enabled.
- **Draft continuation qo_len=1** → v2 kernel **excludes qo_len==1** (`:849`) and MMA excludes fp8 → falls to `flash_attn_sm120_fp8_kv_decode_paged` (**CUDA-core, the slow one**). The draft does 2–3 of these per step over the full 128K KV.
- **Prefill/suffix** → `flash_attn_sm120_fp8_kv_paged` general kernel.
- Split-KV is correctly tuned: `_DECODE_TARGET_SPLITS_PER_REQ=64`, `decode_fixed_kv_split_size` derived from per-slot capacity (`direct_model_runner.py:1027`).

**Theory**: 8.4 GB KV/step ÷ ~1 TB/s effective ≈ **8–9 ms floor** for target attention alone; at 5.8× FlashInfer slack, real cost is far above the bandwidth floor → the kernel is compute/occupancy-inefficient, not bandwidth-saturated. **This is the highest-value kernel target.**

---

## 4. GDN overhead analysis (focus #3)

- 48 GDN layers; per decode token each runs: `in_proj` GEMM → `causal_conv1d_update` → `fused_sigmoid_gating_delta_rule_update` (spec) / `fused_recurrent_gated_delta_rule_packed_decode` (non-spec) → `out_proj` GEMM (`qwen_gdn_linear_attn.py:_forward_core:1260`).
- nsys: **8% of GPU time at batch=1** (`PROGRESS.md:3261`) — real but bounded; ~9 kernel launches/layer.
- **GDN state management is no longer a per-step cost**: Phase 2 (2026-07-18) removed snapshot/restore/recompute-forward (`mtp_verify_and_commit_batch:4909` docstring); the spec-decode mechanism (`_ssm_spec_row:83`, `build_gdn_metadata_spec_batch:1097`) writes K+1 dedicated rows and selects via `num_accepted_tokens` — **zero rollback**. This was previously 56% of round wall time and is already fixed.
- **Overlap potential**: GDN (48 layers) and attention (16 layers) are sequential within `model.forward`; they can't overlap without restructuring the layer loop. Not a near-term win.

---

## 5. CUDA Graph analysis (focus #4)

- **Captured**: target verify (`CapturedBatchDecodeGraph:5061`, qo_len=4) + draft step0/continuation (`CapturedMTPDraftStepGraph:5634`). `compute_logits` is inside the graph; argmax/`.tolist()` is outside (small).
- **`_fill_buffers` cost** (`:5292`): builds ~8 Python lists + ~8 `torch.tensor(...).copy_()` H2D per replay — small but real; the only per-step CPU work in graph mode.
- **THE blocker**: graphs need `num_slots ≥ 2×batch` (`:5176`) — the extra `batch_size` slots are **permanent warmup reservations**, and `allocate_fixed_slot_kv_caches:525` allocates **full `blocks_per_slot` KV for every slot including warmup-only ones**. At 128K (`blocks_per_slot≈8200`), doubling slots OOMs → **warm bench forces `enable_cudagraph=False`** (`prefix_cache_warm_throughput_check.py:552`: "cudagraph doubles num_slots => OOM at long ctx"). **This is why the long-context numbers are measured on the slow eager path.** Production server uses `blocks_per_slot=512` (8K ceiling, `server/engine.py:130`) so cudagraph fits — but it can't serve 128K.
- No graph-breaking inside the captured region; the breakage is the eager step0 fallback (§2).

---

## 6. Server scheduler analysis (focus #5)

- `_step` (`server/engine.py:634`): admit (blocking prefill) → **one** `mtp_verify_and_commit_batch` round → process decisions → `await asyncio.sleep(0)`. Single event loop, no background thread.
- **Idle path** sleeps `idle_sleep_s=0.005` (`:764`) only when no active slots — irrelevant during sustained decode.
- Per-step Python is modest: `decisions` loop, EOS/max_tokens check, `committed_tokens.extend`. The `await asyncio.sleep(0)` is a cheap yield. **Not a meaningful bottleneck** vs the GPU work. The real scheduler gap is **blocking admission/prefill** (no prefill↔decode interleaving) — that's the TTFT problem, not the decode loop.

---

## 7. MTP draft model analysis (focus #6)

- `Qwen3_5MultiTokenPredictor` (`qwen3_5_mtp.py:63`): embed + 2× RMSNorm + `fc` (2·hidden→hidden GEMM) + **1 full-attention layer (own KV)** + lm_head.
- **Not batched across K**: step0 + (K−1) **sequential autoregressive** forwards (`_mtp_run_continuation_steps:3312`), each attending over the full 128K KV with the **slow qo_len=1 kernel**. Native proposes K drafts in fewer forwards (tree attention).
- Draft KV is tiny (1 layer) but the **attention read over 128K is not** — 3 draft forwards × ~0.5 GB KV each ≈ 1.5 GB extra KV traffic/step on the slow kernel.

---

# Prioritized optimizations

## ⚡ Quick wins (< 1 hour each, measurable)

**Q1. Enable CUDA graphs at long context by shrinking warmup-slot KV** — *highest leverage*
- **What**: In `allocate_fixed_slot_kv_caches:525`, allocate a **small** `blocks_per_slot` (e.g. 16 blocks) for the warmup-reserved slots (the last `num_slots//2` logical slots), full size only for production slots. The addressing (`_initial_block_table:65`, `_physical_slot`) already supports per-slot block counts; `capture()` only writes a few warmup positions.
- **Why**: removes the "cudagraph doubles num_slots ⇒ OOM" constraint (`prefix_cache_warm_throughput_check.py:41-42,552`), letting the 128K bench run the **graph path** and delete the eager metadata-churn overhead (§2).
- **Expected**: +10–20% steady-state decode at long context (removes 4×/step eager `torch.tensor`/metadata rebuild). **Risk**: low (warmup slots are never replayed against; `:5540` guards them). **Complexity**: low.

**Q2. Graph-capture the ragged step0 resync via padding**
- **What**: relax the uniformity gate at `:3470-3475` — pad ragged `num_new_tokens` to a small bucket (1/2/3/4) and capture per-bucket graphs, masking padded rows in accept logic.
- **Why**: step0 currently falls back to eager every step at 70% acceptance (§2). **Expected**: +3–6% (one fewer eager forward/step). **Risk**: medium (masking correctness). **Complexity**: low-medium.

**Q3. Try `SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL=1`**
- **What**: set the env var (`sm120_gqa.py:139`) in the warm bench + server; it routes verify to the native-FP8 QK/PV-MMA kernel.
- **Why**: free A/B on the verify attention path. **Expected**: unknown, possibly +5–15% on attention-bound long context. **Risk**: low (correctness-gated; verify near-tie). **Complexity**: trivial.

**Q4. Benchmark with cudagraph ON + report steady-state decode separately from TTFT**
- **What**: add a steady-state-only tok/s metric (exclude TTFT) to `native_warm_compare.py`/`prefix_cache_warm_throughput_check.py`.
- **Why**: the current metric conflates a 25.7 s TTFT with decode, hiding that steady-state decode is already ~parity. **Expected**: no perf change, but **correctly targets** the remaining work. **Complexity**: trivial.

## 🔧 Medium-term (1–3 days)

**M1. Cross-step interleaved chunked prefill (Phase B / INV8)** — *the TTFT killer*
- **What**: implement `notes/2026-07-20-inv8-chunked-hit-prefill-plan.md` Phase B — allow "partially-prefilled" slots, advance one `effective_chunk` per `_step`, interleaving decode rounds between chunks (matches native `--enable-chunked-prefill`).
- **Why**: this is **~60-70% of the reported gap** (§1A). Phase A (intra-admission chunking) only gave −10.7% TTFT (`PROGRESS.md:49-79`) because it doesn't interleave. Native's 4.4 s vs our 25.7 s TTFT is exactly this.
- **Expected**: warm TTFT 25.7 s → ~5–7 s; reported tok/s 105 → ~140+ at 128K (approaching native 146.85). **Risk**: medium (scheduler state machine: free→prefilling→active). **Complexity**: medium.

**M2. Replace/augment SM120 decode attention with FlashInfer for long KV**
- **What**: route the qo_len∈[1,4] decode/verify path to FlashInfer's batch-decode (the native baseline) when KV is large, keeping SM120 for prefill; or port FlashInfer's split-KV decode strategy into the SM120 kernel.
- **Why**: 5.8× kernel gap at 131K is the biggest steady-state item (§3); native uses FlashInfer for both main+MTP. **Expected**: +15–30% steady-state decode at ≥64K. **Risk**: medium-high (two backends, KV-layout/scale conventions, #37554 fp8-scale caveat at `sm120_gqa.py:790`). **Complexity**: medium-high.

**M3. Batch the K draft proposals into one tree-attention forward**
- **What**: replace step0 + (K−1) sequential `_mtp_run_continuation_steps` with a single K-token tree/chain forward in the 1-layer draft model.
- **Why**: cuts 4 forwards/step → 2 (§7), removes 2× slow qo_len=1 attention over 128K. **Expected**: +5–10% steady-state. **Risk**: medium (draft KV write/rollback semantics, `_mtp_forward:2814` prior_kv_len invariant). **Complexity**: medium.

## 🏗️ Long-term (architectural)

**L1. 48-layer GDN fusion** — fuse the ~9 kernels/layer (`in_proj`+conv1d+delta-rule+gating+`out_proj`); 8% ceiling (`PROGRESS.md:3341`) but real.
**L2. NVFP4 GEMM weight-layout / quant-dequant fusion** — 76% of GPU time (`PROGRESS.md:3261`); the largest absolute ceiling. Evaluate `项目实施规划.md` Phase 7 order (input-proj > MLP gate-up > MLP down > o_proj > MTP > lm_head).
**L3. Acceptance-length recovery** — the draft/target kernel-numerics divergence (3.3 vs 4.85) is structural; options: distill/retune the draft against the SM120 target, or use target-kernel-matched draft logits. High value (~20-25% of gap) but research-grade.

---

## Recommended sequence
1. **Q4 + Q1 + Q3** (today): fix the measurement, unlock cudagraph at long context, A/B the native-fp8 kernel.
2. **M1** (this week): the single biggest reported-tok/s lever (TTFT).
3. **M2** (next): the biggest steady-state-decode lever (attention kernel).
4. **Q2 / M3**, then **L1/L2** as the GEMM/GDN ceilings.

**Key caveat**: all % estimates are derived from the nsys breakdown (`PROGRESS.md:3245`, batch=1/short-ctx) + the warm-bench math + source structure; I could not run GPU benchmarks in this read-only pass. **Q4 (separate steady-state vs TTFT metric) should be done first** — it will confirm whether the real target is M1 (TTFT, my primary hypothesis) or M2 (decode kernel), and prevent spending effort on a decode loop that's already near parity.

Want me to start on any of these? Q1 (warmup-slot KV shrink to unlock long-context cudagraph) and Q4 (metric split) are both low-risk and I can implement + verify them directly.