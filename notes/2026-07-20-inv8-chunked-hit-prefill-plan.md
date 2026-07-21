---

# INV8 Lift: Chunked Hit-Path Suffix Continue-Prefill — Implementation Plan

## 1. Current Architecture Summary

**How hit-path prefill works now:**

The unified production entrypoint `mtp_prefill_with_cache` (runtime/direct_model_runner.py:4586) handles all prefill admission. For cache-hit slots (L > 0):

1. **Restore**: `restore_cached_prefix(slot, prompt, L)` restores attention KV blocks [0, L) and the GDN checkpoint at L.
2. **Monolithic suffix prefill**: ONE `_forward_batch(hit_slots, suffix_per_slot, hit_L, qo_len=suffix_lens, commit=True, is_decode=False)` processes ALL hit slots' ragged suffixes in a single forward call.
3. **MTP draft sync**: ONE `_mtp_sync_and_propose_batch(...)` over the full suffix hidden states.
4. **Publish**: `_publish_committed_blocks` for future deeper hits.

At c=4 with 10K suffixes, this batches 4×10,240 = 40,960 tokens into one forward pass, causing 28.8s warm TTFT and 92.9 GiB peak memory.

**Where the "unchunked" constraint lives:**

- `mtp_prefill_with_cache` line ~4635: the **INV8 ragged-suffix-chunking guard** explicitly demotes hit slots with suffixes > `chunk_size` to the cold path rather than chunking them. The comment reads: *"Do NOT lift the ragged+chunk limit here."*
- The hit-path code issues ONE `_forward_batch` over ragged suffixes with no chunk loop.
- `mtp_prefill_batch`'s chunked loop (line ~3850) requires `is_uniform_len` — it advances all slots' chunk boundaries in lockstep from a single shared counter.

**Existing chunking mechanism (cold path):**

`mtp_prefill_batch` (line 3576) already has a fully working chunked prefill for **uniform-length** prompts:
- Processes `ceil(prompt_len / chunk_size)` sequential chunks.
- Each chunk is a genuine paged-KV continuation (`kv_lengths` grows, `commit=True`).
- GDN carries over via `has_initial_state` (False for chunk 0 of a fresh slot, True thereafter).
- Draft model step-0 is chunked in lockstep (each chunk's target hidden states fed to that chunk's draft forward).
- Chunk-boundary GDN checkpoints are materialized at block-aligned boundaries.
- `_mtp_run_continuation_steps` handles the K-1 autoregressive tail after the last chunk.

**Scheduler architecture:**

The server engine (`server/engine.py`) runs a single-threaded async event loop:
- `_step()`: (1) admit waiting requests → prefill, (2) run ONE decode/verify round for all active slots.
- Admission is **blocking**: all prefills complete before decode runs.
- No concept of "partially prefilled" — a slot transitions atomically from free → active (decode-ready) via `_activate_slot`.
- `EagerEngine` (runtime/engine.py) has states PREFILL → DECODE → COMPLETED with no intermediate state.

---

## 2. Proposed Changes

### Approach: Two-Phase Implementation

**Phase A (minimal, high-leverage): Intra-admission chunked suffix prefill**
Chunk the hit-path suffix within the same `_step` call, bounding per-forward token count to `chunk_size × num_hit_slots`. Does NOT interleave with decode, but bounds activation memory and enables the ragged+chunk combination.

**Phase B (full parity): Cross-step interleaved chunked prefill**
Allow slots to be "partially prefilled" — process one chunk per `_step`, interleaving decode rounds between chunks. Matches native vLLM's `--enable-chunked-prefill` behavior.

---

### Phase A: Intra-Admission Chunked Suffix Prefill

#### 2.1 New method: `_chunked_hit_continue_prefill`

Location: `runtime/direct_model_runner.py`, near `mtp_prefill_with_cache`.

```python
def _chunked_hit_continue_prefill(
    self,
    hit_slots: list[int],
    hit_prompts: list[list[int]],
    hit_L: list[int],
    chunk_size: int,
) -> dict[int, dict]:
    """Chunked ragged suffix continue-prefill for hit slots.
    
    Generalizes mtp_prefill_batch's uniform-length chunked loop to
    per-slot ragged suffixes starting at per-slot kv_lengths=[L_s].
    Each chunk processes min(chunk_size, remaining_s) tokens per slot;
    slots whose suffix is exhausted drop out of subsequent chunks.
    """
    k = self.num_speculative_tokens
    suffix_per_slot = [p[L:] for p, L in zip(hit_prompts, hit_L)]
    suffix_lens = [len(sfx) for sfx in suffix_per_slot]
    num_slots = len(hit_slots)
    
    # Per-slot chunk progress
    chunk_offset = [0] * num_slots  # tokens consumed so far per slot
    all_hidden_chunks: list[torch.Tensor] = []  # accumulate for draft
    
    while any(chunk_offset[i] < suffix_lens[i] for i in range(num_slots)):
        # Active slots for this chunk (suffix not yet exhausted)
        active_mask = [chunk_offset[i] < suffix_lens[i] for i in range(num_slots)]
        active_indices = [i for i, m in enumerate(active_mask) if m]
        
        # Per-slot chunk tokens and lengths
        chunk_tokens = []
        chunk_lens = []
        active_slots = []
        active_kv_lens = []
        for i in active_indices:
            start = chunk_offset[i]
            end = min(start + chunk_size, suffix_lens[i])
            chunk_tokens.append(suffix_per_slot[i][start:end])
            chunk_lens.append(end - start)
            active_slots.append(hit_slots[i])
            active_kv_lens.append(hit_L[i] + start)  # current kv position
        
        # ONE forward over this chunk's active slots
        logits_chunk, hidden_chunk = self._forward_batch(
            active_slots,
            chunk_tokens if max(chunk_lens) > 1 else [t[0] for t in chunk_tokens],
            active_kv_lens,
            qo_len=chunk_lens,  # ragged per-slot
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        
        # Draft model step-0 sync for this chunk (lockstep)
        # ... (mirrors mtp_prefill_batch's chunked draft logic)
        
        # Advance offsets
        for idx, i in enumerate(active_indices):
            chunk_offset[i] += chunk_lens[idx]
        
        # Chunk-boundary GDN checkpoints (block-aligned)
        # ... (mirrors mtp_prefill_batch's P3.2 logic)
    
    # Final anchor + K-1 continuation from last chunk's outputs
    # ... (mirrors mtp_prefill_batch's tail logic)
```

#### 2.2 Modify `mtp_prefill_with_cache` hit-path block

Replace the monolithic `_forward_batch` + `_mtp_sync_and_propose_batch` with:

```python
# In mtp_prefill_with_cache, HIT set block:
if hit_idx:
    # ... restore each slot (unchanged) ...
    
    if chunk_size is not None and max(suffix_lens) > chunk_size:
        # NEW: chunked ragged suffix continue-prefill
        result.update(self._chunked_hit_continue_prefill(
            hit_slots, hit_prompts, hit_L, chunk_size
        ))
    else:
        # EXISTING: monolithic ragged suffix (short suffixes, unchanged)
        suffix_logits, suffix_hidden = self._forward_batch(...)
        # ... existing code ...
```

#### 2.3 Remove the INV8 ragged-suffix-chunking guard

The guard at line ~4635 that demotes oversized hit slots to cold:

```python
# REMOVE this block:
if chunk_size is not None and len(hit_idx) >= 2:
    hit_suffix_lens = [...]
    if len(set(hit_suffix_lens)) > 1:
        for i in list(hit_idx):
            if ... > chunk_size:
                hit_idx.remove(i)
                cold_idx.append(i)
```

Replace with: the chunked path handles ragged suffixes natively.

#### 2.4 Wire `chunk_size` from the server engine

In `server/engine.py` line ~726:

```python
# Change:
prefill_result = self.runner.mtp_prefill_with_cache(new_slots, new_prompts)
# To:
prefill_result = self.runner.mtp_prefill_with_cache(
    new_slots, new_prompts, chunk_size=_DEFAULT_PREFILL_CHUNK_SIZE
)
```

#### 2.5 Handle MTP draft state across chunks

The draft model's step-0 sync must be chunked in lockstep (same as cold path):
- Each chunk's target hidden states feed that chunk's `_mtp_forward_batch` call.
- `slot_draft_sync_len` advances by `chunk_len` after each chunk.
- Only the LAST chunk's draft logits produce the anchor.
- K-1 continuation steps run once after the final chunk (via `_mtp_run_continuation_steps`).

Key difference from cold path: `start_pos_list` for the draft is `[L_s + offset]` (not `[offset]`), because the draft attends over the restored [0, L_s) blocks.

#### 2.6 Handle GDN state across chunks

**No new mechanism needed.** Proven by the cold chunked path:
- After `restore_cached_prefix`, `slot_gdn_initialized[slot] = True`.
- Each `_forward_batch` call with `commit=True` advances GDN state in-place.
- `build_gdn_metadata_batch` reads `slot_gdn_initialized` → `has_initial_state=True` for all chunks after restore.
- The chunked FLA kernel handles arbitrary `qo_len` per request (its designed use case).

---

### Phase B: Cross-Step Interleaved Chunked Prefill (Full Native Parity)

#### 2.7 New slot state: `PARTIAL_PREFILL`

In `server/engine.py`, add a `prefilling: dict[int, dict]` map alongside `active`:

```python
self.prefilling: dict[int, dict] = {}
# Per-slot chunk progress:
# {
#     "req": GenerationRequest,
#     "prompt_ids": list[int],
#     "L": int,              # hit boundary
#     "suffix_offset": int,  # tokens consumed so far
#     "suffix_len": int,     # total suffix length
# }
```

#### 2.8 Modify `_step()` to interleave

```python
async def _step(self) -> None:
    # 1. Advance partially-prefilled slots by ONE chunk
    if self.prefilling:
        self._advance_prefill_chunks()  # one chunk_size per slot per step
    
    # 2. Admit NEW requests (only if no prefilling slots, or if free slots available)
    #    New admissions start their first chunk here
    
    # 3. Decode round for all ACTIVE (fully prefilled) slots
    if self.active:
        # ... existing verify_and_commit_batch ...
```

#### 2.9 `_advance_prefill_chunks` method

```python
def _advance_prefill_chunks(self) -> None:
    """Process one chunk_size of suffix prefill for each partially-prefilled slot."""
    completed = []
    for slot, state in self.prefilling.items():
        offset = state["suffix_offset"]
        remaining = state["suffix_len"] - offset
        this_chunk = min(remaining, self.prefill_chunk_size)
        
        # Forward one chunk
        chunk_tokens = state["prompt_ids"][state["L"] + offset : state["L"] + offset + this_chunk]
        self.runner._forward_batch(
            [slot], [chunk_tokens], [state["L"] + offset],
            qo_len=this_chunk, commit=True, is_decode=False, ...
        )
        # Draft lockstep (if last chunk: produce anchor + K-1)
        
        state["suffix_offset"] += this_chunk
        if state["suffix_offset"] >= state["suffix_len"]:
            completed.append(slot)
    
    # Transition completed slots to active
    for slot in completed:
        state = self.prefilling.pop(slot)
        self._activate_slot(slot, state["req"], state["anchor"], state["drafts"])
```

---

## 3. Invariant Preservation Analysis

| Invariant | Status | How Preserved |
|---|---|---|
| **INV1** (cold-vs-hit exact diff) | ✅ Unchanged | Restore mechanism untouched; chunking only affects the suffix forward AFTER restore |
| **INV2** (block hash chain) | ✅ Unchanged | `_publish_committed_blocks` called at same boundaries; chunk-boundary checkpoints use same chained hash |
| **INV3** (content-hash agreement) | ✅ Unchanged | Hash computation is per-block, independent of how tokens were forwarded |
| **INV4** (GDN checkpoint fidelity) | ✅ Unchanged | GDN state at chunk boundaries is the live state (same as cold chunked path's P3.2 checkpoints) |
| **INV5** (eviction safety) | ✅ Unchanged | Reserve-before-forward order maintained; blocks allocated per-chunk via `_ensure_blocks` |
| **INV6** (slot freshness) | ✅ Unchanged | Fresh-slot check happens BEFORE restore (unchanged); chunking starts after restore |
| **INV7** (draft sync correctness) | ✅ Preserved | Draft step-0 chunked in lockstep; `slot_draft_sync_len` advances per-chunk; K-1 tail unchanged |
| **INV8** (ragged admission) | 🔄 **Lifted** | The guard is removed; ragged+chunk now handled by the new chunked loop |
| **INV9** (eviction never races) | ✅ Unchanged | Single-event-loop; no new concurrency introduced |

**Critical invariant for chunking correctness:**
- `_forward_batch` with `commit=True` advances `slot_kv_len` by `qo_len` per call. After N chunks, `slot_kv_len = L + sum(chunk_lens) = L + suffix_len = prompt_len`. Identical to the monolithic path's final state.
- GDN `has_initial_state` is True for all chunks after restore (set True by restore, stays True). The FLA kernel reads/writes the same physical GDN state row, advancing it chunk by chunk. Numerically identical to one large forward (proven by `mtp_chunked_prefill_check`).

---

## 4. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **Ragged chunk loop correctness** — slots dropping out mid-batch could corrupt metadata | High | Active-slot mask per chunk; `_forward_batch` already handles arbitrary slot subsets (proven by verify's per-group batching) |
| **Draft hidden-state accumulation** — draft step-0 needs the FULL suffix's hidden states for `_mtp_sync_and_propose_batch` | Medium | Phase A: accumulate hidden chunks and concatenate for the final draft call (same as cold path). Phase B: only the last chunk's hidden states are needed for anchor; intermediate chunks' draft forwards are for KV-cache population only |
| **GDN numerical drift across many chunks** — bf16 accumulation over 5+ chunks | Low | Already proven benign by cold chunked path (`mtp_chunked_prefill_check`: ssm rel-diff 0.095% at 128K). Same kernel, same accumulation pattern |
| **Memory: hidden-state accumulation** — storing all chunks' hidden states for the final draft call | Medium | Phase A: hidden states are `[chunk_len × num_slots, hidden_dim]` per chunk; at 8192×4×3584×2B = 224 MiB per chunk, 5 chunks = 1.1 GiB. Acceptable. Alternative: stream draft step-0 per-chunk (cold path already does this) |
| **Phase B scheduler complexity** — partial-prefill slots interacting with admission/eviction | High | Phase B is a separate, gated change. Phase A delivers the memory/TTFT bound without scheduler changes |
| **Regression on short-suffix hits** — the common case (suffix < chunk_size) must be byte-for-byte unchanged | Medium | Gate: `if max(suffix_lens) > chunk_size` → chunked path; else → existing monolithic path (untouched) |
| **`_mtp_sync_and_propose_batch` with ragged `num_new_tokens`** — already supported (2026-07-17 recompute round) | Low | Proven by `mtp_ragged_recompute_verify_check`; no new mechanism needed |

---

## 5. Expected Performance Improvement

### Phase A (intra-admission chunking)

**Warm TTFT at 128K/c=4 (4×10K suffix):**
- Current: 28,769 ms (one 41K-token forward)
- Chunked (8192 tokens/chunk): 5 sequential chunks × ~5.5s each ≈ **~27s** (marginal TTFT improvement alone — total compute is the same, just split)
- **Memory benefit**: peak activation drops from 41K×hidden to 8192×4×hidden per forward (~5× reduction in transient working set). Stays well under 95 GiB.

**Key insight**: Phase A alone does NOT significantly improve TTFT because the total compute is unchanged and there's no decode interleaving. Its value is:
1. Bounding activation memory (enables 200K/c≥2 under 95G ceiling).
2. Enabling Phase B's interleaving.
3. Eliminating the OOM/throttle risk at high concurrency.

### Phase B (cross-step interleaved chunking — full native parity)

**Warm TTFT at 128K/c=4:**
- Native vLLM: 4,417 ms (chunked prefill interleaved with decode)
- Expected ours: **~4,000–6,000 ms** (same chunking granularity, same kernel)
- Improvement: **~5–7× TTFT reduction** (28.8s → ~4-6s)

**Aggregate warm throughput:**
- Current: 83.24 tok/s
- Native: 146.85 tok/s
- Expected ours (Phase B): **~120–140 tok/s** (closing 80–95% of the gap)
- Remaining gap: native's slightly higher acceptance length (4.85 vs ~4.0) contributes ~15–20%

**Why the improvement:**
- Decode rounds run BETWEEN prefill chunks → decode latency is hidden behind prefill compute.
- Per-step token count bounded at 8192 → no single step blocks for 28s.
- TTFT for each request is spread across ~5 steps, but other requests' decode progresses concurrently.

---

## 6. Implementation Order

### Step 1: Phase A — Chunked ragged suffix prefill (1–2 days)

1. Implement `_chunked_hit_continue_prefill` in `runtime/direct_model_runner.py`:
   - Per-slot chunk offset tracking.
   - Active-slot mask per chunk (slots drop out when suffix exhausted).
   - Lockstep draft step-0 per chunk (stream hidden states, don't accumulate).
   - Chunk-boundary GDN checkpoints (reuse P3.2 logic).
   - Final anchor + `_mtp_run_continuation_steps` tail.

2. Wire into `mtp_prefill_with_cache`:
   - Gate: `chunk_size is not None and max(suffix_lens) > chunk_size`.
   - Remove INV8 ragged-suffix-chunking guard.

3. Wire `chunk_size=_DEFAULT_PREFILL_CHUNK_SIZE` from `server/engine.py`.

4. **Validation**: 
   - `mtp_chunked_prefill_check` (existing, cold path — must stay green).
   - New: `prefix_cache_chunked_hit_check` — hit-path chunked vs monolithic token-identical comparison.
   - `prefix_cache_warm_throughput_check` at 128K/c=4 (memory must stay < 95G).
   - `mtp_w1s_our_runtime_perf --batched --cudagraph` (zero-regression gate).

### Step 2: Phase B — Cross-step interleaved prefill (2–3 days)

5. Add `self.prefilling` state to `ServerEngine`.
6. Implement `_advance_prefill_chunks` (one chunk per step per slot).
7. Modify `_step()` ordering: advance prefilling → admit new → decode active.
8. Handle admission during active prefill (new requests wait or join the prefill batch).
9. **Validation**:
   - E2E warm throughput at 128K/c=4 (target: TTFT < 6s, agg tok/s > 120).
   - Correctness: token-identical output vs Phase A (chunking granularity must not affect greedy output).
   - `mtp_async_arrival_check` (mixed prefill/decode under async load).

### Step 3: Hardening (1 day)

10. `mtp_prefill_warm_continue` (P4b session-affinity): apply same chunking for long suffixes.
11. `_prefill_cold_with_populate`: apply chunking for the cold phase-1 (enables 200K/c≥2 under 95G).
12. Update `notes/prefix-cache-design.md` INV8 to document the lifted constraint.
13. Benchmark comparison vs native FlashInfer at 128K/c=4 (target: ≥0.85× native throughput).

---

## Summary

The minimal high-leverage change is **Phase A**: a chunked ragged suffix loop in `mtp_prefill_with_cache`'s hit block, reusing the proven cold-path chunking primitives (`_forward_batch` with growing `kv_lengths`, lockstep draft step-0, `_mtp_run_continuation_steps` tail). This bounds memory and removes the INV8 guard. **Phase B** (cross-step interleaving) delivers the full TTFT/throughput parity with native vLLM by allowing decode rounds between prefill chunks — this requires a new `prefilling` slot state in the server engine but no kernel or model changes.