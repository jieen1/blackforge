# Prefix-cache implementation log

Tracks P0-P4 of `notes/prefix-cache-design.md` as they land. One section
per phase, appended in order (not rewritten) so the history of what was
actually built/verified stays intact; detailed narrative belongs in git
history, this file stays a compact per-phase record (problem, what was
built, verification evidence, next-phase readiness) per this project's
established documentation convention.

---

## P0 — Block-table indirection substrate (2026-07-19)

**Status: DONE, verified, zero regression. Ready for P1.**

### What was built

Per `notes/prefix-cache-design.md` sec 5's "P0 — Block-table indirection
substrate": a behavior-identical refactor introducing
`DirectModelRunner.block_table: list[list[int]]`, one physical-block-id
list per logical slot, initialized by a new "thin allocator"
(`_initial_block_table(slot, blocks_per_slot)`, `runtime/
direct_model_runner.py`) to exactly the contiguous range the old
`arange`-based formula always computed. Nothing about runtime behavior
changes in this phase — the table just re-expresses today's addressing
through an indirection layer that P1 (dynamic free-list allocator) will
later populate differently.

Routed through the indirection, gated by a new `enable_block_table: bool
= False` constructor parameter on `DirectModelRunner` (default `False`,
following this project's established feature-flag convention, e.g.
`enable_cudagraph`):

1. **Read path** — `build_attention_metadata` (single-request) and
   `build_attention_metadata_batch` (batched) gained an optional
   `block_table`/`block_tables` parameter. `None` (default) preserves the
   exact prior `arange`-based construction byte-for-byte — required
   because `build_attention_metadata` is also called by
   `runtime/vllm_stage_c_baseline.py` (an unrelated, untouched diagnostic),
   which never passes this parameter.
2. **Write path** — `DirectModelRunner._slot_mapping`/`_slot_mapping_batch`
   now branch on `self.enable_block_table` to source each token's physical
   block id from `self.block_table[slot]` instead of the inline formula.
3. **CUDA-graph path** — `CapturedBatchDecodeGraph._fill_buffers` now
   branches on `runner.enable_block_table` for both `kv_page_indices` and
   `slot_mapping` construction.
4. **Thin allocator** — `_initial_block_table`, called once per slot at
   `DirectModelRunner.__init__` time (unconditionally — cheap, small
   Python lists — so the dedicated equivalence test can check it
   regardless of the flag; only *consulted* by 1-3 above when the flag is
   on).

**Beyond the design doc's literal four touch-points** (a deliberate,
low-risk scope addition, not scope creep into P1): `CapturedMTPDraftStepGraph
._fill_buffers` (the MTP draft model's own CUDA-graph fill path) does the
identical contiguous-addressing computation as `CapturedBatchDecodeGraph
._fill_buffers` and was fixed with the same code pattern, so P1's
dynamic allocator doesn't inherit a silent gap in the sibling graph class.
Not separately driven through a dedicated numeric-equivalence test this
round (would require loading the MTP draft model just for this narrow
check) — covered by the full regression battery at the flag's default
`False`, and by code-pattern identity with the class that IS directly
tested.

**Untouched, per the design doc**: GDN state handling, `_physical_slot`,
reserved-physical-slot-0 handling. The MTP draft-model call sites
(`_mtp_forward`/`_mtp_forward_batch`) reuse the exact same
`build_attention_metadata`/`_batch` functions the target model uses (not a
second implementation), so they are also correctly routed.

**Instrumentation** (§25.9's cheap recommendation, folded into this
round): `server/engine.py` gained `_log_prefix_overlap` — for every
newly-admitted request, measures exact longest-common-token-prefix
overlap against (a) other requests admitted in the *same* round (pattern
A, fan-out) and (b) a bounded rolling window of the last 64 admitted
requests' prompts (pattern B / incidental cross-request sharing), logging
per-request samples plus running event counts into `self.stats`. Purely
additive — never consulted by admission/scheduling/generation logic.

### Verification

1. **New dedicated test** (`benchmarks/prefix_cache_block_table_check.py`):
   - Arange-equivalence (pure Python, no GPU): `_initial_block_table`
     matches the old inline `arange` formula for every slot at 4 different
     (num_slots, blocks_per_slot) shapes. **PASS**.
   - Real-GPU numeric equivalence: one `DirectModelRunner`, same
     prompt/slot, run twice (`enable_block_table=False` then `=True`) —
     single-request path (`build_attention_metadata`/`_slot_mapping`),
     batched path (`build_attention_metadata_batch`/`_slot_mapping_batch`,
     both prefill- and decode-shaped calls), and `CapturedBatchDecodeGraph
     ._fill_buffers` (exercised directly, no full graph capture needed).
     All 6 checks **bytewise-identical** (not near-tie — pure addressing
     equivalence within one process, so any difference at all would be a
     real bug): `single_request_greedy_token_match`,
     `single_request_logits_bytewise_equal`, `batched_anchor_tokens_match`,
     `batched_prefill_logits_bytewise_equal`,
     `batched_decode_logits_bytewise_equal`,
     `graph_fill_buffers_page_indices_equal`,
     `graph_fill_buffers_slot_mapping_equal`,
     `graph_fill_buffers_kv_page_indptr_equal`. **Overall PASS**.

2. **Full `benchmarks/mtp_*_check.py` battery (all 11 scripts) + design
   doc's own P0-called-for `cudagraph_eager_parity_check.py`**: fresh
   GPU-idle-verified run, all 12 **PASS** (`mtp_accept_reject_check`,
   `mtp_async_arrival_check`, `mtp_batch_verify_check`,
   `mtp_chunked_prefill_check`, `mtp_gdn_rollback_check`,
   `mtp_multiround_check`, `mtp_prior_kv_len_fix_check`,
   `mtp_ragged_prefill_check`, `mtp_ragged_recompute_verify_check`,
   `mtp_real_draft_check`, `mtp_verify_cudagraph_check`,
   `cudagraph_eager_parity_check`). No tracebacks, no `FAIL` strings, in
   any log.

3. **4K/c=4 headline** (`mtp_w1s_our_runtime_perf --batched --cudagraph
   --repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`): all 3
   reps produced `total_committed_tokens=4116` and
   `draft_acceptance_rate_pct=70.29204431017119` — **bit-identical** to
   this project's long-established baseline (unchanged across dozens of
   prior measurements in `notes/2026-07-18-session-review-and-next-steps.md`).
   `num_drafts=1324`, `num_draft_tokens=3972`, `num_accepted_tokens=2792`
   also identical across all 3 reps. Throughput 145.6/146.3/149.9 accepted
   tok/s (mean 147.3), within this project's established measurement
   variance (no regression).

4. **`mtp_verify_cudagraph_check.py` re-run specifically** (called out
   separately since this phase touches the CUDA-graph fill-buffers path):
   included in and confirmed by item 2 above — **PASS**.

5. **Bonus, beyond the design doc's explicit P0 gate**:
   `benchmarks/cudagraph_decode_regression.py` (a *real* capture+replay of
   `CapturedBatchDecodeGraph`, not just a direct `_fill_buffers` call) —
   **PASS**, with the SAME extreme-shape coverage (96%-of-capacity slot,
   freshly re-prefilled 1-token slot alongside larger ones) this project's
   test already establishes, all at the flag's default `False`.

Every item bit-identical or bytewise-equal as required; nothing needed
fixing.

### Readiness for P1

P1 (dynamic free-list allocator + reference counting, still no cross-slot
sharing) can proceed directly on top of this substrate: `block_table[slot]`
already exists and is already consulted end-to-end (metadata, slot-mapping,
CUDA-graph fill) whenever `enable_block_table=True`; P1's only job is to
change what `_initial_block_table`'s replacement (a real `BlockPool`) hands
out — non-contiguous ids instead of a fixed contiguous range — and to wire
`reset_slot`/`_finish_request` to free blocks back to the pool (design
doc's R10). No known open gap blocks this. `enable_block_table` stays
`False` by default until P1's own dedicated allocator test
(`prefix_cache_allocator_check.py` per the design doc) is green.
