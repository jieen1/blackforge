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

---

## P1 — Dynamic free-list allocator + reference counting (2026-07-19)

**Status: DONE, verified, zero regression. Ready for P2.**

### What was built

Per `notes/prefix-cache-design.md` sec 5's "P1 — Dynamic free-list
allocator + reference counting": `runtime/direct_model_runner.py` gained
`Block` (dataclass: `block_id`, `ref_cnt`, `block_hash` — the last unused
this phase, kept for P3) and `BlockPool` — a real free-list allocator over
the shared physical attention-KV block pool. `BlockPool.__init__(num_blocks,
reserved=RESERVED_PHYSICAL_SLOTS)` carves out `[0, reserved)` (just block 0
in production, INV7) and seeds a FIFO free queue (`collections.deque`) with
every other id. `allocate(n)` pops `n` ids from the front (`ref_cnt: 0→1`
each); `free(block_ids)` decrements `ref_cnt` and, at 0, appends each id to
the tail. Both reject any id `< reserved` (defense in depth beyond the
constructor's own exclusion), and `allocate`/`free` raise on exhaustion/
double-free respectively rather than silently misbehaving.

`DirectModelRunner.__init__` now constructs one `self.block_pool =
BlockPool(num_blocks=(num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot)`
(same `num_blocks` formula P0 already used for the KV tensor's own
allocation — unchanged, still shared with `vllm_stage_b_baseline.py`) and
initializes `self.block_table` to **empty lists** (`[[] for _ in
range(num_slots)]`), replacing P0's `_initial_block_table`-based
up-front full-`blocks_per_slot` population. `_initial_block_table` itself
is untouched (still imported/called directly by `prefix_cache_block_table_
check.py`'s arange-equivalence check) — it just no longer populates
`self.block_table` at construction time.

A new `DirectModelRunner._ensure_blocks(slot, kv_len_needed)` grows
`self.block_table[slot]` on demand (append-only — INV6) to
`ceil(kv_len_needed / block_size)` physical ids, raising the same
capacity-exceeded `RuntimeError` `build_attention_metadata`/`_batch`
already used (checked *before* touching the pool, so a doomed request
never consumes a block). Wired into every one of the six existing
`enable_block_table`-gated read sites identified in P0: `_attention_metadata`
(single-request target), `_forward_batch` (batched target, via a per-slot
loop over `kv_lengths`+`qo_lens`), `_mtp_forward`/`_mtp_forward_batch`
(single/batched draft — the draft layer shares the target's block-id
namespace per the design doc's sec 3.1), and both `CapturedBatchDecodeGraph
._fill_buffers`/`CapturedMTPDraftStepGraph._fill_buffers` (CUDA-graph
per-replay refill). Each call site's growth target is exactly the same
`new_kv_len`/`prior_kv_len + num_new_tokens` value that call's own metadata
builder already computes internally, so a single growth call ahead of
metadata construction covers slot-mapping construction too (verified by
inspection, not assumed, for all six sites). `reset_slot` (design doc's R10
risk) now frees a slot's held blocks back to the pool and clears its table
to `[]`, driven by the table's own contents (not `enable_block_table`'s
current value) so a slot that was never grown (flag off its whole life) is
a safe no-op.

Still **zero cross-slot sharing**: every block has exactly one referencer
(`ref_cnt` never exceeds 1 in this phase); the only thing P1 changes is
*where* a slot's blocks physically live, not what content ends up in them.

### Verification

1. **New dedicated `benchmarks/prefix_cache_allocator_check.py`** (pure
   Python, no GPU — see its own docstring for why): 7 checks, all
   **PASS** — `alloc_free_correctness`, `ref_cnt_bookkeeping`,
   `free_queue_ordering` (confirmed strict FIFO oldest-freed-first, across
   two separate free/reallocate rounds), `inv7_reserved_block_zero`
   (block 0 never handed out across repeated exhaust/recycle cycles, never
   accepted by `free`, construction rejects `reserved<1`),
   `append_only_growth_inv6` (a position's physical id never changes
   across idempotent/growing `_ensure_blocks` calls; capacity check raises
   before consuming a block), `reset_slot_frees_blocks` (real block
   release + table clear + safe no-op on a never-grown slot), and
   `fragmentation_after_churn` (an explicit allocate/reset/reallocate churn
   recipe leaves a single slot's own `block_table` provably non-contiguous,
   e.g. `[1, 5, 7]` — not just claimed). The allocator/free-logic checks
   call the REAL, unmodified `DirectModelRunner._ensure_blocks`/`reset_slot`
   as unbound methods against a lightweight duck-typed stand-in (not a
   reimplementation) — genuine production-code coverage without a GPU/model
   load.

2. **`prefix_cache_block_table_check.py` (P0's own test) re-run and
   updated**: the arange-equivalence check and the four logits-equality
   checks (single-request, batched prefill/decode) — unaffected, **PASS**
   unchanged. The two CUDA-graph-fill sub-checks that used to assert raw
   block ids were BYTEWISE IDENTICAL between `enable_block_table=False`/
   `True` (`graph_fill_buffers_page_indices_equal`/`_slot_mapping_equal`)
   tested a P0-specific incidental fact (P0's allocator was a pure
   relabeling of the same ids) — P1 legitimately changes this by design
   (dynamic, non-contiguous placement), so real correctness never depended
   on it. Replaced with `graph_fill_buffers_page_indices_valid_on`/
   `_slot_mapping_valid_on` (in-range, excludes reserved block 0, no
   accidental cross-slot id sharing) plus `graph_fill_buffers_kv_page_
   indptr_equal`, kept unchanged (page *counts* per request never
   depended on allocator identity). Full re-run: **PASS**, including the
   real-GPU numeric-equivalence pass (bytewise-identical logits, prefill
   + decode, single-request and batched).

3. **Fragmented CUDA-graph re-runs (INV5 — the core proof this phase
   exists to deliver), both with a real GPU/model, both new
   `--enable-block-table` flags added (default `False`, preserving each
   script's original invocation/behavior byte-for-byte)**:
   - `benchmarks/cudagraph_decode_regression.py --enable-block-table`:
     added `_fragment_pool_via_churn` — sponges up the pool down to a
     small working set sized to the upcoming big allocation, then rotates
     it via real `prefill`/`reset_slot` cycles on slot 3 + 2 scratch
     slots — before slot 3's existing "push to near-capacity" step.
     Result: slot 3's real block_table came back as `[1289..1407, 1, 8, 9,
     1280, 1284]` (123 blocks, **provably non-contiguous** — wraps from
     1407 to 1), and the SAME captured graph replayed correctly across it
     (signal-probe identity checks **PASS** for all 4 slots, both the
     extreme-mixed-kv_len and tiny-kv_len-neighbor steps). Default
     (flag-off) re-run: **PASS**, unchanged.
   - `benchmarks/mtp_verify_cudagraph_check.py --enable-block-table`:
     added `_rotate_pool` (same sponge+rotate recipe, applied globally
     before `mtp_prefill_batch` even runs, via 2 new scratch slots). Result:
     **all 8** real slots (4 `mtp_slots` + 4 `ref_slots`) ended up with
     non-contiguous block tables, and the full 8-round force-pattern
     battery (organic/mixed-reject/fully-ragged/uniform-reject, verify
     graph at batch 4 and 2, draft-step graph at qo_len 1/2) still
     **PASSED** — `per_slot_ok` all `true`, all 4 coverage gates (verify
     graph b4/b2, draft-step qo2/qo1, actually *replayed* not just
     precaptured) `true`, `fragmentation_proof.ok: true`. Default
     (flag-off) re-run: **PASS**, unchanged (`fragmentation_proof: null`).

4. **Full `benchmarks/mtp_*_check.py` battery (all 11 fast scripts) +
   `cudagraph_eager_parity_check.py`** — same 12-script set P0's own gate
   used (this project's established "fast battery" convention;
   `mtp_sustained_realistic_workload_check.py`, the 63.5-minute realistic-
   workload E2E test, is a separate, already-characterized slow check —
   see its own PROGRESS.md entry — not part of this per-phase regression
   gate, matching P0's precedent): fresh GPU-idle-verified run, all 12
   **PASS** (`mtp_accept_reject_check`, `mtp_async_arrival_check`,
   `mtp_batch_verify_check`, `mtp_chunked_prefill_check`, `mtp_gdn_
   rollback_check`, `mtp_multiround_check`, `mtp_prior_kv_len_fix_check`,
   `mtp_ragged_prefill_check`, `mtp_ragged_recompute_verify_check`,
   `mtp_real_draft_check`, `mtp_verify_cudagraph_check`,
   `cudagraph_eager_parity_check`). All run at `enable_block_table`'s
   default `False` (none of these scripts pass the flag) — this is the
   "P1 changes nothing when off" half of the regression gate.

5. **4K/c=4 headline** (`mtp_w1s_our_runtime_perf --batched --cudagraph
   --repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`): all 3
   reps produced `total_committed_tokens=4116` and
   `draft_acceptance_rate_pct=70.29204431017119` — **bit-identical** to
   the established baseline (unchanged since P0, and every prior
   measurement in `notes/2026-07-18-session-review-and-next-steps.md`).
   `num_drafts=1324`, `num_draft_tokens=3972`, `num_accepted_tokens=2792`
   identical across all 3 reps too. Throughput 141.8/142.2/144.3 accepted
   tok/s (mean 142.8), within this project's established measurement
   variance (no regression). This run uses the default `enable_block_
   table=False` (the harness doesn't pass the flag) — confirms P1's
   changes are a true no-op for the flag-off production path.

Every item passed or matched exactly as required; nothing needed fixing
beyond one self-inflicted test-design bug in the first draft of
`_fragment_pool_via_churn` (the initial recipe relied on the pool's
naturally-huge ascending virgin supply to self-fragment via small churn
alone — it never did, since the untouched tail is far larger than any
realistic churn; fixed by explicitly sponging the working set down to
just above the upcoming allocation's real need before rotating it) and one
scratch-slot placement bug (initial scratch slot indices collided with
`CapturedBatchDecodeGraph`'s own permanently-reserved warmup range,
producing a `RuntimeError: slot N is not fresh` — fixed by placing scratch
slots between the real slots and the warmup range, not past it).

### Readiness for P2

P2 (fan-out fork, Pattern A same-round sharing) can proceed directly on
top of this substrate: `BlockPool.allocate`/`free` are real, tested,
production primitives; a slot's blocks are provably tolerant of
non-contiguous placement end-to-end (metadata, slot-mapping, CUDA-graph
replay, both target and draft models). What P2 adds on top — the one
thing P1 deliberately does NOT do — is `ref_cnt > 1` (a block referenced
by more than one slot's `block_table`) and the `touch`/multi-referencer
semantics that come with it; `BlockPool`'s `allocate`/`free` API is
already shaped to accept that extension without a caller-visible
interface change (per `BlockPool`'s own docstring). No known open gap
blocks P2's start.
