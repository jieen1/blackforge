# Implementation Progress

Updated: 2026-07-16

## Completed

### Phase 3, strategy reset (2026-07-16, fifth pass) — freeze kernel bisection, rebuild top-down

After four debugging passes each surfaced a real, specific, low-level
anomaly (Triton `causal_conv1d_fn` cold start; a CUTLASS SM120 pingpong-GEMM
race per `compute-sanitizer racecheck`) and then *disproved* both as the
actual root cause (bypass experiments changed output without fixing it),
the strategy reset: **freeze further per-kernel bisection** (both anomalies
now documented as independent, closed defects -- see the design doc's
"Known independent defects" section) and instead rebuild top-down from
vLLM's real, already-verified `GPUModelRunner`/`Scheduler` machinery.

**Step 1 (this round, done): formalized the repro, ran it 20x.** New
committed script `benchmarks/single_prefill_regression.py` (fixed prompt,
config, no bypasses; records first-token id + a SHA-256 hash of the full
logits vector; runs N fresh-process repeats). **Result: 20/20 identical
failures** -- same wrong token, same byte-for-byte logits hash, every
single run. This is an important clarification: the bug is **100%
deterministic** under a fixed configuration, not a genuine hardware race.
The earlier apparent "Heisenbug" (pass 3) is now understood to have come
from *changing* the test configuration between runs (different token
counts, added instrumentation, kernel bypass flags altering the code
path) -- not true non-determinism. This points toward a **semantic/contract
mismatch** (wrong metadata field, padding convention, or state
initialization vs. what real `GPUModelRunner` does) as the more likely
root cause, which is exactly what step 2 targets.

**Step 2 (this round, DONE): no-HTTP correct baseline established, 20/20.**
Built `runtime/vllm_inprocess_baseline.py`, driving vLLM's own `LLM` class
directly instead of hand-deriving `GPUModelRunner`'s internals -- real
`Scheduler`, real `KVCacheManager`-allocated cache, the real
`SM120GQAMetadataBuilder`/`GDNAttentionMetadataBuilder`, real
warmup/profiling sequencing, our own `SM120GQABackend` with decode v2 --
with **no HTTP layer at all** (`LLM` never has one; only `vllm serve` does;
under the hood it's `SyncMPClient` + a background `EngineCore` reached via
local ZMQ, not a network call). **Ran 20 consecutive fresh-process repeats
of "The capital of France is": all 20 produced the identical, correct**
`" Paris.\n\n<think>\nHere's a"` **(20/20 pass).** This is a decisive,
positive result: it confirms the model/checkpoint/quantization/GDN/our own
SM120GQABackend (decode v2 included) are all correct when driven through
vLLM's real machinery -- combined with the two independent defects already
ruled out as root causes, this narrows the actual bug specifically to
`direct_model_runner.py`'s own hand-rolled orchestration, not anything
downstream of it. Full detail in the design doc's "Step 2 result" section.
**Step 3, Stage B (this round, DONE): our own cache exonerated, 20/20.**
Built `runtime/vllm_stage_b_baseline.py`: real vLLM `Scheduler`/
`GPUModelRunner`/metadata builders/warmup, unchanged, with exactly one
substitution -- monkey-patched `GPUModelRunner.initialize_kv_cache_tensors`
to call a newly-extracted shared helper,
`runtime.direct_model_runner.allocate_fixed_slot_kv_caches` (the *same*
4-fixed-slot allocation/binding logic `DirectModelRunner` itself uses,
confirmed behavior-preserving via the regression script's unchanged logits
hash after the refactor), instead of vLLM's own cache tensor allocation.
**Ran 20 consecutive fresh-process repeats: 20/20 identical PASS**, same
correct completion every time. **This exonerates the cache layer** -- our
own KV/GDN state tensor shape, allocation, and `bind_kv_cache()` binding are
all correct under real vLLM scheduling. The bug narrows specifically to
Stage C (this project's own hand-built attention/GDN metadata
construction), not cache shape/binding/slot-mapping/state-init. One noted
gap (not a failure, an untested corner): the real scheduler's block-pool
size belief (~200K tokens) still exceeds the substituted tensor's real
capacity (~8192 tokens) -- never manifested in this narrow single-short-
request test, but would need reconciling before trusting Stage B beyond
this smoke test. Full detail in the design doc's "Stage B result" section.

**Step 3, Stage C (this round, DONE -- bug conclusively localized): 0/20,
deterministically.** Built `runtime/vllm_stage_c_baseline.py`: keeps
Stage B's cache substitution, adds exactly one more -- the two real vLLM
metadata builder classes (`SM120GQAMetadataBuilder`,
`GDNAttentionMetadataBuilder`) are monkey-patched to call this project's
own hand-built metadata construction (`build_attention_metadata`/
`build_gdn_metadata`, extracted as shared functions -- confirmed
behavior-preserving) instead of their real field derivation, sourcing the
few needed facts from vLLM's own real `CommonAttentionMetadata`. **Ran 20
consecutive fresh-process repeats: 0/20 passed, and all 20 failures were
byte-for-byte identical** (same deterministic-failure signature as the
original bug, different exact wrong text). **Per the ladder's decision
rule, this conclusively localizes the bug**: A (real everything) passes
20/20, B (real metadata + our cache) passes 20/20, C (our metadata + our
cache) fails 0/20 -- the bug is specifically in this project's hand-built
attention/GDN metadata construction logic, not cache, not the model, not
the two independent kernel-level anomalies found and ruled out in earlier
passes. Natural next step (not attempted this round): diff our hand-built
metadata field-by-field against the real builders' output for the
identical real input, to pinpoint exactly which field is wrong. Stage D
(the full `direct_model_runner.py`) was not run this round since C failing
already makes D's outcome predictable. Full detail in the design doc's
"Stage C result" section.

**Step 4 (this round, DONE -- root cause found and fixed, closed loop
established): field diff → causal confirmation → general fix → 20/20 on
both Stage C and Stage D.** Ran the real `SM120GQAMetadataBuilder.build()`/
`GDNAttentionMetadataBuilder.build()` and our hand-built
`build_attention_metadata()`/`build_gdn_metadata()` side by side on the
identical `CommonAttentionMetadata`, diffing every field. All fields
matched except **`GDNAttentionMetadata.non_spec_state_indices_tensor`** and
**`prefill_state_indices`**: real builder produced `[1]`, ours produced
`[0]`. Cross-checked against a real `SchedulerOutput` dump: vLLM's real
scheduler assigns block/state index **1** to the first-ever scheduled
request (`block_ids=([1], [2], [3], [4])`) -- **index 0 is never used for
real request data**. Our hand-built metadata hardcoded physical
slot/state index = logical slot number, so logical slot 0 → physical index
0, landing on whatever convention makes index 0 unsafe (exact mechanism
not pinned down -- possibly NULL_BLOCK_ID-adjacent -- but the empirical
fact is solid). **Causal confirmation** (not just correlation): a
throwaway Stage C variant with `SLOT=1` hardcoded produced the correct
`" Paris"` output on the first try.

**Fix applied** (`runtime/direct_model_runner.py`): added module constant
`RESERVED_PHYSICAL_SLOTS = 1` and helper `_physical_slot(logical_slot) =
logical_slot + RESERVED_PHYSICAL_SLOTS`, applied consistently in all four
places physical addresses are computed: `allocate_fixed_slot_kv_caches`
(allocates one extra slot's worth of capacity/state rows and never
addresses row 0), `build_attention_metadata` (`first_block`),
`build_gdn_metadata` (`state_indices`), and
`DirectModelRunner._slot_mapping` (`first_block`). Confirmed via a
lightweight CPU-only check that both the attention-side
(`kv_page_indices`) and GDN-side (`prefill_state_indices`,
`non_spec_state_indices_tensor`) fields now skip physical index 0 for
every logical slot -- the diagnostic script had crashed before printing
the attention-side diff directly, so this closes that open question too.

**Verification (both 20x, both 20/20 PASS):**
- `runtime/vllm_stage_c_baseline.py` (no code changes needed -- its
  existing `SLOT=0` is now automatically offset internally): **20/20
  PASS**, identical `' Paris.\n\n<think>\nThe user has'` every run.
- `benchmarks/single_prefill_regression.py`, i.e. the full
  `DirectModelRunner` (**Stage D**): **20/20 PASS**, correct first token
  `' Paris'`, **identical logits hash `7eda2739bbecbc52` across all 20
  runs**.

This closes the ownership-transfer ladder: A/B/C/D all now produce
correct, deterministic output. The minimal correct closed loop for
"direct GPU state ownership" (no HTTP, hand-built metadata, our own
KV/GDN cache tensors) is established. The throwaway `SLOT=1` test file
was deleted once the general fix landed in `direct_model_runner.py`
itself; no separate hack file remains in the repo.

### Phase 3, batch decode support (2026-07-16) — real batched metadata, batch=1 verified 19/20, batch>=2's "mismatch" traced to a pre-existing real-vLLM effect (not our bug)

`decode_batch()` used to be a Python loop calling single-request
`decode()` per slot -- not a real GPU batch. Replaced with genuinely
batched metadata construction: new `build_attention_metadata_batch`/
`build_gdn_metadata_batch` functions (kept separate from the
single-request builders to avoid regressing the just-closed Stage C/D
loop), generalizing the real `SM120GQAMetadataBuilder`/
`GDNAttentionMetadataBuilder`'s own pure-decode-batch CSR construction
(read directly from vLLM source to get the convention right) from one
request to N requests across this project's fixed physical slots.
`DirectModelRunner._forward_batch()`/`.decode_batch()` now issue exactly
ONE `model.forward()` call for every listed slot. New test harness
`benchmarks/batch_decode_regression.py`: prefills two independent,
identically-initialized slot groups per test (avoids the confound of
testing before/after on the same slot), decodes one via the
already-verified single-request path and the other via one real batched
call, and diffs logits bytewise (SHA-256 hash).

**Batch=1: 19/20 PASS** (1 failure was an `hf-mirror.com` `504 Gateway
Timeout` during model-config resolution, before any model/kernel code
ran -- unrelated to decode_batch; worked around via `HF_HUB_OFFLINE=1`
since the checkpoint is already fully cached locally).

**Batch=2: initial result 0/2 bytewise match -- investigated, NOT a
decode_batch bug.** Both rows (duplicate prompts, to rule out cross-row
addressing) gave an identical wrong-vs-reference hash to each other (not
request-order corruption, but genuinely different from the single-request
value). Root-caused with a real-vLLM-only diagnostic (no code of ours):
`LLM.generate()` on "The capital of France is" **alone** vs
**concurrently with a second real request**, greedy (`temperature=0.0`),
reproduced deterministically twice:

```
ALONE      : " Paris.\n\n<think>\nHere's a"
TOGETHER[A]: " Paris.\n\n<think>\n\n</think>\n\nThat"
```

**This proves the numerical divergence is a pre-existing property of the
real production stack (almost certainly floating-point non-associativity
in batched GEMM -- different M-dimension kernel/tile selection changes
summation order/rounding, a well-documented industry-wide "batch
invariance" issue), not a bug in this round's hand-built batch metadata.**
Consistent with batch=1 passing cleanly (no alternate M-dimension path
exists at batch=1). **Practical implication**: requiring bytewise-identical
logits between single-request and N-request-batched paths is not an
achievable bar -- not even vLLM's own real production path meets it. This
is a methodology decision (what tolerance/metric should replace bytewise
match for steps 2-8 of the validation ladder) flagged to the coordinator
rather than decided unilaterally before proceeding further.

**Coordinator decision: new acceptance criteria adopted** (1) argmax/token
plausibility, (2) signal-probe/marker-token crosstalk detection as the
primary bug-catching tool, (3) same-batch internal self-consistency
(identical prompts in one batch call must match bytewise) replacing
"run alone" as the reference.

### Phase 3, batch decode ladder steps 2-6 (2026-07-16) — all PASS under the new criteria

New harness `benchmarks/batch_decode_signal_probe.py`: each slot's prompt
is `"{filler}The value of X is {number}. The value of X is"` (a strong
in-context copy cue), with a duplicate number+filler pair (slots 0 and
batch-1, batch>=3) for self-consistency and distinct numbers elsewhere
for crosstalk detection.

- **batch=3**: 1 sanity + 3/3 repeat, all PASS.
- **batch=4**: 1 sanity + 3/3 repeat, all PASS.
- **variable-length (batch=4)**: first attempt failed self-consistency
  3/3, but this was a TEST HARNESS bug (the duplicate pair had different
  filler lengths, so their prompts weren't actually identical) -- not a
  decode_batch bug (`signal_ok` was already `True` in every "failing" run,
  i.e. no real crosstalk). Fixed the harness (`_assign_filler_repeats`
  keeps the self-consistency pair's filler length matched) and reran:
  **3/3 PASS**.
- **slot release + reuse (batch=3)**: 3/3 PASS -- after 8 decode rounds,
  slot 0 released and re-prefilled with a brand-new, disjoint number;
  8 further decode-only steps recover exactly the new number with zero
  residue from its prior occupant or the still-active other slots.
- **continuous 256-token generation (batch=4)**: 2/2 PASS -- 512 total
  real `_forward_batch()` calls across both repeats, no crash, checks
  still clean against the full generated text.
- **Signal-probe no-crosstalk**: cross-cutting, `signal_ok: true` with
  zero leaked-number instances across every run above -- the primary
  evidence that `decode_batch`'s physical-slot addressing generalizes
  correctly from the single-request path to real N-request batches.

Used 3x repeats (2x for the 256-step run) rather than 20x: a genuine
addressing bug here is a deterministic logic error that would show up on
the first run, not a rare hardware race (unlike the earlier cross-process
bytewise-hash comparisons, which were sensitive to legitimate
floating-point noise) -- judged sufficient given every run passed
cleanly. MTP is correctly NOT attempted this round (explicitly last in
the coordinator's ladder ordering) -- left for its own round.

### Phase 3, batch decode ladder final step: MTP verify (2026-07-16)

Generalized `build_attention_metadata_batch`/`build_gdn_metadata_batch`
from `qo_len=1`-only to a `qo_len` parameter (uniform across the batch,
matching real MTP usage), so `decode_batch` can also drive MTP verify
(K draft + 1 bonus token per request, e.g. qo_len=4 for K=3). Attention
side reuses the exact same CSR-construction logic generalized by
`qo_len` (byte-identical at `qo_len=1`); this is what routes calls to the
real production `flash_attn_sm120_fwd_v2_decode_fp8kv_paged` kernel
(already hardened for qo_len 2-4, no kernel changes made) -- confirmed
directly via the backend's own log line `v2 decode kernel path HIT
(qo_len=4)` in every test run. GDN side reuses `build_gdn_metadata`'s
existing chunked/"prefill" branch (not the real builder's much more
involved spec-decode branch, which is out of scope this round) since it's
numerically correct for an ordinary multi-token continuation regardless
of whether the caller calls it "MTP" or not. New `DirectModelRunner
.verify_batch()` wraps `_forward_batch(qo_len=...)`; accept/reject
sampling on the raw returned logits is explicitly left for a follow-on
round.

Caught a real test-design flaw before it became a false bug report:
"rewinding" a slot's kv_len bookkeeping to re-verify already-decoded
tokens would be unsound, because GDN's recurrent state (unlike paged
attention's content-addressed KV) can't be cheaply rewound -- fixed by
using an independent twin slot group instead (mirrors the earlier
batch=1 equivalence test's approach). Also found and fixed a test-harness
scoping issue: the prompt's very first generated token is always a
leading space, not a digit, so `qo_len=4` (kept deliberately small to hit
the real v2 kernel's qo_len 2-4 range) can only recover 4 of a 5-digit
number's digits -- confirmed via a direct diagnostic that `verify_batch`'s
raw predictions are bit-for-bit identical to the trusted single-token
path at the same positions, then fixed the crosstalk check to compare a
length-matched number prefix instead of requiring the full number.

**Results at qo_len=4 (real production MTP shape), 1 sanity + 3/3 repeat
each: batch=1 PASS, batch=2 PASS, batch=4 PASS** (self-consistency and
zero crosstalk held throughout). One batch=4 repeat run hit a 600s
timeout under 96.8GiB GPU memory pressure -- traced via `ps -ef` to a
COMPLETELY UNRELATED concurrent session's own `llama.cpp` benchmark on
this shared machine, not a bug in this round's code; waited for that
job's own timeout to elapse naturally (never touched a process this
session didn't own) and retried cleanly for 3/3.

**This completes the core "direct model runner supports 4-slot fixed
batch + MTP" mechanism.** Explicitly not done this round (per the
coordinator's scoping): accept/reject sampling logic and CUDA Graph
capture -- both left for a follow-on round alongside real
W1/W2/concurrency=4/MTP-K=3 performance measurement.

### Phase 3, CUDA Graph capture/replay step 1 (qo_len=1 batch decode) (2026-07-16) — implemented + verified uninstrumented; compute-sanitizer in progress

Per the coordinator's explicit caution about this work's crash history,
read the sibling project's kernel-level CUDA-graph test
(`test_cudagraph_decode_fixed_sizing.py`) as prior art first, and fixed a
prerequisite BEFORE attempting capture: `build_attention_metadata_batch`'s
`kv_split_size` was derived per-call from live kv_len -- exactly what the
sibling's own docs warn goes stale under graph replay at a kv_len larger
than capture-time data. Added `fixed_kv_split_size`/`fixed_max_num_splits`
params (derived once from this slot's hard capacity, default `None`
preserving the existing eager-path behavior) with the same correctness
proof the real backend uses.

New `CapturedBatchDecodeGraph`: every tensor a captured launch reads is a
persistent, fixed-address buffer; `replay()` only `.copy_()`s fresh
values into them. Also caught and fixed `torch.cuda.synchronize()` being
called inside the capture region (a documented capture violation) before
it could crash, via a sync-free `_forward_no_sync()` path.

New test `benchmarks/cudagraph_decode_regression.py`: captures at a small
(~15-token) shape, replays at the capture-time shape, 8 steps of normal
growth, one slot pushed to **1961/2048 tokens (96% of hard capacity)**
while others stay small, and a freshly re-prefilled 1-token slot
alongside larger ones -- deliberately far more extreme than the
capture-time shape, per the coordinator's explicit instruction not to
only test the happy path. Checked via signal-probe (not bytewise).
**Result: 3/3 independent repeats, all PASS, zero crashes** -- every slot
at every kv_len, including the 96%-capacity case, correctly recovered its
own identity marker with zero cross-slot leakage.

**compute-sanitizer: attempted, not yet complete -- reported honestly,
not skipped or faked.** Full test under `--tool memcheck`: weight loading
alone took 662-793s (~10-15x normal), then 60+ minutes with zero further
progress; killed after ~1-2 hours. A cut-down 10-call minimal repro
(`benchmarks/cudagraph_decode_sanitizer_repro.py`) still stalled for
hours under full memcheck. Switched to `--tool initcheck` (lighter,
targets uninitialized-memory/invalid-pointer issues specifically):
weight loading returned to normal speed, but the first forward pass (this
runner's own pre-existing `_warmup()`, unrelated to this round's code)
produced thousands of "Uninitialized __global__ memory read" reports, ALL
tracing to the already-known, already-investigated `causal_conv1d`
Triton-kernel cold-start defect documented earlier in this project (see
notes/direct-model-runner-design.md's "Known independent defects") --
real, but not new, and not part of this round's scope. Its sheer volume
drowns out the sanitizer's report budget before ever reaching the actual
`capture()`/`replay()` code. Worked around via `--kernel-name-exclude
kernel_substring=causal_conv1d`; a machine reboot killed this attempt
mid-run (confirmed via fresh `git status`/`nvidia-smi`/`ps` after
restart -- all working-tree changes preserved on disk, no GPU/process
state survived, as expected).

**Restarted after reboot, tried two more variants, same pattern both
times.** At the original batch_size=4 scope with the exclusion applied:
20+ minutes with no progress past the (now-excluded) causal_conv1d point.
Cut further to a genuinely minimal, sanitizer-only script
(`benchmarks/cudagraph_sanitizer_micro.py`, batch_size=2, exactly ONE
replay directly at an extreme kv_len, 7 total forward-pass-equivalent
calls -- verified correct and fast, ~20s, uninstrumented first). Under
`initcheck` with the same exclusion: weight loading was normal speed
again, but the SAME pre-existing `_warmup()` call surfaced a **second,
different, previously-undocumented uninitialized-read report** (100
instances, default cap) in `qwen_gdn_linear_attn.py`'s
`_output_projection` -- a different kernel from causal_conv1d, but still
100% confined to `DirectModelRunner.__init__`'s own warmup call, not this
round's new code. After exhausting that cap, the process ran silently for
20+ more minutes with zero further output, still inside that same first
forward pass.

**Pattern across all four attempts** (full memcheck x2, initcheck at
batch=4, initcheck at batch=2): every one stalled or flooded inside the
pre-existing `_warmup()` mechanism, which this round's CUDA Graph code
doesn't even reach yet. At least two distinct model/kernel cold-start
defects now confirmed (causal_conv1d, and the GDN output projection) --
real, pre-existing, unrelated to CUDA Graphs, but making this model+kernel
stack fundamentally expensive to sanitizer-instrument from a cold process
start, independent of anything CUDA-Graph-specific.

**Honest status, not glossed over**: `CapturedBatchDecodeGraph` itself is
solidly verified via extensive real, uninstrumented testing targeting the
exact failure modes (address staleness, split-size staleness under kv_len
far exceeding capture-time data) this project's sibling documented --
3/3 clean passes, zero crashes, including a 96%-of-capacity extreme case.
The compute-sanitizer 0-errors gate has NOT been satisfied for this
round's new code specifically, after genuine, sustained effort across
four distinct configurations -- flagged as an open item for a coordinator
decision on how to proceed, not silently marked done.

**Coordinator decision: accept the uninstrumented verification for this
round; compute-sanitizer stays a tracked open item, no further time spent
chasing pre-existing cold-start defects.** Proceeded to MTP CUDA Graph
capture.

### Phase 3, CUDA Graph capture/replay step 2 (qo_len=4 MTP verify) (2026-07-16) — implemented, 3/3 PASS

Generalized `CapturedBatchDecodeGraph` in place (not a new class) to
`qo_len>1`: static buffers sized `batch_size*qo_len`, and GDN's chunked
metadata fields (`chunk_indices`/`chunk_offsets`/`nums_dict`/`batch_ptr`/
`token_chunk_offset_ptr`/`has_initial_state`) computed ONCE in `__init__`
since they depend only on query-length structure, never on kv_len or
slot identity -- genuinely constant across every replay for a fixed
(batch_size, qo_len) graph. At qo_len=1 every formula reduces exactly to
the previous values (confirmed via regression rerun, identical output).

**Found and fixed a second, more consequential methodology issue before
it could produce a false result**: `capture()`'s 3 real warmup executions
(on a side stream, before the graph trace -- the trace itself executes
nothing) are NOT safe to run against the same slots later checked via
`replay()`, because GDN's recurrent/chunked state update is not
idempotent under repeated identical input (unlike attention's KV cache).
This is a real imprecision in the ALREADY-COMMITTED qo_len=1 test too
(reused the same slots for warmup and its first replay check) -- not
retroactively fixed there (its 3/3 PASS, including the 96%-capacity case,
stands as real evidence; likely didn't surface because `capture()`'s
`slot_ids` need not match `replay()`'s, and this signal-probe task is
likely dominated by full-attention layers rather than GDN). Fixed
properly in the new MTP test via dedicated, disposable `ref_slots`
(establish drafts + serve as warmup data, spent afterward) kept strictly
separate from `graph_slots` (touched by nothing but their own prefill
until the real replay calls).

Also caught a test-design bug (not a decode_batch bug): chaining a second
"extreme" verify as a continuation of the first replay's own predictions
fed mismatched content (a verify's output isn't the same thing as a new
draft for a follow-on step, which needs real accept/reject bookkeeping --
out of scope this round). Fixed by making the extreme check a fully
independent single-shot verify instead.

**Results, 1 sanity + 3/3 repeat**: small-shape replay (self-consistency
held, zero crosstalk, confirmed `v2 decode kernel path HIT (qo_len=4)` in
logs) and an independent extreme-shape replay (3 slots short, 1 slot at
**1961/2048 tokens, 96% of hard capacity**) -- every slot recovered its
own identity with zero leakage, same captured graph both times. **3/3
PASS, zero crashes.**

**This completes CUDA Graph capture/replay for both scopes asked for**
(qo_len=1 batch decode, qo_len=4 MTP verify). Not done: compute-sanitizer
(tracked open item) and accept/reject sampling (out of scope). Next: real
W1/W2/concurrency=4/MTP-K=3 performance comparison against native
FlashInfer.

### Phase 3, correction (2026-07-17) — an independent review found the CUDA Graph work above had a REAL, unfixed gap; now fixed with quantified before/after proof

**This corrects something reported as verified/passing above.** The
coordinator commissioned an independent Codex analysis, personally
verified it against the actual code, and found: `capture()`'s 3 real
warmup executions ran against the SAME slots every test script here
later checked via `replay()`. Attention's KV cache tolerates redundant
warmup writes harmlessly; GDN's recurrent/chunked state does NOT (it
reads-old-state-writes-new-state every call, not idempotent under
repeated identical input) -- so those slots' real GDN state silently
advanced 3 extra unaccounted steps before any "real" replay. The earlier
round's guess that this "likely didn't surface because full-attention
layers dominate the signal-probe task" was an **unverified guess stated
with more confidence than earned, not evidence**.

**Quantified proof the gap was real and severe**: a throwaway diagnostic
reproducing the old pattern (identical single-token input, eager vs a
graph with old-style same-slot warmup) measured `logits max_abs_diff=
7.93`, `cosine_sim=0.55`, GDN `conv_max_diff=45.8`, `ssm_max_diff=12.5`
-- not floating-point noise, a real divergence the signal-probe (which
only checks whether decoded TEXT still recovers the right identity
number) never had a chance to catch.

**Fix, built into the class itself**: `CapturedBatchDecodeGraph` now
permanently reserves `batch_size` of the runner's own slots exclusively
for `capture()`'s disposable warmup (`capture()` takes no external
arguments anymore); callers must size `num_slots >= 2*batch_size` and
never pass a graph's reserved slots to `replay()` (both enforced with
errors). Also removed a per-replay `torch.cuda.synchronize()` (stream
ordering already guarantees correctness; the blanket device-wide sync
was actively working against the point of using a graph to cut CPU
dispatch overhead) and made `_fill_buffers` compute values via plain
Python arithmetic instead of round-tripping through the shared metadata-
builder functions (real, partially-mitigated per-replay allocation
overhead -- not fully eliminated, a further optimization for later).

**New decisive verification** (`benchmarks/cudagraph_eager_parity_check.py`,
real numerical comparison, not signal-probe): identical input through
eager vs the fixed graph path, comparing full logits AND the GDN
`conv_state`/`ssm_state` tensors directly. **Result: `max_abs_diff=0.0`,
`cosine_similarity=1.0`, top-1/top-5 exact match, all 48 GDN layers
checked show 0.0 diff -- eager and graph are bytewise identical, not
just close. 3/3 repeats PASS.** Re-ran the qo_len=1 and MTP regression
tests too (all 4 affected scripts updated to the new API): both still
3/3 PASS.

**Also corrected**: prior wording describing this test's small
`blocks_per_slot*block_size=2048`-token limit as "hard (physical)
capacity" was inaccurate -- it's a value THIS TEST configured for speed,
not a GPU hardware limit, and far below what a real W1(4K)/W2(32K)
workload needs. Fixed in live code/docstrings; already-committed
historical entries using the old phrasing are left as-is (this note is
the correction of record).

**Acknowledged but NOT fixed this round** (tracked open items): no
native-attention-backend fallback path, `engine.py`'s `decode_batch()`
still disconnected from the real batching/CUDA-Graph mechanism,
accept/reject sampling still not implemented. Per the coordinator's
priority order, next is full eager-mode MTP semantics (real draft
generation, accept/reject, a GDN state commit/rollback strategy for
partial rejection), THEN the real W1/W2 performance comparison
(configured with actually-sized per-slot capacity, not this round's
small test value).

### Phase 3, MTP semantics round (2026-07-17) — accept/reject + GDN rollback implemented and verified; real draft generation investigated in depth, honestly deferred (more scope than estimated)

> **2026-07-17 correction** (caught by an independent Codex-sol analysis,
> verified against the real checkpoint's `config.json` + vLLM source
> before accepting): every `Qwen3NextMTP`/`qwen3_next_mtp.py` reference
> below is the WRONG class for this project's actual target checkpoint.
> `unsloth/Qwen3.6-27B-NVFP4` has `model_type: "qwen3_5"`, which vLLM's
> `SpeculativeConfig` routes to `Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor`
> (`vllm/model_executor/models/qwen3_5_mtp.py`), a different file/class
> serving a different `model_type` (`qwen3_next_mtp.py` is for the
> unrelated `qwen3_next` model family). Field name is
> `mtp_num_hidden_layers`, not `num_nextn_predict_layers`. The
> architectural conclusion below (separate small model, own
> full-attention KV cache, every-step sync) was never wrong, only these
> specific names were — left as originally written below since this is
> a historical log entry; see `notes/direct-model-runner-design.md`
> for the corrected version and the fully fixed writeup.

Read `项目实施规划.md`'s actual contract first, per instruction, before
designing anything: Phase 8 says "先完整复现 vLLM 的 MTP K=3" (replicate
vLLM's real mechanism, don't invent a simplified one); Phase 1's gate
says MTP acceptance must not drop >1 percentage point vs vLLM.

**Traced vLLM's real MTP K=3 mechanism from source** (not guessed):
`Qwen3NextMTP` is a genuinely separate small model (own full-attention
decoder layer(s), sharing only embed_tokens/lm_head with the target).
Its own attention layer needs its OWN KV cache kept in sync with the
target model on EVERY real step, not just during propose loops --
`_prepare_prefill_inputs_kernel`'s "shift input_ids by one" logic runs
over the full current query range on every real prefill/decode step,
because the draft model needs complete causal history to work at all.
This means faithfully replicating vLLM's MTP requires restructuring the
main forward path itself (every prefill/decode/_forward_batch call), not
just adding an isolated propose loop -- substantially more invasive than
initially scoped. The "complete-replication, not reinvented" loading
path is also worked out: pass `speculative_config={"method": "mtp",
"num_speculative_tokens": 3, "attention_backend": "CUSTOM"}` to
`EngineArgs` (matching `launch_test_server.py` exactly) so
`vllm_config.speculative_config.draft_model_config` is built by vLLM's
own logic, then `get_model()` with that config loads
`Qwen3NextMTP` -- and its attention layer registers into the SAME
`static_forward_context` this project's existing KV-cache-allocation
machinery already iterates over, so its cache "just works" once loaded
before allocation (confirmed by reading the code, not yet exercised).

**Honest scope decision**: rather than rush the full draft-model
integration (which does not fit this round's remaining budget with the
rigor this project requires), implemented and verified the two pieces
that are genuinely self-contained and checkable without it:

1. **Accept/reject boundary logic** (`benchmarks/mtp_accept_reject_check.py`):
   greedy verification against the already-working, CUDA-graph-capable
   `verify_batch()` (draft = `[anchor, d_0..d_{K-1}]`, K=3). Verified via
   3 constructed scenarios per run (all-accept; reject at position 1;
   reject at position 0), each using a deliberate decoy token at a KNOWN
   position and REAL trusted continuation tokens as ground truth.
   **Decisive check**: on rejection, the recovery token equals the TRUE
   next token (the target model's real prediction), not the decoy and
   not garbage. **3/3 repeats, all PASS**, exact token-id comparison.
2. **GDN state rollback -- "Option A" (snapshot/restore)**:
   `DirectModelRunner.snapshot_gdn_state()`/`restore_gdn_state()`. Chosen
   over Option B (exploit chunked FLA's own chunk boundaries for a
   cheaper partial recompute) because it's simple to verify in complete
   isolation and doesn't depend on unverified assumptions about whether
   K=3 aligns safely with FLA's chunk granularity (not ruled out, just
   not attempted this round). Verified via
   `benchmarks/mtp_gdn_rollback_check.py`: a real "detour" (4 genuine
   extra decode steps) followed by restore, compared against a twin
   slot that never took the detour. **Result: `logits_exact_equal=true`
   (bytewise identical), all 48 GDN layers show 0.0 diff. 3/3 repeats,
   all PASS.**

**Not implemented this round** (tracked for its own dedicated round):
real draft generation via `Qwen3NextMTP` -- needs the draft model loaded
+ its KV cache kept in sync on every real step + the K-step propose loop
+ wiring the two verified pieces above into a real end-to-end cycle +
comparing acceptance rate against a real vLLM MTP server (the ≤1pp
gate). Reported honestly as larger in scope than initially estimated,
per the coordinator's explicit invitation to do so rather than force a
rushed finish -- the concrete design above is the starting point for
that round, not a re-investigation.

### Phase 3, MTP draft-sync evidence chain + worst-case redesign scope + pragmatic fallback proposal (2026-07-17)

Design/research-only round (explicitly no heavy GPU ops), following up
the MTP semantics round above. Full detail in
`notes/direct-model-runner-design.md`'s "2026-07-17 follow-up" section;
summary here.

**Question**: does every real target-model step truly need a synchronous
draft-model forward pass, or is there a lighter alternative (e.g. only
sync when actually about to propose)? **Answer: confirmed decisively,
no lighter alternative exists in vLLM's real implementation.** Exact
evidence chain: `vllm/v1/worker/gpu/model_runner.py:1114`'s
`execute_model()` is the single unified per-step entry point for both
prefill and decode (not two dispatch paths); `:1456-1479` calls
`self.speculator.propose(...)` right after `postprocess_sampled()`,
gated ONLY by `self.speculator is not None` (no decode-only/every-N-steps
condition); the SAME call is duplicated at `:582-623` for the dummy/
warmup run, confirming it's treated as a mandatory part of every step's
contract, not an optional side channel. Mechanism-level reason it must
be per-step: `_prepare_prefill_inputs_kernel`
(`autoregressive/speculator.py:510-519`) shifts-by-one over the FULL
current step's query range on every real step, feeding the draft model's
own attention layer a gap-free causal history — a transformer attention
layer has no valid "catch up later," every position must be written in
order or later positions attend over a hole.

> **2026-07-17 correction** (caught by the independent Codex-sol analysis
> mentioned below, verified against the checkpoint's real `config.json`
> before accepting): the draft model class this section originally
> named is wrong — see the correction note in the "MTP semantics round"
> entry above. Corrected below to `Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor`
> / `mtp_num_hidden_layers`. Point C is additionally refined per sol
> (see below): the sync call needs to be part of every round's state
> machine, but does NOT need to be duplicated into every public forward
> entry point — it can be centralized at one funnel point.

**Worst-case redesign scope** (design-level only): (A) model loading —
add `speculative_config` to `build_vllm_config`, load `Qwen3_5MTP` via
a second `get_model()` call BEFORE KV-cache allocation so its attention
layer auto-registers into the existing allocation machinery (small,
low-risk, reuses existing generic code). (B) KV sizing — draft model's
own attention layer needs the same per-slot capacity as the target's 16
full-attention layers; ~1/16 ≈ 6.25% more attention-KV memory per slot,
GDN memory unaffected (draft model has no GDN layers). (C) **the
expensive part, revised**: rather than duplicating a draft-model call
into every public forward entry point (`prefill`/`decode`/`_forward`/
`_forward_batch` individually), centralize it at the ONE point they all
already funnel through — right after the target model's own
forward+logits produce this step's hidden state — via a single internal
method every entry point calls through. This still roughly doubles real
forward-pass count per step and the CUDA-Graph-captured path still needs
it captured as part of the same graph; the change from the original
write-up is WHERE the call lives (one funnel point, not scattered), not
whether it can be skipped or made cheaper. (D) the K-step autoregressive
propose loop layers on top of C once a synced draft KV cache exists. (E)
wiring already-verified `verify_batch`/`determine_accept_reject`/
`snapshot_gdn_state`/`restore_gdn_state` into a real cycle, tracking
per-slot `committed_len`/`draft_sync_len`/pending draft tokens/
speculative-write KV range/GDN snapshot generation — comparatively
contained once C exists. **Bottom line: A-B-D-E are small/contained, C
is the real cost driver**, though C is one centralized funnel point, not
a sprawl across the whole runtime (softer than the original write-up).

**Pragmatic fallback evaluated**: self-drafting — feed K real greedy
single-token decodes (via the already-verified qo_len=1 path) as the
"draft," submit through the REAL qo_len=K+1 `verify_batch`/CUDA-graph
path exactly as production MTP would. **Gets right**: exercises the
actual production kernel shape that determines launch-gap/throughput
(this round's real open question) using 100% already-built, already-
verified infrastructure, zero new engineering. **Gets wrong, by
construction**: acceptance rate will read ~100% (the "draft" IS the
target model's own output) vs. this project's own earlier ~63-66%
measurement for real MTP on this model — must be reported as an
explicitly-labeled optimistic upper bound on accepted-tokens/s, never
conflated with a real acceptance-rate number, and does not by itself
satisfy Phase 8's "faithfully replicate vLLM's MTP" mandate.

**Sol's independent analysis returned** (see the next section for the
adopted two-phase route and this round's actual probe work): confirmed
the architecture-level conclusion (separate model, own KV cache, every
real step needs sync) but caught the wrong-class error above, refined
"restructure the whole forward path" down to "centralize at one
boundary," and recommended a strictly time-boxed trace-driven
performance probe (no real drafter) before committing to the full
faithful integration — adopted, see next section.

### Phase 3, capacity expansion to real W1/W2 scale (2026-07-17) — empirically measured, expanded, full MTP suite re-verified with zero regressions

Full detail in `notes/direct-model-runner-design.md`. Built `benchmarks/
capacity_w1w2_check.py` to measure real GPU memory via `nvidia-smi`
rather than trust a hand derivation. Findings: persistent KV cache
scales trivially with context (~34KB/token across all 17
position-dependent attention layers; GDN state is fixed-size regardless
of context length) — never the real constraint. The real open question
was peak transient ACTIVATION memory during one non-chunked 32768-token
prefill: measured at **~22GB** on top of ~39GB persistent (weights + 8
slots' KV/GDN cache). Decisive check: does memory GROW per additional
slot's sequential prefill (would mean concurrency=4 doesn't actually
fit) or PLATEAU (this runtime never batches prefills across slots, so
each one's transient memory should be freed before the next starts)?
Measured across 4 sequential W2-sized prefills: **63353 MiB flat across
all 4, zero growth**, out of 97887 MiB total (~36GB headroom). **Conclusion**:
`block_size=16` (unchanged) + `blocks_per_slot=2560` (40960 tokens/slot,
~21% margin over W2's 33792 need) works for BOTH W1 and W2 with the
SAME config — no need to split configs or reduce concurrency, contrary
to the coordinator's stated contingency; memory was never the real
constraint once measured directly.

**Full MTP correctness suite re-verified at the new capacity** (per the
coordinator's explicit instruction not to assume bigger capacity is
automatically safe — `blocks_per_slot` feeds directly into this
runtime's address arithmetic): updated all 5 MTP test scripts'
`blocks_per_slot` from 128 to 2560 and re-ran every one —
`mtp_prior_kv_len_fix_check.py`, `mtp_accept_reject_check.py`,
`mtp_gdn_rollback_check.py`, `mtp_real_draft_check.py`,
`mtp_multiround_check.py`. **All pass, byte-for-byte identical results
to the smaller-capacity runs** (same near-tie finding reproduces
identically), confirming the address logic generalizes correctly at
20x the previous `blocks_per_slot` value.

Capacity (the coordinator's item 3) is now done. Moving to the real
W1/W2 acceptance-rate comparison against native vLLM next.

### Phase 3, MAJOR CORRECTION (2026-07-17) — real structural bug in the propose loop, caught by independent Codex-sol review AFTER steps 1-6 were reported passing; fixed, re-verified, other findings triaged

**This corrects work reported as verified above (the "steps 5-6" and
Phase-2 entries below).** Before starting step 7 (W1/W2), an independent
Codex-sol review of the Phase 2 implementation came back at
REQUEST-CHANGES level. The coordinator personally re-read
`_mtp_sync_and_propose`'s source and confirmed the core finding before
relaying it — right discipline, matching this project's standing rule
of verifying an independent review's claims before acting on them.

**The bug (confirmed real, fixed)**: `_mtp_sync_and_propose`'s
exploratory propose loop passed the intentionally-frozen
`slot_draft_sync_len` as EVERY exploratory step's `prior_kv_len`, but
the real physical write position keeps advancing each iteration. For
K=3 (production), the 1st exploratory step happens to still be correct
(immediately follows step 0, where frozen and real still coincide) —
the 2nd is not: its attention metadata claimed a shorter history than
where its own K/V actually landed, so it silently failed to attend to
the previous exploratory step's own contribution. Every real K=3
proposal's 3rd draft token was computed against an incomplete causal
history. **Why steps 3-4 didn't catch this** (confirmed methodology
gap): they only checked shape/vocab-range, never per-step numerical
content — this bug produces the right shape at every step, only the
content is wrong from the 2nd exploratory step on.

**Fix**: `_mtp_forward` now takes an explicit `prior_kv_len` argument
instead of reading `slot_draft_sync_len` internally;
`_mtp_sync_and_propose` tracks its own local `running_prior_kv_len` that
advances every exploratory iteration, while the persistent field still
only advances once (after step 0) — decoupling "what this call's
attention needs" from "what cross-round bookkeeping remembers" fixes
both correctly.

**Decisive re-verification** (not another shape check), `benchmarks/
mtp_prior_kv_len_fix_check.py`: (1) white-box invariant
(`prior_kv_len == start_pos` at every propose-loop call) — zero
mismatches at K=3 and K=5, the actual decisive proof. (2) black-box
numerical demonstration, letting old-buggy and fixed semantics
propagate autoregressively side by side — at K=3 the two sequences
happened to match exactly for this specific prompt (reported honestly,
not suppressed: a single missing context token didn't flip this
position's greedy choice); at K=8 (stress) they matched for 4 tokens
then diverged completely, concrete quantified proof the bug has real
consequences once the gap compounds and that the fix eliminates them.

**Other findings from the same review, verified individually (not
accepted wholesale)**: (1) `reset_slot()` didn't clear
`slot_draft_sync_len`/`slot_pending_draft_tokens` — confirmed, an
immediate bug for any reused slot, fixed. (2) GDN snapshot generation
wasn't slot-bound and wasn't marked consumed after restore — confirmed
(a cross-slot restore could pass the generation check by coincidence),
fixed with a slot-id tag + consumed marker. (3)
`CapturedBatchDecodeGraph.replay()` unconditionally committed, no
`commit` flag, inconsistent with the eager path's new semantics —
confirmed (not yet triggering an observed failure since CUDA graph
integration into MTP accept/reject isn't wired up yet), fixed with a
matching `commit` parameter. (4) methodology gap (shape-only checks) —
confirmed, addressed for this specific bug, not retrofitted across
every existing verification step. (5) capacity
(`block_size=16×blocks_per_slot=128=2048` tokens/slot vs. W1's
4096/W2's 32768 need) — confirmed real and NOT yet fixed, the concrete
next blocker before step 7 can start.

**Regression check**: re-ran the full existing MTP test suite after all
4 fixes (`mtp_real_draft_check.py` — needed one direct-call-site update
for the new required argument, caught immediately by the signature
change; `mtp_multiround_check.py`; `mtp_accept_reject_check.py`;
`mtp_gdn_rollback_check.py`; `mtp_trace_driven_probe.py`) — all pass,
identical results to before the fixes, confirming no regressions.

### Phase 3, multi-round + 4-slot isolation (2026-07-17) — verification gradient steps 5-6 done, mtp_decode design-simplified away, one benign near-tie found and root-caused

Full detail in `notes/direct-model-runner-design.md`'s "steps 5-6"
section. Before writing either test, reasoning through what a real
multi-round loop needs eliminated the previously-planned, not-yet-built
`mtp_decode()` coordinator entirely: `mtp_verify_and_commit` already
computes everything needed (the newly-committed real tokens + the
target's own hidden states for them) to ALSO resync the draft's KV and
propose the NEXT round's K tokens in the same call, by generalizing the
EXISTING `_mtp_sync_and_propose` (previously only called from
`mtp_prefill` over "the whole prompt") to "this round's newly committed
range" instead. `mtp_verify_and_commit` now returns `next_anchor`/
`next_draft_tokens` directly; a multi-round loop is just feeding these
back in as the next call's `anchor`/`draft_tokens` -- no separate decode
coordinator, matching real vLLM's own design (propose immediately after
commit) more closely than the earlier plan.

**Step 5 (multi-round, concurrency=1)**, `benchmarks/mtp_multiround_check.py`:
8 real rounds on one slot (organic accept/reject from the draft's own
proposals, plus a forced-decoy reject every 3rd round), with an
independent reference slot replaying the same real committed tokens
every round and comparing next-token predictions -- round-by-round, not
just a final check. Bookkeeping invariant (`slot_draft_sync_len ==
slot_kv_len`) held 8/8 rounds every run. Content match was 7/8 exact;
the 1 mismatch was root-caused (not dismissed) by dumping actual logit
values: two candidate tokens were in a genuine exact tie in one kernel
path (`25.375` both) vs. a clear lead for one of them in another kernel
path (`24.25` vs `24.0`), both far above the 3rd-place candidate --
confirming a real, inherent near-tie in the model's own distribution
(the same "different kernel dispatch path, small floating-point
difference" phenomenon already documented in step 4), not state
corruption. Added a logit-margin-aware tolerance to the test
(`NEAR_TIE_LOGIT_MARGIN=2.0`, distinct real candidates are typically
8-13+ units apart here) -- **8/8 rounds pass with this tolerance,
bit-identical and stable across 2 independent full runs** (same round
index, same tokens, both times -- deterministic, not noise).

**Step 6 (4-slot concurrent isolation)**, same script: 4 different
prompts (France/Japan/Germany/Italy capitals, an eyeballable
signal-probe layer per the coordinator's explicit request for both
methods), 4 MTP slots + 4 independent reference slots, interleaved
round-robin (not 4 sequential independent runs) so a cross-slot
addressing bug would have to manifest under real interleaving. Result:
no two slots' committed sequences were ever identical (contamination
sanity check), and all 4 slots pass the numerical-twin check 100%
cleanly (zero mismatches, even at strict exact-match) across both runs.
Slot 0 (France, same prompt as step 5) reproduces the IDENTICAL
near-tie behavior whether run in isolation or interleaved with 3 other
slots -- itself evidence against cross-slot contamination, which would
plausibly behave differently depending on interleaving order, not
reproduce identically both ways.

**Scope note stated honestly**: step 6 interleaves independent
SINGLE-slot calls across 4 slots, not one batched call spanning all 4 at
once (would need generalizing the MTP coordinator methods to accept
slot lists, matching how the target-only `_forward_batch`/`verify_batch`
already do -- not done this round, a real gap for a genuinely concurrent
production loop).

**Not attempted this round**: step 7 (real W1/W2 ≤1pp acceptance-rate
gate against native vLLM -- needs a real vLLM MTP server and comparable
workload, substantially larger, better scoped as its own round) and
step 8 (CUDA graph integration, explicitly last in the gradient). Not
blocked on anything technical -- a scope/pacing call given how much real
GPU time this round already used (4 full model-load test runs + 2
targeted diagnostic runs).

### Phase 3, real MTP draft model + centralized state machine (2026-07-17) — Phase 2 / sol's "Option A", verification gradient steps 1-4 done, real bug found and fixed

Full detail in `notes/direct-model-runner-design.md`'s "Phase 2" section.
Loaded the REAL `Qwen3_5MTP` draft model via vLLM's own
`load_eagle_model()` (same function real vLLM's `MTPSpeculator` calls --
not hand-rolled), built a centralized MTP-cycle funnel
(`_mtp_forward`/`_mtp_sync_and_propose`/`mtp_prefill`/
`mtp_verify_and_commit` in `runtime/direct_model_runner.py`) so draft-sync
logic lives in ONE place rather than scattered into every entry point
(the original `prefill`/`decode`/`decode_batch`/`verify_batch` are
untouched), added the explicit per-slot state Codex-sol asked for
(`slot_draft_sync_len`, `slot_pending_draft_tokens`,
`slot_gdn_snapshot_gen` with stale-snapshot rejection), and fixed
`_forward_batch`'s physical-write-vs-committed conflation for real
(`commit: bool` param; `verify_batch` now defaults to `commit=False`).

**A real bug was found and fixed this round**: the recompute-on-reject
path fed `decision["committed"]` directly as input tokens, which is
WRONG token alignment (should be `[anchor] + committed[:-1]`, since the
anchor/greedy token has no KV entry until fed back in as a later call's
input, matching `prefill()`/`decode()`'s existing contract) — would have
silently written wrong content into the KV cache while still passing
every shape/bookkeeping check. Found via direct reasoning about
position-alignment semantics, not a test failure — then a genuine
content-correctness check (independent reference-slot replay + real
continuation, comparing next-token predictions) was added specifically
to catch this class of bug going forward.

**Verification gradient, steps 1-4, real numerical twin checks (not
signal-probe), stable across 3 independent runs** (`benchmarks/
mtp_real_draft_check.py`): (1) target prefill hidden/logits align
exactly between a plain runner and an MTP-loaded runner — loading the
draft model doesn't perturb the target. (2) `embed_tokens`/`lm_head`
sharing confirmed via `data_ptr()` identity (genuinely the same tensor);
draft's shifted first pass produces correct shape, all finite. (3) real
`mtp_prefill()` produces exactly K=3 valid draft tokens. (4) both a
real-draft-proposal scenario (draft's own actual output, whatever
accept/reject outcome results — not asserted to any specific value,
since that's the draft's own prediction quality, not this coordinator's
correctness) and a forced-reject scenario pass: kv_len tracks the real
committed length, GDN state gets repaired, and the content-correctness
cross-check agrees with an independent reference replay. All 4 steps:
**PASS**. One honestly-recorded nuance: the content check's hidden-state
`allclose` is `false` while cosine similarity is 0.997 and the greedy
token still matches exactly — attributed to two different kernel
dispatch paths (prefill-shaped vs. decode/verify-shaped) computing the
same math, consistent with this project's established practice of using
cosine-sim + exact greedy-token agreement as the correctness bar, not
literal hidden-state allclose.

**Not attempted this round** (remain for future rounds): step 5
(multi-round continuation, needs a not-yet-built `mtp_decode()`
coordinator), step 6 (4-slot concurrency isolation), step 7 (the real
W1/W2 ≤1pp acceptance-rate gate against native vLLM), step 8 (CUDA graph
integration — last in sol's gradient, untouched).

### Phase 3, trace-driven scheduling-overhead probe (2026-07-17) — sol's Phase 1, GPU-busy% ~100% found and confirmed stable across 2 runs

Full detail in `notes/direct-model-runner-design.md`'s "second follow-up"
section. Independent Codex-sol analysis returned this round; two things
accepted after independent re-verification (checked the real
checkpoint's `config.json` + vLLM source directly, not taken on trust):
(1) the wrong-draft-model-class error in the prior round's sections
(corrected there in place — `Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor`
from `qwen3_5_mtp.py`, not `Qwen3NextMTP`), (2) the "must sync every
step" conclusion refined to "must be part of every round's state
machine, but the call site can be centralized at one funnel point, not
scattered into every public forward entry point."

Built and ran `benchmarks/mtp_trace_driven_probe.py` (sol's recommended
"方案C", strictly time-boxed, this round's actual scope): NO real
drafter — a synthetic, seeded accept/reject trace (K=3 Bernoulli(p)
per-token trials) drives the real, already-verified `verify_batch`
(qo_len=4, concurrency=4), `snapshot_gdn_state`/`restore_gdn_state`, and
a real committed-length recompute forward on any non-full-accept round —
measuring ONLY control-plane/scheduling overhead via CUDA-event
GPU-busy-time vs. wall-clock time. GPU/process state confirmed clean via
`nvidia-smi`/`ps` before each run, per standing discipline.

**Result, stable across 2 independent runs (fresh process each time,
not cherry-picked)**: GPU-busy% is ~98-101% (indistinguishable from
100% within measurement noise) across ALL THREE tested configs
(p=1.0 best-case/never-reject, p=0.65 realistic-shape, p=0.0
worst-case/always-reject) — i.e. essentially ZERO launch-gap/scheduling
overhead in this runtime's own eager-mode call sequence, regardless of
how much rollback/recompute work each round does.

**Interpretation**: for this workload's shape (64-layer/27B/batch=4/
qo_len=4 forward), real GPU compute (100-420ms/round) already dwarfs any
Python dispatch cost — this runtime's OWN call path has no further
launch-gap to squeeze via CUDA graphs at this granularity. This does NOT
undercut the project's premise: the overhead being targeted lives in
native vLLM's Python scheduler/block-manager/HTTP layer, which this
minimal runtime never had in the first place (by construction) and which
this probe doesn't measure at all — the real answer still needs the
actual W1/W2 vs. native comparison (task #85, still blocked on real
draft-model integration). The useful takeaway: since our own residual
overhead is already ~0%, any gap that comparison finds is a fully
capturable win, not partially eaten by our own dispatch cost — a
positive signal for continuing the Phase 2 investment. Caveat stated
honestly: `_forward_batch`/`verify_batch`'s forced double
`torch.cuda.synchronize()` per call structurally prevents observing any
cross-call async pipelining — the kind of gain CUDA graphs mainly help
with tends to matter for SMALL, decode-shaped calls, not this
large/compute-dominated verify shape, consistent with "capture graph"
being the last step of sol's verification gradient, not an early one.

**Scope discipline**: per the coordinator's explicit "这一轮先做探针(阶段1)…
严格限时" instruction, Phase 2 (loading real `Qwen3_5MTP`, building the
full centralized incremental MTP state machine) was NOT started this
round, despite sol's overall recommendation to move to it immediately
regardless of phase 1's result — that describes the recommended route
across rounds, not a mandate to compress both into one round. Phase 2
remains the explicit next step.

### Phase 3, main-line redirect (2026-07-16) — direct model runner, replacing the HTTP bridge

The sibling `sm120-flash-attention` project's attention-kernel-tuning main
line hit diminishing returns today (decode v2/prefill v2's "beats native"
claims were both overturned, and the final split-KV hypothesis was falsified
too), so main development effort moved to this project. Per the new
direction: remove the HTTP bridge to a separate vLLM server
(`runtime/vllm_bridge_backend.py`, commit `b28942c`) and have this runtime's
own process directly own the GPU KV/GDN state for 4 fixed slots, reusing
existing kernels (FlashInfer NVFP4 GEMM, sm120-flash-attention's decode
v2/prefill v2, vLLM's own GDN implementation) rather than reinventing them.

**Design**: `notes/direct-model-runner-design.md` -- the concrete mechanism
(reusing `EngineArgs.create_engine_config()`, `get_model()`,
`bind_kv_cache()`, `set_forward_context()`, all real vLLM primitives, none
reimplemented), per-slot KV/GDN tensor layout, and hand-built
per-request attention/GDN metadata for this round's single-request scope.

**Implementation**: `runtime/direct_model_runner.py` -- loads the real
`unsloth/Qwen3.6-27B-NVFP4` model in-process (no separate server, no HTTP),
allocates and binds real per-slot KV cache (attention) and state (GDN
conv/ssm) tensors, and drives `model.forward()` directly for prefill/decode.

**Status: runs end-to-end without crashing, but output is INCORRECT --
not yet a working closed loop.** This is reported honestly, not glossed
over: the point of this round was exactly to verify correctness under direct
GPU-state ownership, and that verification found a real, unresolved bug.
Concrete findings (full detail in the design doc's "Current state" section):
- Found and fixed one real bug already: `ForwardContext.slot_mapping` (a
  field *separate* from `attn_metadata`) was never populated, so KV-cache
  writes silently never happened at all. Fixed; KV cache now genuinely gets
  written.
- Ruled out: KV cache dtype mismatch, FP8 default-scale-quality (the
  HTTP-bridge round used identical FP8-KV settings and got correct output),
  and `positions`/mrope shape.
- **Chased the `conv_state`-all-zero lead to ground per the coordinator's
  direction (2026-07-16, later same day)**: confirmed the conv1d's own
  input is real/non-degenerate (not the problem). Then found, in complete
  isolation (a ~30-line script calling vLLM's `causal_conv1d_fn` directly,
  no model, no runtime code involved) that **the first-ever call to this
  Triton kernel in a process silently returns an all-zero result**; every
  later call at the same shape is correct. This is a genuine, reproducible
  bug, independent of anything this project wrote. Added a `_warmup()` step
  (mirrors real vLLM's own pre-serving warmup pass) to work around it.
  **However: the warmup fix did NOT change the real model's wrong output**
  (tried both a 1-token and a shape-matched 5-token dummy warmup; identical
  wrong completion both times). Follow-up isolated tests show the bug is
  messier than "first call bad, rest fine" -- interleaving different
  prompt-length shapes did not self-correct the way repeating one shape
  did, so there is some additional, not-yet-characterized state at play.
  Separately, and still unresolved regardless: `conv_state` remained zero
  even in isolated calls whose *output* was otherwise fully correct --
  likely two distinct issues, not one. Full blow-by-blow, including the
  next specific debugging steps, is in the design doc's "deep dive" section
  -- this is genuine, reportable progress (a real bug, isolated and
  partially characterized), not a dead end, but it is **not yet fixed**.
- **Third pass (2026-07-16, same day, following the coordinator's exact
  3-step order): a material revision, not a fix.** (1) An unrelated Triton
  kernel run first does NOT warm up `causal_conv1d_fn`'s first call --
  worse, it breaks every subsequent call too, ruling out "any GPU kernel
  primes a global CUDA/Triton state." (2) Instrumented all 48 real GDN
  layers within one actual forward pass: in one run, **every single
  conv1d call was fully correct (non-zero) throughout the whole model** --
  yet the final generated token was still wrong. This means the isolated
  cold-start bug found earlier is likely **not, by itself, the cause of
  the wrong output**. (3) A near-identical rerun with only *additional
  read-only* instrumentation (checking `conv_state` before/after) flipped
  to all-zero throughout -- a classic Heisenbug signature (behavior
  changes under observation), pointing to a real race condition somewhere
  in this stack rather than a deterministic missing-parameter bug. Tried
  disabling `async_scheduling` + explicit `torch.cuda.synchronize()` as a
  race-motivated fix -- did not change the (still wrong) output. **The
  actual root cause of the wrong "Paris" answer remains unidentified** --
  next step is `compute-sanitizer --tool racecheck` on a minimal repro
  rather than continued print-based bisection, since instrumentation
  itself has now been shown to change the outcome. Full detail in the
  design doc's "third pass" section.
- **Fourth pass (2026-07-16, same day): ran racecheck, found a real
  specific hazard, then disproved it as the root cause -- decisive but
  still not a fix.** `compute-sanitizer --tool racecheck` on the minimal
  single-prefill repro found **100 consistent "Potential RAW hazard"
  reports**, every one in the *same* kernel and thread pair: CUTLASS's
  SM120 warp-specialized "pingpong" GEMM (`cutlass_scaled_mm_sm120`, used
  by `CutlassFP8ScaledMMLinearKernel` for one of GDN's FP8 W8A8-quantized
  linear projections), Write Thread 63 racing Read Thread 128 across many
  shared-memory tiles. This is a real, specific, reproducible localization
  -- and it directly rules out a `direct_model_runner.py`-level
  synchronization bug (the race is between two CUDA threads *inside one
  kernel launch*; nothing at the Python orchestration level can reach
  intra-kernel warp synchronization). One caveat: TMA/mbarrier-synchronized
  warp-specialized kernels are a documented source of racecheck false
  positives, so this alone doesn't prove a genuine CUTLASS bug.
  **Immediately tried the obvious bypass**: `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel`
  forced a fallback to a plain PyTorch FP8 kernel (confirmed via log:
  "Selected ChannelWiseTorchFP8ScaledMMLinearKernel"). Result: **the output
  changed (proving this kernel matters) but is still wrong** (still not
  "Paris", just differently wrong) -- **decisive evidence this specific
  race, real as it is, is not the (sole) root cause**. After four full
  passes, two independent, real, specific low-level findings have been
  surfaced (the conv1d cold-start bug, this CUTLASS race) and both have
  been shown to be real but insufficient to explain the wrong output alone
  -- suggesting multiple independent issues in this unusual
  direct-forward-pass usage pattern, not one single root cause. Full
  detail, including exact repro commands, in the design doc's "fourth
  pass" section -- **this is the point to decide, with the coordinator,
  whether to keep root-causing at this depth or pivot to a more
  conservative strategy** (e.g. a correctness-first baseline that doesn't
  bypass vLLM's own scheduler/executor for the fragile parts, even at a
  performance cost).

**Do not read this as "single prefill+decode achieved."** The mechanism
(model loading, KV/GDN tensor ownership, metadata plumbing) is real,
substantial, verified-working infrastructure; the actual *output* is still
wrong, and shipping this as a claimed milestone would misrepresent that.

### Phase 0 — Baseline contract

- Frozen W1 (4K input / 1K output) and W2 (32K / 1K) workloads for
  concurrency 1 and 4.
- Added the baseline record template in `notes/phase-0-baseline.md`.
- Verified CUDA execution on RTX PRO 6000 Blackwell (SM120) with
  PyTorch 2.11.0+cu130.

### Phase 1 — Correctness oracle

- Added golden fixture definitions, numerical comparison metrics, and a
  read-only PyTorch forward-hook capture utility.
- Oracle captures are detached to CPU and can be saved as safetensors without
  modifying the local, dirty `~/vllm` checkout.

### Phase 2 — Loader and packing prerequisites

- Validated the Unsloth NVFP4 checkpoint: 5 shards, 1,968 indexed tensors,
  168 packed NVFP4 tensors, and matching safetensors headers/scales.
- Added config validation for the 64-layer topology: 16 full-attention and
  48 GDN layers.
- Added on-demand, index-directed tensor reading; no full checkpoint load is
  performed by metadata tools.

### Phase 3 — Eager control plane

- Implemented fixed four-slot lifecycle management, hybrid KV/GDN cache
  metadata, and prefill/decode request control flow.
- All slots retain stable logical addresses; release/reset increments the slot
  generation to prevent stale state reuse.

### Phase 3 continued — first real (non-mock) model backend

- Added `runtime/vllm_bridge_backend.py`: the first real `prefill`/`decode`
  `OpRegistry` implementation. It does not yet drive model layers directly
  (see "Real scope of this bridge" below) -- it drives the real
  `unsloth/Qwen3.6-27B-NVFP4` checkpoint (16 full-attention + 48 GDN layers)
  through a real, isolated vLLM server (the same
  `sm120-flash-attention/vllm_integration/launch_test_server.py` this
  machine already uses for kernel validation) over its OpenAI-compatible
  HTTP API. This matches `项目实施规划.md`'s own phased-rollout intent:
  "每个算子最开始可以调用 vLLM/FlashInfer/torch，之后逐个替换成自研 kernel."
- Added `benchmarks/real_forward_smoke.py`, a live-server integration script
  (not part of `pytest -q` -- AGENTS.md's testing guideline says unit tests
  must not require downloading weights or a live server) covering:
  1. single prefill + short decode,
  2. continuous generation (tested at 48 tokens; contention-bounded, see
     below -- the mechanism is identical at any length),
  3. four real concurrent requests (capacity=4) with distinct marker codes,
     checking each recovers only its own code,
  4. release one of four slots, submit a new request, confirm the freed
     physical slot is reused with `generation + 1` and produces the new
     request's own content with zero leakage from any prior occupant (the
     vLLM issue #37554 class of risk this project's root CLAUDE.md flags:
     stale GDN dummy-forward state leaking into a reused slot).
- All four ran against the real model on 2026-07-15 and passed:
  ```
  1) "The capital of France is" -> " Paris...\nThat is correct. Paris is the
     capital and largest" -- PASS
  2) 48 real tokens generated, coherent text -- PASS
  3) slot-0.."falcon-9182", slot-1.."harbor-3305", slot-2.."cinder-7716",
     slot-3.."meridian-6640" -- each recovered only its own code, zero
     cross-slot leakage -- PASS
  4) freed slot_id=0 reused with generation 0->1; new request recovered its
     own code "thistle-5540" with zero leakage from any of the four prior
     occupants' codes -- PASS
  ```
- Decode v2 (this machine's fastest attention kernel, `+8.6%` over native
  FlashInfer end-to-end in `sm120-flash-attention`) was exercised for real:
  the server was launched with `SM120_GQA_USE_V2_DECODE_KERNEL=1` and
  `SM120GQABackend` registered under `--attention-backend CUSTOM`, confirmed
  from the server log (`registered AttentionBackendEnum.CUSTOM ->
  ...sm120_gqa.SM120GQABackend`, once for the parent, once for the spawned
  EngineCore child). Prefill v2 is not yet wired into `SM120GQABackend` as of
  this run (confirmed via `git log` on the other project before starting --
  only decode v2 is registered there), so prefill still used the old kernel;
  re-run once prefill v2 lands there.

#### Real scope of this bridge (what it does and does not prove)

This is a deliberate, honestly-scoped first increment, not the full Phase 3
target from `项目实施规划.md`. What it proves: the control plane
(`EagerEngine`/`HybridCache`/`FixedSlotManager`) drives a **real** 64-layer
hybrid model correctly through prefill, decode, continuous generation, and
slot release/reuse, with no mocks anywhere in the loop. What it does **not**
yet do: our `HybridCache` does not own the physical GPU KV/GDN state
addresses -- vLLM's own engine still owns those internally. Getting there
requires extracting vLLM's `Qwen3_5Model`/`qwen_gdn_linear_attn.py` layers
(`vllm/model_executor/models/qwen3_5.py`, 642 lines;
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, 1828 lines)
and driving their `forward()` directly against our own slot-addressed
buffers, bypassing vLLM's scheduler/`KVCacheManager`/`Mamba2Metadata`
construction entirely. No existing vLLM test or example does this at real
model scale (checked `tests/model_executor/test_qwen3_5_quantization.py`:
mocked-config unit tests only, not a real-weights harness) -- this is real,
substantial follow-on engineering, not a short next step. Also not yet done:
`--return-tokens-as-token-ids` (the bridge currently reconstructs token ids
locally via `transformers` tokenizer round-trip on `max_tokens=1` text
fragments -- correct in every run observed today, but see the design note in
`vllm_bridge_backend.py` for the residual BPE-boundary risk).

#### Cross-fork GPU contention encountered and root-caused

Requests intermittently stalled 12-27s (a single-token completion should be
near-instant). Root-caused to a **different, parallel task in the sibling
`sm120-flash-attention` project** running `compute-sanitizer --tool
memcheck` against its own kernel test (`test_prefill_v2_paged.py`) on the
same single GPU -- `compute-sanitizer` is well known to serialize/slow GPU
access 10-50x, and this machine has exactly one GPU. Not a bug in this
runtime or the bridge. The isolated test server (port 8100) was stopped
(`stop_test_server.py`, confirmed via
`nvidia-smi --query-compute-apps` returning empty) immediately after the
four correctness checks passed, to free the GPU for that other task's own
decisive benchmark -- **nsys full-model time-breakdown (attention/GDN/NVFP4
GEMM/launch-gap attribution) was deferred for this reason** and needs a
dedicated GPU window (a fresh `nsys launch`-wrapped server start; nsys
cannot attach retroactively to an already-running server).

### Phase 3 continued — full-model nsys time breakdown (2026-07-15, later same day)

Captured with `nsys launch --session-new=... --trace=cuda,osrt --cuda-graph-trace=node --
python launch_test_server.py ...` (the isolated test server, decode v2 live via
`SM120_GQA_USE_V2_DECODE_KERNEL=1`), then `nsys start`/`nsys stop` wrapped around
`real_forward_smoke.py` tests 1+2 (one real prefill + ~46 real single-token
decode steps against the actual model). Trace:
`sm120-flash-attention/vllm_integration/profiles/qwenruntime_full_model_20260715_182606.nsys-rep`
(11 MB, kept on disk).

**GPU kernel time, by category (91,727 kernel launches, 904.76 ms total GPU-busy time):**

| Category | Share | Time |
|---|---:|---:|
| GEMM (NVFP4/FP8 linear layers -- QKV/MLP/o_proj/lm_head) | **76.0%** | 687.2 ms |
| Other (norm/elementwise/copy/misc fused epilogues) | 8.8% | 79.4 ms |
| GDN (48 linear-attention layers: chunk/conv/delta-rule kernels) | 8.0% | 72.5 ms |
| Sampling / logits / argmax | 3.7% | 33.1 ms |
| NVFP4/FP4 quant-dequant (Triton fused epilogues, separate from the GEMM itself) | 2.0% | 18.1 ms |
| Attention (16 full-attn layers, incl. decode v2) | 1.5% | 13.3 ms |
| KV-cache / paged-attention scheduling infra | 0.1% | 1.2 ms |

**GEMM dominates by a wide margin -- nearly 10x the next category.** This is
consistent with this project's own established finding (see root CLAUDE.md's
validation-methodology notes and the Phase 3 NVFP4 P-requantization work)
that quantize/dequantize cost dominates over raw matmul on this hardware;
here it shows up at the whole-model level, not just inside one attention
kernel. Attention's 1.5% share is genuinely small in context -- decode v2's
prior +8.6% end-to-end win was won by optimizing a category that was never
going to move overall model latency by more than a couple of percent on its
own; the ceiling for *any* further attention-kernel work is bounded by this
1.5%, GDN's ceiling is bounded by 8.0%, and GEMM's is not meaningfully bounded
at all by comparison.

**CPU/launch gap: measured, but not at production scale -- flagged, not
resolved.** GPU kernel busy time (904.76 ms) covers only ~14.7% of the
captured wall-clock span (6.14 s); most of the remainder is real inter-kernel
gaps (91,613 gaps under 2 ms each still sum to 4.36 s, dwarfing the 58 gaps
over 2 ms that sum to 0.88 s -- so this is pervasive small per-kernel
dispatch overhead, not a few big stalls). This capture used the correctness
smoke test's workload: **single request, batch=1, qo_len=1 decode, no MTP**
-- the worst case for launch-overhead-to-compute ratio, since real
production traffic (concurrency=4, MTP K=3 -> qo_len=4 verify batches) does
more GPU work per kernel launch, which should shrink this ratio. This number
should **not** be read as "vLLM's real production CPU/launch gap is ~85%" --
it is an upper bound from an unrepresentative micro-workload, and needs a
dedicated concurrency=4/MTP-shaped capture before it can settle
`项目实施规划.md`'s Phase 0 gate ("如果 CPU/launch gap <3%，不优先重写 C++ scheduler").

**MTP verify/acceptance: not measured this round.** This capture deliberately
avoided `--with-mtp` to keep the bridge's token-bookkeeping simple and
reliable for a first real capture. No data exists yet on MTP's own kernel
share or CPU overhead from this runtime.

## Verification

- `./.venv/bin/python -m pytest -q`: 27 passed.
- `./.venv/bin/ruff check .`: passed.
- `./.venv/bin/python tools/verify_cuda.py`: SM120 CUDA tensor smoke test
  passed.
- `./.venv/bin/python -m benchmarks.real_forward_smoke`: all 4 real-server
  checks passed (2026-07-15, see above).

## Environment and Current State

- Project-local `.venv` contains the CUDA runtime, PyTorch, safetensors, and
  vLLM Python dependencies. It is ignored by Git.
- A separately managed `~/.venvs/vllm` runs local vLLM `0.25.1.dev0` when
  needed. Do not modify its source checkout.
- No vLLM process is currently active (isolated test server stopped after
  this round's verification; GPU confirmed free via
  `nvidia-smi --query-compute-apps`).
- Added `requests`/`transformers` as an optional `serving` dependency group
  in `pyproject.toml` for the bridge backend and smoke script.

## "下一刀切哪" -- revisited against the real nsys data (2026-07-15)

The candidate list was 48-layer GDN fusion / NVFP4 GEMM weight layout / MTP
verify-acceptance / CPU launch-scheduling overhead. Against the real
kernel-time breakdown above:

- **NVFP4 GEMM / weight layout: confirmed as the clear top priority.** 76.0%
  of GPU kernel time, ~10x every other category combined. Nothing else comes
  close to this ceiling. This is now a data-backed conclusion, not a guess.
- **48-layer GDN fusion: real, but demoted below GEMM.** 8.0% of GPU kernel
  time -- a genuine opportunity (today's chunk_fwd/gated_delta_rule/conv/etc
  are 9 separate kernel launches per layer), but its total ceiling (even a
  100% reduction) caps out at 8% of whole-model GPU time. `项目实施规划.md`'s
  own Phase 6 gate ("如果 profiling 证明 GDN 是主要时间占比，这一阶段的优先级
  高于 NVFP4 attention") does not fire here: GDN is not the majority, GEMM is.
- **MTP verify/acceptance: still an open question, not addressed by this
  capture.** This round's trace deliberately avoided `--with-mtp`; there is
  currently zero data on its kernel share or CPU overhead from this runtime.
  Needs its own dedicated capture before it can be ranked.
- **CPU launch/scheduling overhead: real signal, but from an
  unrepresentative single-request/batch=1/no-MTP workload** (see above) --
  neither confirmed nor ruled out as the top-line bottleneck at production
  concurrency. Given GEMM's raw kernel-busy-time dominance holds even before
  factoring in any launch gap, it does not currently outrank GEMM as the
  next cut, but deserves a concurrency=4/MTP-shaped re-measurement rather
  than being dismissed outright.

## Next Work, in priority order (real time/complexity estimate per item)

1. **NVFP4 GEMM / weight layout work** -- now the data-confirmed top
   priority (76.0% of GPU kernel time). Concretely: profile which specific
   GEMM shapes dominate (QKV/MLP-gate-up/MLP-down/o_proj/lm_head -- the
   kernel-name-level `cuda_gpu_kern_sum` report already distinguishes several
   `cutlass::device_kernel` variants by shape/instance-count, a next capture
   should break these out by call site), then evaluate `项目实施规划.md`'s
   own Phase 7 priority order (input-proj > MLP gate-up > MLP down > o_proj
   > MTP proj > lm_head) against those real shapes before picking one.
2. **A concurrency=4/MTP-shaped nsys capture** to settle the CPU/launch-gap
   question and get real MTP verify/acceptance kernel-time data -- both
   currently open from this round's single-request/no-MTP capture. Needs (2)
   below first (today's bridge cannot easily drive 4 real concurrent
   MTP-shaped requests through the HTTP text API without a lot of extra
   bookkeeping; `real_forward_smoke.py`'s existing 4-slot test proves the
   mechanism but at max_new_tokens=1, not a real MTP verify batch).
3. **Own the physical GPU KV/GDN state addresses directly** (the real
   remaining Phase 3 target) -- extract vLLM's `Qwen3_5Model` and
   `qwen_gdn_linear_attn.py` layer modules and drive their `forward()`
   directly against our `HybridCache`'s slot-addressed buffers, replacing
   today's HTTP bridge. This is substantial (multi-session) engineering, not
   a quick follow-on -- see "Real scope of this bridge" above for the exact
   files and why no existing vLLM harness shortcuts it.
4. **48-layer GDN fusion** -- real but secondary (8.0% ceiling); worth doing
   once GEMM work is underway, not before.
5. **Wire prefill v2 in** once it lands in `SM120GQABackend` on the
   sibling project (only decode v2 is registered there as of this round).
6. Re-run `real_forward_smoke.py`'s continuous-generation check at the full
   256-1000 token range once (2) or (3) removes the current bridge's
   per-step full-HTTP-roundtrip cost (today's design re-sends the whole
   growing prefix as text each step; prefix caching should make this cheap
   GPU-side, but the Python/HTTP/tokenizer overhead per call is real and
   was not separately measured this round).
