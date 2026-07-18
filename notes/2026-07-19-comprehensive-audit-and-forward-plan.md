# 2026-07-19 Comprehensive audit + forward plan (independent cold review #2)

Status: independent, skeptical audit + a genuinely fresh forward assessment.
Scope: a from-source re-audit of `qwen-sm120-runtime` after the entire
2026-07-18 review's prioritized plan (Phase A, Phase B, D1, D2, D3) was
executed to completion across `notes/2026-07-18-session-review-and-next-steps.md`
sections 10-21 -- plus my own independent judgment on what (if anything) is
worth doing next now that the gap picture has shifted substantially (the
runtime now *beats* native at several shapes it previously lost at).

Method: `runtime/direct_model_runner.py` read **in full** (3960 lines, up
from the 3387 the prior review read -- +573 lines of chunked prefill,
ragged prefill, and mid-flight-admission plumbing landed since). Every one
of the 7 correctness suites re-run fresh from clean processes. The 4K/c=4
headline re-measured (3 reps). Every substantive code claim below carries a
`file:line` citation, re-derived from the current source, not relayed from
the session's own write-ups. GPU/process hygiene (`pgrep -af`,
`nvidia-smi`) verified idle before starting (2332 MiB baseline, 0-1% util,
no compute apps) and before every run.

---

## 1. Bottom line

1. **The single most important finding is reassuring on code, but there is
   exactly ONE genuinely open correctness item, and the code itself reports
   it honestly.** `benchmarks/mtp_async_arrival_check.py` -- the mid-flight
   slot-admission driver built in §21.2 -- returns `passed: false` (exit 1)
   by design, because at the one round whose batch composition mixes two
   long-running slots with two freshly-admitted ones it hits a 7.9375-logit
   reference divergence (far above `NEAR_TIE_LOGIT_MARGIN=2.0`). This is
   the only `passed`-gated script in the tree that does not go green, and
   it is the correct top priority. It was investigated (self-healing in one
   round, decoded to a "continue vs. break a repeated phrase" near-tie, no
   cross-slot addressing bug found) but NOT closed to the project's own
   Phase-A end-to-end-generation-quality bar. Details §3.4 / §4.1.

2. **The code is clean and internally consistent after all the layering.**
   Full read of the now-3960-line `direct_model_runner.py`: no dead code,
   no scaffolding (`grep` for `print(`/`TODO`/`breakpoint`/`pdb` returns
   zero), the D3 grad-disable fix is present (`:872`), the removed
   `_const_gdn_extra`/`TARGET_SPLITS` constructs survive only in
   explanatory comments, and the four features layered on this session
   (chunked prefill, ragged prefill, mid-flight admission, the spec-decode
   GDN mechanism) interact through a small number of **shared, already-
   verified primitives** rather than parallel copies. The one bootstrap
   invariant every path depends on (`slot_num_accepted_tokens[s] = 1`) is
   set consistently across all three prefill branches. Details §3.1-3.3.

3. **Fresh regression battery: all 7 suites PASS** (§3.2), re-run by me
   from clean processes, not relayed. **Fresh 4K/c=4 headline: my 3-rep
   mean is 162.638 accepted tok/s** (162.482 / 162.861 / 162.571 -- a
   remarkably tight 0.38 tok/s spread) (§3.3), with
   `total_committed_tokens=4116` / `draft_acceptance_rate_pct=70.29204431017119`
   bit-identical to every prior measurement in project history. This sits
   near the top of the doc's own ~142-166 range and is ~1.13x *faster* than
   native's 144.54.

4. **One real, precisely-bounded untested-in-combination gap** (§3.4):
   chunked-prefill + CUDA-graph was exercised (§20) and is sound by a
   disjoint-code-path argument. But **ragged-prefill + CUDA-graph** and
   **mid-flight admission (a *growing* batch composition) + CUDA-graph**
   have never been exercised together -- every ragged/admission test runs
   `enable_cudagraph=False`. The disjoint-path argument covers most of the
   risk (prefill is always eager), but the growing-batch-composition ×
   graph-lookup interaction is genuinely novel. This is an extension of the
   already-flagged §21.2 open item, not a new surprise.

5. **Working tree is clean; the three previously-flagged stray files
   (`README.md`, `benchmarks/phase0_nsys_gap_ledger_diag.py`,
   `项目实施规划.md`) are all committed** (§3.5). No new stray changes.

6. **Forward assessment (§4):** the D1/D2/D3 sweep is genuinely complete and
   the "chase the last few %" question is still correctly closed. The two
   pieces of *new* value are (P0) closing the §21.2 admission numerical
   finding to the Phase-A bar, and (P1) a single **realistic coding-agent
   workload** end-to-end test that would simultaneously (a) validate the
   actual production target, (b) naturally exercise the untested
   ragged/admission × cudagraph combinations, and (c) serve as the
   multi-hour stability probe the session never ran. The 16K/c=4 residual
   (1.814x) is a genuinely different (compute-bound) class and is **not**
   worth chasing with the same techniques that closed 64K/c=4.

---

## 2. What I reviewed and how

- Read `runtime/direct_model_runner.py` in full (3960 lines) --
  reconciled against `notes/2026-07-18-session-review-and-next-steps.md`
  §§10-21, `notes/2026-07-17-post-ragged-round-next-steps.md`, and
  `PROGRESS.md`'s recent sections.
- Traced every claimed "still-live" retained primitive
  (`snapshot_gdn_state`/`restore_gdn_state`, chunked `verify_batch`) to its
  real callers via `grep` across `benchmarks/`.
- Re-ran all 7 correctness suites fresh, one process at a time, GPU
  verified idle before each.
- Re-ran the 4K/c=4 W1-S headline (`--batched --cudagraph --repeats 3`).
- Re-ran the mid-flight-admission driver (`mtp_async_arrival_check.py`) to
  confirm its `passed: false` is reproducible, not stale.
- Verified `git status` clean and the three named files tracked.

---

## 3. Part 1 -- audit findings

### 3.1 Code cleanliness and the "no dead code" claim, re-verified

- **No scaffolding.** `grep -nE '\b(print\(|breakpoint\(|pdb\.|TODO|FIXME|XXX|HACK)\b'`
  over `runtime/direct_model_runner.py` returns zero non-comment hits.
- **D3 grad fix present and correct.** `torch.set_grad_enabled(False)` is
  the first statement in `DirectModelRunner.__init__` (`:872`), before any
  model construction. `grep` for `grad` finds only this line plus its
  explanatory comment (`:847-871`).
- **Removed constructs leave no live references.** `_const_gdn_extra` and
  `TARGET_SPLITS` appear only inside comments describing their removal
  (`:3259`, `:3312`, `:3497`); the live split-KV constant is
  `_DECODE_TARGET_SPLITS_PER_REQ = 64` (`:1027`).
- **The retained "legacy" primitives are genuinely still live, not dead.**
  `snapshot_gdn_state` (`:1760`) / `restore_gdn_state` (`:1858`) are called
  by 6 benchmark files (`mtp_gdn_rollback_check.py`,
  `mtp_batch_divergence_diag.py`, `mtp_real_draft_check.py`,
  `mtp_trace_driven_probe.py`, `mtp_slot_identity_pinpoint_diag.py`,
  `phase0_nsys_gap_ledger_diag.py` -- confirmed by `grep`). The chunked
  `verify_batch` (`:1653`) is still called by `decode_batch` (`:1650`) and
  several diagnostics. **No production verify path calls snapshot/restore
  any more** -- confirmed: `mtp_verify_and_commit` (singular, `:2100`) now
  calls `verify_batch_spec` (`:2198`) with no snapshot/restore/recompute,
  exactly as §17 (Phase B) claimed.

### 3.2 Fresh regression battery -- all 7 PASS (re-run, not re-reported)

Every suite re-run from a clean process, `CUDA_HOME`/`PATH` pinned to 13.3,
venv `/home/bot/.venvs/vllm`, GPU verified idle before each:

| Suite | Invocation | Result |
|---|---|---|
| `mtp_gdn_rollback_check.py` | `--repeat 3` | **3/3 PASS** (bytewise GDN-state restore across 48 layers) |
| `mtp_batch_verify_check.py` | (default) | **PASS** -- `check0..check3` all true, `no_cross_contamination_signal: true` |
| `mtp_ragged_recompute_verify_check.py` | (default) | **PASS** -- all 3 sub-checks true (incl. `check1_ragged_recompute`, `check2_mixed_ragged_and_full_accept`) |
| `mtp_verify_cudagraph_check.py` | (default) | **PASS** -- all 4 replay-coverage flags true (`verify_graph_batch4_replayed`, `verify_graph_batch2_replayed`, `draft_step0_qo2_graph_replayed`, `draft_continuation_graph_replayed`) |
| `mtp_chunked_prefill_check.py` | (default) | **PASS** -- check0/1 (`gdn_all_sane: true`), check2 (singular ref), check3 (`any_value_mismatch_on_shared_prefix: false`), check4 (natural-language prompt) all true |
| `mtp_ragged_prefill_check.py` | (default) | **PASS** -- check0 (`all_anchors_ok_within_tolerance: true`, `gdn_all_sane: true`), check1 (`no_cross_contamination_signal: true`) |
| `mtp_async_arrival_check.py` | (default) | **`passed: false` (exit 1) -- BY DESIGN**, reproduces the §21.2 finding, see §3.4/§4.1 |

The first six are the canonical regression battery this project runs after
every production change; all green. The seventh is the mid-flight-admission
driver, whose `passed: false` is the honest signal on the one open item --
see §3.4.

### 3.3 Fresh headline re-measurement

`python -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph
--repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`, fresh process,
GPU verified idle before:

| Rep | accepted tok/s | draft accept % | committed toks | temp start->end |
|---|---:|---:|---:|---:|
| 1 | 162.482 | 70.29204 | 4116 | 56C -> 67C |
| 2 | 162.861 | 70.29204 | 4116 | 67C -> 71C |
| 3 | 162.571 | 70.29204 | 4116 | 71C -> 73C |
| **mean** | **162.638** | 70.29204 | 4116 | -- |

The 0.38 tok/s spread across my three reps is far tighter than the ~24
tok/s spread the session saw *across rounds* (142.504 in §11.7 to 165.730
in §19.5) -- because my reps ran back-to-back in one warm process (clock
pinned at 2250 MHz for reps 2-3 as temperature climbed 56C->73C), whereas
the cross-round spread mixed cold and warm process starts. This is itself
corroboration of the thermal explanation below: within one thermal regime,
the number is stable to <0.3%.

`total_committed_tokens = 4116` and `draft_acceptance_rate_pct =
70.29204431017119` are bit-identical across all 3 reps and to every prior
measurement in project history -- the strongest within-runtime determinism
cross-check available, confirming zero change to generation correctness.

**On the doc's own thermal caveat**: the session's own reported means for
this exact command drifted across rounds -- 142.504 (§11.7) -> 147.656
(§13.7) -> 148.193 (§17.5) -> 147.931 (§18.2) -> 165.730 (§19.5) -> 156.939
(§21.3) -- with the code path provably byte-identical (`chunk_size`
defaults `None`, uniform fixture) and acceptance/commit counts bit-
identical throughout. The doc attributes the spread to Max-Q thermal state
(§19.5 noted a cool 53C/2272MHz run; §21.3 noted a 70-73C run). **My read:
this is a credible explanation, and it is corroborated, not merely
asserted** -- the correctness signals being bit-identical across the entire
spread rules out any code/determinism cause, so the residual variance can
*only* be environmental (thermal/clock) on a documented thermally-limited
part. The honest consequence is that the 4K/c=4 headline should be quoted
as a *range* (~142-166 tok/s depending on thermal state), all of which sit
at or above native's 144.54, i.e. parity-or-better; a single-run "X tok/s"
headline over-states precision this card can't deliver run-to-run.

### 3.4 Untested-in-combination analysis (the audit's core ask)

I checked every feature-interaction the session layered on, specifically
looking for combinations each round only ever tested in isolation.

**What IS covered:**
- **chunked prefill + ragged**: explicitly, safely blocked --
  `mtp_prefill_batch` raises `NotImplementedError` (`:2773-2781`) if
  `chunk_size` is set with non-uniform prompt lengths. Correct guard, not a
  silent mis-chunk.
- **chunked prefill + CUDA-graph**: exercised at 16K/c=4 (§20) and argued
  sound by a disjoint-code-path proof I re-verified from source:
  `mtp_prefill_batch`'s chunked loop never references
  `self.enable_cudagraph` (`:2882-2981`), and the only graph-eligibility
  gate (`_mtp_sync_and_propose_batch:2529-2534`) requires
  `num_new_tokens <= _MAX_DECODE_QO_LEN (=16)`, which no 8192-token chunk
  can ever satisfy -- so prefill is *provably* eager, chunked or not, and
  the graph only ever touches the (independently verified) decode/verify
  round loop. Sound.
- **shrinking batch composition + CUDA-graph**: covered by
  `mtp_verify_cudagraph_check.py` (batch4 and batch2 graphs both replayed,
  §3.2), and every `_get_verify_graph(len(slots), k+1)` lookup (`:3070`) is
  keyed on the live active-slot count, so a batch that *shrinks* round to
  round just looks up a smaller precaptured graph.

**What is NOT covered (the real gap):**
- **ragged-length prefill + CUDA-graph**: `mtp_ragged_prefill_check.py`
  constructs its runner with `enable_cudagraph=False` (`:374`). The
  disjoint-path argument above *does* extend here (a ragged prefill's
  `qo_len` list is passed to `_forward_batch` with `is_decode=False`, so it
  is eager exactly like a uniform prefill), so the risk is low -- but it is
  argued, not demonstrated.
- **mid-flight admission (a GROWING batch) + CUDA-graph**: this is the one
  genuinely novel interaction. `mtp_async_arrival_check.py` runs
  `enable_cudagraph=False` (`:374`, `num_slots=26`). When a slot is
  admitted mid-flight, `len(slots)` *grows* between rounds, so
  `mtp_verify_and_commit_batch` would look up a *larger* precaptured verify
  graph than the previous round used. Two things have never been jointly
  exercised: (i) growing (not just shrinking) the graph batch size mid-
  stream, and (ii) the `num_slots >= 2*batch_size` reservation constraint
  (`:3242`) interacting with an admission pool -- with cudagraph, the
  runner must be sized `num_slots = 2*concurrency`, and admitted slots must
  stay within the first `concurrency` logical indices, which the current
  eager `num_slots=26` driver never has to respect. Neither is a known bug;
  both are genuinely untested. **This falls squarely inside the already-
  flagged §21.2 "mid-flight admission is NOT production-ready" boundary** --
  it is not a new undiscovered gap, it is a precise statement of what the
  §21.2 follow-on must cover.

**Whether to build a combined test now**: I judged it *not* the cheapest
first move. A synthetic "ragged + admission + chunking + cudagraph all at
once" microtest would be substantial to write correctly (it needs the
cudagraph slot-reservation sizing, an admission pool that respects the
first-`concurrency` index constraint, and a reference oracle for the
growing-composition rounds) -- and it would duplicate most of what a single
realistic-workload E2E test (§4.4) delivers for free. I flag the gap
precisely here and fold the joint coverage into the §4 P1 recommendation
rather than build a throwaway now.

### 3.5 Working tree and stray files

`git status` is clean. All three previously-flagged files are tracked:
`README.md`, `benchmarks/phase0_nsys_gap_ledger_diag.py`,
`项目实施规划.md` (confirmed via `git ls-files`). HEAD (`f60989c`) is the
ragged-prefill + admission commit; it includes the production change to
`runtime/direct_model_runner.py` (+121 lines) plus the two new benchmark
checks. No unexpected working-tree changes. **Nothing to fix here** -- the
one small hygiene fix this round would have made (§20.6's hardcoded
`"passed"` literal) was already landed by a prior round (`:441` now calls
`_sanity_check_reps(reps)`, confirmed).

---

## 4. Part 2 -- fresh forward assessment

The 2026-07-18 review's own Phase C conclusion ("chasing the last few % at
4K/c=4 is not worth it") **still holds** -- but every D-series item it
raised is now *done*, and the gap picture has inverted at long context
(64K/c=4 went from categorically blocked to 1.29x *faster* than native).
So the operative question is no longer "which D-item next" but "is anything
left that's worth real effort, or is this a natural stopping point?" My
independent answer: there is exactly one P0, one high-value P1, and a
correctly-closed rest.

### 4.1 P0 -- close the §21.2 mid-flight-admission numerical finding

This is the only `passed: false` gate in the tree and the honest blocker to
calling continuous-batching (D2) "done" rather than "structurally
demonstrated." I re-ran `mtp_async_arrival_check.py` fresh and reproduced it
exactly: `passed: false`, `correctness_ok: false`, all 6 requests finished,
the divergence a `near_tie_margin: 7.9375` at round 13's mixed batch
composition -- bit-identical to §21.2's reported value, confirming it is a
deterministic effect, not run-to-run randomness. The finding was
investigated but explicitly *not* validated against the project's own
Phase-A bar (real reference-vs-ours token-sequence comparison over a full
generation) -- §21.2 says so itself and recommends exactly this.

**Concretely**: run the Phase-A methodology (§10 -- greedy decode vs a
trusted reference, near-tie-margin analysis of every root divergence)
*specifically over mixed-admission-round batch compositions*, not the
uniform-arrival shape §10 already cleared. The question to answer: is a
7.9-logit divergence at a fresh-slot-joins-long-running-slots round *typical*
for that batch shape (a real, if benign, cross-slot batching-order effect
that needs numerical hardening), or was it a rarer large-sample of the same
benign near-tie noise floor everything else in this project sits at? Until
that is answered, mid-flight admission cannot be signed off.

- **Gate (pass):** divergences at mixed-composition rounds occur only at
  documented near-ties and the divergent tails stay semantically equivalent,
  at a rate no worse than native's own kernel-order noise -- same bar Phase A
  (§10) used.
- **Falsifier:** mixed-composition rounds diverge at a materially higher
  rate/magnitude than uniform-composition rounds -- in which case the cross-
  slot batching order at admission needs real numerical work, not a
  tolerance bump.
- **Effort:** small-to-medium (0.5-1 GPU-day) -- the Phase-A harness already
  exists in the session scratchpad; the new work is driving it at admission
  compositions.

### 4.2 P1 -- one consolidated *realistic coding-agent workload* E2E test

This is the highest-value *new* work and directly answers the user's own
framing (this serves THIS machine's real coding-agent workload, not a
generic benchmark). The session accumulated ~15 separate synthetic
shape/scenario tests; none models an actual coding-agent request stream.

**Deliverable**: a single driver that replays a realistic trace -- real
code-context prompt lengths (a spread, not a uniform 4096), real generation
lengths (short edits *and* long completions), realistic concurrency
arrival/departure (staggered, up to c=4), over a longer wall-clock window
(tens of minutes, not one batch) -- **run WITH `--cudagraph`**, against the
native-vLLM reference for both throughput and output quality.

Why this one test is worth more than more synthetic sweeps:
1. It **naturally exercises the untested combinations** §3.4 flagged
   (ragged arrival + mid-flight admission + varied context, *with* cudagraph)
   -- closing that coverage gap as a side effect of testing the real thing,
   instead of a throwaway microtest.
2. It is the **multi-hour-stability probe** the session never ran -- the D3
   memory fix is verified flat over 1107 rounds (~20 min, §11), which is
   strong but not a real production-length (hours/days) guarantee; extending
   this driver's duration is the cheapest way to get that evidence.
3. It **re-validates the real target** -- the many synthetic W1-S numbers
   use a sequential-ascending-token-id input the repo itself documents as
   *raising* acceptance rate above genuine text (`workloads.py:137-148`), so
   the real coding-agent acceptance rate (and therefore the real ours/native
   ratio) is still unmeasured on representative input.

- **Gate:** serves the trace without falling back to the slow singular path,
  end-to-end throughput at parity-or-better vs native, output quality within
  Phase-A near-tie tolerance, and memory flat over the full window.
- **Note:** the `W1_R`/`W2_R` representative fixtures are still "NOT YET
  DEFINED" (`workloads.py:31`) -- this test is the natural place to finally
  define them.

### 4.3 The 16K/c=4 residual (1.814x) -- do NOT chase with the same techniques

§14 already root-caused this to genuine near-linear-scaling compute in the
single-shot prefill forward (attention+FFN over real tokens), *not* a bug:
the one alternative kernel (v2 prefill) was measured **slower** at this
shape (§14.3), host-side metadata and the decode round-loop were measured
flat (§14.2/§14.4), and native's scheduler was confirmed identical
(§14.5). Chunking narrowed it (2.080x->1.814x, §20) by bounding working set,
but cannot reduce total FLOPs.

**My independent read**: this is a *categorically different* problem from
the ones this session closed. 64K/c=4 was **memory-capacity-bound**
(chunking removed a hard ceiling -> crossover to 1.29x faster); 4K/c=4 and
the c=1/c=2 wins are **dispatch/scheduler-overhead-bound** (the hand-rolled
loop's structural advantage). 16K/c=4 is **compute-bound with no identified
inefficiency** -- the same techniques that produced the other wins do not
apply. It is also a genuine local worst-case in an otherwise-favorable
curve (4K parity-or-better, 32K 1.116x, 64K 1.29x *faster*), i.e. an
anomaly, not a trend. Recommendation: **accept and document it as a known
compute-bound cell, do not open a new optimization line** unless a
realistic workload (§4.2) actually shows 16K-ish concurrent prefill
dominating real serving time (the synthetic sweep says it might not -- real
coding-agent context is rarely a uniform 16K across 4 simultaneous fresh
arrivals).

### 4.4 Remaining production-readiness gaps beyond D1/D2/D3

A skeptical reviewer should not call this "production ready" without noting:

1. **Request-level error handling / graceful degradation is absent.** A
   request whose context exceeds the per-slot capacity raises a hard
   `RuntimeError` mid-prefill (`build_attention_metadata:192-196`,
   `build_attention_metadata_batch:439-443`) -- in a real batch this
   crashes the *whole batch's* forward, not just the offending request.
   Real serving needs to reject/truncate the single over-long request and
   let the others proceed. Similarly, malformed inputs (mismatched
   `slots`/`token_ids`/`kv_lengths` lengths) raise `ValueError` rather than
   being rejected at an admission boundary. This is genuinely out of scope
   for the benchmark-driven work done so far, but it is a real gap before
   "production."
2. **No hours/days continuous-operation evidence.** Longest continuous run
   this session was ~1107 rounds / ~20 min (D3, §11). The grad-fix makes
   allocated+reserved memory provably flat, which is strong -- but thermal
   throttling behavior, driver stability under WSL2, and any slow
   accumulation outside the torch allocator over a multi-hour session are
   unverified. Folds naturally into §4.2.
3. **The runtime is still benchmark-driven, not server-integrated in this
   repo.** `runtime/engine.py`/`server/` exist but the validated path is
   the benchmark harness. Actual vLLM-backend integration (the sibling
   `sm120-flash-attention` project's Phase 5) is a separate, larger effort
   and out of this repo's current scope -- noted for completeness, not
   recommended as the immediate next step.

### 4.5 What is genuinely DONE (do not re-open)

- **D1 shape sweep**: complete across c∈{1,2,4} × {4K,16K,32K,64K}. The
  gap table is filled; 64K/c=4 crosses over to faster-than-native.
- **D3 memory leak**: root-caused (missing grad-disable), fixed, verified
  flat over 1107 rounds with ~34% headroom. Closed.
- **Phase A (uniform-arrival generation quality)**: PASS (§10), backed by a
  real token-sequence comparison.
- **Phase B (singular↔batch mechanism unification)**: done -- one GDN verify
  mechanism, `check0` back to bit-exact (§17).
- **Chasing the 4K/c=4 last-few-%**: still correctly not worth it -- the gap
  is inside the thermal noise band (§3.3), and the runtime is already at
  parity-or-better there.

---

## 5. Recommendation (prioritized)

| Pri | Item | Why now | Effort |
|---|---|---|---|
| **P0** | Close the §21.2 mid-flight-admission finding via Phase-A methodology on mixed-admission compositions (§4.1) | Only `passed: false` gate in the tree; honest blocker to signing off D2/continuous-batching | 0.5-1 GPU-day |
| **P1** | One realistic coding-agent workload E2E test, run WITH cudagraph, over a longer window (§4.2) | Validates the real target; closes the ragged/admission × cudagraph coverage gap (§3.4) as a side effect; doubles as multi-hour stability + defines the missing `W*_R` fixtures | 1-2 GPU-days |
| P2 | Request-level error handling / graceful over-capacity rejection (§4.4.1) | Needed before real serving, but not before more validation | 1 day |
| -- | Chase 16K/c=4 further; chase 4K/c=4 last few % | Compute-bound / inside noise -- different class, low expected value (§4.3) | not recommended |

If the user's goal is "is this trustworthy enough to actually serve THIS
machine's coding agent?", the honest answer is: **the core mechanism is
sound and the code is clean, but two things gate a real production sign-off
-- (P0) the admission-composition numerical finding, and (P1) a single test
on genuinely representative input over a realistic duration.** Both are
modest, well-scoped, and together would convert "extensively benchmarked on
synthetic shapes" into "validated on the real workload." Absent those, this
is a natural, comprehensive stopping point for the *optimization* line --
the remaining work is *validation*, not more speed.

---

## 6. Falsifiers for this audit's own conclusions

- If the P0 Phase-A check on admission compositions comes back clean (mixed-
  composition divergences track native within near-tie noise), then §4.1
  downgrades to "was worth checking, now closed" and continuous-batching is
  signed off.
- If a realistic-workload run (§4.2) shows the acceptance rate on real code
  text is materially lower than the synthetic 70.3%, the ours/native ratio
  at 4K/c=4 shifts and the "parity-or-better" headline must be re-quoted on
  real input.
- If the ragged/admission × cudagraph combination (§3.4), once exercised,
  reproduces a divergence the eager path did not, the disjoint-path argument
  is falsified and the graph batch-composition-change path needs its own
  correctness work.
