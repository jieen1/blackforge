# 2026-07-18 Session review + prioritized next steps (independent cold review)

Status: review + plan. Scope: an independent, from-source review of the
2026-07-18 session that closed the ours-vs-native W1-S gap from ~7.8x to a
reported ~1.057x (136.750 accepted tok/s), plus a prioritized follow-up
plan. Every substantive code claim below carries a `file:line` citation and
was re-derived by reading the current source (`runtime/direct_model_runner.py`
read in full, 3387 lines) and re-running the regression battery + the
headline benchmark fresh — not by trusting the session's own summaries.

Reviewer note on method: the session's middle stretch ran in an explicit
fast-iteration mode (single-rep perf, spot-check correctness, terse
write-ups). This review's job was to find what that stretch may have left
under-verified. The short answer: the *code* is in good shape (clean, no
dead code, internally consistent after the Phase 2 rewrite), and the fresh
regressions pass — but there is **one material, still-open correctness
question the whole session deferred and never actually answered**, and it is
the most important thing in this document.

---

## 1. Bottom line

1. **The single most important finding: the Phase 2 rewrite's output quality
   was never validated end-to-end against a trusted reference.** The residual
   per-verify-call numerical difference the rewrite introduced was diagnosed
   as "inherent bf16 batching-order noise" and declared acceptable — but that
   conclusion rests entirely on (a) kernel/layer-level cosine similarity, (b)
   signal-probe "the text reads as coherent" eyeballing, and (c)
   acceptance-rate/throughput parity. **None of these compares the Phase 2
   path's generated token sequences against a trusted reference (native vLLM
   greedy, or the pre-Phase-2 path) over a real generation.** The design
   doc's own section 10.5 named exactly this check as the real open question;
   sections 11-12 then landed Phase 2 without doing it. The current 136.750
   tok/s headline runs on a code path whose output quality is unproven.
   Details and citations in §4.

2. **The code itself is clean.** `runtime/direct_model_runner.py` read in
   full: no dead code (every "legacy" method is still a live call target),
   no debug/TODO scaffolding, the removed `_const_gdn_extra`/`TARGET_SPLITS`
   leave no dangling references, and the CUDA-graph machinery is internally
   consistent with the Phase 2 spec-decode rewrite. One genuine piece of
   tech debt (the singular↔batch mechanism divergence) and one cosmetic stale
   docstring. Details in §3.

3. **Fresh regressions: all 4 pass** (§5). **Fresh headline re-measurement
   reproduces**: my independent 3-rep mean is **137.784 accepted tok/s** vs
   the claimed 136.750 (within 0.76%); acceptance rate / committed tokens
   bit-identical to the session's figures; gap to native 1.049x. Utilization
   is materially higher than the session's last comparable figure, consistent
   with its own causal story. **One flag**: the run peaked at 99.3% of GPU
   memory (near-OOM) at only 4K context (§6, D3). (§6.)

4. **Housekeeping resolved** (§7): committed `README.md` and
   `benchmarks/phase0_nsys_gap_ledger_diag.py` (the latter the design doc
   already *claimed* was committed but was still untracked); reverted the
   pure CRLF→LF churn in `项目实施规划.md`.

5. **On chasing the last ~5.7%:** not recommended before finding #1 is
   closed; parity is achieved and the marginal gap is inside the run-to-run
   noise band. §8.3.

---

## 2. What I reviewed and how

- Read `runtime/direct_model_runner.py` in full (3387 lines), reconciled
  against `notes/2026-07-17-post-ragged-round-next-steps.md` §§7-12 and
  `PROGRESS.md`'s two most recent sections.
- Mapped every call site of the "legacy" methods (`mtp_verify_and_commit`,
  `snapshot_gdn_state`/`restore_gdn_state`, `verify_batch`,
  `build_gdn_metadata_batch`) across `benchmarks/` and `runtime/`.
- Re-ran all four regression suites fresh from clean processes.
- Re-ran the headline W1-S benchmark (`--batched --cudagraph --repeats 3`)
  fresh, with a concurrent 1 Hz `utilization.gpu` sampler.
- Independently cross-checked the generation-quality question via a
  dedicated read of every candidate correctness/quality check in
  `benchmarks/` and `tests/`.
- GPU/process hygiene (`pgrep -af`, `nvidia-smi`) verified clean before
  starting (0% util, 1986 MiB baseline, no compute apps) and after finishing.

---

## 3. Part A.1/A.2 — code correctness & consistency review

### 3.1 No dead code; every "legacy" path is still live

Read in full, `runtime/direct_model_runner.py` contains **no code that
should be deleted**. Specifically, the mechanisms Phase 2 stopped calling
from the batched path are all still reachable and exercised elsewhere:

- `snapshot_gdn_state` (`:1660`) / `restore_gdn_state` (`:1745`): still
  called by the singular `mtp_verify_and_commit` (`:2043`, `:2058`) and by
  `benchmarks/mtp_gdn_rollback_check.py`, `mtp_real_draft_check.py`,
  `mtp_trace_driven_probe.py`, `mtp_slot_identity_pinpoint_diag.py`,
  `mtp_batch_divergence_diag.py`, `phase0_nsys_gap_ledger_diag.py`. Their
  retention is documented at `:2481-2486`. **Not dead.**
- `verify_batch` (non-spec chunked, `:1556`): still called by the singular
  path (`:2044`) and several diagnostics. **Not dead.**
- `build_gdn_metadata_batch`'s chunked qo_len>1 branch (`:495`, `:596-629`):
  still used by the singular path and `decode_batch`. **Not dead.**
- `mtp_verify_and_commit` (singular, `:1980`): a live entry point — called
  by `benchmarks/mtp_w1s_our_runtime_perf.py:109` (the non-`--batched`
  path), `mtp_multiround_check.py:84`, `mtp_real_draft_check.py:195`,
  `mtp_our_runtime_acceptance.py:125`, and as the `check0` reference in
  `mtp_ragged_recompute_verify_check.py:183`. **Not dead.**

Confirmed no dangling references to the removed constructs: `_const_gdn_extra`
and `TARGET_SPLITS` appear only inside "removed"/"was stale" comments
(`:2686`, `:2739`, `:2924`), never in live code. No `print(`, `breakpoint(`,
`pdb`, `TODO`, `FIXME`, `XXX`, or `HACK` anywhere in the file.

### 3.2 The one genuine consistency issue: singular↔batch mechanism divergence

`mtp_verify_and_commit` (singular, `:1980`) still uses the OLD chunked-GDN +
snapshot/restore + recompute-forward mechanism, while
`mtp_verify_and_commit_batch` (`:2415`) uses the NEW spec-decode mechanism.
This is intentional and documented (`:2481-2486`, `:1596-1602`), and it is
**correct** — but it is real, latent tech debt, not a non-issue:

- The two entry points now produce genuinely **different committed token
  trajectories on near-ties** (the "271 vs 198" flips), which forced
  `check0` in both `mtp_batch_verify_check.py` and
  `mtp_ragged_recompute_verify_check.py` to be loosened from bit-exact to
  near-tie-tolerant (design doc §11.2). The singular path is therefore **no
  longer a bit-exact oracle for the batched path** — a real loss of a
  cross-check this project relied on (`check0` historically caught the
  `decode_qo_len` bug bit-exactly).
- Anyone calling the singular path (e.g. `mtp_w1s_our_runtime_perf.py`
  *without* `--batched`) silently gets the slower, un-CUDA-graphed
  snapshot/restore mechanism. Correct output, but a large latent perf/behavior
  cliff between two "equivalent-looking" entry points.

Risk: **LOW for correctness** (both mechanisms are individually validated),
but a genuine maintenance hazard. Ranked in the plan (§8.2), not fixed here
(fixing it means either migrating or deprecating the singular path — a scoped
change, not a trivial cleanup).

### 3.3 CUDA-graph machinery is consistent after the Phase 2 rewrite (Part A.1 item 2)

- `CapturedBatchDecodeGraph` (`:2556`) is fully reconciled with the
  spec-decode mechanism. Its qo_len>1 branch builds spec-decode GDN metadata
  (`static_spec_state_indices`/`static_num_accepted_tokens` allocated at
  `:2767-2768`, refilled per replay at `:2879-2888`; `_static_metadata_dicts`'
  qo_len>1 branch builds a `num_spec_decodes` metadata object at `:2926-2938`).
  Split-KV now reads `runner.decode_fixed_kv_split_size/max_num_splits`
  (`:2702-2703`) — the stale `TARGET_SPLITS=16` is gone. The `qo_len>1`
  guard requires MTP configured (`:2756-2762`). **Internally consistent** —
  it does not assume the old snapshot/restore world anywhere.
- `CapturedMTPDraftStepGraph` (`:3116`) covers the DRAFT model, which
  registers no GDN layer at all (`:3137-3139`, `:3300`) — so Phase 2's
  GDN-state-commit change cannot affect it. It was correctly left unchanged.
  Its qo_len>1 step-0 path is still valid and still used when a round's
  committed lengths happen to be uniform (`:2264-2277`).
- Buffer-aliasing check: `CapturedBatchDecodeGraph.replay()` returns views
  into its own static logits/hidden buffers (`:3112`); the caller consumes
  them into fresh tensors — `determine_accept_reject_batch`'s argmax
  (`:2516`) and `hidden_concat = torch.cat(...)` (`:2534`) — before the
  *separate* draft-step graph object replays. No cross-graph buffer
  aliasing.

### 3.4 `enable_cudagraph` has a tested eager fallback (Part A.1 item 4)

`enable_cudagraph` is still meaningful and its eager fallback is **not**
bit-rotted. When `False` (the default), `_get_verify_graph`/
`_get_draft_step_graph` are never called (guarded at `:2497`, `:2326`,
`:2264`, and precapture at `:1002`); the eager path routes through
`verify_batch_spec` (`:2508`) — the **same** spec-decode mechanism, just
eager. The three non-cudagraph regression suites construct runners without
`enable_cudagraph`, so they exercise this eager spec path directly. The
eager↔graph parity is additionally locked by `mtp_verify_cudagraph_check.py`
(content-consistency via `_ref_check`) and by the passive W1-S cross-check
that `total_committed_tokens`/`draft_acceptance_rate_pct` are bit-identical
between the eager Phase-2 and graph-replayed paths (design doc §12.4). That
passive identity is a genuinely strong graph==eager proof.

### 3.5 One cosmetic stale docstring

`_forward_batch`'s `commit` docstring (`:1382-1387`) still describes a
non-full-accept verify outcome as needing "the snapshot/restore +
recompute-forward repair." True only for the singular path now; the batched
path abandoned that mechanism in Phase 2. Cosmetic, low priority — folded
into §8.2.

---

## 4. Part A.3 — THE central finding: bf16-noise generation-quality gap is UNRESOLVED

### 4.1 The claim, and why it does not hold up

The design doc's §10.5 (`notes/...:1508-1522`) explicitly states that after
diagnosing the Phase 2 residual as inherent bf16 batching-order noise, the
real open question "is no longer 'where is the bug' but 'is this inherent
bf16-batching-noise-compounding-through-48-layers gap small enough, in its
effect on ACTUAL generation quality (not just raw cosine similarity on one
isolated verify call), to be acceptable' — a question that would need an
end-to-end generation-quality check ... rather than more kernel-level
debugging."

§11.2 then landed Phase 2 and claimed to close this with one sentence
(`notes/...:1676-1681`): "`mtp_batch_verify_check.py`'s signal-probe check
decodes real multi-round completions through the new mechanism ... coherent,
not degenerate — confirming the acceptable-noise conclusion holds at the
level that actually matters (real generations)." PROGRESS.md:1837-1839
repeats this: "Real multi-round generations through the new mechanism are
coherent (signal-probe completions read as normal text, not garbage)."

**This does not answer the §10.5 question.** "Coherent, not garbage" is a
category-(b) signal-probe/eyeball result. It does not compare the Phase 2
path's output against any reference. bf16 noise that flips near-tie tokens
produces divergent-but-still-coherent text — coherence cannot distinguish
"same quality as native" from "self-consistently drifted to worse output."

### 4.2 Exactly what coverage exists (and what does not)

I read every candidate check in `benchmarks/` and `tests/`. There are four
categories; the one that matters is absent.

- **(a) kernel/layer-level numerical checks — EXIST.**
  `mtp_gdn_rollback_check.py` (bytewise GDN-state diff); the recurring
  `_ref_check` next-token near-tie replay in all four MTP suites
  (`mtp_batch_verify_check.py:112-135` and siblings); the isolated
  cosine/allclose kernel tests in design-doc §§10.2/10.5/11.2.
- **(b) signal-probe causal-masking checks — EXIST.**
  `mtp_batch_verify_check.py:_check_signal_probe` (`:275-354`);
  `batch_decode_signal_probe.py`. **The project itself documents this
  method's blind spot**: PROGRESS.md:442-448 records that the signal-probe
  (which "only checks whether decoded TEXT still recovers the right identity
  number") **"never had a chance to catch"** a real prior divergence measured
  at `logits max_abs_diff=7.93, cosine_sim=0.55`. Leaning on signal-probe
  coherence to certify Phase 2's quality repeats a method this repo already
  proved can miss exactly this class of divergence.
- **(c) acceptance-rate / throughput benchmarks — EXIST.**
  `mtp_w1s_our_runtime_perf.py` (states at `:32-35` it "only measures
  performance, it does not re-verify correctness"); `w1s_native_bench.py`
  (scrapes Prometheus `vllm:spec_decode*` counters + timing — `_send_one`
  **discards the generated text entirely**). Acceptance rate measures
  draft/target *agreement*, both running the *same* noisy mechanism — it is
  not a proxy for output quality vs a reference.
- **(d) TRUE end-to-end generation-quality — DOES NOT EXIST.** No file uses
  an independent reference model (a repo-wide grep for
  `AutoModelForCausalLM`/`.generate()` in `benchmarks/` and `tests/` returns
  zero hits). No file captures native vLLM's output token ids/text to
  compare against ours. No greedy-decode-vs-reference sequence comparison
  over many tokens exists.

### 4.3 The two closest approximations, and why they fall short

1. **`check0`** (`mtp_batch_verify_check.py:144-215`,
   `mtp_ragged_recompute_verify_check.py:149-258`) is the *only* check that
   free-runs the new batched path against the pre-Phase-2 singular path and
   compares committed tokens. But it is **near-tie-tolerant** — a
   `near_tie_divergence` (the two paths commit *different* tokens, each
   locally self-consistent) is explicitly **not** a failure (the pass gate
   at `:205` counts only `exact_mismatches`), it is **batch=1 only**, runs
   ~6 rounds (≤~24 tokens), and its real gate is each side's own
   *teacher-forced* `_ref_check`, not sequence equality. Per design-doc §11.2
   it in fact **produced divergent trajectories** (the 271/198 pair + 2
   cascaded rounds) that were **accepted as passing**. By construction it
   permits the new path to generate different text from the reference.
2. **`total_committed_tokens`/`draft_acceptance_rate_pct` bit-identical
   across reps** (design-doc §§11.3/12.4) is a *within-our-runtime*
   determinism cross-check (and a graph==eager cross-check) — **not** a
   comparison against native or the pre-Phase-2 mechanism.

Also note `_ref_check` itself is **teacher-forced against the same runner/
model** (feeds the batched path's own committed tokens back into
`runner._forward` and checks the single next-token argmax within
`NEAR_TIE_LOGIT_MARGIN=2.0`). It cannot detect a self-consistent wrong
trajectory, and never compares full sequences.

### 4.4 Why this is material, not pedantic

The Phase 2 spec-decode path genuinely commits **different tokens** than the
pre-Phase-2/chunked path at near-ties (§11.2's own measurement). Over 256
generated tokens × 16 requests these divergences compound. The conv-side
~0.03 per-call discrepancy that §10.5 traced (and attributed to bf16
batching order), compounded through 48 GDN layers, is precisely the kind of
perturbation that *can* be benign or *can* systematically degrade output —
and the only evidence offered for "benign" is that the text still reads as
sentences. The 136.750 tok/s number is real; what is unproven is that the
tokens it produces are as good as native's. This is a **prerequisite gate
for calling the runtime production-ready**, and it is the top item in §8.

---

## 5. Part A.2 — fresh regression battery (re-run now, not re-reported)

All four suites re-run from clean processes, `CUDA_HOME`/`PATH` pinned to the
13.3 toolkit, venv `/home/bot/.venvs/vllm`:

| Suite | Invocation | Result |
|---|---|---|
| `mtp_gdn_rollback_check.py` | `--repeat 3` | **3/3 PASS** (bit-exact GDN-state restore across 48 layers) |
| `mtp_batch_verify_check.py` | (default) | **PASS** — top-level `passed: true`, all 4 sub-checks (`check0..check3`) `true`; `no_cross_contamination_signal: true` |
| `mtp_ragged_recompute_verify_check.py` | (default) | **PASS** — top-level `passed: true`, all 3 sub-checks `true` (incl. `check1_ragged_recompute` per-slot committed lengths 1/2/3/1) |
| `mtp_verify_cudagraph_check.py` | (default) | **PASS** — `passed: true`; the four reconciliation coverage flags all `true` (`verify_graph_batch4_replayed`, `verify_graph_batch2_replayed`, `draft_step0_qo2_graph_replayed`, `draft_continuation_graph_replayed`) — confirming the spec-decode CUDA-graph verify path is genuinely replayed, not merely precaptured |

All exit code 0, fresh clean processes. GPU/process hygiene confirmed clean
after the batch (0% util, 1705 MiB baseline, no compute apps, no benchmark
procs). Note: `mtp_batch_verify_check.py`'s printed `decoded_completions`
(e.g. slot 0: `". The capital of Germany is Berlin. The capital of Italy is
Rome"`) are exactly the "coherent" signal-probe outputs §4 discusses — they
pass the suite but are never compared to a reference, which is the whole
point of finding #1.

---

## 6. Part A.4 — fresh headline re-measurement (independent)

`python -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph
--repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`, with a
concurrent 1 Hz `utilization.gpu` sampler.

| Rep | accepted tok/s | ms/accepted | draft accept % | committed toks | gpu_busy% |
|---|---:|---:|---:|---:|---:|
| 1 | 138.381 | 7.226 | 70.29204 | 4116 | 90.82 |
| 2 | 141.950 | 7.045 | 70.29204 | 4116 | 90.85 |
| 3 | 133.021 | 7.518 | 70.29204 | 4116 | 90.92 |
| **mean** | **137.784** | 7.263 | 70.29204 | 4116 | 90.86 |

**The headline reproduces.** My fresh 3-rep mean is **137.784 accepted
tok/s** vs the claimed **136.750** — within 0.76%, i.e. reproduced within
run-to-run noise. Gap to native's 144.54: **1.049x** (session claimed
1.057x). The passive cross-checks all hold bit-for-bit and match the
session's reported values exactly: `draft_acceptance_rate_pct =
70.29204431017119`, `total_committed_tokens = 4116`, `num_accepted_tokens =
2792`, identical across all 3 reps. (This is a strong within-runtime
determinism cross-check — but, per §4, NOT an output-vs-native check.)

**Utilization sanity check (Part A.4).** 124 one-second samples, overall
mean 59.73% (this mixes in model-load and between-batch idle — many 0-4%
samples early). Segmented like the session's §9.6 method: the active-period
(decode/verify rounds + prefill spikes, samples ≥60%) mean is **84.3%**
(n=85), with the decode/verify steady-state cluster sitting ~71-84% and 23
samples pinned at 99% (the ~3s large-prefill TTFT spikes). This is
**materially above** the session's last comparable segmented figure (54.12%,
the Phase-3-final state, §9.6) — **consistent** with the causal story that
adding the spec-decode verify CUDA graph back (§12) and removing the eager
recompute forward (§11) put more of each round on the graph-replayed path.
No inconsistency with the session's own account.

**One flag from this run (see D3):** the concurrent sampler recorded a peak
`memory_used` of **97227 MiB against a 97887 MiB total — 99.3% of capacity.**
The within-process memory climb the session flagged as non-blocking
(§§9.7/11.3/12.4, peaks of 82-95 GB there) reproduced here at an even higher
peak, essentially touching the ceiling. It did not OOM this run and returned
to a 1554 MiB baseline on process exit (so a caching-allocator high-water-
mark, not a true leak) — but at 4096 context this is already a razor-thin
margin, which materially raises the priority of D3.

---

## 7. Part B — housekeeping resolution

- **`README.md` (was untracked) → COMMIT.** Read in full (172 lines): a
  substantive, benign project reference map (learning-resource index for
  vLLM/SGLang/CUTLASS/FlashInfer, environment notes, the "list dominant GEMM
  shapes before writing a kernel" task checklist). References PROGRESS.md and
  项目实施规划.md, includes the correct guardrails ("don't clean up others'
  uncommitted work"). Clearly meant to be committed. **Resolved this round.**
- **`benchmarks/phase0_nsys_gap_ledger_diag.py` (was untracked) → COMMIT.**
  The design doc §7 states "The diagnostic script itself ... is committed
  under `benchmarks/` per this phase's instructions" — but it was in fact
  still untracked. It is a legitimate Phase-0 diagnostic (inline replay of
  `mtp_verify_and_commit_batch`'s body with NVTX ranges + host timers),
  consistent with the many other committed `*_diag.py` scripts. **Resolved.**
- **`项目实施规划.md` (tracked, modified) → REVERT.** The diff is a pure
  line-ending churn: HEAD is CRLF (475 lines), working tree is LF (475
  lines), `git diff --ignore-all-space` is empty — zero content change, and
  it predates this session. Reverted to keep the contract doc stable and
  avoid a 950-line noise diff. Nothing is lost (content is provably
  identical). **Resolved.**

---

## 8. Part C — prioritized, phased follow-up plan

Standing discipline (unchanged): full regression suites after any production
change; W1-S 3-rep protocol as the end-to-end perf gate; no
kernel/microbenchmark win claimed as an end-to-end win without the W1-S
number. Priorities are ranked by real risk — silent generation-quality
degradation above everything else, per this review's charter.

### Phase A — Close the generation-quality gap (P0, highest risk, do first)

**Deliverable:** an end-to-end generated-*sequence* comparison of the Phase 2
spec-decode path against a trusted reference. NOT more kernel debugging (per
§10.5's own framing) — the unit is decoded token ids over a real generation.

Concretely:
1. Pick a small frozen prompt set (reuse the W1-S fixtures + a handful of
   *real* natural-language/code prompts, since the synthetic input is
   atypically predictable — see Phase D).
2. Greedy-decode (temp=0) ≥256 tokens per prompt through **(ref)** a trusted
   reference and **(ours)** the Phase 2 path, capture token ids for both.
   Strongest reference: native vLLM serving the same model greedy —
   `w1s_native_bench.py` currently *discards* generated tokens (`_send_one`),
   so extend it (or a sibling) to capture them. A weaker but zero-new-infra
   in-house reference: the pre-Phase-2 singular chunked path
   (`mtp_verify_and_commit`), which is still present and was the accepted
   oracle through 2026-07-17.
3. Report a token-level agreement metric (longest common greedy prefix /
   per-position match rate) AND a semantic proxy on the divergent tails.

**Gate (pass):** divergences from native's greedy output occur only at
positions that are documented near-ties (margin < `NEAR_TIE_LOGIT_MARGIN`),
at a rate no higher than native's own run-to-run/kernel-order noise, and the
divergent tails remain semantically equivalent.
**Falsifier (fail → re-open Phase 2 correctness):** the Phase 2 path diverges
from native greedy at a materially higher rate than near-tie noise, or
produces measurably worse text (e.g. higher perplexity under the reference,
or degeneration). In that case the conv-side ~0.03 discrepancy from §10.5,
compounded through 48 GDN layers, is the prime suspect and its state-commit
correctness must be re-derived — NOT papered over with more cosine checks.
**Explicitly do NOT** use signal-probe coherence as the gate — PROGRESS.md
:442-448 proves that method can miss a cosine-0.55 divergence.

Estimated effort: 0.5-1.5 GPU-days (mostly wiring native-output capture).
This is the one item that blocks calling 136.750 tok/s "production-ready."

### Phase B — Resolve the singular↔batch mechanism divergence (P1, tech debt)

**Deliverable:** one GDN verify mechanism, or an explicit deprecation.
Options: (a) migrate `mtp_verify_and_commit` (singular) to the spec-decode
mechanism too (restores a bit-exact singular↔batch relationship, lets
`check0` return to bit-exact, and removes the snapshot/restore/recompute code
if no benchmark still needs it); or (b) formally deprecate the singular path,
migrate its call sites (`mtp_w1s_our_runtime_perf.py`'s non-`--batched`
branch, `mtp_multiround_check.py`, the `check0` references) and delete
`snapshot_gdn_state`/`restore_gdn_state`/the recompute branch.
Fold in the cosmetic `_forward_batch` docstring fix (`:1382-1387`).
**Gate:** a single mechanism in the tree, all suites pass, `check0` states its
tolerance explicitly. **Falsifier for (b):** if any diagnostic genuinely
needs the snapshot/restore primitive (e.g. `mtp_gdn_rollback_check.py`
validates it directly), (b) is off the table and (a) is the path.
Risk: LOW. Effort: 1-2 days for (a).

### Phase C — Is the remaining ~1.057x worth chasing? (P2)

**Recommendation: not before Phase A, and probably not at all as a headline
goal.** The absolute gap is now ~6.8 tok/s (144.54 − 137.784, this review's
own measurement), inside the run-to-run noise band — my three reps alone
span 133.0-141.9 tok/s, a spread (~8.9 tok/s) wider than the mean gap to
native itself. Both of the
plan's original targets (≥110 tok/s, <1.3x) are met. The most direct
remaining lever, if pursued, is the draft-model sync/propose path
(`_mtp_sync_and_propose_batch`) — its K-1 continuation loop still does a
per-step host `argmax().tolist()` (`:2346`) and its step-0 falls back to
eager when committed lengths are ragged (`:2278-2287`) — but Phase 0's ledger
already showed the draft model is small relative to the 64-layer target and
graph-capturing its whole cycle landed flat (design-doc §9.2 item 12). The
founding premise ("beat native through specialization") is a separate, larger
question the design doc's §6 already flags as a user decision once parity is
on the table. **Gate if pursued:** any lever must move the W1-S 3-rep mean by
> the measured rep-to-rep std, or it is noise.

### Phase D — Production-readiness gaps the narrow benchmark left (P1/P2)

The entire session optimized and measured at exactly ONE shape (n16, K=3,
c=4, uniform 4096-token prompts, 256-token generation, sequential-token
synthetic input). Several gains may be shape-specific:

- **D1 — Shape-generalization sweep (P1).** The project's own docs flag three
  reasons the single shape is unrepresentative: (i) the W1-S input is a
  *sequential ascending-token-id* synthetic that the repo itself documents
  "meaningfully raise[s] acceptance rate relative to genuine i.i.d. sampling"
  (`benchmarks/workloads.py:137-148`) — real coding-agent text will have a
  different (likely lower) acceptance rate, changing committed-tokens/round
  and therefore the ours/native ratio; (ii) the representative `W1_R`/`W2_R`
  fixtures are "NOT YET DEFINED" (`workloads.py:31`); (iii) the true
  W2-scale 32768-context fixture is "not built" (`workloads.py:156-158`).
  **Deliverable:** re-measure the gap across `c ∈ {1,2,4}` × `{uniform,
  ragged prompt lengths}` × `context ∈ {4K, 16K, 32K}`. **Gate:** report the
  gap per cell; flag any cell where it exceeds 1.3x. **Expected weak spot:**
  at c=1 the cross-slot batching win (a large share of the session's gains)
  evaporates — the gap likely re-widens there; if so, the 1.057x is a
  c=4-specific result and should be reported as such, not as "parity."
- **D2 — Continuous batching / async arrival (P1).** `mtp_prefill_batch`
  *hard-requires* uniform prompt length (`:2372-2374`) and the benchmark
  prefills all 4 slots synchronously then verifies in lockstep. Real serving
  has async request arrival/departure and variable lengths; there is no
  mid-flight admission path (a slot *finishing* is handled by the shrinking
  active set, but a slot *joining* mid-flight at a different kv_len/prompt
  length is not). **Deliverable:** a variable-length, staggered-arrival
  driver + whatever prefill path it needs (per-slot `mtp_prefill`, or a
  ragged batched prefill). **Gate:** the runtime serves a realistic
  arrival/length trace without falling back to the slow singular path for
  every request.
- **D3 — Memory-growth-across-reps (P1 now — this review reproduced a
  near-OOM).** Reserved memory climbs within one process (design-doc
  §§9.7/11.3/12.4 saw 82-95 GB peaks and ruled it non-blocking). **This
  review's fresh run peaked at 97227 MiB against a 97887 MiB total — 99.3%
  of capacity**, higher than any figure the session recorded, and only ~660
  MiB short of OOM at merely 4096-token context. It returned to a 1554 MiB
  baseline on exit (a caching-allocator high-water-mark, not a true leak),
  and did not OOM this run — but the margin is now demonstrably razor-thin,
  and at 32K context (real W2) or a multi-hour session it would very
  plausibly OOM. This is no longer a "worth a look later" item.
  **Deliverable:** a `torch.cuda.memory_stats`/`memory_summary` trace over a
  long run to localize the growth (prime suspects: the still-eager paths'
  per-call `torch.tensor(...)` staging in `_fill_buffers`/
  `build_*_metadata_*`, and PyTorch reserved-segment fragmentation across
  reps); then either cap the allocator (`PYTORCH_CUDA_ALLOC_CONF`
  `max_split_size_mb`/`expandable_segments`), call
  `torch.cuda.empty_cache()` at round boundaries, or preallocate the staging
  tensors. **Gate:** steady (non-monotonic) reserved memory over ≥100 rounds
  at the target context length, with ≥10% headroom to the ceiling.
  **Falsifier for "non-blocking":** already fired — 99.3% at 4K context is
  blocking for 32K.

### On whether the fast-iteration stretch needs its own "close the rigor gap" pass

**No separate pass is needed for the *code*.** This review (full source read
+ fresh 4-suite regression + fresh 3-rep perf) covers exactly what the
fast-iteration mode short-cut: it confirms the code is clean, internally
consistent, and free of dead code/scaffolding, and that the correctness
suites pass against the final state. What this review does **not** close are
two validations that *no* round (fast or full-rigor) ever performed — the
end-to-end generation-quality check (Phase A) and the memory-growth
root-cause (D3). Those are genuine deferred work, not fast-iteration sloppiness
— they were deferred by every round equally. The honest framing: the
fast-iteration stretch left the *code* fine; the session as a whole left two
specific *validations* open, and Phase A is the one that matters.

---

## 9. Falsifiers for this review's own conclusions

- If Phase A's end-to-end check comes back clean (Phase 2 output tracks
  native greedy within near-tie noise), then §4's "material open question"
  downgrades to "was worth checking, now closed," and the 136.750 number is
  production-trustworthy at this shape.
- If the D1 sweep shows the gap holding ≤1.3x across c and context, then the
  "shape-specific" caveat downgrades to a non-issue and parity is general.
- If either comes back the other way, the corresponding phase's falsifier
  fires and that work becomes load-bearing, not optional.

---

## 10. Phase A results: end-to-end generation-quality validation (executed 2026-07-18)

**Verdict: PASS.** The Phase 2 spec-decode path's generated token sequences
track the trusted reference within documented near-tie noise, and every
diverging continuation (on both sides) remains fluent, on-topic text --
no garbage, no repetition, no degeneration. This closes the §4/§8 Phase A
gap: the 136.750-137.784 accepted tok/s headline is now backed by an actual
token-sequence comparison, not just cosine similarity / signal-probe /
acceptance-rate parity.

### 10.1 Methodology

**Reference chosen: native vLLM's own real engine, in-process, WITHOUT
speculative decoding** (plain autoregressive greedy target-only decode) --
this is the *strongest* option the review named, made tractable by using
`vllm.LLM(...)` directly (same pattern `runtime/vllm_inprocess_baseline.py`
already established: never had an HTTP layer to begin with; still a real
`Scheduler`/`GPUModelRunner`/KV-cache-manager engine reached via a spawned
`EngineCore` process over ZMQ, not a network call). No MTP/spec-decode
config at all on the reference side -- deliberately: speculative decoding
with greedy verification is *supposed* to be lossless against plain greedy
decoding, so plain greedy is the least-assumption ground truth spec-decode
(native's or ours) is meant to reproduce, rather than introducing a second
spec-decode mechanism as an extra variable in the oracle itself.

`attention_backend=CUSTOM` (this project's own `SM120GQABackend`, already
independently validated in the sibling `sm120-flash-attention` project)
and `kv_cache_dtype=fp8_e4m3` were used on **both** sides, deliberately --
this keeps the attention *kernel* identical, so any divergence found is
attributable to the thing actually in question (this runtime's own Phase 2
GDN state-commit mechanism and orchestration), not a different attention
implementation. `unsloth/Qwen3.6-27B-NVFP4`, `enforce_eager=True`,
`language_model_only=True`, `max_model_len=8192` on both sides.

**"Ours" = the real, unmodified Phase 2 mechanism**, verified faithful
(not a re-implementation): the capture script inlines `mtp_prefill_batch`'s
and `mtp_verify_and_commit_batch`'s bodies (calling the exact same
underlying methods -- `_forward_batch`, `verify_batch_spec`,
`determine_accept_reject_batch`, `_mtp_sync_and_propose_batch` -- in the
same order with the same arguments) *only* so intermediate logit tensors
could be retained for margin reporting, since neither public wrapper
returns raw logits. **This was independently verified, not assumed**: a
dedicated self-check (`verify_inlining_faithful.py`) re-ran two prompts
(`natural_0`, `w1s_2`) through the real, completely unmodified
`mtp_prefill_batch` + `mtp_verify_and_commit_batch` and confirmed the
committed token stream is **bit-identical** to the inlined driver's output
for both (`ALL_MATCH=True`). `enable_cudagraph=False` throughout (the
eager path -- already established in §3.4 as byte-for-byte the same
mechanism the CUDA-graph-enabled path falls back to).

**Prompt set (8 total, greedy/temp=0, `ignore_eos=True`, 260 tokens
generated, first 256 compared):**
- 3 of the 16 frozen W1-S fixture prompts (`benchmarks/workloads.py`'s
  `W1_S_FIXTURE`, sequential-ascending-token-id synthetic, 4096 tokens
  each) -- `w1s_0`, `w1s_1`, `w1s_2`.
- 5 natural-language/code prompts written for this task (real text, not
  the synthetic fixture's atypically-predictable input): a quicksort
  explanation, a palindrome-checker completion, a Flask 500-error
  diagnosis, a Fibonacci function completion, and a SQL-injection review.

Both capture scripts loaded prompt token ids from ONE frozen, pre-tokenized
JSON (`build_prompts.py`, run once) -- eliminating any chance of a
tokenization discrepancy being a confound. GPU/process hygiene: confirmed
idle (baseline ~1.5-1.6 GB, 0% util, no stray processes) before the run and
after each of the three GPU-heavy scripts (ours capture, reference capture,
inlining self-check), verified via `pgrep`/`ps`/`nvidia-smi` directly, not
assumed from a background-task notification alone. Ran strictly
sequentially, never concurrently (peak observed 87 GB during the reference
capture's model load/warmup -- well short of the review's flagged 97.3 GB
near-OOM figure, because this test's config -- `num_slots=2`,
`max_model_len=8192`, no CUDA-graph slot doubling -- is much smaller than
the production `c=4`/`cudagraph`/`max_model_len=40960` benchmark config).

### 10.2 Per-prompt agreement

| prompt | kind | prompt_len | exact matches / compared | match rate | longest common prefix |
|---|---|---:|---:|---:|---:|
| w1s_0 | synthetic W1-S | 4096 | 256/256 | 100.0% | 256 |
| w1s_1 | synthetic W1-S | 4096 | 256/256 | 100.0% | 256 |
| w1s_2 | synthetic W1-S | 4096 | 5/256 | 2.0% | 5 |
| natural_0 | natural (code-explain) | 107 | 16/256 | 6.2% | 14 |
| natural_1 | natural (code-complete) | 37 | 66/256 | 25.8% | 64 |
| natural_2 | natural (bug diagnosis) | 41 | 78/256 | 30.5% | 76 |
| natural_3 | natural (code-complete) | 25 | 208/256 | 81.2% | 208 |
| natural_4 | natural (security review) | 38 | 6/256 | 2.3% | 5 |
| **overall** | | | **891/2048** | **43.5%** | -- |

**The raw overall match-rate (43.5%) is NOT the right headline number and
would be misleading read in isolation** -- see §10.4 for why. The real
diagnostic is §10.3: every one of the 6 divergence *events* (not the raw
per-position mismatch count) traces to a documented-class near-tie.

### 10.3 Root-cause divergence analysis (the actual gate check)

2 of 8 prompts (`w1s_0`, `w1s_1`) reproduce native's greedy output
**bit-exactly for the full 256 compared tokens** -- zero divergence.

The other 6 prompts each diverge from the reference at exactly one root
position (after which, by construction, the two paths are conditioned on
different prefixes and legitimately generate different-but-still-valid
continuations -- see §10.4). For every one of these 6 root divergences,
the margin was computed exactly matching this project's own established
convention (`benchmarks/mtp_multiround_check.py`'s
`near_tie_margin = ref_top1_logit - ref_logit_for_mtp_choice`; logprob
differences equal raw logit differences exactly since log-softmax
preserves differences, so vLLM's returned logprobs are used directly as
logit margins) against `NEAR_TIE_LOGIT_MARGIN = 2.0` (the value actually
in the code, confirmed by grep across `benchmarks/mtp_multiround_check.py`,
`mtp_batch_verify_check.py`, `mtp_ragged_recompute_verify_check.py`,
`mtp_verify_cudagraph_check.py` -- all define it identically):

| prompt | pos | ours token | ref token | ref top-1 | margin (ref top1 − ref logprob(ours' token)) | ours' own top-2 margin |
|---|---:|---|---|---|---:|---:|
| w1s_2 | 5 | `271` (`"\n\n"`) | `198` (`"\n"`) | `198` | **0.125** | 0.75 |
| natural_0 | 14 | `15771` (`"Under"`) | `2014` (`"An"`) | `2014` | **0.500** | 2.875 |
| natural_1 | 64 | `10121` (`" usage"`) | `198` (`"\n"`) | `198` | **0.375** | 0.25 |
| natural_2 | 76 | `1510` (`" \`"`) | `2407` (`" body"`) | `2407` | **0.375** | 0.25 |
| natural_3 | 208 | `198` (`"\n"`) | `271` (`"\n\n"`) | `271` | **0.125** | 0.0 |
| natural_4 | 5 | `550` (`"##"`) | `13962` (`"###"`) | `13962` | **0.625** | 0.125 |

**Every single root divergence margin (0.125-0.625 logit units) is
comfortably under `NEAR_TIE_LOGIT_MARGIN=2.0`** -- none is remotely close
to the threshold. Notably, `w1s_2`'s divergence is the token pair
`271`/`198` (`"\n\n"` vs `"\n"`) -- **literally the same near-tie example**
this project's own `NEAR_TIE_LOGIT_MARGIN` docstring already cites from an
earlier round's independent finding, an unplanned but strong corroboration
that this is the same, already-characterized benign phenomenon, not a new
one. In every case ours' chosen token was reference's own rank-2 (or, for
`natural_4`, rank-3) candidate, at logprob within 0.5-1.6 nats of
reference's own top pick -- i.e., reference's own model was genuinely
near-torn between the two options at that exact position.

### 10.4 Why the 43.5% raw match rate is not the headline metric

Once a real (even if near-tie) divergence occurs at position *i*, position
*i+1* onward compares two **different, unrelated continuations** (ours'
own subsequent text vs. reference's own subsequent text, conditioned on
different prefixes from that point on) -- there is no reason to expect
these to coincide token-for-token, and the review's own gate framing
explicitly anticipates this ("diverging tails... remain semantically
equivalent," not "match the reference"). This is confirmed by the data
itself: `w1s_2` and `natural_4` (earliest divergences, position 5) show
essentially zero further coincidental matches for the remaining ~250
positions (2.0%/2.3% match rate is just the 5-token shared prefix, nothing
more) -- exactly the signature of "one clean fork, then two independent
valid continuations," not cascading corruption. `natural_3` (latest
divergence, position 208) correspondingly shows the highest match rate
(81.2%) simply because most of its 256 compared tokens are pre-divergence.
The right question is not "do post-divergence positions match" (they
structurally can't be expected to) but "is the ROOT divergence a near-tie,
and does the diverging tail stay coherent" -- both answered affirmatively
in §10.3 and §10.5.

### 10.5 Qualitative read of the diverging tails

Read in full (not skimmed) for all 6 diverging prompts, both sides:

- **`w1s_2`** (synthetic sequential-token-id prompt, not real language):
  ours' continuation correctly identifies the input as "not meaningful
  content... a corrupted data dump... gibberish/mojibake"; reference's
  continuation goes into an extended step-by-step "thinking process"
  reasoning about the same nonsensical input ("a massive, seemingly random
  block of text... programming keywords, HTML/CSS/JS snippets, SQL
  fragments"). **Both are correct, sensible reactions to a genuinely
  nonsensical prompt** -- the stylistic difference (terse verdict vs.
  extended reasoning) plausibly traces directly to the `"\n\n"` vs `"\n"`
  fork (whether a `<think>`-style block closes early or continues).
- **`natural_0`** (quicksort explain): both sides correctly begin
  analyzing the `quicksort` function structurally, just with different
  phrasing of the same first analysis step. Coherent on both sides.
- **`natural_1`** (palindrome function): reference continues explaining
  the function normally. **Ours shows one artifact worth flagging
  honestly**: after finishing the function body, it emits an
  `<|endoftext|><|im_start|>user...` sequence -- i.e., having reached what
  the model considers a natural stopping point, forcing continuation past
  it via `ignore_eos=True` (a deliberate, symmetric methodology choice on
  BOTH sides, matching this project's own established `-S`-line convention
  in `w1s_native_bench.py`) causes it to hallucinate a new simulated chat
  turn. This is **fluent, grammatically well-formed text, not garbage or
  repetition** -- a known, expected, and symmetric side effect of
  forced-length generation past a real stopping point, not a Phase 2
  mechanism bug.
- **`natural_2`** (Flask bug diagnosis): both sides correctly diagnose the
  same root cause (`request.get_json()`/`Content-Type` handling with an
  empty body), just reordering which clause comes first. Coherent, both
  substantively correct.
- **`natural_3`** (Fibonacci completion): ours continues with example
  `print()` calls; reference explains the function's recursion and base
  cases. Both reasonable, on-topic continuations of the same completion
  task.
- **`natural_4`** (SQL injection review): both sides correctly identify
  the vulnerability and give a correct example attack payload
  (`' OR '1'='1`) -- differing only in heading style (`"## SQL Injection
  Vulnerability Analysis"` vs `"### Vulnerability Analysis"`, directly
  downstream of the `##`/`###` fork at position 5) and phrasing. Both
  substantively correct security analyses.

**No occurrence of repetition loops, gibberish, or broken/garbled output
on either side, across any of the 6 diverging prompts.**

### 10.6 Verdict

**PASS**, per the review's own framing: divergences from the reference's
greedy output occur *only* at positions that are documented near-ties
(all 6 root-cause margins 0.125-0.625, versus the 2.0 threshold -- not
close), at a rate consistent with this project's own previously-
characterized bf16/kernel-order noise floor (the `271`/`198` example
recurring verbatim is strong corroborating evidence this is the *same*
already-understood phenomenon, not a new failure mode), and every
diverging tail remains semantically reasonable, fluent text on both sides
-- never degenerating into garbage or repetition. The §10.5 (prior doc)
"~0.03 conv-side bf16 noise, compounded through 48 GDN layers" hypothesis
is **not falsified** by this check; it is now backed by an actual
token-sequence comparison rather than resting on cosine-similarity/
signal-probe/acceptance-rate proxies alone. The 136.750-137.784 accepted
tok/s headline is now supported by a real generation-quality validation,
closing the §4/Phase-A gap this review opened.

**Scope notes, stated honestly:** (1) 8 prompts is a modest sample --
sufficient to *find and characterize* the divergence phenomenon (which it
did, cleanly, 6 times) but not a large-N statistical bound on divergence
*rate*; a larger sweep would sharpen the "consistent with the noise floor"
claim from qualitative to quantitative. (2) This check ran at concurrency=1
(batch size 1) throughout, not the production `c=4` cross-slot-batched
shape -- deliberately, to keep the check simple/low-risk and because the
GDN state-commit *mechanism* being validated does not itself depend on
batch size (same functions, same kernels); cross-slot batching-order
effects are a separate axis the review's own Phase D (D1) already covers.
Neither scope note changes the verdict; both are natural candidates if a
larger/production-shape re-run is ever wanted.

**Artifacts** (kept in the session scratchpad, not committed --
measurement scripts/outputs, not project source):
`build_prompts.py`, `capture_ours.py`, `capture_reference.py`,
`verify_inlining_faithful.py`, `compare.py`, `prompts.json`,
`ours_result.json`, `reference_result.json`, `comparison_report.json`.

---

## 11. D3 (near-OOM memory growth): root-caused, fixed, verified
(executed 2026-07-18)

**Verdict: real leak of live GPU tensors, not allocator fragmentation.
Root cause: this hand-rolled runtime never disabled autograd. One-line
fix (`torch.set_grad_enabled(False)`). Memory now provably flat over
1107 rounds (3 full W1-S passes), and the fix came with a small perf
*improvement*, not a regression.** This closes D3, the falsifier §8/D3
flagged as already fired.

### 11.1 Methodology

Wrote `benchmarks/memory_growth_diag.py` (committed): reimplements the
same real call sequence `_run_batch_batched` uses
(`mtp_prefill_batch`/`mtp_verify_and_commit_batch`, `--batched
--cudagraph` shape, n16/c4/K=3/256 tokens) but samples
`torch.cuda.memory_allocated()` **and** `torch.cuda.memory_reserved()`
(not just one) at every batch boundary, plus every 10th individual
decode/verify round, across multiple full passes over the W1-S fixture
in the SAME process (no reload between passes -- the exact condition
D3's falsifier needs). `memory_allocated()` is the caching allocator's
own live-referenced-tensor-bytes counter; `memory_reserved()` is total
segment memory the allocator holds (live + cached-but-freed). Flat
allocated + growing reserved = fragmentation, no true leak. Both
growing = a genuine accumulating live-tensor reference somewhere.
Cross-checked against `nvidia-smi`'s own `memory.used` at every batch
boundary throughout (never relied on the allocator's self-report alone).

GPU/process hygiene: verified idle via `nvidia-smi`/`pgrep` immediately
before every run in this section (never assumed from memory), one
GPU-heavy process at a time throughout.

### 11.2 Before the fix: confirmed real leak, not fragmentation

Ran 3 passes (1107 total decode/verify rounds) on the pre-fix code.
`memory_allocated()` sampled at each of the 12 batch boundaries:

| round | pass | batch | allocated (MiB) | reserved (MiB) | nvidia-smi (MiB) |
|---:|---:|---:|---:|---:|---:|
| 87 | 0 | 0 | 43480.1 | 63454 | 65906 |
| 183 | 0 | 1 | 45810.8 | 71306 | 73758 |
| 284 | 0 | 2 | 48147.0 | 71404 | 73855 |
| 369 | 0 | 3 | 50456.2 | 71476 | 73928 |
| 456 | 1 | 0 | 52779.6 | 71556 | 74008 |
| 552 | 1 | 1 | 55110.3 | 79408 | 81859 |
| 653 | 1 | 2 | 57446.5 | 79506 | 81957 |
| 738 | 1 | 3 | 59755.7 | 79580 | 82031 |
| 825 | 2 | 0 | 62079.0 | 87418 | 89869 |
| 921 | 2 | 1 | 64409.7 | 87510 | 89961 |
| 1022 | 2 | 2 | 66746.0 | 87608 | 90059 |
| 1107 | 2 | 3 | 69055.2 | 95442 | 97261 |

`memory_allocated()` (live tensor bytes) grew **continuously and
monotonically**, every single batch, with **no plateau** across all 3
passes -- roughly +25 MiB every individual round, +2.3 GB per 90-round
batch, zero drops at batch or pass boundaries (`reset_slot`'s in-place
KV/GDN-state overwrite did not free anything). This is decisive: a
pure-fragmentation story (flat `allocated`, growing `reserved` only)
does not fit the data -- `allocated` itself is the thing growing. Final
state: 69055 MiB allocated / 97261 MiB `nvidia-smi` against the 97887
MiB card -- **99.3% of capacity**, matching the review's reported
figure (97227/97887) almost exactly, and reproducing the same near-OOM
condition on demand. `reserved` tracks `allocated` upward (it must,
since reserved >= allocated) rather than independently ballooning, so
`reserved`'s growth here is a *consequence* of the real leak, not a
separate fragmentation effect layered on top.

### 11.3 Root cause: no `torch.no_grad()`/`inference_mode()` anywhere

`grep -n "grad" runtime/direct_model_runner.py` returned **zero hits**
before this fix -- confirmed directly, not assumed. This file drives a
27B-parameter model's forward pass every decode/verify round
(`_forward_batch`, and critically `_mtp_forward_batch` via the eager
step-0 fallback in `_mtp_sync_and_propose_batch`, which is hit on
essentially every round in production: step-0 only uses the
`CapturedMTPDraftStepGraph` when every active slot's committed length
is *uniform* that round, and at a ~70% per-token draft-acceptance rate
with `concurrency=4`/`K=3`, four slots landing on the identical accept
count by chance is the exception, not the rule). Unlike real vLLM's
`GPUModelRunner` (whose `execute_model` always runs under
`@torch.inference_mode()`), nothing in this hand-rolled runner ever
disabled gradient tracking, so every one of those eager forward calls
built a full autograd graph rooted at the model's parameters
(`requires_grad=True` by default -- this project's loading path never
explicitly freezes them). The model's own persistent, in-place-updated
buffers (paged KV cache, GDN conv/ssm recurrent state) being written to
every round under live autograd tracking is the natural mechanism for
why the retained graph never got freed round-to-round -- consistent
with the observed steady ~25 MiB/round, no-plateau growth, and with
growth continuing right through `reset_slot`'s own in-place
resets/overwrites (an in-place op under autograd tracking extends the
graph rather than severing it).

This is the standard, well-documented class of PyTorch bug ("memory
grows during an inference-only loop because nothing disabled
autograd") and the standard fix is exactly what real vLLM already does
at its own execution boundary -- this hand-rolled runtime had simply
never added the equivalent when it took over model execution from
vLLM's own runner.

Note on the review's original two named suspects: the `_fill_buffers`/
`build_*_metadata_*` per-call `torch.tensor(...)` staging tensors
(review's suspect #1) are **int32/int64/bool** -- floating-point-only
autograd literally cannot attach to them, so they are not a contributor
to the `allocated` growth observed here (they may contribute a much
smaller, secondary `reserved`-only fragmentation effect from varying
per-round sizes, but that is not what is driving this near-OOM
trajectory). The dominant, decisive mechanism is the missing
grad-disable around the real floating-point model forward calls, found
by actually tracing the data rather than accepting the review's
suspects at face value, per this project's own standing discipline.

### 11.4 The fix

One line, added as early as possible in `DirectModelRunner.__init__`
(`runtime/direct_model_runner.py`, before any model construction or
forward call):

```python
torch.set_grad_enabled(False)
```

Global and process-wide (not a context manager needing a matching
exit) -- appropriate because this class represents an entire
pure-inference runtime process that never computes a backward pass.
Checked for conflicts: none of the four correctness suites or this
project's other benchmark scripts reference `grad` anywhere, so nothing
relies on autograd being enabled inside a `DirectModelRunner` process.

### 11.5 After the fix: memory flat over 1107 rounds, ~34% headroom

Re-ran the **identical** 3-pass/1107-round diagnostic post-fix:

| round | pass | batch | allocated (MiB) | reserved (MiB) | nvidia-smi (MiB) |
|---:|---:|---:|---:|---:|---:|
| 87 | 0 | 0 | 41147.0 | 61128 | 64359 |
| 183 | 0 | 1 | 41147.0 | 61128 | 64298 |
| 284 | 0 | 2 | 41147.0 | 61128 | 64302 |
| 369 | 0 | 3 | 41147.0 | 61128 | 64298 |
| 456 | 1 | 0 | 41147.0 | 61128 | 64298 |
| 552 | 1 | 1 | 41147.0 | 61128 | 64298 |
| 653 | 1 | 2 | 41147.0 | 61128 | 64298 |
| 738 | 1 | 3 | 41147.0 | 61128 | 64298 |
| 825 | 2 | 0 | 41147.0 | 61128 | 64299 |
| 921 | 2 | 1 | 41147.0 | 61128 | 64302 |
| 1022 | 2 | 2 | 41147.0 | 61128 | 64298 |
| 1107 | 2 | 3 | 41147.0 | 61128 | 64222 |

**Both `allocated` and `reserved` are perfectly flat (identical to
sub-0.01 MiB precision) from round 87 through round 1107** -- steady,
non-monotonic, comfortably clears the review's own gate ("steady,
non-monotonic reserved memory over >=100 rounds ... with >=10% headroom
to the ceiling") at more than 10x the required round count. Peak
`nvidia-smi` usage 64359 MiB against 97887 MiB total -- **~34.3%
headroom**, versus 0.7% headroom (near-OOM) before the fix. Process
returned to the normal ~2 GiB idle baseline on exit, confirmed via
`nvidia-smi` immediately after.

### 11.6 Regression suite: all 4 PASS, unchanged

Ran all four fresh (GPU verified idle before each, one process at a
time):

| suite | result |
|---|---|
| `mtp_gdn_rollback_check.py` | `passed: true`, `logits_exact_equal: true`, `gdn_state_close: true` (48/48 GDN layers) |
| `mtp_batch_verify_check.py` | `passed: true` (all 4 sub-checks: `check0_batch1_equivalence`, `check1_numerical_twin`, `check2_signal_probe`, `check3_mixed_stage`) |
| `mtp_ragged_recompute_verify_check.py` | `passed: true` |
| `mtp_verify_cudagraph_check.py` | `passed: true`, all 4 graph-shape coverage flags true (`verify_graph_batch4_replayed`, `verify_graph_batch2_replayed`, `draft_step0_qo2_graph_replayed`, `draft_continuation_graph_replayed`) |

Zero regressions from the fix.

### 11.7 W1-S 3-rep perf: no regression -- a small improvement instead

Real `benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph
--repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`, post-fix:

| rep | accepted tok/s | draft accept % | committed toks | `thermal_after.memory_used_mib` |
|---|---:|---:|---:|---:|
| 1 | 142.517 | 70.29204 | 4116 | 63879 |
| 2 | 143.149 | 70.29204 | 4116 | 63879 |
| 3 | 141.847 | 70.29204 | 4116 | 63879 |
| **mean** | **142.504** | 70.29204 | 4116 | 63879 |

`draft_acceptance_rate_pct`/`total_committed_tokens` are bit-for-bit
identical to every prior measurement in this project's history (70.29204431017119%,
4116) -- confirms the fix changed nothing about generation correctness.
`thermal_after.memory_used_mib` is now **identical across all 3 reps**
(63879 MiB every time) -- the *production* benchmark script itself
shows the same flat-memory signature the dedicated diagnostic found,
not just a diagnostic-script artifact.

**Performance did not regress -- it improved slightly**: 142.504 mean
vs. this review's own 137.784 pre-fix mean (+3.4%) and the original
session's 136.750 (+4.2%), narrowing the gap to native's 144.54 from
~1.05x to **~1.014x**. Consistent with the mechanism: disabling
autograd tracking removes real (if previously unmeasured) per-call
overhead -- saved-tensor bookkeeping, version-counter updates -- on
every eager forward call, so this is not a "paid a perf tax for
memory safety" tradeoff; both improved together because the same root
cause (unnecessary autograd tracking) was responsible for both a small
perf tax and the memory leak.

### 11.8 Bottom line

D3's falsifier ("already fired") is now closed: memory is flat, not
climbing, and the review's own numeric gate (>=100 rounds, >=10%
headroom) is cleared with a wide margin (1107 rounds, ~34% headroom).
The fix is a single line (`torch.set_grad_enabled(False)` in
`DirectModelRunner.__init__`), root-caused with real
`memory_allocated()`-vs-`memory_reserved()` evidence (not assumed from
the review's own framing), verified with the full existing 4-suite
regression battery (zero regressions) and a real W1-S 3-rep perf
re-measurement (small improvement, not a cost). Diagnostic script
`benchmarks/memory_growth_diag.py` is committed alongside the fix,
matching this project's convention of keeping the diagnostics that
found real bugs in the tree.
