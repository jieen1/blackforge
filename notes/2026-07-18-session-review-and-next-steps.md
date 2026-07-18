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

---

## 15. The skipped D1 cell measured: 32K/c=4 (executed 2026-07-18)

**Task scope: measurement only, per explicit instruction -- no production
code touched.** Section 12.5 deliberately skipped this exact cell
("32K/c=4 for this runtime: skipped deliberately... given 16K/c=4 already
peaked at 99.2% of GPU memory... this was a judgment call to not push
through recklessly"). That was before the D3 memory-leak fix (§11) and
the D1 vocab-logits fix (§13) landed, both of which independently lower
memory pressure at this shape; native's own 32K/c=4 number (32.941 accepted
tok/s, 94338/97887 MiB = 96.4%) was already known from §12's sweep. This
section fills in the missing cell with today's HEAD (which already
includes the grad-disable fix, the vocab-logits fix, and the cudagraph
correction from §14).

### 15.1 Fixture and command

Confirmed `benchmarks/fixtures/d1_ctx32k_prompts.json` exists (3.8 MiB) and
matches `D1_CTX32K_FIXTURE` in `benchmarks/workloads.py:215-222` exactly:
16 prompts, each verified to be exactly 32768 tokens
(`prompt_token_ids[i]` length checked directly, not assumed from the
JSON's own `prompt_len` field), same tokenizer/formula/seed as
`D1_CTX16K_FIXTURE`.

Command, following the same pattern §12/§13/§14 used for `ctx16k` (the
`--fixture` choices are wired in `mtp_w1s_our_runtime_perf.py:386`,
confirmed by reading the script directly rather than guessing):

```
python -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph \
  --fixture ctx32k --concurrency 4 --num-requests 4 --max-tokens 256
```

`--num-requests 4` was chosen to exactly match concurrency (one batch, no
repeats), mirroring native's own 32K/c=4 bounding choice from §12.1 ("4 at
32K, vs. the full 16 at 4K") and minimizing exposure given this cell's
known history of memory risk. Single rep, matching this project's
established D1-sweep convention (single rep, not the 3-rep protocol used
for the 4K/c=4 headline). `CUDA_HOME`/`PATH` pinned to the 13.3 toolkit;
GPU/process idleness verified via `nvidia-smi`/`pgrep` immediately before
starting (2450 MiB baseline, 0% util, no matching processes) and
immediately after finishing (1995 MiB, 0% util, no stray processes).

### 15.2 Memory monitoring methodology

A background loop sampled `nvidia-smi --query-gpu=memory.used,memory.total
--format=csv,noheader` every 5 seconds, started before the benchmark
process and stopped after it exited, logging every sample (not just
before/after) to catch the actual peak. Full log (31 samples spanning the
whole run, timestamps are unix seconds):

```
...(idle 2450-2544 MiB)...
t+20s   24378-24381 MiB   (model weights loading)
t+81s   28866 MiB         (weights loaded, runner init)
t+94s   46492 MiB         (transitioning into prefill)
t+99s   82768 MiB         (prefill/decode running)
t+99..126s  82768-82776 MiB, stable plateau for ~30s
t+131s  back to 2113 MiB  (process exited)
```

**Peak observed: 82776 MiB / 97887 MiB = 84.6% of capacity.** This is
comfortably below any near-OOM concern (well short of the 90%+ threshold
this task flagged for extra caution, and far below the 99.2%/99.3% peaks
this project hit before the D3/D1 fixes landed). The plateau held steady
for ~30 seconds across the whole prefill+decode body of the run (not a
single 5-second spike that 5s-granularity sampling might have missed),
giving good confidence this is a faithful peak, not an undersampled
transient. No safety action (no `--num-requests` reduction, no abort) was
needed -- the run completed cleanly on the first attempt.

### 15.3 Result

```json
{
  "num_requests": 4, "max_tokens": 256, "concurrency": 4, "k": 3,
  "batched": true, "cudagraph": true,
  "wall_s_e2e": 34.889,
  "num_drafts": 344, "num_draft_tokens": 1032, "num_accepted_tokens": 686,
  "draft_acceptance_rate_pct": 66.473,
  "total_committed_tokens": 1030,
  "accepted_tokens_per_sec": 29.522,
  "ttft_mean_ms": 29051.7,
  "gpu_busy_pct": 90.831
}
```

**Measured: 29.522 accepted tok/s.**

### 15.4 The gap, and a genuinely surprising trend reversal

Gap = native / ours = 32.941 / 29.522 = **1.116x** (native ~11.6% faster).

**This is UNDER this project's own 1.3x flag threshold** -- unlike 16K/c=4
(2.080x, flagged), 32K/c=4 does **not** flag. Put plainly: the gap got
*better*, not worse, going from 16K to 32K -- the opposite of what a
naive "residual near-linear-scaling compute cost, extrapolated further"
reading of §14 would predict.

The mechanism, checked directly from the two known numbers at each
context length (not re-profiled this round -- a re-profile is the natural
next step if this is ever chased further, explicitly not attempted here
per this task's measurement-only scope):

| | 16K -> 32K ratio (2x context) |
|---|---:|
| **ours** (58.638 -> 29.522 tok/s) | **1.986x** slower -- almost exactly linear |
| **native** (121.960 -> 32.941 tok/s) | **3.702x** slower -- far worse than linear |

**This CONFIRMS the §14 near-linear-scaling finding for our own runtime's
compute** -- 1.986x for a 2x context increase is about as clean a
near-linear-scaling confirmation as this project has measured anywhere.
What it does **not** confirm is any assumption that the gap-to-native
would stay flat or widen: native's own throughput degrades
*super*-linearly over this same doubling (3.702x, i.e. worse than
doubling attention's O(L^2) share would alone predict for a system where
FFN cost -- which scales linearly -- still made up the majority of cost at
16K). Because native's curve falls faster than ours, the ratio between
them shrinks as context grows, even though our own absolute throughput is
also dropping. This is a real, source-unconfirmed (i.e., not re-profiled
this round) but numerically clean observation, precisely the kind of
"worth a quick note, not a new investigation" finding this task's
instructions anticipated.

**Worth flagging for a future task** (not investigated further here, no
code touched, no new hypothesis tested): if this trend continues, there
may exist a context length beyond which this runtime's near-linear-scaling
compute cost curve actually crosses native's super-linear one, i.e. a
context length at which this runtime is *faster* than native rather than
merely closer to it. A 48K or 64K spot-check (native + ours, single
request/single batch to bound cost, same safety discipline as this
section) would confirm or refute this directly -- flagged as a natural,
cheap follow-up, not attempted this round.

### 15.5 Safety summary

No safety concerns materialized. Peak memory (84.6%) was well under any
threshold requiring `--num-requests` reduction or an abort; the single
planned run (`--num-requests 4`, one batch) completed on the first
attempt. GPU/process state confirmed idle before and after via direct
`nvidia-smi`/`pgrep` checks, not assumed.

### 15.6 Bottom line

| Context | Concurrency | Native tok/s | Ours tok/s | Gap (native/ours) | Flag |
|---|---:|---:|---:|---:|---|
| 16K | 4 | 121.960 | 58.638 | 2.080 | **>1.3x -- FLAG** |
| 32K | 4 | 32.941 | **29.522** | **1.116** | under 1.3x -- no flag |

The previously-skipped 32K/c=4 cell is now measured: **29.522 accepted
tok/s**, a **1.116x gap** to native (not flagged), peak memory **84.6%**
(safe, no near-OOM). The near-linear-scaling story from §14 holds for our
own runtime's compute (confirmed cleanly, 1.986x for 2x context); it does
**not**, however, mean the gap-to-native worsens proportionally --
empirically the opposite happened here, because native's own scaling
degrades faster than ours over this range. No new code was written or
changed this round; this is a pure measurement addition to the D1 sweep.

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

---

## 12. Phase D1 results: shape-generalization sweep (executed 2026-07-18)

**Verdict: the review's own falsifier fired, but on the OPPOSITE axis it
predicted.** The specific prediction ("at c=1 the cross-slot batching win
evaporates, the gap likely re-widens there") is **refuted** -- c=1 (and
c=2) are, in fact, where this runtime's advantage over native is
**largest** (up to ~1.45x faster than native, not slower). But a real,
previously untested weak spot was found on a different axis entirely:
**concurrent (c=4) batched prefill at long context (16K)**, where the
gap explodes to **4.85x slower than native** and the run peaked at
**99.2% of GPU memory capacity** (97110/97887 MiB) -- worse than
anything this project has measured before, including the D3 near-OOM
incident. This is reported in full per this project's own convention:
a "gains don't generalize this way" finding is exactly as valuable to
record as a clean parity result.

### 12.1 Methodology

**Grid covered**: concurrency c in {1, 2, 4} (the project's own scope
contract, `项目实施规划.md:23`, caps concurrency at 4) x context in
{4K, 16K}, plus a 32K spot-check at c=4 for native only (ours skipped
there for a documented safety reason, see 12.4). All cells: single rep
(NOT the 3-rep protocol), `--max-tokens 256`, `SM120_GQA_USE_V2_DECODE_
KERNEL=1` on both sides (matching this project's own established
practice for a same-kernel comparison, PROGRESS.md:2034), same model
(`unsloth/Qwen3.6-27B-NVFP4`), same `kv_cache_dtype=fp8_e4m3`.

**4K cells**: the existing frozen `w1s_prompts.json` (`W1_S_FIXTURE`,
n=16), full fixture, both sides -- directly comparable to every prior
headline number in this doc.

**16K/32K cells**: `W1_S_FIXTURE`'s own generation formula/seed/
tokenizer had no fixture at these lengths (`workloads.py`'s own
docstring already flagged the true W2-scale 32768 fixture as "not
built"). Two new, clearly-labeled, SAME-formula/SAME-seed constructed
fixtures were built for this task only:
`benchmarks/fixtures/d1_ctx16k_prompts.json` and
`d1_ctx32k_prompts.json` (16 prompts each, prompt_len=16384/32768),
added as `D1_CTX16K_FIXTURE`/`D1_CTX32K_FIXTURE` in
`benchmarks/workloads.py` with an explicit docstring that they are
**not** the official W2/W2-S line -- just this sweep's own synthetic
extension, following `generate_synthetic_fixtures.py`'s existing
convention. Wired into both `mtp_w1s_our_runtime_perf.py --fixture
ctx16k/ctx32k` and `w1s_native_bench.py --fixture ctx16k/ctx32k
--num-requests N` (a `--num-requests` slicing flag was added to the
native client, mirroring this runtime's own script, to bound cost at
long context: 8 requests at 16K, 4 at 32K, vs. the full 16 at 4K).

**Ragged prompt lengths**: confirmed genuinely unsupported, not just
undocumented -- `mtp_prefill_batch` (`runtime/direct_model_runner.py
:2400-2401`) raises `ValueError("mtp_prefill_batch requires every
slot's prompt to have equal length")` if lengths differ, exactly the
constraint the prior review's D2 section named. **Out of scope for this
sweep per the task's own instruction** -- not worked around with a
hack; flagged as a real, standing gap (tracked under D2, unchanged).

**Execution order and safety**: native's server was launched ONCE
(`launch_test_server.py --with-mtp --kv-cache-dtype fp8_e4m3 --model
unsloth/Qwen3.6-27B-NVFP4`, default `--max-model-len 262144`, already
enough headroom for every context length tested) and all 6 native cells
were run against it sequentially, then cleanly stopped
(`stop_test_server.py`, confirmed 0 compute processes + GPU back to
~2GB baseline) before starting any of this runtime's own process
(each invocation of `mtp_w1s_our_runtime_perf.py` reloads the full
model in a fresh process) -- this machine's tight 23GB system RAM
relative to the 21.81GB checkpoint (documented in the design doc's own
"two real infrastructure incidents" section) makes running both sides'
model loads concurrently a real risk, so the two legs were kept
strictly sequential, exactly as prior rounds did. `nvidia-smi`/`pgrep`
verified idle immediately before every leg, not assumed from memory.

### 12.2 The gap table

| Context | Concurrency | Native tok/s | Ours tok/s | Gap (native/ours) | Flag |
|---|---:|---:|---:|---:|---|
| 4K | 1 | 41.554 | 60.326 | **0.689** (ours 1.45x faster) | -- |
| 4K | 2 | 78.421 | 96.809 | **0.810** (ours 1.23x faster) | -- |
| 4K | 4 | 140.125 | 141.322 | **0.992** (parity, within noise) | -- |
| 16K | 1 | 30.476 | 37.190 | **0.819** (ours 1.22x faster) | -- |
| 16K | 4 | 121.960 | 25.137 | **4.852** (native 4.85x faster) | **>1.3x -- FLAG** |
| 32K | 4 | 32.941 | not measured | n/a | skipped, see 12.4 |
| 32K | 1 | not measured | not measured (aborted) | n/a | skipped, see 12.4 |

The 4K/c=4 cell cross-checks cleanly against this doc's own established
numbers: this single fresh rep (140.125 native / 141.322 ours) lands
inside the noise band of the already-verified 3-rep means (144.54
native / 142.504 ours, section 11.7) -- both sides' single-rep samples
fall within the ~9 tok/s rep-to-rep spread section 6 already
characterized, not a new discrepancy.

### 12.3 c=1 is NOT the weak spot -- the review's specific prediction is refuted

The review's falsifier (section 8/D1) predicted the cross-slot-batching
win would "evaporate" at c=1 and the gap would "re-widen." **The
opposite happened**: at both 4K and 16K, c=1 (and c=2 at 4K) are where
this runtime is *furthest ahead* of native (up to 1.45x faster), and
the advantage shrinks (not grows) as concurrency rises toward 4. The
likely mechanism, consistent with everything else this project has
already found: native vLLM's own per-step engine-loop/scheduler
overhead is a largely fixed *per-step* tax that amortizes worse at low
concurrency (fewer tokens committed per step to spread it over), while
this runtime's hand-rolled loop already eliminated most of that
overhead from the very first round of this project (before cross-slot
batching was ever added) -- so at c=1, where no batching benefit is
even possible on either side, the comparison mostly measures "removed
scheduler overhead" (a genuine, real, and apparently LARGER advantage
than previously reported, since it had never been isolated from the
batching win before this sweep) rather than "batching benefit," and
that removed-overhead advantage does not depend on concurrency.

### 12.4 The actual weak spot found: concurrent long-context prefill, both a throughput collapse AND a near-OOM

16K/c=4 is a materially worse result than anything in this project's
history: **4.85x slower than native** (not the ~1.3x ceiling the review
worried about), TTFT ballooning to a mean of **34.0 seconds** (p99 52.1s,
vs. 2.9s at 4K/c=4), and GPU memory peaking at **97110/97887 MiB (99.2%
of capacity)** -- higher even than the pre-D3-fix near-OOM figure
(97227/97887, section 6) this same doc already flagged as urgent. The
"warm" second batch within the same rep (21s wall vs. the first batch's
61s, suggesting a one-time capture/compile cost on the first batch) was
still only ~49 tok/s once isolated -- still far below native's 121.96,
so this is not purely a cold-start artifact.

**Source-grounded hypothesis for why (not a fix, out of scope for this
measurement task)**: `mtp_prefill_batch` (`runtime/direct_model_runner
.py:2377-2416`) issues exactly ONE non-chunked `_forward_batch(...,
qo_len=prompt_len, ...)` call covering every concurrent slot's FULL
prompt length in a single kernel launch -- confirmed by direct reading,
there is no chunking anywhere in this path. At 16K/c=4 that is a single
shot attending over qo_len(16384) x concurrency(4) = 65536 query
positions across every layer at once. Native vLLM, by contrast, runs
with `--enable-chunked-prefill` (`max_num_batched_tokens=8192`,
`launch_test_server.py`'s default) regardless of concurrency, so no
single native step ever processes more than 8192 tokens. This is a
plausible, code-citation-backed explanation for both symptoms at once
(large one-shot transient working set -> near-OOM; a shape far outside
anything this project's kernel work has tuned/validated for ->
throughput collapse) -- but it is a hypothesis from reading the code,
not a profiled root cause; confirming it would need an `nsys`/memory
trace, which is out of scope for this measurement sweep and is called
out below as the natural follow-up.

Critically, this is **not** simply "long context is slow" -- c=1 at
16K (batch=1, same 16384-token prefill, no concurrent multiplication)
was fine (ours *faster* than native, 12.4). The problem is specifically
the *product* of concurrency and context length in this one non-chunked
batched-prefill call, not either factor alone.

### 12.5 What was skipped, and why (stated explicitly, not silently)

- **32K/c=4 for this runtime**: skipped deliberately. Given 16K/c=4 already
  peaked at 99.2% of GPU memory and the mechanism above scales with the
  concurrency x context product, attempting a cell with double the context
  at the same concurrency risked a real OOM -- this was a judgment call to
  not "push through recklessly" per this task's own safety instruction,
  backed by concrete evidence (the 16K/c=4 result itself) rather than a
  guess. Native's own 32K/c=4 cell WAS run safely (94338 MiB, 96.4%,
  32.941 tok/s) since native's paged-KV allocator did not show the same
  scaling pathology.
- **32K/c=1 for this runtime**: attempted, then aborted after ~4.5 minutes
  stuck in weight-loading (vs. the usual 15-90s every other cell in this
  sweep took) with no forward progress in the log and fluctuating RSS --
  most likely a cold-page-cache effect from the preceding large-memory
  16K/c=4 run, not a capability problem (an earlier, smaller dry-run at
  this exact shape -- 1 request, 32 tokens -- had already completed
  cleanly, peaking at 71004 MiB, confirming the shape itself works
  mechanically). Killed cleanly (GPU/RAM confirmed back to idle
  immediately after) rather than let an anomalous run consume further
  budget for a cell whose likely answer (c=1 stays fine, per the 4K/16K
  pattern in 12.3) was already well-supported by two other c=1 data
  points.
- **32K/c=1 for native, and c=2 at 16K/32K**: not run at all, to keep the
  total sweep within budget -- deprioritized because the c=1-vs-c=4
  pattern was already unambiguous from the cells that WERE run, and c=2
  behaves as an intermediate point at 4K (0.81, between c=1's 0.69 and
  c=4's 0.99) with no reason to expect a qualitatively different story at
  longer context.
- **Ragged prompt lengths**: out of scope per the task's own instruction,
  confirmed genuinely blocked (12.1), tracked under the existing D2 item,
  unchanged.

### 12.6 Revised priority: this is now more urgent than the ~1.057x-chasing question Phase C considered

Phase C (section 8) concluded the remaining ~5-6 tok/s gap at 4K/c=4 was
not worth chasing. That conclusion is unaffected. But this sweep found
something that changes the priority ordering of the existing D-series
items: **16K/c=4's 99.2% memory peak is a more acute near-OOM signal
than the one that motivated D3** (which was fixed), and it recurs at a
shape (16K context, c=4, MTP K=3) squarely inside this project's own
stated target bucket (`项目实施规划.md`'s own three context buckets are
4K/16K/32K, concurrency 1-4) -- i.e., this is not an out-of-contract
stress test, it is the actual production shape space this runtime is
supposed to serve. Recommended next step (not attempted this round,
consistent with this task's own scope boundary against touching
`direct_model_runner.py`'s core logic): profile and chunk
`mtp_prefill_batch`'s single non-chunked forward call the same way
native's chunked-prefill already does, then re-run this exact 16K/c=4
cell as the gate for whether the fix worked.

### 12.7 Bottom line

- The review's specific c=1 prediction: **refuted** -- c=1 is this
  runtime's best relative shape, not its worst, at both context lengths
  tested.
- A real, more serious weak spot exists on a different axis: **concurrent
  batched prefill at long context** -- 16K/c=4 is 4.85x slower than
  native and comes within 0.8% of the GPU's full memory capacity.
- The 1.014x-1.057x "parity" headline from the earlier sections of this
  document is **real but shape-specific**, confirmed exactly as the
  review's own falsifier framed it (section 9): it holds at c=4/4K (and
  is actually a conservative floor relative to c=1-2/4K, where this
  runtime is well ahead of native) but does **not** generalize to
  concurrent long-context prefill, where it inverts sharply. Reporting
  "parity" without this qualification would have been misleading.

---

## 13. D1 follow-up: 16K/c=4 root-caused, fixed, verified (executed 2026-07-18)

**Verdict: the "no chunking" hypothesis section 12 offered is REFUTED as the
primary mechanism. The real root cause, found by direct instrumentation, not
inference: `mtp_prefill_batch`'s target-model forward AND its draft-model
step-0 sync forward both project EVERY position of the full
`concurrency x prompt_len` batch through the vocab head
(`compute_logits`), when only each slot's own LAST position is ever read.
At this shape (vocab_size=248320) that is a `[65536, 248320]` bf16 tensor
(~30.3 GiB) computed TWICE per prefill call, of which only 4 of 65536 rows
(0.006%) are ever used. Fixed with an opt-in, zero-risk-to-other-callers
parameter. Result: the 16K/c=4 gap to native narrows from 4.85x slower to
2.63x slower (throughput +84.6%, 25.137 -> 46.394 accepted tok/s), and the
99.2%-of-capacity near-OOM peak drops to 55.4%. The 4K/c=4 headline does
not regress (142.504 -> 147.656 mean, actually a small genuine
improvement). A real bug in this fix's own first draft was caught by the
existing regression suite and corrected before landing -- see 13.5.**

### 13.1 Methodology: confirming/refuting the chunking hypothesis with real evidence

Per this task's charter, the hypothesis was not assumed. Three independent
lines of investigation, in order:

1. **Read `mtp_prefill_batch`/`_forward_batch`/`_mtp_forward_batch` in full**
   (`runtime/direct_model_runner.py`). Confirmed: `mtp_prefill_batch`
   (`:2486` in the pre-fix file) issues exactly ONE `_forward_batch(...,
   qo_len=prompt_len, ...)` call for the target model, covering every
   listed slot's full prompt in one `model.forward()` -- no host-side
   chunking loop anywhere in this path, exactly as section 12 found.
   Additionally (not previously noted): the draft/MTP model's own step-0
   resync, reached via `_mtp_sync_and_propose_batch` ->
   `_mtp_forward_batch`, is called with the SAME `num_new_tokens=prompt_len`
   -- so the draft model ALSO does one giant non-chunked forward, not just
   the target model.
2. **Read native vLLM's real scheduler** (`/home/bot/vllm/vllm/v1/core/sched/
   scheduler.py:476-477`): confirmed `token_budget` (from
   `max_num_batched_tokens`, 2048 in this exact config per the launched
   run's own log line `Chunked prefill is enabled with
   max_num_batched_tokens=2048`) caps `num_new_tokens` PER SCHEDULER STEP
   across the WHOLE running batch, regardless of concurrency -- native
   never processes more than 2048-8192 tokens in one step, confirming the
   doc's citation. **But also read `gpu_model_runner.py:2192,4386`**:
   `logits_indices = query_start_loc[1:] - 1` and `sample_hidden_states =
   hidden_states[logits_indices]` -- native NEVER projects more than one
   row per running request through the LM head, in EVERY step (decode,
   chunked-prefill, or otherwise). This second mechanism turned out to be
   the one that actually matters here, not the chunking budget.
3. **Profiled the real call directly** (new script,
   `benchmarks/mtp_prefill_batch_memory_diag.py`, committed): monkey-patches
   `runner.model.forward`/`compute_logits` and
   `runner.mtp_model.forward`/`compute_logits` with thin wrappers recording
   wall time + `torch.cuda.memory_allocated()` before/after each real call,
   then calls the REAL, unmodified `mtp_prefill_batch(slots, prompts)` at
   the exact c=4/`D1_CTX16K_FIXTURE` shape. GPU/process hygiene verified
   idle via `nvidia-smi`/`pgrep`/`ps` immediately before every run in this
   section (one process at a time throughout; confirmed a mid-session
   coordinator check that flagged `nvidia-smi --query-compute-apps` as
   empty during model load was a known false-negative of this machine's
   WSL2 driver setup, not a hung process -- re-verified directly via `ps -p
   <pid> -o pcpu,stat,etime` showing 90%+ CPU and running state).

### 13.2 Pre-fix profile: the smoking gun

| call | time | memory delta | output shape | rows actually read |
|---|---:|---:|---|---:|
| `target_model.forward` | 12.18s | +640 MiB | `[65536, 5120]` bf16 | all (real compute) |
| `target_model.compute_logits` | **1.06s** | **+31040 MiB** | `[65536, 248320]` bf16 | **4 / 65536** |
| `draft_model.forward` (step0) | 0.44s | +640 MiB | `[65536, 5120]` bf16 | all (real compute) |
| `draft_model.compute_logits` (step0) | **15.20s** | **+31040 MiB** | `[65536, 248320]` bf16 | **4 / 65536** |
| `draft_model.forward`/`compute_logits` (steps 1-2) | ~0.02s total | ~0 | `[4, ...]` | all (already small) |

Total `mtp_prefill_batch` wall time: **29.197s**. Peak `torch.cuda
.max_memory_allocated()`: **95702 MiB**; `nvidia-smi` after: **97094 MiB
(99.2% of the 97887 MiB card)** -- matches section 12's reported
97110/97887 almost exactly, confirming this diagnostic faithfully
reproduces the flagged shape and configuration (`num_slots=concurrency`,
`enable_cudagraph=False` -- back-derived as section 12's own likely
configuration, since a `2x` cudagraph-reserved `num_slots` would have
made the KV-cache baseline alone already exceed capacity once this
waste is added, which did not happen).

**Two findings, not one:**
- The two `compute_logits` calls alone account for **16.26s of the 29.2s
  total wall time (55.7%)** -- MORE than the two real model forward passes
  combined (12.62s) -- for work that is 99.994% discarded.
- The SECOND huge allocation (draft model's, needing another 30.3 GiB when
  ~64 GiB is already resident) took **15.2s -- ~14x longer** than the
  essentially-identical-shaped FIRST allocation (1.06s, at a lower ~32 GiB
  baseline). This is direct evidence of near-OOM allocator-pressure
  pathology (consistent with this machine's own documented WSL2
  memory-allocation-is-slower gotcha) COMPOUNDING the waste, not just
  "a big GEMM takes longer."

### 13.3 Verdict on the hypothesis

**Refuted in its literal form, confirmed in spirit.** Chunking the
attention/FFN forward pass itself would not have fixed this: chunking a
GEMM into N pieces does not reduce its total FLOPs, only reorganizes
scheduling -- if `mtp_prefill_batch` were chunked but still called
`compute_logits` on every position of every chunk, the SAME total ~60 GiB
of wasted vocab-head compute would just be spread across more, smaller
calls, with the SAME total waste and (per the peak-memory mechanism above)
still enough transient pressure at c=4/16K to risk the near-OOM condition
depending on how the chunks overlap.

The real, chunking-orthogonal, and much larger-magnitude (`vocab_size`
=248320x factor, not a small constant factor) inefficiency is: this
runtime's own `mtp_prefill_batch` never adopted native's `logits_indices`
mechanism (project only the position(s) actually needed through the vocab
head) for its OWN batched-prefill path -- despite this exact idea being
directly available by inspection of the very vLLM code this whole runtime
is built alongside.

### 13.4 The fix

Added an opt-in `logits_last_position_only: bool = False` parameter to
`_forward_batch` and `_mtp_forward_batch` (default `False`, preserving
EVERY existing call site byte-for-byte -- `decode_batch`/`verify_batch`/
`verify_batch_spec` genuinely need every position's logits for real MTP
verification and never pass it), plus a `step0_logits_last_position_only`
parameter threaded through `_mtp_sync_and_propose_batch`. When set, the
hidden-state tensor is `index_select`-gathered down to one row per slot
(each slot's own last position) BEFORE the vocab-head projection, instead
of after -- the full, un-gathered `hidden_states` is still returned
unchanged where callers need it (e.g. the draft-model sync step still
needs the target model's FULL per-position hidden states to correctly
resync its own recurrent/attention state over the whole prompt; only the
`compute_logits` INPUT is sliced). Only `mtp_prefill_batch` sets these
flags to `True`, since it is the only caller that already only reads each
slot's own last-position logits/draft-token (confirmed by reading its own
body: `target_logits[i * prompt_len + prompt_len - 1]` for the anchor, and
`_mtp_sync_and_propose_batch`'s own `index_select(0, last_idx_tensor)` for
the first draft token -- both discard every other row already).

Files: `runtime/direct_model_runner.py` (`_forward_batch`,
`_mtp_forward_batch`, `_mtp_sync_and_propose_batch`, `mtp_prefill_batch`);
new diagnostic `benchmarks/mtp_prefill_batch_memory_diag.py` (committed,
matches this project's convention of keeping diagnostics that found real
bugs in the tree).

### 13.5 A real bug in this fix's own first draft, caught by the existing regression suite

The first version of this fix added a defensive `RuntimeError` inside
`_mtp_sync_and_propose_batch`, asserting that
`step0_logits_last_position_only=True` could never coincide with the
captured-draft-step-graph branch (reasoning: `mtp_prefill_batch`'s prompt
length is always far larger than `_MAX_DECODE_QO_LEN=16`, so the graph
branch's own size gate should never fire for it). **This assumption was
wrong** -- running the full regression suite immediately surfaced it:
`mtp_verify_cudagraph_check.py` deliberately calls `mtp_prefill_batch`
with a SHORT prompt (`"The capital of France is"`, a handful of tokens)
under `enable_cudagraph=True`, specifically as its own regression check
that the draft-step graph branch is correctly reachable from
`mtp_prefill_batch`'s step-0 call (see that file's own docstring, lines
89-95, written well before this round). In that legitimate configuration
`num_new_tokens_list[0] <= _MAX_DECODE_QO_LEN` IS true, the graph branch
IS taken, and the defensive check fired a hard `RuntimeError`, breaking a
previously-passing suite.

**Real fix**: track whether step 0's return value was ACTUALLY gathered to
last-position-only (`step0_already_last_only`), which is only true when
the EAGER branch ran AND the flag was requested -- not simply the raw
`step0_logits_last_position_only` parameter value. When the graph branch
is taken, the optimization is silently skipped (harmless: that branch only
fires for `num_new_tokens <= 16`, far too small for the vocab-head cost to
matter) and the pre-existing full-row `index_select` runs exactly as
before. No `RuntimeError` needed. Re-ran `mtp_verify_cudagraph_check.py`
after this correction: **PASS**, all 4 coverage flags true including
`draft_step0_qo2_graph_replayed: true` -- confirming the graph path was
genuinely exercised through this exact code and handled correctly. This is
exactly the kind of thing this project's standing "run the full
regression suite, not just the numbers" discipline exists to catch --
credited to the suite, not found by inspection first.

### 13.6 Before/after: the 16K/c=4 cell

Real profiling call (`mtp_prefill_batch_memory_diag.py`, same shape,
`num_slots=4`, no cudagraph):

| | pre-fix | post-fix |
|---|---:|---:|
| `target_model.compute_logits` | 1.06s, +31040 MiB | 0.0014s, +2 MiB |
| `draft_model.compute_logits` (step0) | 15.20s, +31040 MiB | 0.0014s, +3 MiB |
| total `mtp_prefill_batch` wall time | 29.197s | 12.689s |
| peak `torch` allocated | 95702 MiB | 45674 MiB |
| `nvidia-smi` peak | 97094 MiB (99.2%) | 54214 MiB (55.4%) |

Real end-to-end W1-S run at this shape (`mtp_w1s_our_runtime_perf.py
--batched --fixture ctx16k --concurrency 4 --num-requests 8 --max-tokens
256`, single rep, matching section 12's own D1 methodology -- 8 requests /
2 sequential batches of 4 concurrent slots, no `--cudagraph` since
back-deriving section 12's actual configuration from its memory-peak
number showed it did not use `--cudagraph` either):

| | pre-fix (section 12) | post-fix (this section) |
|---|---:|---:|
| accepted tok/s (ours) | 25.137 | **46.394** |
| native tok/s (unchanged) | 121.960 | 121.960 |
| gap (native/ours) | **4.852x slower** | **2.629x slower** |
| TTFT mean | 34.0 s | **12.5 s** |
| TTFT p99 | 52.1 s | **12.7 s** |
| GPU memory peak | 97110/97887 MiB (99.2%) | **54216/97887 MiB (55.4%)** |
| `gpu_busy_pct` | not reported | 90.83% (healthy) |

**Throughput improved 84.6% (25.137 -> 46.394 tok/s) and the near-OOM
condition is resolved** (55.4% peak, ~34 GB headroom instead of ~0.8%).
The remaining 2.63x gap to native is a real, smaller, and structurally
different problem -- see 13.8.

### 13.7 Confirmed no regression: 4K/c=4 headline and full 4-suite battery

**Headline** (`mtp_w1s_our_runtime_perf.py --batched --cudagraph --repeats
3 --max-tokens 256 --concurrency 4 --fixture n16`, fresh process, GPU
verified idle before):

| rep | accepted tok/s |
|---:|---:|
| 1 | 146.940 |
| 2 | 148.647 |
| 3 | 147.381 |
| **mean** | **147.656** |

vs. the established **142.504** baseline (section 11.7) -- **no
regression; a small genuine improvement (+3.6%)**, consistent with the fix
also trimming a smaller amount of the same waste at 4K context (`qo_len
=4096`, ~8x smaller than 16K's waste, but not zero). `draft_acceptance_rate
_pct` (70.29204431017119%) and `total_committed_tokens` (4116) are
bit-identical to every prior measurement in this project's history across
all 3 reps -- confirms zero change to generation correctness/determinism.
Note `mtp_prefill_batch`'s `qo_len=4096` here still exceeds
`_MAX_DECODE_QO_LEN=16`, so this run exercises the EAGER (optimization-
active) branch throughout, same as the 16K cell -- the graph-path
fallback in 13.5 is not exercised by either of these production shapes,
only by the short-prompt regression test.

**Full regression suite**, fresh processes, GPU verified idle before each,
one process at a time:

| suite | result |
|---|---|
| `mtp_gdn_rollback_check.py --repeat 3` | **3/3 PASS** |
| `mtp_batch_verify_check.py` | **PASS** (`check0`..`check3` all true) |
| `mtp_ragged_recompute_verify_check.py` | **PASS** (all sub-checks true) |
| `mtp_verify_cudagraph_check.py` | **PASS** (after the 13.5 correction; all 4 coverage flags true) |

Zero regressions.

### 13.8 What remains open: the residual ~2.63x gap

The confirmed, dominant root cause (wasted full-position vocab-head
projection) is fixed. The REMAINING gap is smaller and different in
character: `target_model.forward` itself -- the real attention+FFN
compute over 65536 positions across 64 layers -- still takes ~12.2s in
one uninterrupted call, and total wall time for the real 2-batch/8-request
run (44.3s) is dominated by this real compute, not by any further
identifiable waste. Two candidate explanations for the residual gap,
NEITHER confirmed here (reporting precisely rather than guessing, per this
task's own instruction):

- **Native's chunking may still matter, but for a different reason than
  raw compute reduction**: interleaving a 2048-token prefill chunk with
  other scheduler work could let native's engine loop overlap
  host-side/scheduling latency with GPU compute across chunks in a way a
  single giant blocking call cannot -- a scheduling-flexibility argument,
  not a FLOPs argument.
- **This runtime's attention kernel may be less efficient at this
  specific huge-`qo_len` prefill shape than at the small-`qo_len`
  decode/verify shapes it was tuned for** -- `build_attention_metadata_
  batch` explicitly routes `qo_len > _MAX_DECODE_QO_LEN` to a different
  ("general/chunked") kernel dispatch than the specialized fast decode
  kernel; whether that dispatch is well-tuned for `qo_len=16384` has not
  been separately profiled.

**Not attempted this round, and correctly so per this task's scope**: real
host-side chunking of `mtp_prefill_batch`'s forward call (splitting
`prompt_len` into e.g. 2048-token pieces, matching native's own
`max_num_batched_tokens`) is a plausible next lever for the residual gap,
but it is a materially larger, riskier, structural change than this
round's fix -- it requires correct incremental `kv_len`/position
bookkeeping across chunks for BOTH the target and draft models, careful
handling of the causal mask at chunk boundaries, and its own dedicated
correctness re-validation, not just a parameter default. This round's fix
was scoped to the CONFIRMED, dominant, safely-opt-in-fixable root cause;
the residual gap is real but smaller (2.63x, not 4.85x) and its own root
cause is not yet confirmed by direct profiling -- recommended as the next
D1 follow-up if pursued, not forced here.

---

## 14. D1 second follow-up: the residual ~2.63x gap, root-caused (executed 2026-07-18)

**Verdict: no single bug. Five candidate mechanisms were checked with real
profiling evidence, not assumption; four are refuted or shown negligible,
and one -- an asymmetric benchmark configuration (16K/c=4 was never measured
WITH `--cudagraph`, unlike the 4K/c=4 "parity" headline, which always uses
it) -- is real and, once corrected, recovers a substantial fraction of the
gap: 16K/c=4 goes from 46.394 to 58.638 accepted tok/s (+26.4%), narrowing
the gap to native (121.960, unchanged) from 2.629x to 2.080x. The remaining
~2.08x is a genuine, profiled, near-linear-scaling compute cost (prefill's
own forward pass), not a further bug -- see 14.6 for why this is not chased
further this round.**

### 14.1 Methodology

Reused this project's own Phase-0 nsys gap-ledger convention (notes/2026-
07-17-post-ragged-round-next-steps.md section 7): `nsys profile -c
cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt`, `nsys
export --type sqlite`, then direct SQL against `CUPTI_ACTIVITY_KIND_KERNEL`/
`_MEMCPY` and `NVTX_EVENTS` (not GUI eyeballing). `CUDA_HOME`/`PATH` pinned
to the 13.3 toolkit throughout; GPU/process idleness verified via
`nvidia-smi`/`pgrep`/`ps` immediately before every run, never assumed.

Two new diagnostics (both committed, following this project's convention of
keeping diagnostics that found real, quantified findings):

- `benchmarks/d1_prefill_shape_nsys_diag.py`: calls the real, unmodified
  `mtp_prefill_batch` at BOTH ctx16k (slots 0-3) and ctx4k (slots 4-7) in
  ONE process/ONE model load (avoids a second ~90s cold-start reload),
  each call under its own top-level NVTX range plus per-call sub-ranges
  (`target_model.forward`/`compute_logits`, `draft_model.forward`/
  `compute_logits`, same monkey-patch technique
  `mtp_prefill_batch_memory_diag.py` established) -- lets the kernel-family
  ledger be sliced per shape without editing `direct_model_runner.py`.
- `benchmarks/d1_decode_round_kvlen_diag.py`: after prefilling both slot
  groups, runs N real `mtp_verify_and_commit_batch` rounds (organic,
  feeding each round's real anchor/draft output into the next, exactly
  like `mtp_w1s_our_runtime_perf.py`'s own `_run_batch_batched`) on each
  group independently, timing every round, to isolate whether the
  decode/verify round-loop itself scales with kv_len.

All runs used the real production shape: concurrency=4, `unsloth/Qwen3.6-
27B-NVFP4`, `kv_cache_dtype=fp8_e4m3`, `SM120_GQA_USE_V2_DECODE_KERNEL=1`
(this project's own established convention for a same-kernel comparison).

### 14.2 Hypothesis 1 (compute scaling worse than linear): mostly refuted for the forward pass as a whole, confirmed but non-dominant for attention specifically

Direct nsys measurement of the real `mtp_prefill_batch` call, same process,
same model load:

| | ctx16k (qo_len=16384) | ctx4k (qo_len=4096) | ratio | token-count ratio |
|---|---:|---:|---:|---:|
| `target_model.forward` wall | 12.094s | 2.566s | **4.71x** | 4.0x |
| `mtp_prefill_batch` total wall | 12.627s | 2.695s | **4.69x** | 4.0x |

Only ~17-18% worse than perfectly-linear scaling for a 4x token-count
increase -- NOT a dramatic quadratic blowup. Kernel-family breakdown
(`CUPTI_ACTIVITY_KIND_KERNEL`, classified by kernel name) of the SAME two
calls:

| kernel family | ctx16k ms (% of wall) | ctx4k ms (% of wall) | ratio |
|---|---:|---:|---:|
| GEMM/FFN (`device_kernel`) | 5787.2 (45.8%) | 1370.8 (50.9%) | 4.22x |
| **attention** (`flash_attn_fwd_kernel_fp8kv`, 16-17 launches) | **2009.0 (15.9%)** | **136.9 (5.1%)** | **14.68x** |
| GDN/FLA | 842.9 (6.7%) | 205.9 (7.6%) | 4.09x |
| elementwise/norm/misc | 2835.7 (22.5%) | 694.8 (25.8%) | 4.08x |
| memcpy | 144.7 (1.1%) | 32.9 (1.2%) | 4.40x |
| no-kernel/no-memcpy gap | 313.8 (2.5%) | 86.3 (3.2%) | 3.64x |

The prefill NVTX range is **96.4% kernel-active** at ctx16k (96.4%
at ctx4k too) -- confirms prefill itself is essentially 100% real
GPU-bound compute at BOTH shapes, not host-dispatch-bound (unlike the OLD
decode/verify-round ledger in section 7, which found ~90% of round wall
time was host-side gap). This directly refutes hypothesis 4 (host-side
metadata-building Python loops scaling badly) for the prefill call: the
gap fraction barely changes between shapes (2.5% vs 3.2%) and is tiny in
absolute terms either way.

The attention kernel's own time DOES scale close to quadratically (14.68x
for a 4x token-count increase, consistent with causal self-attention's
inherent O(L^2) FLOPs -- expected, and equally unavoidable for native,
since chunking a causal prefix-attention computation does not change its
total FLOP count, already established in section 13.3). Its RELATIVE
weight in the whole forward pass triples (5.1% -> 15.9%), but GEMM/FFN
(which scales near-linearly, as expected since FFN cost is per-token) still
dominates at both shapes -- so attention's disproportionate growth is real
but is not, by itself, the dominant driver of the forward pass's overall
(near-linear) scaling behavior.

### 14.3 Hypothesis 3 (attention kernel dispatch): the already-built "v2" prefill kernel was tried and empirically REFUTED as a fix for this shape

Source reading first: `SM120GQAImpl.forward()` (`vllm/v1/attention/backends/
sm120_gqa.py`, confirmed identical to this project's own reference copy at
`sm120-flash-attention/vllm_integration/sm120_gqa_snapshot/sm120_gqa.py`)
routes `is_decode=False` calls (i.e., every real `mtp_prefill_batch`
target-model forward, regardless of prompt length -- `decode_qo_len` is
ALWAYS 0 when `is_decode=False`) to `flash_attn_sm120_fp8_kv_paged` (the
"general" kernel) UNLESS `SM120_GQA_USE_V2_PREFILL_KERNEL=1`, which this
project's runtime never sets (confirmed: `grep`-ing every benchmark script
for `SM120_GQA_USE_V2_PREFILL_KERNEL` returns zero hits, unlike
`SM120_GQA_USE_V2_DECODE_KERNEL=1`, set in ~30 scripts). The module
comment claims the v2 kernel (`flash_attn_sm120_fwd_prefill_v2_fp8kv_paged`)
is "verified 11.7-15.1% faster than native FlashInfer at the dense/
fixed-shape vertical slice" -- a real, plausible-looking lever.

**Correctness pre-check** (required before considering enabling it): ran
all 3 of that kernel's existing standalone correctness scripts
(`kernel/tests/test_prefill_v2_correctness.py`,
`test_prefill_v2_causal_probe.py`, `test_prefill_v2_paged.py`) fresh on
this machine -- **38/38 cases PASS** (cosine >0.999 vs. F.sdpa throughout),
including the exact production shape (QH=24/KVH=4, head_dim=256,
page_size=16 AND page_size=784, varlen batches, chunked-prefill-
continuation, causal-mask signal probes at page/tile boundaries).

**Performance measurement** (`d1_prefill_shape_nsys_diag.py` re-run with
`SM120_GQA_USE_V2_PREFILL_KERNEL=1`, log-confirmed via `sm120_gqa.py:990
"v2 prefill kernel path HIT"`, same shape, same process pattern):

| | general kernel (baseline) | v2 kernel | delta |
|---|---:|---:|---:|
| `target_model.forward`, ctx16k | 12.094s | 12.644s | **+4.5% slower** |
| `target_model.forward`, ctx4k | 2.566s | 2.651s | **+3.3% slower** |
| attention kernel time, ctx16k (nsys) | 2009.0ms (`flash_attn_fwd_kernel_fp8kv`) | 2333.0ms (`flash_attn_prefill_v2_fp8kv_paged`) | **+16.1% slower** |

**The v2 kernel is empirically slower, not faster, at this runtime's real
batched/paged/`page_size=16`/concurrency=4 shape** -- directly contradicting
its own validation claim, which was evidently measured at a different
(likely single-request, larger-page_size, "dense/fixed-shape vertical
slice") configuration that does not generalize to ours. Reported honestly
as a dead end for THIS shape; **not enabled** -- no change made to any
`SM120_GQA_USE_V2_PREFILL_KERNEL` default.

### 14.4 Hypothesis 3b (decode/verify round-loop scaling with kv_len): measured directly, found flat

`d1_decode_round_kvlen_diag.py`, 20 real organic rounds per group, same
process, same model load, no cudagraph (isolating this specific
mechanism):

| | kv_len~16384 | kv_len~4096 | ratio |
|---|---:|---:|---:|
| mean round time (`mtp_verify_and_commit_batch`) | 127.418ms | 120.164ms | **1.060x** |

Essentially flat for a 4x kv_len increase. Source-grounded explanation:
`self.decode_fixed_kv_split_size`/`max_num_splits` (`__init__`,
`runtime/direct_model_runner.py:1013-1016`) are derived ONCE from this
runner's `blocks_per_slot * block_size` capacity ceiling (targeting 64
splits/request), NOT from each call's live kv_len -- so a LONGER-context
slot actually gets MORE real split-KV parallelism (at kv_len=16384,
`ceil(16384/640)=26` real splits; at kv_len=4096, `ceil(4096/640)=7`),
which roughly compensates for its larger per-request attention cost. The
decode/verify loop (post-Phase-2-rewrite spec-decode-GDN mechanism) is
NOT a source of disproportionate 16K-specific slowdown when measured in
eager mode.

### 14.5 Hypothesis 2/E (CUDA-graph asymmetry): the real, dominant, verified factor

Source reading confirmed: `mtp_prefill_batch` NEVER takes the captured-graph
branch, at ANY context length -- `_mtp_sync_and_propose_batch`'s own
graph-eligibility gate requires `num_new_tokens_list[0] <= _MAX_DECODE_QO_LEN
(=16)` (`direct_model_runner.py:2377-2381`), and `mtp_prefill_batch` always
calls it with `num_new_tokens=prompt_len` (4096 or 16384, both `>>16`) -- so
prefill is correctly, symmetrically eager at BOTH shapes; this is not where
the asymmetry lives.

The real asymmetry is in how the two numbers being compared were actually
produced: the 4K/c=4 "parity" headline (section 11.7/13.7) is always run
with `--cudagraph` (`mtp_w1s_our_runtime_perf.py --batched --cudagraph
...`), which captures/replays the DECODE/VERIFY round loop via
`CapturedBatchDecodeGraph`/`CapturedMTPDraftStepGraph`. The 16K/c=4 number
this doc has reported so far (both the original 4.85x and the follow-up
2.63x) was run WITHOUT `--cudagraph` (section 12.1/13.6 explicitly say so).
This is an asymmetric methodology, not a native-vs-ours architectural gap.

**Direct re-measurement**, same command as the existing 16K/c=4 line
(`mtp_w1s_our_runtime_perf.py --batched --fixture ctx16k --concurrency 4
--num-requests 8 --max-tokens 256`) with `--cudagraph` added (the ONLY
difference -- `num_slots` auto-doubles to 8 per the script's own existing
`num_slots = 2 * concurrency if cudagraph else concurrency` logic, single
rep, GPU verified idle before):

| | without `--cudagraph` (prior number, section 13.6) | with `--cudagraph` (this section) |
|---|---:|---:|
| accepted tok/s (ours) | 25.137 → 46.394 (post D1-fix) | **58.638** |
| native tok/s (unchanged) | 121.960 | 121.960 |
| **gap (native/ours)** | **2.629x slower** | **2.080x slower** |
| TTFT mean | 12.5s | **12.458s** (unaffected, as expected -- prefill is eager at both) |
| `gpu_busy_pct` | 90.83% | 90.76% |
| GPU memory peak (nvidia-smi) | 54216/97887 MiB (55.4%) | 64050/97887 MiB (65.4%, no near-OOM concern) |
| draft_acceptance_rate_pct | (not directly comparable, different rep) | 71.94% (plausible, consistent with the ~70-72% range this project's history reports) |

**+26.4% throughput** (46.394 → 58.638 tok/s) from a pure configuration
change (no code touched) -- the SAME `--cudagraph` flag the 4K/c=4 headline
already relies on. Back-derived decomposition: since TTFT (~12.46s,
cudagraph-invariant) dominates total wall time at this shape (~35s for 2
batches of 4 requests), the round-loop's own internal speedup is actually
much larger than +26% in isolation -- round-loop-only time (total wall
minus 2x TTFT) drops from **~19.3s to ~10.1s** across the 2 batches, close
to a 2x improvement, consistent with cudagraph eliminating most of the
per-round Python/launch-dispatch gap this project's Phase-0 ledger (section
7.4) already quantified for the (pre-Phase-2) round mechanism -- it is
diluted to +26% overall only because prefill/TTFT, which cudagraph cannot
touch, is the larger share of wall time at 16K specifically (unlike at 4K,
where TTFT is a much smaller fraction of total wall time, so the same
round-loop speedup shows up as a smaller relative contribution to the
already-near-parity 4K number).

Hypothesis E (native's scheduler giving it a compute/dispatch overlap
advantage) was separately checked and refuted directly from vLLM source:
`vllm/config/scheduler.py`'s `SchedulerConfig.get_scheduler_cls()` selects
`AsyncScheduler` only `if self.async_scheduling` (truthy); native's own
`launch_test_server.py` never sets `async_scheduling`, so it stays at its
default `None` (falsy) and native resolves to the SAME synchronous
`Scheduler` this runtime's `build_vllm_config` explicitly selects
(`async_scheduling=False`). No asymmetry here -- both sides run the
identical (non-async) scheduling model.

### 14.6 What remains open: the residual ~2.08x gap, and why it is not chased further this round

After the cudagraph correction, the remaining ~2.08x gap is dominated by
the prefill/TTFT cost itself (~12.46s, of which 12.09-12.64s is
`target_model.forward`'s real compute, directly profiled in 14.2-14.3 to
scale close to linearly with token count -- not a bug -- with the one
concretely-available alternative kernel (v2 prefill) directly measured to
be SLOWER, not faster, at this shape). There is no further low-risk,
bounded lever identified by this round's profiling: both kernel-dispatch
alternatives available today (general vs. v2) were tried; host-side
metadata construction and the decode/verify round loop were both directly
measured and ruled out as disproportionate-at-16K factors; native's
scheduling model was confirmed identical, not more overlapped.

The one remaining, NOT-yet-tried lever is the same one section 13.8 already
flagged and declined to attempt: real host-side CHUNKING of
`mtp_prefill_batch`'s single giant forward call (e.g. into 2048-token
pieces, matching native's own `max_num_batched_tokens`). This round's own
measurements sharpen why this is genuinely uncertain rather than a likely
win: chunking would not reduce total FLOPs (section 13.3), and this
round's own profiling shows the forward pass ALREADY scales close to
linearly with token count with no evidence of a fixable inefficiency at
the current single-shot granularity -- so chunking's plausible benefit, if
any, would have to come from a qualitatively different mechanism (e.g.
overlapping host-side dispatch of chunk N+1 with GPU execution of chunk N,
or reduced peak working-set enabling better cache locality) rather than
"removing waste," and is unconfirmed by any measurement in this round. It
is also, as section 13.8 already noted, a materially larger and riskier
change (correct incremental `kv_len`/position bookkeeping across chunks
for BOTH the target and draft models, causal-mask correctness at chunk
boundaries, its own dedicated correctness re-validation) than anything
attempted this round -- correctly out of scope per this task's own
structural-change boundary, recommended as the next D1 follow-up if ever
pursued.

### 14.7 What was changed, and verification

**No file under `runtime/` was modified.** This investigation found no
code bug in the current HEAD -- both CUDA-graph support and the v2 prefill
kernel already existed, already worked correctly, and were already
covered by existing tests; the gap was a previously-unexamined,
undocumented ASYMMETRY in how the 16K/c=4 shape was being benchmarked
relative to the 4K/c=4 headline (methodology, not implementation). The
corrective action is: **future context-length sweeps of this shape should
use `--cudagraph`**, matching the 4K/c=4 headline's own established
convention, and the 58.638 tok/s / 2.080x-gap number (not 46.394 / 2.629x)
is now this project's best-known, correctly-configured result for 16K/c=4.

Two new diagnostic scripts added and committed (`benchmarks/
d1_prefill_shape_nsys_diag.py`, `benchmarks/d1_decode_round_kvlen_diag.py`),
following this project's established convention of keeping diagnostics
that produced real, quantified findings.

**Verification** (full standing rigor, per this task's own instruction):

- **4K/c=4 headline: no regression, because no code changed.** Since zero
  lines of `runtime/direct_model_runner.py` (or any other production file)
  were touched, the previously-verified 147.656 tok/s mean (section 13.7,
  3 reps) is mathematically unaffected -- re-running it would exercise a
  byte-identical code path. Not re-run this round to conserve GPU time for
  the new measurements above; flagged explicitly rather than silently
  assumed.
- **Full regression suite: 4/4 PASS**, fresh processes, GPU verified idle
  before each, one process at a time:

  | suite | result |
  |---|---|
  | `mtp_gdn_rollback_check.py --repeat 3` | **3/3 PASS** |
  | `mtp_batch_verify_check.py` | **PASS** (`check0`..`check3` all true, `no_cross_contamination_signal: true`) |
  | `mtp_ragged_recompute_verify_check.py` | **PASS** (all sub-checks true) |
  | `mtp_verify_cudagraph_check.py` | **PASS** (all 4 coverage flags true: `verify_graph_batch4_replayed`, `verify_graph_batch2_replayed`, `draft_step0_qo2_graph_replayed`, `draft_continuation_graph_replayed`) |

  Zero regressions -- expected, since no production code changed, but
  confirmed rather than assumed, per this project's standing rigor.

### 14.8 Bottom line

- The ~2.63x 16K/c=4 gap left open by section 13 is **not one bug**: four
  candidate mechanisms (prefill-forward superlinear scaling, the general-
  vs-v2 attention kernel choice, decode/verify round-loop kv_len scaling,
  native scheduler overlap) were checked with real profiling evidence and
  found refuted or non-dominant.
- The one REAL, verified, and substantial factor: **the 16K/c=4 number was
  never measured with `--cudagraph`**, unlike the 4K/c=4 headline. Adding
  it (a pure configuration change, zero code risk, already covered by the
  existing regression suite) recovers **+26.4% throughput**, narrowing the
  gap from **2.629x to 2.080x**.
- The remaining ~2.08x is a genuine, directly-profiled, near-linear-scaling
  compute cost in the single-shot prefill forward pass -- not a bug, and
  not fixable by either available kernel choice (v2 measured slower, not
  faster, at this shape). The only remaining lever is real prefill
  chunking, a structurally bigger, riskier change correctly left as a
  scoped recommendation for future work, not attempted this round.
