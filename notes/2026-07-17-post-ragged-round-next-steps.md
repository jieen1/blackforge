# 2026-07-17 Post-ragged-round next steps: independent re-diagnosis and revised plan

Status: proposal (research/planning round, no production code touched).
Scope: answers four questions -- (1) is the "16-CTA / 16.7% occupancy is the
root cause" diagnosis complete? (2) should the pre-existing plan
`~/.claude/plans/sharded-discovering-summit.md` be executed next? (3) is
raising concurrency above 4 the right alternative? (4) what should actually
happen next. Everything below was re-derived by reading the current source,
not by trusting prior rounds' summaries; every substantive claim carries a
file:line citation.

---

## 1. Bottom line

**Neither the existing plan (FlashInfer-style occupancy-driven split
scheduling) nor raising concurrency is the right next lever.** Both attack
"how many CTAs does the attention kernel get" -- but native vLLM runs the
**same attention kernel, at the same c=4 shape, with the same fixed-split
config** (`launch_test_server.py` defaults to `--attention-backend CUSTOM`;
confirmed in `notes/direct-model-runner-design.md:3746-3750`) and still
achieves 144.54 accepted tok/s vs our 18.54. Whatever separates us from
native at c=4 is by construction **not** a property of that kernel's
occupancy at this concurrency.

Re-reading the runner with fresh eyes surfaces a set of concrete,
source-confirmed structural differences between our per-verify-round work
and native's -- several of them never priced by any prior round, and one of
them (a ~600MB-per-round synchronous GPU→CPU GDN state snapshot) plausibly
larger than everything previously investigated combined. The revised plan
is: **measure the round with `nsys` once (Phase 0), then remove the four
structural per-round costs in measured-impact order (Phases 1-3), and only
then revisit split micro-tuning (Phase 4).** Target: within ~1.3x of native
(≥110 accepted tok/s) at W1-S; native's own number is the existence proof
that this is reachable at c=4 with these exact kernels.

---

## 2. Re-diagnosis: the occupancy story is true but cannot be the main story

### 2.1 The two "GPU busy" metrics contradict each other, and the docs reconciled them incorrectly

- Our "~95% GPU-busy" comes from ONE `torch.cuda.Event` pair bracketing the
  **entire** `mtp_verify_and_commit_batch` call
  (`benchmarks/mtp_w1s_our_runtime_perf.py:179-190`). Event `elapsed_time`
  measures the wall-clock span between the two events' completion on the
  stream -- it **includes every intra-call gap** where the GPU sits idle
  waiting for Python to dispatch the next kernel, build metadata on the
  host, or finish a `.item()` round-trip. It is a "fraction of wall time
  inside bracketed regions" metric, not a "fraction of time a kernel is
  executing" metric. (The earlier trace-probe reading of ~98-101% -- above
  100% -- is itself a tell that this metric is span-based, not
  kernel-active-based.)
- The coordinator's `nvidia-smi utilization.gpu` ≈ 30% is, per NVML's own
  definition, "percent of time over the sample period during which one or
  more kernels was executing". It does **not** measure SM occupancy of a
  running kernel -- a 16-CTA kernel counts as 100% "utilized" while it runs.
  The design doc's reconciliation ("busy% = is a kernel running; utilization
  = how much of the SM array one launch uses",
  `notes/direct-model-runner-design.md:3695-3703`) has the two definitions
  **swapped**: it assigned the occupancy interpretation to the one metric
  that actually measures kernel-resident time.

**Implication, stated plainly**: `utilization.gpu` ≈ 30% is direct evidence
that for roughly **70% of wall time, no kernel is executing at all** during
our batched MTP run. That is the opposite of "launch/dispatch idle time was
never the bottleneck." The ~95% figure never had the resolution to see this.
(Copy-engine activity -- see §2.3 item 1 -- also does not count toward
`utilization.gpu`, consistently explaining how both numbers coexist.)

### 2.2 Why 16-CTA occupancy cannot explain ours-vs-native

The `ncu` finding (16 CTAs, 16.7% occupancy for
`flash_attn_decode_v2_fp8kv_paged_split`, design doc:3820-3853) is real and
correctly measured. But:

- Native launches the **same kernel** at the same batch=4, qo_len=4 shape,
  with the same `_DECODE_TARGET_SPLITS_PER_REQ = 64` fixed-split derivation
  (`vllm/v1/attention/backends/sm120_gqa.py:172,378-379`) that our Finding-2
  fix now replicates (`runtime/direct_model_runner.py:759-762`). Native's
  verify-attention grid is the same ~112 CTAs as ours post-fix. Same
  occupancy, 7.8x different end-to-end throughput ⇒ occupancy of this
  kernel is not the differentiator.
- Scale check: the entire split-KV lever, going from 1 split to 7 splits
  (7x more CTAs), bought +13% end-to-end (16.61→18.78, PROGRESS.md
  Finding 2). The attention kernel's whole time share is bounded by that
  observation. It cannot hide a 7.8x.

Low per-launch occupancy at c=4 is the correct explanation for why **native
itself** cannot go much faster than 144 tok/s on this GPU (why c=4 doesn't
saturate 188 SMs). It is not the explanation for why **we** are 7.8x below
native. Prior rounds' summary (design doc:4050-4059, and the
`project_kernel_milestone_reversals_2026_07` memory) conflated these two
questions.

### 2.3 What actually differs from native, per verify round (source-confirmed)

Our batched round (`mtp_verify_and_commit_batch`,
`runtime/direct_model_runner.py:1809-1963`) vs native's (real vLLM step with
CUDA graphs + spec-decode-aware GDN):

1. **GDN state snapshot to CPU, every round, every slot**
   (`direct_model_runner.py:1873` → `snapshot_gdn_state:1214-1219`, which
   does `.detach().to("cpu", copy=True)` per layer). Volume: ssm_state per
   layer per slot = 48 v-heads × 128 × 128 × fp32 ≈ 3.15MB
   (`config.json: linear_num_value_heads=48, linear_key_head_dim=128,
   linear_value_head_dim=128`); × 48 GDN layers ≈ **151MB per slot**, × 4
   slots ≈ **~604MB of synchronous, pageable D2H PCIe traffic per verify
   round** (384 individual blocking copies), plus the mirror-image H2D on
   restore for every recompute slot (84.4% of rounds have ≥1;
   `restore_gdn_state:1249-1253`). At realistic pageable-PCIe-under-WSL2
   rates (3-8GB/s) this is plausibly **75-200ms of the measured 292.8ms
   fully-batched round** -- the single largest unpriced candidate. It is
   round-count-invariant (identical in every configuration ever
   benchmarked), which also cleanly explains why three rounds of
   launch-count reduction left round time flat. Native does none of this
   (see item 3). The docstring's rationale for CPU residency ("avoid
   holding extra persistent GPU memory", :1183-1186) buys ~600MB of VRAM
   at a machine measured to have ~36GB headroom (PROGRESS.md capacity
   round).
2. **A second full target-model forward on 84.4% of rounds**
   (`direct_model_runner.py:1909-1919`): the recompute branch re-runs all
   64 layers (plus an lm_head `compute_logits` whose output is discarded --
   `_forward_batch:1087` always computes logits; the caller at :1909 keeps
   only hidden states). Native never re-forwards the target on partial
   accept: its GDN kernels commit state only for accepted tokens (item 3),
   and hidden states for the accepted prefix are already rows of the verify
   pass's output.
3. **Chunked-prefill GDN kernels instead of the spec-decode fused-recurrent
   path, 48 layers, every verify round**
   (`build_gdn_metadata_batch:533-580` declares every qo_len>1 batch as
   `num_prefills=N, num_spec_decodes=0` with chunk metadata). Native's
   builder takes the real spec branch
   (`vllm/v1/attention/backends/gdn_attn.py:198-288`), driving
   `causal_conv1d_update(..., num_accepted_tokens=...)` and the
   fused-recurrent spec kernels with per-position intermediate state slots
   (`qwen_gdn_linear_attn.py:29,1318,1345-1357`) -- which is precisely the
   mechanism that makes items 1 and 2 unnecessary for native. Our code
   documents this as a numerically-correct scope choice
   (:491-504) -- correct, but its *cost* (chunk-64-shaped kernels + extra
   launches for 16 real tokens, ×48 layers, plus forcing the
   snapshot/recompute architecture) was never measured.
4. **Eager execution with ~10+ full-device syncs and per-slot `.item()`
   round-trips per round** vs native's CUDA-graphed step:
   `_forward_batch:1086,1088` and `_mtp_forward_batch:1648,1650` each
   contain two `torch.cuda.synchronize()`; a round runs 2 target + up to 6
   draft forwards, plus per-slot `int(logits.argmax().item())` loops
   (:1723-1729,1743-1751,1792-1793). Native's server runs with CUDA graphs
   by default (`--enforce-eager` is opt-in:
   `vllm_integration/launch_test_server.py:16,26`), and the W1-S native leg
   did not pass it. Combined with §2.1's ~70%-no-kernel-executing evidence,
   eager dispatch starvation is a first-class suspect, **contrary to the
   sessions' repeated "launch overhead ruled out" conclusion, which rested
   on the span-based metric**.
5. **Full-vocab logits computed for all prompt positions at prefill**:
   `mtp_prefill_batch` runs `_forward_batch` with `qo_len=prompt_len`
   (:1780-1788) and `_forward_batch:1087` computes logits for every row --
   4×4096 rows × 151936 vocab (~25 TFLOP + a multi-GB activation) when only
   4 last-position rows are used (:1792). Secondary (once per request), but
   free to fix.

None of these five require new kernels; items 1-4 all have a working native
reference implementation already running on this machine.

---

## 3. Verdict on `sharded-discovering-summit.md`

**Its FlashInfer reading is accurate; its premise is no longer the right
lever; do not execute it as the next step.**

- Verified against real source this round: `max_grid_size =
  num_blocks_per_sm * num_sm` via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  (`flashinfer/include/flashinfer/attention/scheduler.cuh:~178-190`);
  `padded_batch_size = split_kv ? max_grid_size / gdy : batch_size` under
  CUDA graphs (:~441-444); the per-call
  `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch` binary search
  (:~198); the dummy-shape CUDA-graph anchor comment (:~578-586). The
  plan's four "已经读懂" findings all check out. Notably the same code also
  shows `if (batch_size * gdy >= max_grid_size) split_kv = false` --
  FlashInfer itself treats split-KV purely as a fill-the-GPU device, which
  at our shape would yield roughly 2-3x more CTAs than our current
  64-target derivation (~23 vs 7 splits at kv_len≈4096). Given 1→7 splits
  bought +13%, 7→23 is worth single-digit percent at best.
- The plan was written against a different problem: the **closed**
  attention-kernel line's 1.37x backend-vs-FlashInfer gap (its own Context
  section cites 20.07 vs 14.66 ms/step). That line was formally stopped
  after both its gates failed
  (`sm120-flash-attention` commits `0f56006`, `bdc9e7e`); `0f56006`
  additionally measured that at real chunked-prefill shapes split-KV
  contributes ~0.48% weighted because grids already exceed 188 SMs -- i.e.
  the plan's lever was measured near-zero on the prefill side, and §2.2
  bounds it to single-digit percent on the decode/verify side.
- Its six Phase-0 completion criteria, re-judged for the runtime problem:
  item 5 (the gap ledger with a bounded unexplained residual) is exactly
  right and is adopted as this plan's Phase 0; item 2 (how vLLM's
  dummy-shape reaches the builder) becomes relevant only in Phase 3 (graph
  wiring); items 1, 3, 4, 6 serviced the closed 1.37x question and should
  not be revived for this gap.
- Salvage list: occupancy-driven `max_grid_size` as a replacement for the
  hand-tuned splits-target constant is a legitimate Phase-4 refinement;
  the fixed-buffer + early-exit CUDA-graph-safety pattern it describes is
  the pattern Phase 3 already uses.

## 4. Verdict on raising concurrency above 4

**Rejected as a primary direction.**

- Out of contract: the project's own scope table fixes 并发 at 1-4
  (`项目实施规划.md:23`); the workload is 4 coding agents, and the stated
  goal is per-request speed for them, not aggregate throughput.
- Not necessary: native's 144.54 tok/s **at c=4** with the same kernels is
  an existence proof that c=4 does not cap this GPU at 18.54. The gap to
  close is ours-vs-native at c=4, and §2.3 accounts for it without any
  concurrency change.
- The math it does affect: verify-attention CTAs scale linearly with slots
  (~28/slot post-split-fix), so c=8/c=16 would raise that one kernel's
  occupancy toward saturation -- relevant only to pushing *beyond* native's
  ceiling after parity, at the cost of worse per-request ITL on
  memory-bound forwards. Revisit only if, post-parity, the user explicitly
  wants aggregate throughput over per-agent latency.

---

## 5. Revised plan

Standing verification discipline applies to every phase: full existing MTP
suites re-run after any production change; W1-S 3-rep protocol
(`mtp_w1s_our_runtime_perf.py --batched`, n=16, K=3, c=4,
`spec_decode_num_drafts`-normalized) as the end-to-end gate; no
kernel-or-microbenchmark win claimed as an end-to-end win without the W1-S
number.

### Phase 0 -- one `nsys` gap ledger of the real batched round (0.5-1 GPU-day)

Deliverable: a table decomposing (a) one fully-batched round (~292.8ms
class) and (b) one 2-recompute round into: kernel-active time by family
(GEMM / GDN-chunked / `flash_attn_decode*` / elementwise / draft-model),
`cudaMemcpy` D2H+H2D time, and no-kernel gap time -- with **unexplained
residual ≤25%** (the old plan's own standard). Plus three targeted host
timers, since they are near-free: `snapshot_gdn_state` alone (×4 slots),
one `verify_batch` call, one draft step. Cross-check `nvidia-smi
utilization.gpu` sampling during the same run.

Predictions this ledger tests (each is a falsifier for a later phase):
- P1: D2H/H2D memcpy ≥ 25% of round wall time (→ Phase 1). If it measures
  <10ms/round, Phase 1 is demoted to cleanup and the §2.3-item-1 estimate
  was wrong.
- P2: GDN-family kernel time per verify forward ≫ native's ~8% share, and/or
  the second target forward ≈ a full verify forward's cost (→ Phase 2).
- P3: no-kernel gap time ≥ 30% of round wall time (→ Phase 3). If gaps
  measure small, §2.1's reinterpretation is wrong, the span-metric was
  fine after all, and Phase 3's expected win shrinks to the syncs it
  removes.
- P4: `flash_attn_decode*` ≤ 10% of round time (confirming §2.2's bound on
  the old plan's lever).

### Phase 1 -- GPU-resident GDN snapshot/restore (≤1 day)

Replace `.to("cpu", copy=True)`/`.to(self.device)` in
`snapshot_gdn_state`/`restore_gdn_state` with a preallocated device-side
double buffer (+~604MB VRAM, measured headroom ~36GB) and D2D `copy_`
(~0.4ms at HBM rates vs ~75-200ms over PCIe). API unchanged.
Gate: `mtp_gdn_rollback_check.py` + both full suites pass; W1-S improves by
approximately the Phase-0-measured snapshot share (predicted 1.3-2x if P1
holds). Falsifier: P1 fails ⇒ skip (do it later as hygiene, expect ~0%).

### Phase 2 -- adopt native's spec-decode GDN path; delete snapshot/recompute (3-5 days)

Build the real spec branch of `GDNAttentionMetadata`
(`spec_sequence_masks`/`spec_token_indx`/`spec_state_indices_tensor`/
`num_accepted_tokens`, per `gdn_attn.py:198-288`) for the fixed-slot
runner: allocate K+1 intermediate state rows per slot (~+1.8GB VRAM at
K=3), thread `num_accepted_tokens` from `determine_accept_reject` into the
next round's metadata. This runs verify through
`fused_recurrent`/`causal_conv1d_update` (native's kernels, in the
installed vLLM, exercised in production on this exact model+GPU) instead of
chunk-64 prefill kernels, and makes state commit acceptance-aware --
allowing removal of `snapshot_gdn_state`/`restore_gdn_state` AND the
recompute target forward (accepted-prefix hidden states are rows of
`verify_hidden`, already returned at :1875-1877).
Gate: new committed-content equivalence test (current path vs spec path,
same drafts, forced partial/full rejects) + full suites; W1-S re-measure.
Falsifier: P2 fails (GDN+recompute measured small) ⇒ expected gain is
correspondingly small; still worth doing for Phase 3 (a single-forward
round is much simpler to capture in one graph), but say so honestly.

### Phase 3 -- CUDA-graph the MTP round; kill the sync ping-pong (2-4 days)

`CapturedBatchDecodeGraph` is already bytewise-parity-verified at qo_len=1
and qo_len=4 (PROGRESS.md, `cudagraph_eager_parity_check.py`); wire it into
`mtp_verify_and_commit_batch` for the target verify forward first, then the
K draft steps (batched on-GPU argmax feeding the next step's static input
buffer, one host sync per round instead of ~10+ `synchronize()` +
per-slot `.item()`). Use the old plan's salvaged insight here: anchor
graph-capture worst-case shapes the way FlashInfer/vLLM do (dummy-shape,
not `max_model_len`).
Gate: sanitizer-lite regression set + signal probes; W1-S re-measure;
`utilization.gpu` during steady state should rise sharply if P3 held.
Combined post-Phase-3 target: **≥110 accepted tok/s** (within ~1.3x of
native). If we land materially short with the ledger's residual still
≤25%, the honest conclusion is that the remaining delta is in per-kernel
efficiency at M=16 -- at that point (and only then) do a per-kernel-family
duration diff against a native `nsys` trace at the same shape.

### Phase 4 -- residuals, only if their measured share warrants (opportunistic)

Last-position-only logits at prefill; skip `compute_logits` on
hidden-only calls; occupancy-driven split target (old plan's item 1) if
post-Phase-3 `ncu` shows `flash_attn_decode*` ≥10% of round time.

### Explicit non-goals for this cycle

More coordinator-level batching rounds (three tried, plateaued -- no new
mechanism on the table); concurrency >4 (§4); executing
`sharded-discovering-summit.md` as written (§3); reopening the closed
attention-kernel line (its A4 reopen conditions are unmet -- nothing here
requires a new kernel).

---

## 6. What would prove this whole re-diagnosis wrong

If Phase 0's ledger shows kernel-active time ≈ wall time (P3 false), memcpy
negligible (P1 false), and GDN-family time comparable to native's share (P2
false) -- i.e. the round really is ~293ms of dense, necessary kernel work --
then §2's reinterpretation is wrong, the prior rounds' "the cost lives
inside GPU-busy time" reading was right after all, and the correct next
step becomes the per-kernel ours-vs-native duration diff (same shapes, two
`nsys` traces), not any of Phases 1-3. That check is built into Phase 0
precisely so this plan cannot run on autopilot past its own premise.

Also honest about the endpoint: reaching parity validates that the runtime
can *stand in* for vLLM on this workload; it does not yet validate the
founding premise (beating vLLM through specialization). Whether to pursue
"beyond parity" (persistent/fused decode kernels, prefill-decode overlap --
real specialization advantages a general library can't take) is a user
decision to make once parity-band numbers are on the table, not an
assumption to build into this cycle.

---

## 7. Phase 0 results (executed 2026-07-17)

Profiling-only round, per this document's own Phase 0 scope. No file under
`runtime/` was modified. One new script,
`benchmarks/phase0_nsys_gap_ledger_diag.py`, was added -- it calls
`DirectModelRunner`'s existing, unmodified methods (`snapshot_gdn_state`,
`verify_batch`, `restore_gdn_state`, `_forward_batch`,
`_mtp_sync_and_propose_batch`, plus the module-level
`determine_accept_reject` helper) directly, in the same order and with the
same arguments `mtp_verify_and_commit_batch` itself uses internally
(`runtime/direct_model_runner.py:1809-1963`), so each phase gets its own
NVTX range and host timer -- something not obtainable from outside a single
umbrella call without editing that file. Cross-check: summing the script's
own per-phase timers reproduces the printed round total to within ~1ms
(e.g. round 20: phases sum to 679.88ms vs the printed 680.98ms), confirming
the inline replay is a faithful stand-in for the real call, not a
materially different code path.

Run: `nsys profile -c cudaProfilerApi --capture-range-end=stop
--trace=cuda,nvtx,osrt -o phase0_ledger --force-overwrite=true
python -m benchmarks.phase0_nsys_gap_ledger_diag --num-rounds 50 --fixture n16`,
`CUDA_HOME`/`PATH` pinned to the 13.3 toolkit (`nvcc --version` confirmed
13.3 before running). The profiled region (`cudaProfilerStart`/`Stop`)
excludes model loading; it covers one `mtp_prefill_batch` call (4 slots,
4096-token W1-S prompts) followed by 50 natural (unforced) verify rounds,
then the 3 isolated host-timer calls. A `nvidia-smi
--query-gpu=utilization.gpu --format=csv,noheader -l 1` sampler ran
concurrently, logged to a separate file. Report exported via `nsys export
--type sqlite`; all numbers below come from direct SQL queries against
`CUPTI_ACTIVITY_KIND_KERNEL`/`_MEMCPY`/`_MEMSET` and `NVTX_EVENTS`, not
GUI eyeballing. Kernel-family classification was built from an exhaustive
enumeration of all 82 distinct kernel names actually observed in this
trace (not a guessed regex) -- an initial classifier missed several real
GDN/FLA kernel names (`chunk_fwd_kernel_o`, `recompute_w_u_fwd_kernel`,
`merge_16x16_to_64x64_inverse_kernel`, `_fused_post_conv_kernel`,
`chunk_scaled_dot_kkt_fwd_kernel`, `chunk_local_cumsum_scalar_kernel`) that
would have silently fallen into "elementwise/misc"; caught and fixed by
diffing the classifier's output against the full kernel dump before
trusting any number below.

Of the 50 natural rounds, recompute-slot counts landed on 0 (round 44,
once), 1 (13 rounds), 2 (18 rounds), 3 (8 rounds), 4 (2 rounds) --
consistent with PROGRESS.md's own "0-recompute is the rare case" finding.
Round 44 (the only 0-recompute/"fully-batched" round observed) and round 20
(a representative 2-recompute round, picked from the middle of the run) are
the two analyzed below.

### 7.1 Gap ledger

**Round 44 (0-recompute, fully-batched), wall = 399.648ms:**

| Category | ms | % of wall |
|---|---|---|
| Kernel-active, GEMM | 23.062 | 5.77% |
| Kernel-active, GDN-chunked | 2.098 | 0.52% |
| Kernel-active, flash_attn_decode* | 2.997 | 0.75% |
| Kernel-active, elementwise/norm/misc | 7.034 | 1.76% |
| **Kernel-active, total** | **35.191** | **8.81%** |
| cudaMemcpy D2H | 98.234 | 24.58% |
| cudaMemcpy H2D | 0.018 | 0.00% |
| cudaMemcpy D2D + memset | 0.116 | 0.03% |
| **Memcpy, total** | **98.368** | **24.61%** |
| No-kernel gap (wall - kernel - memcpy) | 266.089 | 66.58% |
| **Unexplained residual** | **0.000** | **0.00%** |

Per-phase breakdown (this round has no recompute slots, so no
`restore`/`recompute_fwd`/`draft_recompute` phases fire):

| Phase | wall ms | kernel ms | memcpy ms | gap ms |
|---|---|---|---|---|
| snapshot (all 4 slots) | 218.110 | 0.000 | 98.222 | 119.888 |
| verify (all 4 slots, 1 forward) | 154.948 | 29.742 | 0.116 | 125.091 |
| draft (full-accept group, 3 batched steps) | 26.589 | 5.449 | 0.026 | 21.110 |

**Round 20 (2-recompute), wall = 679.881ms:**

| Category | ms | % of wall |
|---|---|---|
| Kernel-active, GEMM | 47.925 | 7.05% |
| Kernel-active, GDN-chunked | 3.614 | 0.53% |
| Kernel-active, flash_attn_decode* | 5.556 | 0.82% |
| Kernel-active, elementwise/norm/misc | 12.486 | 1.84% |
| **Kernel-active, total** | **69.581** | **10.23%** |
| cudaMemcpy D2H | 89.557 | 13.17% |
| cudaMemcpy H2D | 27.136 | 3.99% |
| cudaMemcpy D2D + memset | 0.458 | 0.07% |
| **Memcpy, total** | **117.152** | **17.23%** |
| No-kernel gap (wall - kernel - memcpy) | 493.148 | 72.53% |
| **Unexplained residual** | **0.000** | **0.00%** |

Per-phase breakdown:

| Phase | wall ms | kernel ms | memcpy ms | gap ms |
|---|---|---|---|---|
| snapshot (all 4 slots) | 194.707 | 0.000 | 89.542 | 105.165 |
| verify (all 4 slots, 1 forward) | 172.116 | 29.768 | 0.114 | 142.234 |
| restore (2 recompute slots) | 132.892 | 0.000 | 27.335 | 105.557 |
| recompute_fwd (2 recompute slots) | 139.378 | 28.930 | 0.115 | 110.333 |
| draft, full-accept group (2 slots) | 21.441 | 5.624 | 0.022 | 15.793 |
| draft, recompute group (2 slots) | 19.348 | 5.259 | 0.020 | 14.066 |

"Unexplained residual" is 0.000ms in both rounds by direct measurement,
not by definition-only construction: `kernel_active + memcpy` was computed
straight from CUPTI's own activity tables (exhaustive over everything the
GPU executed in each phase's NVTX-bounded window), and gap is the
remainder -- there is no time in either round that these three buckets
fail to account for. The large gap bucket is *not* a mystery, though: see
7.4.

Byte-level cross-check on the D2H snapshot volume: round 44's snapshot
call moved 627,572,924 bytes (~598.4 MiB) D2H in 98.234ms (6.39 GB/s,
inside this doc's own cited "3-8GB/s pageable-PCIe-under-WSL2" range).
Subtracting the small conv_state contribution (~18MB across 4
slots x 48 layers), the ssm_state-only volume is ~609.6MB -- within 1% of
this document's earlier from-source estimate ("~604MB per verify round",
section 2.3 item 1). Round 20's restore (2 of 4 slots only) moved
313,788,118 bytes (~299.3 MiB) H2D in 27.136ms (11.56 GB/s) -- consistent
with restoring exactly half the full 4-slot snapshot volume.

### 7.2 Host timers

Measured via isolated calls (sync-bracketed) at the end of the 50-round
loop, cross-checked against NVTX-projected kernel-active time for the same
calls:

| Timer | Wall ms | Kernel-active ms (NVTX cross-check) |
|---|---|---|
| `snapshot_gdn_state`, slot 0 | 56.131 | -- |
| `snapshot_gdn_state`, slot 1 | 52.064 | -- |
| `snapshot_gdn_state`, slot 2 | 48.748 | -- |
| `snapshot_gdn_state`, slot 3 | 57.576 | -- |
| `snapshot_gdn_state`, sum of the above 4 | **214.519** | 214.137 (0 kernels -- pure memcpy) |
| one `verify_batch` call (4 slots) | **136.128** | 135.868 wall / 29.733 kernel |
| one draft step (`_mtp_forward_batch`, qo_len=1, 4 slots) | **6.218** | 6.537 wall / 1.704 kernel |

The isolated `verify_batch` kernel-active number (29.733ms) matches both
rounds' in-context verify-phase kernel time almost exactly (29.742ms round
44, 29.768ms round 20) -- strong internal consistency, and confirms
`verify_batch`'s cost is essentially fixed regardless of how many slots
will end up needing recompute afterward.

### 7.3 `nvidia-smi utilization.gpu` sample distribution

40 one-second samples fell inside the profiled window
(`1784289665.72`-`1784289703.20`, 37.48s). Segmenting by the printed
`PREFILL` wall time (3.185s) cleanly separates two regimes:

- **Prefill segment** (first ~3.2s, one real 4096-token x 4-slot batched
  forward): samples `[5, 99, 97, 85]` -- a brief, genuine high-utilization
  spike, consistent with this being real, large-GEMM-bound compute at high
  arithmetic intensity, unlike the small decode/verify steps that follow.
- **Decode/verify-round-loop segment** (remaining ~34.3s, 36 samples):
  `mean = 14.47%`, `min = 0`, `max = 20`, distribution concentrated in
  8-20% (`{0:1, 8:2, 9:2, 10:1, 11:2, 12:3, 13:1, 14:6, 15:1, 16:3, 17:3,
  18:3, 19:6, 20:2}`). This is *lower* than the ~30% figure cited elsewhere
  in this project's history, though from a much smaller one-run, 1Hz
  sample than a full W1-S benchmark would give -- reported as measured,
  not reconciled to the historical number.

### 7.4 What the ledger resolves: the ~95% "GPU-busy%" vs ~30%
`utilization.gpu` contradiction, concretely

This ledger directly measures the quantity this document's own tail
(and `direct-model-runner-design.md` around line ~3695-3703) argued was
never actually measured: the literal fraction of round wall time during
which a kernel is executing. It is **8.81% (round 44) / 10.23% (round
20)** -- roughly an order of magnitude below the ~94-95% "GPU-busy%"
figure `mtp_w1s_our_runtime_perf.py`'s own `gpu_busy_pct` has reported all
session. This is not a contradiction once the mechanism is clear: that
metric is `elapsed_time(start_evt, end_evt)` for a `torch.cuda.Event` pair
recorded immediately before/after the whole call, divided by
host-`perf_counter`-measured wall time for the same call -- both the
numerator and denominator span the *entire* call including every internal
gap, so the ratio is close to 1 almost by construction (it measures "did
the bracketed region take about as long on the GPU timeline as on the
host timeline", which is trivially true), not "what fraction of that
region had a kernel actually running". This ledger is the first
measurement this project has made of the latter, and it settles the
question `utilization.gpu`'s ~30% figure was already hinting at: for
~90% of round wall time, no kernel is executing at all.

The gap itself decomposes cleanly into two known, already-hypothesized
mechanisms, not a mystery:
- **~30-31% of round wall time**: gap inside the `snapshot`/`restore`
  phases specifically (round 44: 119.888ms/399.648ms = 30.0%; round 20:
  (105.165+105.557)/679.881 = 31.0%). This phase issues 384 individual
  blocking `.to("cpu", copy=True)` (or the H2D mirror) calls
  (`runtime/direct_model_runner.py:1214-1219`, `:1249-1253`) into freshly
  allocated *pageable* host tensors -- the memcpy-engine-busy time
  (98/89ms) is only about half of the phase's own wall time; the other
  half is host-side per-call overhead (pageable-buffer allocation, the
  synchronous-copy driver path) with no corresponding GPU activity at all.
- **~37-42% of round wall time**: gap inside the compute phases
  (`verify`/`recompute_fwd`/`draft_*`) despite real kernel activity there
  too (round 44: (125.091+21.110)/399.648 = 36.6%; round 20:
  (142.234+110.333+15.793+14.066)/679.881 = 41.5%). The `verify` phase
  alone launches 3634 kernels in 29.742ms of GPU time -- 8.2
  microseconds/kernel on average -- while eager-mode CPU-side dispatch for
  each of those launches (Python/vLLM module-call overhead per layer, not
  captured by any CUDA graph) plausibly costs comparably or more per
  launch. This is exactly the eager-dispatch-starvation mechanism this
  document's section 2.3 item 4 flagged as a suspect, now quantified
  directly rather than inferred from the busy%/utilization contradiction.

### 7.5 Predictions P1-P4: verdicts

**P1 (D2H/H2D memcpy ≥25% of round wall time) -- HELD, with one nuance.**
Strict memcpy-engine-busy time: 24.61% (round 44, 0.4pp under the literal
25% line) and 17.23% (round 20, under). Both are, however, **decisively
above the falsifier's own <10ms/round demotion floor** by roughly an order
of magnitude (89-98ms/round measured, not <10ms) -- the plan's own stated
interpretation of that floor ("if it measures <10ms/round, Phase 1 is
demoted to cleanup") does not apply here by a wide margin. Including the
same mechanism's host-dispatch overhead (the `snapshot`/`restore` phases'
*entire* wall time, not just their memcpy-engine sub-component) raises the
share to 54.6% (round 44) / 48.2% (round 20) of round wall time --
decisively over 25% under that reading. Verdict: **held**, Phase 1 is not
demoted to cleanup.

**P2 (GDN-family kernel time ≫ native's ~8% share, and/or second target
forward ≈ full verify forward cost) -- HELD, but only via the second
clause; the first clause is directly falsified.** GDN-chunked kernel
family time is **0.52-0.53% of round wall time** (1.2-1.4% of a single
verify/recompute forward's own kernel time) -- far *smaller* than native's
cited ~8% share, the opposite of what the first clause predicted; the
chunk-64-shaped GDN kernels are, in absolute terms, cheap. The second
clause is decisively confirmed: round 20's recompute forward's
kernel-active time (28.930ms) is **97.2%** of the verify forward's own
kernel-active time (29.768ms) -- essentially the same real GPU compute,
confirming the recompute branch really does redundantly re-run the full
64-layer target model. Verdict: **held** (via the redundant-full-forward
mechanism, not the GDN-kernel-bloat mechanism the document also floated --
worth correcting for any future round that cites this).

**P3 (no-kernel gap ≥30% of round wall time) -- HELD, decisively.**
Measured 66.58% (round 44) / 72.53% (round 20) -- more than double the
30% threshold in both rounds. This is the single most decisive result in
this ledger and directly resolves the busy%/utilization contradiction
(7.4). Verdict: **held**, and by a wide margin; §2.1's reinterpretation
was correct, not just plausible.

**P4 (`flash_attn_decode*` ≤10% of round time) -- HELD.** Measured 0.75%
(round 44) / 0.82% (round 20) -- roughly an order of magnitude under the
10% bound, confirming section 2.2's argument that this kernel's occupancy
cannot be the ours-vs-native differentiator at this concurrency. Verdict:
**held**.

None of section 6's falsifier conditions triggered (kernel-active time is
not ≈ wall time -- it is 8.8-10.2%; memcpy is not negligible -- it is
89-117ms/round; GDN-family time is not "comparable to native's share" --
it is smaller still). If anything, this ledger is a stronger confirmation
of section 2's re-diagnosis than the plan required: essentially none of
round wall time is dense, necessary kernel work (kernel-active across
*all* families combined is only 8.8-10.2%), so the fallback per-kernel
efficiency investigation section 6 describes is not warranted.

### 7.6 Recommendation

**Proceed to Phase 1 next, exactly as planned, but do not let Phase 3 slip
behind Phase 2 in priority.** Phase 1's target (GPU-resident GDN
snapshot/restore, D2D `copy_` instead of pageable D2H/H2D) is cleanly
quantified here at 89-117ms/round of direct memcpy cost plus a comparable
amount of host-dispatch gap in the same phases (~30-31% of round wall time
combined) -- a fast (≤1 day), low-risk, well-bounded win, and the
falsifier clause it was gated on did not fire. Phase 2's premise (the
recompute branch redundantly re-runs a full target-model forward) is now
directly confirmed via kernel-time ratio (97.2%) rather than inferred, so
it remains justified -- but this ledger also shows Phase 2's other
originally-suspected cost driver (bloated chunk-64 GDN kernels) is not
real; GDN kernel time is a non-issue in absolute terms, so Phase 2's real
payoff is specifically eliminating the redundant forward and the
snapshot/restore mechanic, not "faster GDN kernels." Most importantly:
this ledger's phase-level gap decomposition (7.4) shows the eager-dispatch
gap inside the *compute* phases (verify/recompute/draft -- Phase 3's
target) is, at ~37-42% of round wall time, comparable to or larger than
the snapshot/restore-driven gap (~30-31%, Phase 1/2's target) -- and it is
present in *every* round regardless of recompute count, unlike Phase 1/2's
gains which only apply to recompute-affected rounds (84.4% of rounds, not
100%). This is new information the original phase ordering (1 -> 2 -> 3)
did not have: Phase 3 should be scheduled promptly after Phase 1, not
pushed behind the full 3-5 day Phase 2 effort by default -- worth a
deliberate go/no-go check on sequencing once Phase 1's real measured
impact is in hand, rather than assuming the original ordering still
optimal now that both levers are quantified at comparable size.

Artifacts (not committed to the repo, same convention as this project's
existing `ncu-rep` scratch files): `phase0_ledger.nsys-rep` (40.7MB),
`phase0_ledger.sqlite` (157MB export), `phase0_profiled_run.log`,
`util_sample.log`, and the analysis script `analyze_ledger.py`, all under
this session's scratchpad directory. The diagnostic script itself,
`benchmarks/phase0_nsys_gap_ledger_diag.py`, is committed under
`benchmarks/` per this phase's instructions.

---

## 8. Phase 1 results (executed 2026-07-17)

### 8.1 What changed and why

Replaced `snapshot_gdn_state`/`restore_gdn_state`'s CPU round-trip
(`.detach().to("cpu", copy=True)` on snapshot, `.to(self.device)` on
restore) with a preallocated, GPU-resident, fixed-address per-slot buffer
and a plain D2D `copy_`, exactly per this doc's Phase 1 spec and directly
targeting Phase 0's measured cost (section 7: 89-117ms/round of
memcpy-engine time alone, ~30-31% of round wall time including the
phase's own host-dispatch gap, present in every round since snapshot is
unconditional for all active slots).

`runtime/direct_model_runner.py` changes:

- New `DirectModelRunner._allocate_gdn_snapshot_buffers()`, called once
  from `__init__` right after `_allocate_and_bind_kv_caches()`. Allocates
  `self.gdn_snapshot_conv[name]`/`self.gdn_snapshot_ssm[name]`, one tensor
  per GDN layer shaped `(num_slots, *per_slot_shape)`, dtype/device
  matching the corresponding `kv_caches[name]` tensor. Sizing verified
  against the real call pattern before relying on it (per this round's
  instructions), not assumed: both `mtp_verify_and_commit` and
  `mtp_verify_and_commit_batch` snapshot each active slot AT MOST ONCE per
  round, and any restore for that slot happens later in the SAME round
  (`direct_model_runner.py`'s `mtp_verify_and_commit_batch`, the
  `snapshots = {s: self.snapshot_gdn_state(s) for s in slots}` /
  `self.restore_gdn_state(s, snapshots[s])` pair) — so at most ONE
  snapshot per logical slot is ever outstanding. One buffer entry per
  slot is therefore sufficient; this is deliberately NOT a literal
  ping-pong double buffer (which would cost ~1.2GB instead of ~604MB) --
  the plan doc's own "~604MB" VRAM estimate already assumed this
  one-copy-per-slot sizing (matches Phase 0's own measured D2H byte
  count, ~604MB for a 4-slot round, almost exactly). Indexed directly by
  LOGICAL slot (0..num_slots-1), unlike `kv_caches` (which reserves
  physical index 0 via `RESERVED_PHYSICAL_SLOTS`/`_physical_slot` for an
  unrelated real-vLLM addressing convention this private buffer isn't
  subject to).
- `snapshot_gdn_state` now writes into the persistent buffer via
  `copy_(conv_state[physical])`/`copy_(ssm_state[physical])` (D2D) and
  returns dict values that are VIEWS into that buffer, instead of fresh
  CPU clones. Return type/shape (`dict[str, tuple[Tensor, Tensor]]` plus
  the `__slot__`/`__generation__`/`__consumed__` keys) is byte-for-byte
  the same contract as before.
- `restore_gdn_state` now does a plain `conv_state[physical].copy_(snap_conv)`/
  `ssm_state[physical].copy_(snap_ssm)` (D2D, no `.to(self.device)`
  staging step, since the snapshot tensors are already device-resident).
- All three safety invariants (slot-id check, generation-counter
  staleness check, consumed-once flag) are UNCHANGED — same checks, same
  error messages, still evaluated before any tensor data is touched. This
  matters slightly more than before: because the buffer is now reused
  round-over-round in place, a caller wrongly holding a snapshot object
  across multiple rounds would, without these checks, silently alias
  NEWER data through the same buffer slot instead of failing loudly. The
  checks still fire correctly in that case (verified — see 8.2) because
  they inspect metadata, not tensor contents, and metadata is set before
  any aliasing could matter.
- Fixed-address discipline (buffer allocated once, only ever written via
  `copy_`, never reallocated) follows the same pattern this file's other
  persistent GPU buffers already use (`CapturedBatchDecodeGraph`). This
  code path is eager-only today (`mtp_verify_and_commit`/`_batch` do not
  run inside any CUDA graph capture region; `CapturedBatchDecodeGraph` is
  a separate, not-yet-wired-in mechanism, Phase 3's job) — no immediate
  requirement to be graph-safe, but the discipline was applied anyway so
  Phase 3 doesn't have to revisit this buffer's allocation strategy
  later if GDN snapshot/restore is ever folded into a captured graph.

No new test/diagnostic file was added this round — the three existing
suites (below) were judged sufficient to exercise every behavior this
change touches, including the safety-invariant edge cases.

### 8.2 Correctness verification (all four checks)

1. **`benchmarks/mtp_gdn_rollback_check.py --repeat 3`: 3/3 PASS.** Real
   "detour then restore" test (`num_slots=2`) — bytewise-identical logits
   and all-48-layer 0.0 GDN state diff between a slot that took 4 real
   extra decode steps then restored, vs. a twin slot that never detoured.
   This is the most direct test of the new D2D-buffer restore path's
   correctness (not just its speed).
2. **`benchmarks/mtp_batch_verify_check.py`: PASS, all 4 sub-checks true**
   (`check0_batch1_equivalence`, `check1_numerical_twin`,
   `check2_signal_probe` with zero cross-slot contamination,
   `check3_mixed_stage` forced-reject/full-accept combination) — the
   established full batched-MTP regression suite, unmodified by this
   round's change, run against the new snapshot/restore implementation.
3. **`benchmarks/mtp_ragged_recompute_verify_check.py`: PASS, all 3
   sub-checks true** (`check0_batch1_forced_reject_equivalence`,
   `check1_ragged_recompute` — 5 genuinely-ragged rounds across per-slot
   committed lengths 1/2/2/1 through 1/2/3/1, `check2_mixed_ragged_and_full_accept`)
   — the ragged-qo_len regression suite, likewise unmodified, confirming
   the new buffer's per-slot addressing generalizes correctly to the
   ragged/mixed-stage batched recompute path (multiple slots' GDN state
   independently snapshotted/restored out of the SAME underlying batched
   buffer within one round).
4. **GPU/process hygiene**: `pgrep -af` for every MTP/runner process
   pattern and `nvidia-smi --query-compute-apps`/`--query-gpu` all
   confirmed clean (idle baseline, ~2.9GB/0% util) after each of the
   three checks above and after the perf run in 8.3 — no leaked process
   at any point this round.

All three suites passed on the FIRST attempt with this round's
implementation — no debugging/fix cycle was needed between writing the
code change and it passing correctness.

### 8.3 Performance: real W1-S 3-rep measurement

`python -m benchmarks.mtp_w1s_our_runtime_perf --batched --repeats 3`
(defaults: `max_tokens=256`, `concurrency=4`, fixture `n16`/16 requests,
`K=3`) — identical protocol to every prior W1-S measurement this project
has used.

| Rep | accepted_tokens/s | ms/accepted token | draft acceptance % | gpu_busy_pct (span metric) |
|---|---:|---:|---:|---:|
| 1 | 28.321 | 35.310 | 72.510 | 93.72% |
| 2 | 25.963 | 38.516 | 72.510 | 93.72% |
| 3 | 28.108 | 35.577 | 72.510 | 93.71% |
| **mean** | **27.464** | 36.468 | 72.510 | 93.72% |

`total_committed_tokens` (4112) and `draft_acceptance_rate_pct`
(72.50965...%) are IDENTICAL bit-for-bit across all 3 reps -- expected,
since this change is purely a storage-medium swap and touches nothing in
the accept/reject or draft-generation logic; this identity is itself a
useful passive correctness cross-check on top of the three dedicated
suites above.

**Comparison against the two reference numbers**:
- **vs. the 18.54 tok/s baseline (mean of 18.50/18.75/18.38, the
  immediately-preceding ragged-recompute-batching round): 27.464/18.54 =
  1.481x -- a real +48.1% throughput improvement.**
- **vs. native's 144.54 tok/s: gap narrows from ~7.80x (144.54/18.54) to
  ~5.26x (144.54/27.464).**

**Honest comparison against the plan's own projection** ("approximately
the Phase-0-measured snapshot share (predicted 1.3-2x if P1 holds)"): the
real 1.481x **falls inside the predicted 1.3-2x range -- it matches, not
a miss** -- but lands close to the LOW end of that range, notably below
the midpoint a naive reading of Phase 0's own two numbers would suggest.
Working the arithmetic backward: an overall 1.481x end-to-end speedup
implies roughly 32.5% of TOTAL wall time was removed (1 - 1/1.481); since
the timed W1-S run's wall time is dominated by the decode/verify round
loop (TTFT/prefill is a small, roughly-fixed ~3.0-3.2s slice per batch
against a 145-158s per-rep total), the removed fraction of decode-loop
time alone is closer to ~36%. That sits BETWEEN Phase 0's two candidate
estimates for what Phase 1 would remove: closer to the STRICT
memcpy-engine-only share (17-25%, P1's literal reading) than to the
FULL snapshot/restore-phase share including host-dispatch gap (48-55%,
P1's "including host-dispatch gap" reading section 7.5 also offered).

**Reasoned (not ablation-proven) explanation for landing near the low
end**: the fix eliminates the BLOCKING wait inherent to a pageable D2H/H2D
memcpy and the per-call CPU tensor allocation that used to accompany it
-- but it does NOT reduce the NUMBER of individual host-issued operations
per round (still 384: 2 tensors x 48 layers x 4 slots, same Python `for`
loop structure as before, now doing D2D `copy_` instead of D2H/H2D).
Per-launch Python/CUDA-dispatch overhead for issuing that many small ops
is common to both the old and new implementation and is NOT eliminated
by this change -- only the blocking-transfer-wait portion of the old
phase's cost is. This is consistent with the observed ~36% removed
decode-loop share landing above the strict-memcpy estimate (some
host-dispatch-gap reduction did happen, from removing the CPU allocation
stalls) but well below the full-phase estimate (the residual per-launch
dispatch cost, which is a DIFFERENT mechanism from memcpy, persisted).
This was not confirmed via a dedicated ablation this round -- flagged as
a hypothesis with supporting arithmetic, not a proven mechanism.

### 8.4 Does this change the Phase 2 vs. Phase 3 sequencing question?

Section 7.6's recommendation, made before Phase 1 had a real number, was:
"proceed to Phase 1 next... but do not let Phase 3 slip behind the full
Phase 2 effort by default" -- because Phase 3's target (eager-dispatch gap
inside verify/recompute/draft compute phases, ~37-42% of EVERY round) was
already comparable to or larger than Phase 1/2's target (~30-31% of only
84.4% of rounds).

Phase 1's real result is, if anything, a data point IN FAVOR of that
hedge, not against it -- stated plainly, not as a decision, since that
call belongs to the coordinator: the reasoned explanation in 8.3 for why
Phase 1 undershot the upper end of its own predicted range is that a
persistent-per-launch-dispatch-overhead mechanism (independent of the
memcpy/blocking-wait mechanism Phase 1 actually fixed) ate the
difference. That is EXACTLY the mechanism Phase 3 (CUDA-graph capture,
replacing many discrete host-dispatched launches with one pre-recorded
graph replay) targets directly, and Phase 2 (adopting native's
spec-decode GDN path to delete the redundant recompute forward) does not
-- Phase 2 targets a different, also-real mechanism (confirmed via Phase
0's 97.2% kernel-time-ratio finding) but not this one. Combined with
Phase 3's target already being present in 100% of rounds (not 84.4%) and
already comparable in size, and `CapturedBatchDecodeGraph` already
existing and being independently verified (just not yet wired into
`mtp_verify_and_commit_batch`), Phase 3 looks like the higher-confidence,
likely-larger, and lower-implementation-risk next step relative to Phase
2's larger, riskier lift (building the real spec-decode
`GDNAttentionMetadata` branch from `gdn_attn.py`, 3-5 days per the
original estimate). This is a recommendation for the coordinator to
weigh, not a decision executed this round -- Phase 2's own justification
(the redundant full-forward recompute, real and confirmed) still stands
independently and nothing here invalidates it.

---

## 9. Phase 3 results (executed 2026-07-17/18)

Executed in two passes: an initial pass wiring the verify forward and the
K-1 draft continuation steps (per this doc's own Phase 3 scope, "verify
forward first, then the K draft steps"), then a coordinator-directed
**fast-iteration pass** (explicit instruction: skip the full
suite-per-change/writeup/commit discipline, get real numbers fast, report
each one, only pause the fast loop on a genuine dead end) that kept
picking off the remaining eager-path candidates until hitting one --
Phase 2's own real spec-decode mechanism -- that is architecturally a
different scope class, not a quick win. This section is the "return to
full rigor" consolidation of that whole arc, written after re-running
every correctness suite fresh against the FINAL code state (not
re-reporting the fast-iteration pass's own quick spot-checks) and taking
one proper 3-rep W1-S measurement.

### 9.1 What was built, in the order it landed

**Initial pass (verify forward + K-1 draft steps, per this doc's original Phase 3 scope):**

1. `determine_accept_reject_batch` (module-level, next to the existing
   single-slot `determine_accept_reject`, which is untouched and still
   used by every other caller): computes greedy accept/reject for the
   WHOLE batch via one `argmax(dim=-1)` + a `cumprod`-based "still
   matching the accepted prefix" mask + exactly ONE combined
   `.tolist()`, instead of a per-slot Python loop each doing up to `k+1`
   sequential `.item()` calls (up to 4*(k+1)=16 host round-trips/round
   before this change).
2. Removed the 8 `torch.cuda.synchronize()` calls inside `_forward`,
   `_forward_batch`, `_mtp_forward`, `_mtp_forward_batch` (2 each,
   bracketing `compute_logits`). Same-stream ordering already guarantees
   correctness (identical reasoning this project already used to justify
   removing `CapturedBatchDecodeGraph.replay()`'s own blanket sync,
   2026-07-17 correctness-review round) -- these were pure per-call
   dispatch overhead, not a safety requirement.
3. Vectorized the per-slot argmax loops inside `_mtp_sync_and_propose_batch`
   (both the step-0 harvest and the K-1 continuation loop) the same way:
   one `index_select`/`argmax` + one `.tolist()` per step, covering every
   slot in the batch, instead of a Python loop calling `.argmax().item()`
   per slot per step.
4. New opt-in `DirectModelRunner(..., enable_cudagraph=True)` (default
   `False`, so every existing correctness suite's behavior is
   byte-for-byte unaffected -- see 9.4 for why this had to be opt-in, not
   automatic).
5. `_get_verify_graph(batch_size, qo_len)` / `_precapture_verify_graphs()`:
   lazily-constructed-and-cached (then, later, fully precaptured -- see
   item 9 below) `CapturedBatchDecodeGraph` instances, wired into
   `mtp_verify_and_commit_batch`'s verify forward in place of the eager
   `verify_batch` call, with a documented, tested eager fallback whenever
   `enable_cudagraph` is off or the runner lacks spare slot capacity.
6. Extended `CapturedBatchDecodeGraph` (the pre-existing, already-verified
   class) with a backward-compatible `return_hidden` parameter on
   `replay()` (default `False`, existing callers unaffected) and a
   captured `_static_hidden_states` buffer alongside the existing
   `_static_logits` -- needed because the verify forward's hidden states
   feed the next round's MTP draft resync, and the pre-existing class
   only captured logits.
7. New `CapturedMTPDraftStepGraph` class (draft model, qo_len=1 initially)
   wired into `_mtp_sync_and_propose_batch`'s K-1 continuation loop.

Real W1-S 3-rep result at the end of this initial pass: **47.945 accepted
tok/s** (48.392 / 48.920 / 46.524) -- +74.6% vs the 27.464 Phase-1
baseline, gap to native narrows from 5.26x to 3.01x. A same-protocol 3-rep
run of items 1-3 WITHOUT the graph (`--batched`, no `--cudagraph`) gave
**33.593 tok/s** (32.395 / 34.078 / 34.307, +22.3% alone) -- isolating
sync-removal+batched-argmax's own contribution from the graph's.

### 9.2 Fast-iteration pass: further real wins, one real bug each round

The coordinator's own framing for this pass: quick sanity check (not the
full suite), one real number, report and keep going, stop only on a
genuine dead end. Numbers below are **single-rep** (as instructed at the
time) except where marked -- read as directional evidence, not the
number of record; the number of record is the fresh 3-rep result in 9.3.

8. **GDN snapshot/restore batching** (`torch._foreach_copy_` replacing the
   per-layer Python loop's 2×48=96 individual `.copy_()` launches per
   slot -- 384 total per round across 4 slots -- with 2 fused
   multi-tensor calls per slot, ~8 total per round): correctness holds
   (`mtp_gdn_rollback_check` bytewise-exact), but **measured no net
   throughput effect** -- single-rep 48.810, statistically indistinguishable
   from the 47.9-48.9 range above. Kept anyway (free, correct, and
   strictly fewer launches) but not a real lever: Phase 1 already removed
   the dominant cost (the blocking PCIe transfer); the residual per-launch
   Python/CUDA-dispatch overhead this change targets was apparently
   already small relative to everything else.
9. **Draft K-1 step CUDA-graph capture** (`CapturedMTPDraftStepGraph`,
   item 7 above, now actually wired in and measured): single-rep
   **53.21 tok/s**.
10. **Recompute-forward graph reuse**: the recompute forward is a SECOND
    full 64-layer target-model forward (Phase 0's ledger: 97.2%
    kernel-time ratio vs. the verify forward) that is normally ineligible
    for graph capture because its batch is ragged (each recompute slot's
    committed_len can differ). Opportunistic fix: when every recompute
    slot in a round happens to share the SAME committed_len (a real,
    non-rare case -- greedy rejection position is one draw from a shared
    per-position acceptance distribution across independent slots), reuse
    the existing verify-graph cache (`_get_verify_graph(len(recompute_slots),
    committed_len)`) instead of falling back to eager `_forward_batch`.
    **Real bug caught and fixed during this step**: `CapturedBatchDecodeGraph`'s
    qo_len==1 branch expects FLAT token_ids (one int per slot), but the
    recompute path always builds list-of-per-slot-lists (`real_new_tokens[s]`
    is itself a list, even at length 1) -- a genuinely uniform
    committed_len=1 round (all recompute slots reject at position 0)
    crashed with `RuntimeError: output with shape [4] doesn't match the
    broadcast shape [4, 4]` the first time it was exercised. Fixed by
    flattening `tokens_recompute` before `replay()` specifically for the
    qo_len==1 case (`mtp_verify_and_commit_batch`'s `tokens_for_graph`
    variable). Added dedicated forced-uniform test rounds at both qo_len=2
    and qo_len=1 to `mtp_verify_cudagraph_check.py` to lock this in as a
    permanent regression guard, not just a one-off fix. Single-rep after
    the fix: **58.12 tok/s**.
11. **Full precapture** (extended `_precapture_verify_graphs`/new
    `_precapture_draft_step_graphs` to cover every `qo_len` in
    `1..num_speculative_tokens+1`, not just the verify-only `k+1`, for
    every batch_size the runner has spare capacity for): item 10's
    recompute-reuse path was, until this change, lazily capturing a new
    graph the FIRST time each new `(batch_size, qo_len)` combination
    appeared -- inside the timed measurement region. Moving that one-time
    cost to construction time (matching every other graph's own "pay
    setup cost once" discipline) removed that noise. Single-rep:
    **65.31 tok/s**.
12. **Draft step-0 (resync) generalized to the graph too**:
    `CapturedMTPDraftStepGraph` generalized from qo_len=1-only to
    arbitrary `qo_len` (buffers sized `batch_size*qo_len`, CSR
    `qo_indptr`, etc. -- the same generalization pattern
    `CapturedBatchDecodeGraph` already used for qo_len>1), then wired into
    `_mtp_sync_and_propose_batch`'s step 0 whenever `num_new_tokens` is
    uniform across the calling group (always true for the full-accept
    group's fixed `k+1`; sometimes true for the recompute group).
    **Real near-miss bug caught before ever running it in anger**:
    `mtp_prefill_batch` calls this SAME function with
    `num_new_tokens=prompt_len` (e.g. ~4096 for a real W1-S prompt) --
    also "uniform" by the same check, which would have forced a
    thousands-of-tokens prefill through the decode-kernel dispatch that
    is only validated up to `_MAX_DECODE_QO_LEN=16`
    (`build_attention_metadata_batch`'s own established bound). Caught by
    re-reading the call site before running anything, fixed with an
    explicit `num_new_tokens_list[0] <= _MAX_DECODE_QO_LEN` guard before
    ever attempting the graph path. Single-rep after the fix: **64.01
    tok/s** -- statistically FLAT vs. item 11's 65.31 (within single-rep
    noise, not a real additional win). Kept anyway: it is a real,
    now-verified capability (the draft model's step 0 no longer needs a
    separate code path once its shape is graph-eligible), and Phase 0's
    own ledger explains why the gain is negligible -- the draft model is
    small (few layers) relative to the target model's 64, so even fully
    graph-capturing its whole cycle barely moves a total that's already
    dominated by the target model's captured verify+recompute forwards.
13. **Phase 2 scoping (native spec-decode GDN path) -- investigated, NOT
    implemented, precisely scoped as a real but different-class blocker**:
    the fast-iteration pass's next natural candidate was the general
    ragged recompute-forward case (item 10's reuse only covers the
    uniform special case). Direct investigation (reading the actual
    kernels, not just the metadata struct) found this is a genuine dead
    end for graph capture specifically -- GDN's recurrent state advances
    through every fed token regardless of content, so padding the ragged
    batch to a fixed shape (the obvious workaround) would corrupt the
    final committed state, and one captured graph per exact
    (slot-count, committed-len-combination) shape is combinatorially
    infeasible to precapture. This is NOT a dead end for the underlying
    problem, though: this doc's own Phase 2 (adopt native's real
    spec-decode `GDNAttentionMetadata` branch,
    `vllm/v1/attention/backends/gdn_attn.py:198-288`) sidesteps the whole
    problem class rather than fighting it, by making GDN state commit
    itself acceptance-aware instead of needing snapshot/restore or a
    recompute forward at all. Read the actual kernels (not just the
    builder) to confirm the mechanism is real and get a precise scope:
    - The recurrent/SSM state
      (`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:103-170`,
      the Triton kernel `fused_sigmoid_gating_delta_rule_update_kernel`)
      needs **K+1 dedicated physical state rows per slot**, not 1. Read
      side: `i_t = num_accepted_tokens[i_n] - 1; state_idx =
      ssm_state_indices[i_n, i_t]` -- round N+1 reads whichever of round
      N's K+1 candidate states corresponds to round N's actual accept
      count. Write side: round N's own forward writes to
      `ssm_state_indices[i_n, 0..T-1]` unconditionally, for all K+1
      positions, before anyone knows what will be accepted -- confirmed
      real, this is exactly the "commit is acceptance-aware, no rollback
      needed" mechanism, read directly from the kernel body.
    - The conv1d state is a DIFFERENT mechanism, not the same K+1-slot
      scheme: `causal_conv1d_update`'s own docstring
      (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1103-1107`)
      says `num_accepted_tokens` "indicates the number of accepted
      tokens... conv_state is updated in a sliding window manner" -- ONE
      physical slot per sequence (always column 0 of the same
      block-table-style index tensor), with the kernel doing its own
      internal sliding-window recompute. Conv and SSM state need two
      SEPARATELY correct implementations, not one shared pattern.
    - Concrete scope this implies: (a) restructuring
      `allocate_fixed_slot_kv_caches`'s GDN tensor shapes to add K+1
      addressable ssm rows per slot (conv stays 1) -- a schema change,
      not additive; (b) building the real `spec_sequence_masks`/
      `spec_token_indx`/`spec_query_start_loc`/`spec_state_indices_tensor`
      CSR construction (today's `build_gdn_metadata_batch` builds none of
      these -- always `None`); (c) a new persistent per-slot
      `num_accepted_tokens` runner field threaded from round N's real
      accept/reject into round N+1's metadata, with a defined first-round
      bootstrap; (d) removing snapshot/restore/recompute and rewriting
      `mtp_verify_and_commit_batch`'s control flow around this. This
      matches this doc's own original Phase 2 estimate (3-5 days) -- more
      scope than everything else in this section combined, and the
      failure mode if either kernel's addressing is subtly misunderstood
      is a SILENT GDN state corruption, not a crash. **Coordinator
      decision: hold here, do not implement Phase 2 now** -- this section
      documents the finding as the reason the fast-iteration pass stops
      where it does, not as a rejected idea; whether to invest the 3-5
      days is a call for whoever resumes this line to make.

### 9.3 Final state: what's graph-captured vs. eager, and why

| Call | Captured? | Condition | Why |
|---|---|---|---|
| Verify forward (target model, `mtp_verify_and_commit_batch`) | **Always**, when `enable_cudagraph=True` | any `batch_size` 1..`num_slots//2`, `qo_len=k+1` fixed | Every round submits the same K+1-token draft for every active slot -- always uniform. |
| Recompute forward (target model) | **When uniform** | every recompute slot's `committed_len` this round happens to coincide | Reuses the verify-graph cache at `(len(recompute_slots), committed_len)`. Genuinely ragged rounds (slots with DIFFERENT committed_len) fall back to eager `_forward_batch` -- deliberately, permanently: padding would corrupt GDN's recurrent state, and per-exact-shape graphs are combinatorially infeasible (this is the SAME mechanism that motivates Phase 2, see 9.2 item 13). |
| Draft K-1 continuation steps (draft model) | **Always** | `qo_len=1` uniform by construction, batch = whatever slots are in that step | `CapturedMTPDraftStepGraph`, one entry per encountered batch_size. |
| Draft step 0 / resync (draft model) | **When uniform and `<=16`** | full-accept group: always (fixed `k+1`); recompute group: when committed_len coincides; `mtp_prefill_batch`'s own step 0: NEVER (guarded out, `num_new_tokens=prompt_len` is always `>16`) | Same `CapturedMTPDraftStepGraph` class, generalized to arbitrary `qo_len`. The `_MAX_DECODE_QO_LEN` guard is load-bearing, not decorative -- see 9.2 item 12's near-miss. |
| GDN snapshot/restore | **Never** (always eager) | every round, unconditionally, for every active slot | Not a correctness blocker (Phase 1 already made this GPU-resident/D2D); no measured benefit found to capturing it (9.2 item 8), and it stays outside the captured verify-forward region by design (Phase 1's own writeup: buffers reserved with graph-safe discipline "so Phase 3 doesn't have to revisit this" -- Phase 3 confirms that decision was fine to leave as-is, not that it needed changing). |
| `mtp_prefill_batch`'s own target-model prefill | **Never** (always eager) | once per request batch | Genuine long prefill (thousands of tokens), compute-bound not dispatch-bound (Phase 0's own prefill-segment utilization measurement: 85-99%, already near saturation) -- not the mechanism Phase 3 targets. |

`enable_cudagraph` stays **opt-in, default `False`**. Every existing
correctness suite in this project constructs its own runner with
`num_slots` sized to its OWN real slot count, no spare capacity reserved
for a captured graph's disposable warmup slots (e.g.
`mtp_batch_verify_check.py`'s `ref_slots = [4, 5, 6, 7]` at
`num_slots=8` are real, independently-used reference slots, not reserved
capacity) -- turning graph capture on unconditionally would silently
break these by stealing slots they already use for something else. A
caller that wants the graph path must explicitly pass
`enable_cudagraph=True` AND size `num_slots` to at least twice its real
concurrency (the extra half is reserved, disposable capture-warmup
capacity, never touched by real request traffic) -- see
`benchmarks/mtp_w1s_our_runtime_perf.py`'s `--cudagraph` flag for the
reference caller.

### 9.4 Correctness verification (fresh re-run against the FINAL code state, full suites, no shortcuts)

Every suite below was re-run from a clean process against the code as it
stands at the end of this section, not re-reported from the
fast-iteration pass's own quicker spot-checks:

1. **`mtp_gdn_rollback_check.py --repeat 3`: 3/3 PASS.**
2. **`mtp_batch_verify_check.py`: PASS, all 4 sub-checks** (`check0_batch1_equivalence`,
   `check1_numerical_twin`, `check2_signal_probe`, `check3_mixed_stage`) --
   this suite runs with `enable_cudagraph` at its default `False`, so it
   is the regression guard for the untouched eager path.
3. **`mtp_ragged_recompute_verify_check.py`: PASS, all 3 sub-checks**
   (`check0_batch1_forced_reject_equivalence`, `check1_ragged_recompute`,
   `check2_mixed_ragged_and_full_accept`) -- likewise eager-path,
   unaffected by `enable_cudagraph` defaulting off.
4. **`cudagraph_eager_parity_check.py --repeat 3`: 3/3 PASS** -- the
   pre-existing, already-verified primitive's own bytewise-identical
   parity check, confirming the `return_hidden` extension (9.1 item 6)
   didn't regress its existing behavior.
5. **`cudagraph_mtp_regression.py --repeat 3`: 3/3 PASS** -- likewise, the
   pre-existing qo_len=4 MTP verify signal-probe regression, unaffected.
6. **`benchmarks/mtp_verify_cudagraph_check.py` (new, built this round,
   consolidated from several ad-hoc rounds added during fast iteration
   into one file): PASS.** This is the one that actually exercises the
   NEW graph-capturing code paths through the real
   `mtp_verify_and_commit_batch`/`_mtp_sync_and_propose_batch` entry
   points (`enable_cudagraph=True`), not just the underlying primitives
   in isolation. 8 rounds (organic, mixed, fully-ragged, uniform-qo2,
   uniform-qo1, single-slot-reject, post-ragged-organic) plus a
   batch-size-shrink transition (4->2), verified via the project's
   established independent-reference-replay methodology. Also asserts,
   via a `replay_count` counter added to both graph classes specifically
   for this (cache-key presence alone isn't sufficient evidence --
   precapture populates every entry regardless of whether a round ever
   replays it), that every one of the following was ACTUALLY replayed at
   least once, not just precaptured-but-unused: the verify graph at
   batch_size 4 and 2; the recompute-reuse path at qo_len=2 (round 6) and
   qo_len=1 (round 7); the draft step-0 graph at qo_len=2 (via the
   recompute group's own uniform resync); the draft continuation graph at
   qo_len=1. All confirmed replayed.

GPU/process hygiene (`pgrep`/`nvidia-smi --query-compute-apps`/
`--query-gpu`) confirmed clean after every one of the six suites above
and after the performance run in 9.5 -- no leaked process at any point.

### 9.5 Performance: final real W1-S 3-rep measurement

`python -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph --repeats 3 --max-tokens 256 --concurrency 4 --fixture n16`
(same protocol as every prior W1-S measurement this project has used).

| Rep | accepted_tokens/s | ms/accepted token | draft acceptance % | gpu_busy_pct (span metric) |
|---|---:|---:|---:|---:|
| 1 | 65.354 | 15.301 | 70.814 | 92.93% |
| 2 | 66.522 | 15.033 | 70.814 | 93.00% |
| 3 | 66.582 | 15.019 | 70.814 | 93.05% |
| **mean** | **66.152** | 15.118 | 70.814 | 92.99% |

`total_committed_tokens` (4118) and `draft_acceptance_rate_pct`
(70.814...%) are identical bit-for-bit across all 3 reps -- expected
(this is a deterministic greedy pipeline; every change this round is
purely mechanical, touching neither sampling nor the accept/reject rule
itself), and a useful passive cross-check on top of section 9.4's
dedicated suites.

**Comparison against the reference numbers**:
- **vs. the 27.464 tok/s Phase-1 baseline: 66.152/27.464 = 2.409x --
  a real +140.9% throughput improvement**, entirely from this Phase 3
  arc (sync removal, batched argmax, and the graph-capture wiring in
  9.1-9.2).
- **vs. native's 144.54 tok/s: gap narrows from ~5.26x (Phase 1's own
  end state) to ~2.185x.**
- **vs. the plan's own post-Phase-3 target (>=110 accepted tok/s, "within
  ~1.3x of native"): NOT met.** 66.152 is well short of 110. Per the
  plan's own honest-conclusion clause for this outcome ("If we land
  materially short with the ledger's residual still <=25%, the honest
  conclusion is that the remaining delta is in per-kernel efficiency at
  M=16"): this round's own findings point to a MORE SPECIFIC remaining
  delta than pure per-kernel efficiency, though -- the genuinely ragged
  recompute forward (still eager, real cost per Phase 0's 97.2%
  kernel-time-ratio finding) and the GDN snapshot/restore mechanism
  (still eager, real per-round host-dispatch cost even after both this
  round's and Phase 1's fixes) are both still-live, identified,
  NAMED costs -- not a residual/unexplained gap. Phase 2 (9.2 item 13)
  is the mechanism that would remove both at once; declining to build it
  this round is a scope/risk decision, not evidence the gap is
  unreachable.

### 9.6 GPU utilization: the falsifier check, tracked across the whole arc

Sampled `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1`
concurrently during three points in this arc, segmenting each sample set
into startup/load (<20%), decode/verify steady-state (20-80%), and
prefill spikes (>80%) the same way section 7.3 did:

| Point in the arc | decode/verify steady-state mean | n (samples) |
|---|---:|---:|
| Phase 0 baseline (pre-Phase-3, section 7.3) | 14.47% | 36 |
| After verify-forward graph capture only (9.1's end state) | 43.57% | 211 |
| Final (all of 9.1+9.2) | **54.12%** | 145 |

This is the plan's own falsifier check for whether the eager-dispatch-
starvation hypothesis actually holds ("`utilization.gpu` during steady
state should rise sharply if P3 held") -- it rose monotonically at every
measured point across this arc (roughly 1x -> 3x -> 3.7x), not just once
at the start, which is stronger evidence than a single before/after
snapshot would have been. **The hypothesis is confirmed, decisively, by
this arc's own data, not merely assumed.** The gap between "utilization
tripled" and "throughput only ~2.4x'd, still 2.185x behind native" is
consistent with 9.5's own account: utilization measures whether A kernel
is running, not how much of the remaining wall time is still spent on
the still-eager mechanisms (recompute-forward's ragged case, GDN
snapshot/restore) -- those still show up as real gaps in this same
utilization trace (values in the 20-54% band, not pinned near 100%),
which is itself further, independent confirmation that the NAMED
remaining costs in 9.5 are real, not residual noise.

### 9.7 An observed, unexplained resource-usage pattern (flagged honestly, not investigated further this round)

During the 9.5 measurement, `nvidia-smi`-reported `memory_used_mib`
climbed across reps within the SAME process: 45123 MiB (end of rep 1) ->
72075 MiB (end of rep 2) -> 79924 MiB (end of rep 3, 95531 MiB peak per
the raw log) -- against a ~97887 MiB total, this is uncomfortably close
to the ceiling by the end of rep 3, though it did not OOM in this run.
Confirmed NOT a true unbounded leak in the sense of surviving process
exit: `nvidia-smi` after the process exited showed memory back at the
same ~2653 MiB baseline as before the run started, both times this
measurement was taken (9.1's earlier 3-rep run and this section's final
one). Also confirmed NOT a correctness problem: `total_committed_tokens`
(4118) and `draft_acceptance_rate_pct` were bit-for-bit identical across
all 3 reps regardless of this growth. Plausible (not confirmed)
explanation: PyTorch's caching allocator growing its reserved
high-water-mark across reps due to the many small per-call tensor
allocations on the still-eager paths (fallback recompute, GDN
snapshot/restore's `_fill_buffers`-equivalent per-call `torch.tensor()`
construction) rather than a genuine leak -- not confirmed via a dedicated
allocator-stats trace this round. Flagged here as a real, observed
pattern worth a dedicated look before running this configuration for
much longer than 3 reps back-to-back (e.g. a real multi-hour serving
session), not as a blocking finding for this round's own conclusions.

### 9.8 Honest bottom line

Four real, verified wins landed this session on top of Phase 1's own
+48.1%: sync-removal+batched-argmax (+22.3% alone), the verify-forward
CUDA graph (the largest single jump), recompute-forward graph reuse, and
full precapture (draft step-0 generalization landed flat but is kept for
its own correctness value). Combined: **27.464 -> 66.152 accepted tok/s
(+140.9%), gap to native 144.54 down from 5.26x to 2.185x.** The
>=110/~1.3x target was not met. The eager-dispatch-starvation hypothesis
this whole Phase 3 line was built on is decisively confirmed (utilization
roughly 3.7x'd across the arc, monotonically, not just once). What
remains is not a mystery residual -- it is two specifically-identified,
still-eager mechanisms (the ragged recompute forward, GDN
snapshot/restore) whose fix (Phase 2, adopting native's real
acceptance-aware GDN state commit) is scoped, precise, and real, but is a
3-5 day architecturally-different-class change the coordinator explicitly
chose not to start this round. Whether to invest that time is a decision
for whoever picks this line up next.

---

## 10. Phase 2 implementation attempt (2026-07-18) -- real progress, a genuine residual bug found, not landed this round

Attempted the actual implementation of Phase 2 (native spec-decode GDN
path) per section 9.2 item 13's own scoping, using this project's
standing full-rigor discipline throughout (not fast-iteration mode --
this was explicitly the "do it properly" instruction given the
silent-corruption failure mode). Real progress was made: the hardest
architectural piece (the K+1-row SSM state addressing scheme) is derived
from the real kernels and verified correct in isolation. But a genuine,
smaller correctness gap was found in the full-model integration that was
NOT root-caused despite extensive, disciplined debugging, and this round
stops here rather than force something unverified into production. No
code from this attempt is committed -- the working tree was reverted to
the last real commit (`51a216e`) after writing this section, so the
state is exactly as it was before this attempt started plus this
documentation.

### 10.1 What was built and verified correct

1. **SSM/recurrent state allocation**: `allocate_fixed_slot_kv_caches`
   was extended to give the SSM state tensor `total_physical_slots *
   (1 + num_spec)` rows instead of `total_physical_slots` -- K+1
   dedicated physical rows per logical slot (K = num_speculative_tokens).
   Purely additive/backward-compatible: `num_spec` is derived from
   `vllm_config.speculative_config` (0 when absent, matching every
   non-MTP test exactly as before). Conv state's row count is
   UNCHANGED -- its per-row size already grows automatically via
   `layer.get_state_shape()` (which is passed `self.num_spec` at
   MODEL-CONSTRUCTION time, from the same speculative_config), confirmed
   directly (`conv_shape=(6, 10240)` for K=3, width=4, i.e.
   `width-1+num_spec` in the state_len dimension) -- this was verified
   BEFORE relying on it, not assumed. Re-ran `mtp_gdn_rollback_check.py`
   and `mtp_batch_verify_check.py` immediately after this change (per
   the "verify as soon as something plausibly works, don't wait" rule)
   -- both still passed unchanged, confirming the allocation change is
   safe.
2. **`_ssm_spec_row(logical_slot, col, total_physical_slots, num_spec)`**:
   the addressing scheme, derived from reading the actual kernels (not
   the metadata dataclass's own comments, which turned out to be
   imprecise) --
   `fla/ops/fused_sigmoid_gating.py`'s `fused_sigmoid_gating_delta_rule_update_kernel`
   and `mamba/ops/causal_conv1d.py`'s `causal_conv1d_update`. Key
   derivation: `causal_conv1d_update`'s spec branch hardcodes
   `conv_state_indices=spec_state_indices_tensor[:, 0]` (real vLLM model
   code, not something this project's metadata controls) -- since
   conv_state's row count is unchanged, column 0's VALUE must be a valid
   conv-state row, forcing column 0 == `_physical_slot(slot)` (the SAME
   row this slot's plain/prefill addressing already uses -- this also
   matches real vLLM's own convention,
   `non_spec_state_indices_tensor = block_table_tensor[:, 0]` in
   `gdn_attn.py:219`, where column 0 is shared between spec and non-spec
   addressing too). Columns 1..num_spec have no such constraint and
   address NEW rows appended after the original `total_physical_slots`.
   **Verified correct via a from-scratch isolated unit test of
   `fused_sigmoid_gating_delta_rule_update` directly** (hand-built tiny
   q/k/v tensors, no model involved): a single spec-decode call
   processing 3 tokens with `num_accepted_tokens=1` (bootstrap) matches
   a reference of 3 sequential real single-token decode calls to within
   ~0.001-0.004 (bf16-precision-level noise, confirmed against a TRUE
   zero-noise control -- two identical sequential-decode runs gave EXACT
   0.0). This is the correct, working mechanism -- the "commit is
   acceptance-aware, no rollback needed" idea is real and does work as
   the plan intended.
3. **`build_gdn_metadata_spec_batch`**: real spec-decode
   `GDNAttentionMetadata` construction, simplified for this project's
   narrower scope (MTP verify is ALWAYS pure-spec, never mixed with
   prefill/plain-decode in the same call, unlike real vLLM's general
   builder) -- confirmed by reading `qwen_gdn_linear_attn.py`'s
   `_forward_core` directly that `spec_token_indx`/`non_spec_token_indx`
   are never read in this scope (the `num_prefills==0 and
   num_decodes==0` branch uses the whole batch as the spec batch
   directly, no gather/merge needed).
4. **`verify_batch_spec_gdn`**: an isolated, NOT-wired-in analogue of
   `verify_batch` that routes through the new spec metadata instead of
   the chunked path -- built specifically so the new mechanism could be
   validated against the existing, already-verified chunked path on
   identical input before touching any real control flow (per this
   round's explicit "verify early" instruction), never touching
   `mtp_verify_and_commit_batch` or removing `snapshot_gdn_state`/
   `restore_gdn_state`/the recompute branch (step 4 of the coordinator's
   own implementation spec was never reached).

### 10.2 The residual bug: a real, small, still-unexplained gap in full-model integration

A dedicated parity script
(`benchmarks/phase2_spec_gdn_parity_check.py`, not committed -- see
10.4) compared the new spec path against the existing chunked path on
identical input through the REAL 48-layer model. Initial results showed
large divergence (cosine similarity ~0.83-0.97, logits max diff several
units) -- investigated extensively, in three phases:

**First hypothesis (wrong methodology, real lesson): buffer construction
artifacts in the isolated test.** An earlier version of the isolated
conv1d test used `.transpose(-1,-2).contiguous()` to build the conv
state tensor -- the coordinator (independently verifying before
accepting a "vLLM has a bug" claim, this project's own standing
practice) found that real vLLM's own test
(`tests/kernels/mamba/test_causal_conv1d.py`'s
`test_causal_conv1d_update_with_batch_gather`, lines 244-247)
constructs conv_state via `torch.randn(rows, width-1,
dim).transpose(1,2)` -- a genuine NON-contiguous transposed view, no
`.contiguous()` -- matching `is_conv_state_dim_first()==False` (the
default "SD" layout, confirmed in `mamba_utils.py:27-43`). Redoing the
isolated test with a true transposed view (matching real layout exactly)
did NOT by itself fix the severe corruption found at that point --- but
this was still a real, worth-keeping methodology lesson (buffer
construction in an isolated kernel test needs to match the real model's
exact tensor construction, not just its logical shape).

**Second hypothesis (real test-construction bug, found via the
coordinator's push to re-verify): the isolated test's own
draft-fabrication helper didn't reflect real MTP flow.** The original
isolated model-level test used `_establish_draft` (borrowed from
`cudagraph_mtp_regression.py`'s own helper), which calls the TARGET
model's own plain qo_len=1 decode path (`_forward_batch`) repeatedly to
fabricate a draft sequence. This is a real methodology shortcut this
project's OTHER tests also use, but it does NOT reflect how a real MTP
round works: real drafts come from the SEPARATE draft model
(`self.mtp_model`), which registers no GDN layers at all -- the target
model, in real MTP flow, NEVER does a plain qo_len=1 decode between
prefill and verify. Using `_establish_draft` meant the isolated test was
alternating between `causal_conv1d_update`'s non-spec branch (via the
fabricated-draft calls) and its spec branch (the real verify call) on
the SAME conv_state buffer -- a call pattern that plausibly never
happens in real MTP flow. This was compounded by a genuinely interesting
independent finding while re-reading `gdn_attn.py`'s real builder: **it
reclassifies non-spec decodes as prefills whenever spec-decodes coexist
in the same batch** (`if num_decodes > 0 and num_spec_decodes > 0:
num_prefills += num_decodes; num_decodes = 0`, `gdn_attn.py:247-251`) --
meaning `causal_conv1d_update`'s non-spec/decode branch (the one at
`qwen_gdn_linear_attn.py:1378`) may be close to dead code in any real
production session where MTP is active for a request's entire
lifetime, since conflicting non-spec decodes get rerouted to the
chunked/prefill path instead. This is worth keeping as a real,
independently-useful finding regardless of Phase 2's outcome: it's
evidence AGAINST a broad "vLLM's spec-decode GDN kernel is fundamentally
broken" claim, and IN FAVOR of "an artificial call-alternation pattern
in eager, non-CUDA-graph testing can trigger something real production's
own calling discipline avoids."

Removing the target-model-plain-decode-based draft fabrication (using a
fixed, arbitrary draft token sequence instead, with no plain-decode call
to the target model at all before the spec verify call) did NOT make the
severe corruption reproduce again in that exact form -- but it also did
NOT make the two paths match. **This is the honest, real residual
finding**: a clean, single "cold" verify call (prefill, then ONE chunked
verify vs ONE spec verify, both on identical input, no alternation, no
artificial draft fabrication) at full model scale (48 GDN layers) gives
hidden-state cosine similarity ~0.996 (max diff ~2.4) -- while the
correct control (two independent chunked-path verify calls, no spec
involved at all) gives EXACT 0.0 diff, bit-for-bit identical, at the
same scale. So the true noise floor at 48 layers is zero, and a
0.996-cosine gap is real, not acceptable kernel-algorithm noise.

**Third hypothesis (checked, ruled out, time-boxed): GQA head-count /
`dt_bias`/`A_log` indexing.** The real model's GDN layer has
`num_k_heads=16`, `num_v_heads=48` (a 1:3 expansion ratio, `A_log`/
`dt_bias` shaped `(48,)`) -- a genuine GQA-style asymmetry my earlier
toy isolated test (`H=HV=2`) didn't exercise. Checked whether this could
explain the gap: it doesn't, on inspection -- head decomposition
(`i_h = i_hv // (HV//H)`) is entirely internal to the kernel, derived
from q/k/v/A_log tensor shapes the MODEL constructs (unaffected by this
project's metadata, which only carries per-REQUEST information:
`cu_seqlens`/`ssm_state_indices`/`num_accepted_tokens`, never per-head).
Re-ran the isolated SSM kernel test with a proper zero-noise control
this time: true-control diff is exactly 0.0, and the "spec cold vs
reference" diff is ~0.001-0.004 (position-wise) and ~0.004 (final
state) -- genuine bf16-level noise, not a real bug at the SSM/recurrent
level. The residual gap is therefore NOT explained by GQA heads or
`dt_bias`/`A_log` indexing -- it most likely lives on the CONV side
specifically (the isolated conv1d "cold" test showed ~0.03 max diff at
real DIM=10240 scale, noticeably larger than the SSM side's ~0.001-0.004
and larger than pure bf16 noise would suggest for a comparable
computation), compounding across 48 layers into the observed
0.996-cosine full-model gap. This was not further investigated --
continuing would mean opening a fourth, different hypothesis
(conv-specific state alignment/windowing at the bootstrap transition,
distinct from GQA/heads), which is the explicit signal (per this round's
own time-boxing instruction) to stop rather than continue.

### 10.3 What the next attempt should try first

If someone picks this line up again, the highest-value next step is
almost certainly **the conv side specifically**, not the SSM side (which
is verified working): investigate whether `causal_conv1d_update`'s
"effective `state_len = width - 1 + (seqlen - 1)`" computation (a
PER-CALL value depending on `max_query_len`, confirmed by reading the
wrapper in `causal_conv1d.py:1181-1185`) has some alignment/windowing
subtlety specific to the bootstrap transition (reading a buffer whose
first `width-1` positions were written by the CHUNKED kernel's own
narrower, `state_len=width-1`-only convention, then read via the spec
kernel's wider, call-dependent `state_len` with `conv_state_token_offset
= num_accepted_tokens - 1`) that isn't captured by comparing only the
FINAL output/state values the way this round's tests did. A targeted
next test: dump/compare the conv1d OUTPUT (not just final state) for
ALL 4 verify positions against a hand-computed manual causal convolution
(the same technique this round used successfully to distinguish "real
bug" from "test artifact" for the isolated kernel tests) but built
directly into the FULL MODEL test via a hook/monkeypatch on the conv1d
call, rather than a separate from-scratch isolated kernel test -- this
round's isolated tests kept being one level removed from the actual
full-model computation, which is plausibly why a small conv-specific
discrepancy went uncaught until the full-model comparison.

### 10.4 Housekeeping

No code from this attempt is committed. `runtime/direct_model_runner.py`
was reverted to its state at commit `51a216e` (discarding the additive-
but-incomplete `_ssm_spec_row`/`build_gdn_metadata_spec_batch`/
`verify_batch_spec_gdn`/SSM-allocation-expansion changes described
above) immediately after this section was written, so nothing broken or
half-working is left in the working tree. The exploratory test files
(`benchmarks/phase2_spec_gdn_parity_check.py` and various one-off
isolated kernel repro scripts run via `python3 -c "..."`, never saved as
files) are not committed either -- this section is the complete record
of what was tried and found. GPU/process state confirmed clean via
`pgrep`/`nvidia-smi` before and after this attempt, as always.
