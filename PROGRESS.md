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

**Step 2 (this round, in progress): building a no-HTTP correct baseline
that reuses vLLM's real `GPUModelRunner`/`Scheduler`/`KVCacheManager`
in-process** (HTTP removed, nothing else) -- see below for progress.

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
