# Direct model runner: design (2026-07-15/16)

Replaces the HTTP bridge (`runtime/vllm_bridge_backend.py`, commit `b28942c`)
with an in-process runner that owns GPU KV/GDN state directly, per the
project's re-prioritized main line (2026-07-16: attention-kernel tuning in
the sibling `sm120-flash-attention` project hit diminishing returns --
decode v2/prefill v2's "beats native" claims were both overturned -- so
development effort moved here).

## 2026-07-16, strategy reset after four debugging passes: freeze per-kernel
## bisection, rebuild top-down from vLLM's real GPUModelRunner instead

After four full debugging passes (see the dated sections below) surfaced two
real, specific, independently-confirmed low-level anomalies (a Triton
`causal_conv1d_fn` cold-start bug; a CUTLASS SM120 pingpong-GEMM race per
`compute-sanitizer racecheck`) -- and then *disproved* both as the actual
root cause via direct bypass experiments -- the working theory changed.
**The real, most likely root cause: this runner hand-builds attention/GDN
metadata and calls `model.forward()` directly, but skips a whole contract
`vllm/v1/worker/gpu_model_runner.py`'s real `GPUModelRunner` normally
upholds** -- the distinction between `num_tokens_unpadded`/`num_tokens_padded`
and padding-region handling (e.g. `slot_mapping=-1` for pad slots), the real
metadata builders' persistent-buffer/CUDA-graph-safety conventions, the real
warmup/profiling call sequence, stream/event lifetime management, and how
GDN state initialization interacts with request/batch bookkeeping. This
runner's hand-rolled version approximates each of these individually but was
never checked against the real contract as a whole.

**Per this reset, the per-kernel bisection line (racecheck, kernel
swapping, Triton warmup theories) is now frozen** -- not abandoned as
worthless (both anomalies are real and independently documented below,
each as its own defect), but no longer treated as the search for *the*
root cause. The two sections below ("Known independent defects") capture
them as closed, standalone findings. All further work in this doc follows
a new, top-down plan: (1) formalize the known-wrong repro and characterize
its failure rate precisely (done, see immediately below), (2) build a
"no-HTTP correct baseline" that reuses vLLM's *real*, already-verified
`GPUModelRunner`/`Scheduler`/`KVCacheManager` machinery in-process (HTTP
removed, nothing else), (3) only once that baseline is solid, do a
three-stage ownership transfer (vLLM metadata+vLLM cache -> vLLM
metadata+our cache -> our metadata+our cache -> trimmed direct runner),
comparing intermediate tensors at each stage to localize exactly where
divergence starts.

## Step 1 result: the wrong output is 100% deterministic, not a race

Formalized the ad hoc `/tmp` repro as a committed script,
`benchmarks/single_prefill_regression.py` -- fixed prompt ("The capital of
France is"), fixed model/config (`unsloth/Qwen3.6-27B-NVFP4`,
`kv_cache_dtype=fp8_e4m3`, default kernel selection, no env-var bypasses),
records the first token id/text and a SHA-256 hash of the full logits
vector, and can run N repeats (each a fresh process, since prior rounds
showed process-level state -- Triton kernel cache warmth -- can matter).

**Ran 20 consecutive fresh-process repeats. Result: 20/20 identical
failures** -- same wrong first token every time (`'东'`, id 96265), and
critically **the exact same SHA-256 logits hash on all 20 runs**
(`720406302a42f76f`), meaning the full output logits vector was bit-for-bit
identical across all 20 independent process runs.

**This is an important clarification, not just a confirmation**: the wrong
output is **fully deterministic** under a fixed configuration -- it is not
a true hardware race manifesting randomly run to run. The apparent
"Heisenbug" behavior in the third debugging pass (output flipping between
runs) is now understood to have been caused by *changing* something about
the test between those runs (different token counts, different
instrumentation inserted, different kernel bypass flags) that altered which
code path ran -- not genuine non-determinism under an unchanged
configuration. This matters for where to look next: a 100%-reproducible
deterministic bug is much more consistent with a **semantic/contract
mismatch** (wrong metadata field, wrong padding convention, wrong state
initialization -- exactly the class of bug the strategy reset above is
now targeting) than with a genuine intra-kernel race whose outcome should
vary with scheduling timing. The CUTLASS racecheck hazard and the
causal_conv1d_fn cold-start bug are both still real (see below) but neither
one, on its own, would be expected to produce the *same* wrong answer
byte-for-byte on 20/20 runs -- a genuine race's effect on final output would
be expected to vary at least occasionally if it were the dominant cause.

## Known independent defects (real, documented, NOT treated as root cause)

**Defect 1 -- Triton `causal_conv1d_fn` first-call-in-process returns
all-zero output.** Isolated, reproducible repro (fresh process, `dim=10240`,
`width=4`, real GDN-layer shapes, 4 repeated calls with fresh random
tensors): call#0 all-zero, call#1-3 fully correct. Confirmed NOT the (sole)
cause of this runner's wrong output: instrumenting all 48 real GDN-layer
calls within one real forward pass showed every call producing correct,
non-zero output in one full run that *still* generated the wrong final
token. Real, worth fixing/upstreaming eventually, out of scope for the
current root-cause search.

**Defect 2 -- CUTLASS SM120 warp-specialized "pingpong" GEMM potential race
(compute-sanitizer racecheck).** 100 consistent "Potential RAW hazard"
reports, all in `cutlass_scaled_mm_sm120` (used by
`CutlassFP8ScaledMMLinearKernel` for one of GDN's FP8 W8A8 linear
projections), Write Thread 63 / Read Thread 128, varying shared-memory
offset and CUDA block. Caveat: TMA/mbarrier-synchronized warp-specialized
kernels are a documented source of racecheck false positives, so "potential"
(the tool's own word) is not "confirmed." Confirmed NOT the (sole) cause:
forcing `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` (bypassing
this exact kernel for a plain PyTorch fallback) changed the output but did
not fix it. Real, worth investigating/reporting upstream eventually if
confirmed, out of scope for the current root-cause search.

## What changes vs the HTTP bridge

The HTTP bridge proved the control plane (`EagerEngine`/`HybridCache`/
`FixedSlotManager`) against a *real* model, but the real vLLM engine (a
separate server process) still owned the physical KV cache / GDN state
tensors -- our runtime only sent requests over HTTP. This design removes
that separate process entirely: our own process loads the model, allocates
the KV/GDN tensors, and drives `model.forward()` directly.

## The four things that make "direct ownership" possible (all reused, not
reinvented -- confirmed by reading the actual vLLM source, not guessed)

1. **`vllm.engine.arg_utils.EngineArgs.create_engine_config()`** builds a
   real, fully-resolved `VllmConfig` from the same kind of arguments
   `vllm serve`/`LLM()` accept (model id, `kv_cache_dtype`, `attention_backend`,
   `max_model_len`, ...). We do not hand-roll `VllmConfig` -- construction is
   config-resolution-only, no engine/scheduler is created by this call.
2. **`vllm.model_executor.model_loader.get_model(vllm_config=...)`** loads the
   real model (`unsloth/Qwen3.6-27B-NVFP4`) as a plain `nn.Module`, with real
   weights, real NVFP4-quantized `Linear` layers (already selecting
   `FlashInferCutlassNvFp4LinearKernel` internally -- confirmed from server
   logs in the HTTP-bridge round), and real GDN layers -- no engine, no
   scheduler, no KV cache allocated yet. Every attention layer and every GDN
   layer registers itself into
   `vllm_config.compilation_config.static_forward_context[layer_name]`
   during `__init__` as a side effect of construction -- this is how we later
   discover the exact layer names and layer objects to wire up.
3. **`vllm.v1.worker.utils.bind_kv_cache(kv_caches: dict[str, tensor|tuple],
   forward_context: dict[str, layer], runner_kv_caches: [])`** is the *exact*
   function vLLM's own `GPUModelRunner` uses to attach allocated KV cache
   tensors to layer objects (`forward_context[layer_name].kv_cache =
   kv_caches[layer_name]`). It doesn't care whether the value is a single
   tensor (attention layers) or a `(conv_state, ssm_state)` tuple (GDN layers,
   per `MambaBase.kv_cache: tuple[torch.Tensor, ...]`) -- we call this
   ourselves, once, after allocating our own 4-slot tensors, instead of
   letting `GPUModelRunner.initialize_kv_cache()` do it from a
   dynamically-sized `KVCacheManager`.
4. **`vllm.forward_context.set_forward_context(attn_metadata, vllm_config)`**
   is the context manager `model.forward()` expects to be wrapped in; inside
   it, every attention/GDN layer's own `forward()` calls
   `get_forward_context().attn_metadata[self.prefix]` to get its metadata for
   *this* call. `attn_metadata` is `dict[layer_name, metadata_object]` --
   nothing stops us from constructing that dict by hand for a single request,
   as long as the metadata objects have the fields the backend's `Impl`
   actually reads.

None of (1)-(4) needed to be modified or reimplemented -- they are the same
functions/classes vLLM's own `GPUModelRunner` calls, just invoked directly
by us instead of through the Scheduler/KVCacheManager. This is the whole
point: skip the *dynamic, block-manager-driven* KV allocation, keep
everything else.

## Per-slot tensor layout (4 fixed slots, this round: only slot 0 exercised)

- **Attention KV cache** (16 full-attention layers): one tensor per layer,
  shape `SM120GQABackend.get_kv_cache_shape(num_blocks, block_size,
  num_kv_heads, head_size, cache_dtype_str)` = `(num_blocks, 2, block_size,
  num_kv_heads, head_size)` (NHD layout; `num_blocks = 4 slots *
  blocks_per_slot`, slot `i` owns block range `[i*blocks_per_slot,
  (i+1)*blocks_per_slot)` -- a static partition, not vLLM's dynamic block
  pool). `cache_dtype_str="fp8_e4m3"` matches this project's already-validated
  FP8-KV path.
- **GDN state** (48 GDN layers): call `layer.get_state_shape()` ->
  `(conv_state_shape, ssm_state_shape)` and `layer.get_state_dtype()` ->
  `(conv_dtype, ssm_dtype)` **on the actual layer instance** (no need to
  reconstruct `MambaSpec` by hand) and allocate `torch.empty((4,
  *conv_state_shape), dtype=conv_dtype)` / `torch.empty((4, *ssm_state_shape),
  dtype=ssm_dtype)` per layer -- slot `i` is index `i` along dim 0, again a
  static partition instead of a page table.
- Both are allocated **once**, at startup, and never reallocated -- matching
  the "persistent, fixed physical slot" design this project's HTTP-bridge
  round (`HybridCache`/`FixedSlotManager`) already validated at the control-
  plane level, now extended to the actual GPU tensors.

## Per-step metadata (hand-built for one request in slot 0 -- no CUDA graph,
## no multi-request batching this round)

**Attention (`SM120GQAMetadata`, one shared instance for all 16 layers this
step -- the dataclass encodes request-level info, not per-layer info; the
per-layer KV tensor difference is entirely carried by `layer.kv_cache`, set
once via `bind_kv_cache`)**:
- Prefill (fresh slot, N prompt tokens): `num_reqs=1`,
  `qo_indptr=[0,N]`, `kv_page_indptr=[0, ceil(N/block_size)]`,
  `kv_page_indices=[slot_first_block .. slot_first_block+ceil(N/block_size)-1]`,
  `kv_last_page_len=[N - (ceil(N/block_size)-1)*block_size]`,
  `is_pure_decode=False`, `decode_qo_len=0`.
- Decode (append 1 token, kv_len becomes N+1): same shape family with
  `num_actual_tokens=1`, `is_pure_decode=True`, `decode_qo_len=1`, page
  indices/last-page-len recomputed for the new `kv_len`.

**GDN (`GDNAttentionMetadata`, one shared instance for all 48 layers this
step, same request-level-not-per-layer reasoning)**:
- Prefill: `num_prefills=1`, `num_prefill_tokens=N`, `num_decodes=0`,
  `non_spec_state_indices_tensor=[0]` (slot 0), `has_initial_state=[False]`,
  `chunk_indices`/`chunk_offsets` via
  `vllm.model_executor.layers.fla.ops.index.{prepare_chunk_indices,
  prepare_chunk_offsets}` (reused, not reimplemented) against
  `prefill_query_start_loc=[0,N]` and `FLA_CHUNK_SIZE`, `nums_dict`/
  `batch_ptr`/`token_chunk_offset_ptr` via
  `vllm.v1.attention.backends.utils.compute_causal_conv1d_metadata` (reused).
- Decode: `num_decodes=1`, `num_prefills=0`,
  `non_spec_state_indices_tensor=[0]`, `non_spec_query_start_loc=[0,1]`; the
  chunk-index fields are prefill-only and stay `None`.

This is deliberately much smaller than the real
`GDNAttentionMetadataBuilder.build()` / `SM120GQAMetadataBuilder.build()` --
those handle spec-decode, multi-request batching, and CUDA-graph-safe
persistent buffers, none of which this round's single-request/no-MTP/no-graph
scope needs. The real builders remain the reference for whatever this
minimal version is missing once scope grows (4 concurrent slots, MTP verify
batches, CUDA graph capture) -- **not** something to copy wholesale now.

## Forward loop

```
with set_forward_context(attn_metadata={**{n: attn_meta for n in attn_layer_names},
                                         **{n: gdn_meta for n in gdn_layer_names}},
                          vllm_config=vllm_config):
    hidden_states = model.forward(input_ids, positions)
logits = model.compute_logits(hidden_states)
next_token = logits[-1].argmax(-1)  # greedy, this round
```
`input_ids`/`positions` for prefill are the whole prompt; for decode they are
the single new token id and its absolute position (`prompt_len +
num_generated_so_far`).

## Explicitly out of scope this round (per the coordinator's own bound --
## "不需要一次性做到完整的4并发+CUDA Graph")

- Only slot 0 is exercised; slots 1-3 are allocated but unused.
- No CUDA Graph capture (eager `model.forward()` calls only).
- No real concurrent multi-request batching (the metadata above is
  single-request only).
- No MTP/speculative decode (`decode_qo_len` stays 1, never >1).
- GDN reuses the existing implementation verbatim (this round's own nsys
  full-stack trace put GDN at 8.0% of GPU kernel time -- not the bottleneck,
  not this round's target; see PROGRESS.md).

## Verification plan

Signal-probe methodology (this project's established pattern, reused from
the HTTP-bridge round and the sibling project's own convention): embed a
distinct marker fact, generate a completion, confirm the marker is recalled
correctly. Run once through this new direct runner and compare against the
same prompt's known-good output from the HTTP-bridge round -- if directly
owning GPU state introduced a state-contamination bug (wrong page indices,
wrong GDN slot index, stale `has_initial_state`), this is exactly the kind of
bug a naive "it didn't crash" check would miss.

## Current state (2026-07-16): runs end-to-end, output is WRONG -- not yet
## a working closed loop, and this is being reported as such, not glossed
## over

The full pipeline (`runtime/direct_model_runner.py`) runs without crashing:
model loads with real NVFP4 weights, `SM120GQAImpl` is selected as the real
attention impl, `bind_kv_cache` wires real per-slot tensors onto all 64
layers, and one prefill + several greedy decode steps execute. But the
output for `"The capital of France is"` is not " Paris" -- it is
incoherent, low-confidence, mixed-language garbage (top logit ~8.9, where a
correct/confident completion should be much higher) from the very first
token, i.e. **already wrong within the single prefill forward pass**, before
any decode step or cross-call state reuse is even involved.

**First real bug found and fixed**: `ForwardContext.slot_mapping` is a
*separate* field from `attn_metadata` (see `attention.py`'s
`get_attention_context`: `forward_context.slot_mapping[layer_name]`, read
independently of `attn_metadata_raw[layer_name]`) that tells
`do_kv_cache_update` where to write each new token's K/V. The initial version
of this runner never populated it (`set_forward_context`'s default is `{}`),
so `layer_slot_mapping` was always `None` and `do_kv_cache_update` silently
never ran -- K/V were **never written to the cache at all**. Fixed by adding
`DirectModelRunner._slot_mapping()` (the standard `block_id * block_size +
offset` convention) and passing it via `set_forward_context(...,
slot_mapping=slot_mapping_dict)`. This measurably changed the (still wrong)
output, and post-fix the attention KV cache tensor confirmably contains
non-zero data after prefill (`10240/32768` non-zero elements in layer 0) --
so this fix is real, even though it wasn't sufficient on its own.

**Hypotheses checked and ruled out** (so the next debugging session doesn't
re-derive these):
- KV cache dtype mismatch: `any_attn.kv_cache_torch_dtype` resolves to
  `torch.uint8` for `kv_cache_dtype="fp8_e4m3"` (confirmed via
  `STR_DTYPE_TO_TORCH_DTYPE` in `vllm/utils/torch_utils.py`), matching what
  `SM120GQAImpl.forward()`'s `is_fp8_kv` check expects
  (`key_cache.dtype==torch.uint8 and shape[-1]==head_size`) -- not a dtype
  bug.
- FP8 KV cache default `k_scale`/`v_scale`=1.0 producing wrong results: ruled
  out by comparison -- the HTTP-bridge round (commit `b28942c`) used the
  *exact same* `kv_cache_dtype="fp8_e4m3"` with the same uncalibrated default
  scale, through the real vLLM engine, and got a correct " Paris" answer for
  the same prompt. Since the quantization/scale setup is identical, the bug
  must be in this round's hand-built metadata, not in FP8-KV itself.
- `positions` needing mrope (3, seq_len) shape: checked
  `qwen3_5.py`'s own `@support_torch_compile` docstring -- mrope only
  applies to Qwen2-VL-style interleaved image/video positions; a pure-text
  forward pass correctly uses the plain `(seq_len,)` 1D shape this runner
  already passes. Not the bug.
- The attention kernel dispatch path for `decode_qo_len=0` (my prefill
  setting) is documented in `sm120_gqa.py` as the general/robust path
  ("correct for pure prefill, chunked-prefill continuation, and arbitrary
  mixed prefill+decode batches") and is the same path that already produced
  correct behavior in the real engine for identical metadata field *values*
  -- plausible but not proven innocent; worth revisiting if the GDN lead
  below dead-ends.

## 2026-07-16 continued: deep dive on the conv_state lead -- real progress, not yet a fix

Per the coordinator's explicit direction, chased the `conv_state` lead to
ground rather than jumping to other hypotheses. Order followed: (1) check
whether the conv1d's own *input* is already wrong, (2) compare against how
the HTTP-bridge round's real path triggers the conv-state write, (3) check
the state buffer's binding/lifecycle. Found something real, but it does not
yet add up to a working fix -- reported exactly as far as it goes.

**Step 1 result -- input is fine, but so is `causal_conv1d_fn`'s own output
computation, at least sometimes**: hooked `causal_conv1d_fn` directly
(monkeypatch in `qwen_gdn_linear_attn`'s module namespace, since it's
imported by name). `mixed_qkv` (the conv1d's input) is real, non-degenerate,
non-zero data (51200/51200 non-zero, sane mean/std) every time -- input is
not the problem. But `causal_conv1d_fn`'s **return value** (`out`, the
actual conv1d output that feeds the rest of GDN) was *also* frequently all
Zero in early instrumented runs -- not just `conv_state`. That reframed the
question: state not persisting might be a symptom of the whole kernel
call silently no-op'ing, not an isolated state-write bug.

**Step 2 finding -- reproduced a real `causal_conv1d_fn` bug in complete
isolation, independent of this runtime, the model, or its metadata**: wrote
a minimal standalone script (no model, no `DirectModelRunner`, just
`causal_conv1d_fn` called directly with hand-built random tensors matching
this GDN layer's real shapes: `dim=10240`, `width=4`, `cache_indices=[0]`,
`has_initial_state=[False]`). Result, calling it 4 times in a fresh process
with fresh random inputs each time:
```
call#0: out_nonzero=     0/51200   <- first call: silently all-zero
call#1: out_nonzero= 51200/51200   <- every later call: fully correct
call#2: out_nonzero= 51200/51200
call#3: out_nonzero= 51200/51200
```
This is a genuine, deterministic, repeatable "cold start" bug: **the very
first invocation of this Triton kernel in a process returns an all-zero
result**, independent of any of this runtime's code. This isn't
metadata-construction-specific -- it reproduces with a bare, textbook call.

**Why this matters directly**: real vLLM always runs a profiling/warmup
forward pass before serving any real request (confirmed in this project's
own server logs: `monitor.py:81] Initial profiling/warmup run took 3.64s`).
This runner's first-ever call to the model was the real prefill itself --
exactly the "first call" this cold-start bug breaks. Added `_warmup()` to
`DirectModelRunner.__init__` (runs one throwaway prefill on slot 0, then
calls `reset_slot(0)` so it looks untouched) to mirror this.

**But: the warmup fix did NOT fix the real end-to-end output.** Tried both
a 1-token dummy warmup and a 5-token dummy warmup (matching the real "The
capital of France is" prompt's exact token count) -- both still produced
the identical wrong output (`'东Ё¨¨¨¨...'`, byte-for-byte the same
completion in both cases). So the isolated repro's "just call it twice"
fix does not transfer cleanly to the full 64-layer model.

**Follow-up isolated tests show the bug is messier than "first call bad,
rest fine" -- this is the honest, unresolved part**:
- Repeating the *exact same* shape (`num_tokens=5`, `dim=10240`) 4 times in
  a row, with no other shapes interleaved: call#0 zero, call#1-3 correct
  (matches the simple story).
- But interleaving shapes (`num_tokens=1` then `5` then `5` then `5` then
  `1` then `1` then `1`) gives a different, harder-to-explain pattern: the
  `num_tokens=1` shape recovers after its own first (zero) call and stays
  correct for its next 3 calls, but `num_tokens=5` stayed all-zero for 3
  calls *in a row*, even on its 2nd and 3rd attempts. So "first call at a
  given shape is bad, everything after is fine" is too simple a model --
  there's some additional state (possibly Triton's kernel-variant cache,
  possibly something about calling different shapes back-to-back) this
  investigation has not characterized.
- Given that, the real model's warmup not fixing the real output is
  consistent with "the model's 48 GDN layers each call this kernel with
  their own distinct compiled variant" or some other confound not yet
  isolated -- not proof the cold-start finding is irrelevant, but proof
  a single blanket warmup call is not (yet) the right fix.

**Separately, and still true regardless of the above**: even in the
"working" isolated calls (call#1-3 above, `out` fully non-zero and
correct-looking), `conv_state` itself remained **entirely zero** every
time. So there are likely **two distinct issues** in this area, not one:
(a) the cold-start all-zero-output bug (real, isolated, reproducible,
partially understood), and (b) `conv_state` never actually persisting even
when the surrounding computation is otherwise correct (real, isolated,
reproducible, *not yet investigated in isolation* -- steps 2-3 of the
original plan for this specific sub-question were not reached this round).

**Reproduction scripts** (kept for the next session, not committed --
scratch-only, paths under `/tmp/qwen_check/`, referenced here by content
since the files themselves won't survive): the core minimal repro is ~30
lines -- build `mixed_qkv`/`conv_weights`/`bias` with `torch.randn` at
`dim=10240,width=4,num_tokens=5`, a zeroed `conv_state` of shape
`(1, width-1, dim)` transposed to `(1, dim, width-1)`, `cache_indices=[0]`,
`has_initial_state=[False]`, `query_start_loc=[0,5]`, and call
`causal_conv1d_fn(...)` 4 times in a loop with fresh tensors each iteration
-- call 0 is zero, 1-3 are correct. Worth re-creating as a committed,
permanent regression-probe script (`kernel/tests` equivalent) once this is
actually root-caused, so it can be re-run against future Triton/vLLM
version bumps.

**Next debugging steps** (revised, more specific than last round):
1. Characterize the cold-start bug's real scope: is it keyed by Triton's
   internal kernel-variant cache (which would mean each DISTINCT
   `(dim, width, dtype, num_stages, ...)` compile signature needs its own
   separate warm-up call), or is it about something else entirely (a
   process-wide CUDA/cuBLAS/cuDNN handle lazy-init race, unrelated to
   Triton's own caching)? Try: does calling a *different* Triton kernel
   first (unrelated to causal_conv1d) still leave causal_conv1d's own first
   call broken? If yes, this points away from "per-kernel-variant Triton
   caching" and toward something more global (first CUDA kernel launch of
   any kind in the process, a known class of driver/context lazy-init
   quirk) -- test this specifically, it's cheap.
2. Since the real model's 48 GDN layers likely differ only in weight
   VALUES (not shapes/dtypes -- all should be `dim=10240,width=4`), a
   single real forward pass already calls this kernel 48 times per prefill;
   if "same shape, called repeatedly" were sufficient (as the monotonic
   isolated test suggested), layers 2-48 within the SAME forward call
   should already self-correct even without an explicit separate warmup.
   Instrument all 48 calls within one real forward pass (not just the
   first 2, as this round's hook did) to check whether output nonzero-ness
   turns on partway through the 48 layers -- if it does, that's a
   different, more specific, more fixable finding than "warmup the whole
   model first."
3. Investigate `conv_state` non-persistence (finding (b) above) as its own,
   separate isolated repro -- e.g. check whether calling `causal_conv1d_fn`
   with `has_initial_state=True` and a pre-filled, non-zero `conv_state`
   causes the OUTPUT to actually depend on that pre-filled state (would
   confirm reads work even if writes don't), and try varying
   `cache_indices`/batch size in the isolated script to see if state-write
   ever succeeds under ANY isolated configuration -- this round did not
   find one.
4. Once output is genuinely correct for "The capital of France is" ->
   "Paris" (the stated minimum bar), re-run the *exact* signal-probe
   prompts from the HTTP-bridge round (`benchmarks/real_forward_smoke.py`'s
   marker-code prompts) for a like-for-like comparison, and specifically
   re-test slot release + reuse (`reset_slot`) under this direct ownership
   model -- that is the scenario this whole effort exists to get right
   (vLLM issue #37554's risk class); a mechanism that merely "runs" without
   that guarantee holding is not yet done.

## 2026-07-16, third pass: followed the coordinator's exact 3-step order --
## a major, honest revision, not a fix

Ran all three steps as directed. The headline result: **the isolated
cold-start bug is real, but this round found direct evidence it is
probably NOT the (sole) cause of the wrong final output** -- a materially
different, more sobering conclusion than the previous round's framing
implied. Reporting the full picture, including the parts that don't add up
cleanly yet.

**Step 1 -- unrelated-Triton-kernel warmup**: ran a trivial, unrelated
`@triton.jit` elementwise-add kernel first, then `causal_conv1d_fn`, in a
fresh process. Result: the unrelated kernel does **not** fix
`causal_conv1d_fn`'s first call (still all-zero) -- and, more surprising,
it made every *subsequent* call zero too (previously, repeating the same
causal_conv1d_fn call alone showed call#0 bad / call#1+ good). So "any GPU
kernel warms up some global CUDA/Triton state" is **ruled out** -- whatever
is happening is either specific to `causal_conv1d_fn`'s own kernel-variant
cache, or an interfering-kernel-order effect, not a generic first-kernel-
in-process phenomenon.

**Step 2 -- instrumented all 48 real GDN-layer calls within one actual
forward pass** (both the runner's own `_warmup()` prefill and the
subsequent real "The capital of France is" prefill): every single one of
the 96 total calls (48 in warmup + 48 in the real prefill) showed **fully
non-zero, plausible output** (the ~31520/51200 pattern on roughly every
4th GDN layer is consistent with real SiLU sparsity, not an error). In
other words: **inside the real model, `causal_conv1d_fn` was working
correctly the entire time in this run** -- and yet the model's final
generated token was still wrong (the same `'东'` garbage as every previous
round). This is the load-bearing finding of this pass: if conv1d's own
output is fine throughout a run that still produces wrong text, the
isolated cold-start bug found last round is **not, by itself, sufficient
to explain the wrong output** -- it may be a real, independent bug that
happens to coexist, not the root cause of the "Paris" failure.

**Step 3 -- checked conv_state before/after on that same kind of run,
separately**: here the results stopped agreeing with step 2. In a rerun
with added before/after `conv_states.count_nonzero()` instrumentation
(the *only* code difference from step 2's script -- purely additional
reads, no writes), **all 48 real-prefill calls came back all-zero this
time** (both `out` and `conv_state`), and the model's output changed too
(`' is'` instead of `'东'` as the first token -- still wrong, just
differently wrong). Rerunning nominally-identical code and getting
opposite conv1d-output results, seemingly triggered by adding read-only
diagnostic instrumentation, is a classic Heisenbug signature (behavior
changes when observed) -- strong circumstantial evidence of a genuine race
condition somewhere in this stack, not a deterministic missing-parameter
bug. This also means **the isolated single-process repro from last round,
while real and reproducible in isolation, does not behave identically
inside the full 64-layer model** -- the two contexts are not interchangeable
for debugging purposes.

**Tried, as a race-condition-motivated hypothesis, and it did not fix the
output**: set `EngineArgs(async_scheduling=False)` (this project's log
output had shown "Asynchronous scheduling is enabled" -- a real vLLM
feature this minimal runner does not otherwise replicate the careful
buffer-lifetime handling for) and added explicit `torch.cuda.synchronize()`
calls immediately after `model.forward()` and after `compute_logits()`.
Result: identical wrong output (`'东Ё¨¨¨...'`) to before -- this specific
async/sync hypothesis is now also ruled out, at least in this simple form.

**Where this leaves the investigation, honestly**: three real, reproducible
findings that do not yet form one coherent story:
1. `causal_conv1d_fn`'s first-ever call in an isolated process is
   deterministically all-zero (very reproducible, simple repro).
2. Inside the real model, conv1d's own output has been observed BOTH
   fully-correct-throughout (step 2) AND all-zero-throughout (step 3) across
   different runs of what should be the same code path -- i.e. genuinely
   non-deterministic at the full-model level, not just at cold-start.
3. The model's *final* output has been wrong in every run so far
   regardless of which of the above conv1d behaviors occurred in that run
   -- meaning conv1d correctness this round did not correlate with output
   correctness. The actual root cause of the wrong "Paris" answer is more
   likely elsewhere (or is itself a manifestation of the same
   non-determinism showing up in a different layer/kernel each run) --
   **not yet identified**.

**Next steps, revised given the non-determinism finding**:
1. Stop trying to root-cause this with print/instrumentation-based
   bisection -- this round's own step 2 vs step 3 contradiction shows that
   approach can change the outcome being investigated. Use
   `compute-sanitizer --tool racecheck` (this project's own established
   heavy-but-authoritative tool, already used successfully in the sibling
   project) against a minimal repro to get a real race-condition diagnosis
   instead of continued black-box guessing. Expect it to be slow (10-50x
   per this project's own prior experience) -- run it on the isolated
   single-call repro, not the full 64-layer model, to keep it tractable.
2. Once (1) either confirms or rules out a race condition, re-run the
   exact step-2/step-3 scripts (kept as scratch content in this doc's
   prior section, worth committing as permanent regression probes now)
   several more times each to get an actual failure RATE, not just one
   data point each way -- needed to tell "flaky" from "environment
   changed between runs for an unrelated reason."
3. Given conv1d correctness didn't correlate with output correctness this
   round, broaden the search: instrument the ATTENTION layers' output
   similarly (not just GDN/conv1d) across a couple of runs, to see if
   non-determinism (or a deterministic bug) shows up there instead/also --
   this round only looked at GDN.

## 2026-07-16, fourth pass: compute-sanitizer racecheck -- a real, specific
## localization, with an important caveat

Per the coordinator's explicit direction, stopped print-based bisection and
ran `compute-sanitizer --tool racecheck --racecheck-report all` against the
minimal single-prefill repro (`runner.__init__()`'s own warmup prefill,
`"The capital of France is"`, `unsloth/Qwen3.6-27B-NVFP4`,
`kv_cache_dtype=fp8_e4m3`). GPU verified free via `nvidia-smi`/`ps` before
launch, per standing convention.

**Result: found a real, consistent, specific hazard -- not a dead end.**
Weight loading alone took ~3 minutes under instrumentation (vs. ~15s
normally, consistent with this project's known 10-50x racecheck slowdown
expectation). Once the model reached its own single warmup forward pass,
racecheck immediately started reporting: **100 "Potential RAW hazard"
errors** (its default report cap), every single one of the same shape:

- Same kernel every time: CUTLASS's SM120 **warp-specialized "pingpong"**
  GEMM (`MainloopSm120TmaWarpSpecialized`,
  `KernelTmaWarpSpecializedPingpongSm120`, e4m3 x e4m3 -> bf16, TMA-loaded,
  swizzled shared memory) -- this is the `cutlass_scaled_mm_sm120` kernel
  invoked by `CutlassFP8ScaledMMLinearKernel` /
  `CompressedTensorsW8A8Fp8.apply_weights`, called from
  `qwen_gdn_linear_attn.py:923`'s `forward_cuda` -- i.e. one of GDN's own
  FP8 W8A8-quantized linear projections (this checkpoint mixes NVFP4 for
  most weights with FP8 W8A8 for some GDN-layer linears; confirmed from the
  server log's own "Selected CutlassFP8ScaledMMLinearKernel for
  CompressedTensorsW8A8Fp8" line).
- Same thread pair every time: **Write Thread (63,0,0)**, **Read Thread
  (128,0,0)** -- thread 63 is the last thread of warp 1 (a producer/TMA-load
  warp in this design), thread 128 is the first thread of warp 4 (a
  consumer/MMA warp) -- textbook producer-consumer warp-specialization
  hand-off shape.
- Varying only the shared-memory offset (`0x530` through `0x53f`, a tight
  16-address range -- one tile's worth of a swizzled operand) and the CUDA
  block index (105, 178, 54, 60, 63, 142, 61, ... -- different tiles/blocks
  of the same GEMM, all showing the identical hazard pattern).

This directly answers the coordinator's question 2: **this is not a
`direct_model_runner.py`-level synchronization bug.** The race, if real, is
between two CUDA *threads* inside a *single* CUTLASS kernel launch --
nothing this project's Python-level orchestration code does (or fails to
do) between kernel launches can reach into or fix intra-kernel warp
synchronization. This also retroactively explains why the earlier
`async_scheduling=False` + `torch.cuda.synchronize()` experiment did
nothing: those add synchronization *between* host-issued operations, not
inside a kernel's own warp-specialized pipeline.

**The one important caveat, stated plainly rather than glossed over**:
CUTLASS's Hopper/Blackwell-generation warp-specialized kernels synchronize
producer and consumer warps via **mbarrier** (hardware async-barrier)
primitives, not the plain `__syncthreads()`-style barriers `racecheck`'s
shared-memory hazard detector was originally built to model. This class of
kernel is a **documented source of racecheck false positives** industry-wide
-- the tool can flag a "potential" hazard (its own wording: "Potential RAW
hazard", never "confirmed") between a producer's write and a consumer's read
even when the mbarrier wait genuinely enforces correct ordering, because
racecheck doesn't fully model that synchronization primitive. So this
finding should be read as: **a real, specific, highly consistent signal
pointing at exactly one kernel and one thread pair** -- strong enough to
stop suspecting `direct_model_runner.py`'s own code and to redirect
investigation at this specific CUTLASS kernel -- but **not proof, on its
own, that CUTLASS's SM120 pingpong GEMM has a genuine bug** versus this
being a well-known tool limitation on this kernel class. Distinguishing
those two would need either CUTLASS-level source review of this exact
mbarrier usage, or corroborating evidence (e.g. does the same hazard appear
in a plain vLLM server run of the same shape, outside this project's
runtime, under the same tool?).

**Run terminated after ~31 minutes** (killed manually, GPU verified freed
via `nvidia-smi`/`ps` afterward -- one orphaned `TreeLauncher` helper and
one lingering compute-app entry needed an explicit `kill -9` before GPU
memory actually dropped back to baseline, WSL2's `nvidia-smi` reporting lag
noted elsewhere in this project's own environment docs). The log had
stopped growing for the prior ~10 minutes once the 100-report cap was hit,
and the single warmup forward pass had not yet completed -- the instrumented
run is simply very slow on this specific kernel, consistent with
racecheck's known overhead on shared-memory-heavy, warp-specialized kernels.
Did not run it to completion; no attempt made to reach the real (second)
prefill call under racecheck given the time already spent.

**Tried the bypass experiment immediately (not left for later) -- decisive,
negative result**: set `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel`
(a real, existing vLLM env var -- `is_supported_and_can_implement_kernel()`
in `vllm/model_executor/kernels/linear/__init__.py` checks
`kernel.__name__ in envs.VLLM_DISABLED_KERNELS`, comma-separated) and re-ran
the same single-prefill test, no racecheck this time (just checking output).
Confirmed via log line ("Selected ChannelWiseTorchFP8ScaledMMLinearKernel
for CompressedTensorsW8A8Fp8") that the CUTLASS pingpong kernel was
genuinely bypassed in favor of a plain PyTorch/cuBLAS FP8 kernel with no
CUTLASS warp-specialization at all. Result: **the output changed (from
`'东Ё¨¨¨...'` to `' of of-of of of of...'`) but is still wrong** -- not
"Paris" either way.

This is a real, decisive negative result, not an inconclusive one: the
output *changing* proves this kernel selection genuinely matters to the
computation (ruling out "it was a no-op switch"); the output *still being
wrong* proves **this specific CUTLASS race is not, by itself, sufficient to
explain the wrong final answer** -- avoiding it entirely does not fix
things. So while the racecheck finding above is real and worth reporting
upstream/investigating independently, it is not the root cause this
investigation has been chasing since the "conv_state" lead was first
raised. The actual bug producing wrong output is still elsewhere (or is a
second, still-unidentified instance of the same class of problem, e.g. a
similar race in a *different* kernel that this one experiment didn't touch
-- GEMMs for the main NVFP4-quantized layers are a different code path
entirely and were not covered by this bypass).

**Where this leaves things, honestly, after four full passes on this bug**:
1. A real, reproducible, isolated Triton `causal_conv1d_fn` cold-start bug
   exists (pass 2), but instrumenting the real model showed conv1d output
   can be fully correct throughout a run that still produces wrong text
   (pass 3) -- so this is very likely a real bug, but not THE bug.
2. A real, specific, consistent CUTLASS SM120 pingpong-GEMM race hazard
   exists per racecheck (pass 4, this section), reproducible and precisely
   located (one kernel, one thread pair, tight shared-memory range) -- but
   bypassing that exact kernel changes output without fixing it (this
   section) -- so, by the same logic, likely real but also not THE bug.
3. Both real findings share a pattern worth naming explicitly: this
   environment appears to have **multiple, independent, low-level
   correctness issues** surfacing under this project's unusual
   direct-forward-pass usage pattern (bypassing vLLM's normal
   Scheduler/GPUModelRunner orchestration) -- not one single root cause
   waiting to be found. Chasing each one to ground individually, as directed
   this round, has been valuable (two real, specific, bounded findings) but
   has not yet produced a correct "Paris" output through any configuration
   tried so far.
4. Given four passes have each surfaced a genuine, verifiable finding
   without closing the loop, further undirected bisection has a
   meaningfully uncertain payoff. The decision of whether to keep
   investing in root-causing at this level of depth, versus adopting a
   more conservative strategy (e.g. the coordinator's own suggestion: get a
   *correct* baseline first, even at a performance cost, by not bypassing
   vLLM's own scheduler/executor at all for the parts that seem fragile --
   or by using a much larger `num_stages`/simpler epilogue/a different
   attention-adjacent code path for those specific layers) is exactly the
   kind of call worth surfacing rather than deciding unilaterally.

## Step 2 result (2026-07-16): no-HTTP correct baseline established -- 20/20

Per the strategy reset above, built `runtime/vllm_inprocess_baseline.py`
instead of continuing to hand-derive `GPUModelRunner`'s internals. It
drives vLLM's own `LLM` class directly -- real `Scheduler`, real
`KVCacheManager`-allocated cache, the real `SM120GQAMetadataBuilder` /
`GDNAttentionMetadataBuilder`, real warmup/profiling sequencing, our own
`SM120GQABackend` with decode v2 enabled via
`SM120_GQA_USE_V2_DECODE_KERNEL=1` -- with **no HTTP layer at all** (`LLM`
never has one; only `vllm serve` does). Under the hood it uses
`SyncMPClient` + a background `EngineCore` process reached via local ZMQ,
not a network call -- confirmed this needed the same spawn-safe
`if __name__ == "__main__":` guard `launch_test_server.py` already
documented (hit the identical `RuntimeError` on the first attempt without
it, fixed the same way).

**Result: ran 20 consecutive fresh-process repeats of `"The capital of
France is"` at `max_tokens=8`, temperature 0. All 20 produced the
identical, correct completion**: `" Paris.\n\n<think>\nHere's a"`
(20/20 pass, byte-for-byte identical text every run).

This is a real, decisive, positive result -- not a partial one:
- It directly satisfies step 2's stated bar ("不是性能,是要在同一进程内,
  连续20次都能稳定正确地跑出...Paris").
- It confirms, independently of everything hand-built in
  `direct_model_runner.py`, that **the underlying model, checkpoint,
  quantization, GDN implementation, and this project's own SM120GQABackend
  (decode v2 included) are all correct** when driven through vLLM's real
  scheduling/metadata/cache machinery. Combined with the two independent
  defects already found and ruled out as root causes (Triton
  `causal_conv1d_fn` cold start, CUTLASS SM120 pingpong-GEMM race), this
  narrows the actual bug's location further: it is specifically in
  `direct_model_runner.py`'s own hand-rolled orchestration (metadata
  construction, padding, state initialization, or something else in that
  file), not in anything downstream of it.
- It gives exactly the "stage A" oracle Step 3's ownership-transfer ladder
  needs (vLLM metadata + vLLM cache, verified correct) to compare against
  once that phase is dispatched.

**Not yet done, deliberately out of scope for this round** (per the explicit
instruction to stop after steps 1+2): the three-stage ownership transfer
(vLLM metadata+our cache -> our metadata+our cache -> trimmed direct
runner), per-layer/per-tensor comparison at each stage, and anything about
real batching (`decode_batch()` remains a Python loop over single-request
`decode()` calls in `direct_model_runner.py` -- a known, documented
limitation, not addressed this round), CUDA Graph, or Hy3.

## Stage B result (2026-07-16): our own cache is exonerated -- 20/20

Per the three-stage ownership-transfer ladder, built
`runtime/vllm_stage_b_baseline.py`: real vLLM `Scheduler`/`GPUModelRunner`/
metadata builders/warmup, UNCHANGED, with exactly one substitution --
`GPUModelRunner.initialize_kv_cache_tensors` is monkey-patched to call
`runtime.direct_model_runner.allocate_fixed_slot_kv_caches` (extracted as a
shared helper from `DirectModelRunner._allocate_and_bind_kv_caches`, used
identically by both -- confirmed behavior-preserving by re-running
`benchmarks/single_prefill_regression.py` post-refactor and getting the
*exact same* logits hash as before, `720406302a42f76f`) instead of vLLM's
own `_allocate_kv_cache_tensors`/`_reshape_kv_cache_tensors`. Everything
else -- `initialize_attn_backend`, `initialize_metadata_builders`, the real
`Scheduler`, the real warmup -- is untouched.

**Result: 20/20 identical PASS**, same completion every run
(`" Paris.\n\n<think>\nThe user has"`). No crash, no shape assertion, no
divergence from Stage A's correct behavior.

**This exonerates the cache layer**: our own 4-fixed-slot KV
(attention)/state (GDN) tensor allocation, shape derivation
(`get_kv_cache_shape()`/`get_state_shape()`), and `bind_kv_cache()` wiring
are all correct when driven by real vLLM scheduling. Per the ownership-
transfer ladder's decision rule, the bug is **not** in cache
shape/binding/slot-mapping/state-initialization -- it narrows specifically
to Stage C (this project's own hand-built attention/GDN metadata
construction in `DirectModelRunner._attention_metadata`/`_gdn_metadata`),
which is the next stage.

**One known gap, not exercised by this narrow test, worth flagging**: the
real `Scheduler`'s `kv_cache_config` still reflects vLLM's own large,
profiled block-pool size (~200K tokens here) even though the substituted
tensor only has capacity for `4 slots x 128 blocks x 16 tokens = 8192`
tokens total. This round's single short request (~13 tokens) never
approaches that limit, so the mismatch never manifested -- but this is a
real latent gap (not a "pass," an "untested corner") that would need
addressing (e.g. forcing the scheduler's own block-count belief down to
match, via a smaller profiled `gpu_memory_utilization`/explicit
`kv_cache_memory`) before Stage B could be trusted for anything beyond this
narrow smoke test.

## Stage C result (2026-07-16): the bug is conclusively localized -- 0/20,
## deterministically, and it's specifically the hand-built metadata

Per the ladder, built `runtime/vllm_stage_c_baseline.py`: keeps Stage B's
cache substitution (already verified clean), and adds exactly one more
substitution -- `SM120GQAMetadataBuilder.build()` and
`GDNAttentionMetadataBuilder.build()` (the two REAL vLLM metadata builder
classes) are monkey-patched to call
`runtime.direct_model_runner.build_attention_metadata()` /
`build_gdn_metadata()` (extracted as shared functions from
`DirectModelRunner._attention_metadata`/`_gdn_metadata` -- confirmed
behavior-preserving via the regression script's unchanged logits hash
after this second refactor too) instead of doing the real, production
field derivation. The few facts our hand-built functions need
(`prior_kv_len`, `num_new_tokens`, `is_decode`) are derived from vLLM's own
real, scheduler-computed `CommonAttentionMetadata` (`num_actual_tokens`,
`seq_lens[0]`) rather than re-implementing independent bookkeeping -- this
isolates the metadata *construction logic* as the one variable under test,
not "does our own request/slot bookkeeping also happen to be right."
Everything else (real `Scheduler`, real warmup, Stage B's real-scheduling-
driven cache) stays untouched.

**Result: ran 20 consecutive fresh-process repeats. 0/20 passed -- and,
notably, all 20 failures were byte-for-byte identical**
(`'束\n\n�aser้องagogue衙ires'` every single run,
no crash, no exception). Same deterministic-failure signature as the
original `direct_model_runner.py` bug (Step 1's 20/20 identical failure),
though the exact wrong text differs between the two -- expected, since
Stage C's harness differs slightly in mechanics (single real `LLM.generate`
call under real scheduling vs. `direct_model_runner.py`'s own manual loop),
but both are 100% deterministic wrong answers, not races.

**This is the cleanest localization this entire investigation has
produced**: A (real metadata + real cache) passes 20/20. B (real metadata +
our cache) passes 20/20. C (our metadata + our cache) fails 0/20,
deterministically. Per the ladder's own decision rule, this conclusively
proves **the bug is specifically in this project's hand-built
attention/GDN metadata construction logic**
(`build_attention_metadata`/`build_gdn_metadata` in
`runtime/direct_model_runner.py`) -- not the cache shape/binding/slot-
mapping/state-init (B already exonerated that), not the model, not the
quantization, not the kernels, not the CUTLASS/Triton anomalies found and
ruled out in earlier passes (both real, both independent, neither the root
cause -- now doubly confirmed, since this whole ladder never touches
either of those specific kernels' code paths differently between B and C).

**Not yet done, natural next step (not attempted this round given time
already spent)**: pinpoint exactly *which* field(s) in
`build_attention_metadata`/`build_gdn_metadata` are wrong, by comparing
them value-by-value against what the REAL builders would have produced for
the *identical* real `CommonAttentionMetadata` input at each step (both
patches already have access to the real object at the exact substitution
point -- capturing both the real builder's real output AND our hand-built
output for the same input, then diffing field-by-field, is now a small,
well-scoped follow-on given the harness already exists). Stage D (the
trimmed-down full `direct_model_runner.py`) was not attempted this round
either, since C already failing makes D's outcome predictable (D should
also fail, for the same reason) without needing to re-verify.

## Step 4 result (2026-07-16): root cause found via field diff, fixed, closed loop verified 20/20 on Stage C and Stage D

Followed through on the previous section's "natural next step": wrote a
throwaway diagnostic (`/tmp/qwen_check/stage_c_diff.py`, not committed --
scratch, outside the repo) that wraps the real `SM120GQAMetadataBuilder
.build()`/`GDNAttentionMetadataBuilder.build()` methods (saving originals
before monkey-patching), calls both the real builder and our hand-built
`build_attention_metadata()`/`build_gdn_metadata()` on the *same*
`common_attn_metadata` object (side by side, not sequentially, so no state
changes between the two calls could produce a spurious diff), and diffs
every dataclass field.

**Result**: every field matched except two, both in `GDNAttentionMetadata`:

```
[DIFF] non_spec_state_indices_tensor: real=(torch.Size([1]), [1])  ours=(torch.Size([1]), [0])
[DIFF] prefill_state_indices: real=(torch.Size([1]), [1])  ours=(torch.Size([1]), [0])
```

All other fields -- `num_prefills`, `num_prefill_tokens`, `num_decodes`,
`num_decode_tokens`, `num_spec_decodes`, `num_spec_decode_tokens`,
`num_actual_tokens`, `has_initial_state`, `spec_query_start_loc`,
`non_spec_query_start_loc`, `spec_state_indices_tensor`,
`spec_sequence_masks`, `spec_token_indx`, `non_spec_token_indx`,
`num_accepted_tokens`, `chunk_indices`, `chunk_offsets`,
`prefill_query_start_loc`, `prefill_has_initial_state` -- were identical.
(The diagnostic script crashed right after printing this, on a
`nums_dict` dict-equality comparison triggering an ambiguous-tensor-
boolean `RuntimeError` -- a bug in the throwaway script itself, not fixed,
since the two fields above were already sufficient to act on. The
SM120GQAMetadata side's own diff was therefore never directly printed --
see below for how this was independently confirmed anyway.)

The same diagnostic captured a real `SchedulerOutput` dump for this
request: `block_ids=([1], [2], [3], [4])`. **vLLM's real scheduler never
assigns physical block/state index 0 to real request data** -- it starts
at 1. Our hand-built metadata, by contrast, computed physical
slot/state index directly as `slot` (the logical slot number, 0 for this
project's single-slot scope this round) with no offset -- landing
squarely on the index vLLM's real convention treats as reserved/unsafe.
(The exact underlying mechanism -- whether this is literally a
`NULL_BLOCK_ID = 0` convention in `KVCacheManager`, something block-pool-
allocator-internal, or something else -- was not pinned down. Treated as
an empirically solid fact regardless: real request data is never at index
0, full stop.)

**Causal confirmation, not just correlation**: rather than fix-and-hope,
created a throwaway copy `runtime/vllm_stage_c_slot1_test.py` (Stage C's
baseline with only `SLOT` changed from `0` to `1`, everything else
byte-identical) and ran it. **Single run: PASS**, correct
`" Paris.\n\n<think>\nThe user has"`. This directly demonstrates causation:
changing only the slot-index offset flips the output from wrong to
correct, with no other variable touched.

**General fix applied** to `runtime/direct_model_runner.py` (not the
throwaway test file, which was deleted once this landed): added

```python
RESERVED_PHYSICAL_SLOTS = 1

def _physical_slot(logical_slot: int) -> int:
    return logical_slot + RESERVED_PHYSICAL_SLOTS
```

and applied `_physical_slot()` at all four places a physical address is
computed:
- `allocate_fixed_slot_kv_caches`: `num_blocks = (num_slots +
  RESERVED_PHYSICAL_SLOTS) * blocks_per_slot`, and the GDN conv/ssm state
  tensors are allocated with `num_slots + RESERVED_PHYSICAL_SLOTS` rows
  (one extra slot's worth of capacity, permanently unaddressed at row 0).
- `build_attention_metadata`: `first_block = _physical_slot(slot) *
  blocks_per_slot`.
- `build_gdn_metadata`: `state_indices = torch.tensor([_physical_slot(slot)], ...)`.
- `DirectModelRunner._slot_mapping`: `first_block = _physical_slot(slot) *
  self.blocks_per_slot`.

Because `runtime/vllm_stage_c_baseline.py` already calls these shared
functions with its own `SLOT = 0` (logical), it required **no code
changes** to pick up the fix -- its logical slot 0 now automatically
resolves to physical index 1.

**Independent confirmation that the attention side (not just GDN) had the
same bug**, closing the gap left by the diagnostic script's crash: a
lightweight CPU-only check (no GPU/model involved) calling
`build_attention_metadata()`/`build_gdn_metadata()` directly:

```
RESERVED_PHYSICAL_SLOTS = 1
kv_page_indices (slot=0): [128]  -- starts at 1*128, not 0
prefill_state_indices (slot=0): [1]  -- not [0]
non_spec_state_indices_tensor (slot=2): [3]  -- not [2]
```

confirming both `SM120GQAMetadata.kv_page_indices` and
`GDNAttentionMetadata`'s state-index fields are now consistently offset
for every logical slot.

**Verification, both 20x, both 20/20 PASS** (note: the first 20x attempt at
confirming the throwaway `SLOT=1` hypothesis got contaminated mid-run --
runs 1-4 genuinely passed, but deleting the throwaway test file while its
background 20x loop was still spawning fresh subprocesses caused runs
6-20 to fail with `ModuleNotFoundError` rather than a real bug; run 5's
one `EngineCore init failed` crash is suspected transient port/resource
contention from rapid successive subprocess launches, not reproduced
again -- noted honestly rather than swept under the rug, but not the
signal being tested for):

- `runtime/vllm_stage_c_baseline.py` (real `CommonAttentionMetadata`-driven
  Stage C, now with the general fix, zero code changes needed): **20/20
  PASS**, identical `' Paris.\n\n<think>\nThe user has'` every run.
- `benchmarks/single_prefill_regression.py`, exercising the full
  `DirectModelRunner` end to end (**Stage D**): **20/20 PASS**, correct
  first token `' Paris'`, **identical SHA-256 logits hash
  `7eda2739bbecbc52` across all 20 runs** -- both determinism and
  correctness confirmed simultaneously.

**This closes the ownership-transfer ladder.** A (real everything), B
(real metadata + our cache), C (our metadata + real-scheduler-driven
facts + our cache), and D (the full, real, hand-built
`DirectModelRunner`) all now produce correct, deterministic output. The
"direct GPU state ownership, no HTTP bridge" line has a genuine minimal
correct closed loop as of this round.

## Batch decode support (2026-07-16): real batched metadata implemented; batch=1 verified 19/20; batch>=2's "numerical mismatch" traced to a pre-existing, real-vLLM-too floating-point batch-composition effect, not a decode_batch bug

Following the closed loop above, `decode_batch()` had been a Python loop
over single-request `decode()` calls -- not a real GPU batch. This round
replaced that with genuine batched construction: new module-level
functions `build_attention_metadata_batch`/`build_gdn_metadata_batch` in
`runtime/direct_model_runner.py` (kept SEPARATE from the single-request
`build_attention_metadata`/`build_gdn_metadata` so this new path cannot
regress the just-closed Stage C/D loop), generalizing the real backends'
own pure-decode-batch CSR construction (`SM120GQAMetadataBuilder.build()`'s
`qo_indptr`/`kv_page_indptr`/`kv_page_indices`/`kv_last_page_len`
derivation and `GDNAttentionMetadataBuilder.build()`'s pure-non-spec-decode
branch, both read directly from vLLM source this round to get the exact
convention right) from one request to N requests spanning N of this
project's own fixed physical slots (via the existing `_physical_slot()`
offset). `DirectModelRunner._forward_batch()`/`.decode_batch()` issue
exactly ONE `model.forward()` call covering every listed slot.

**Test harness**: new `benchmarks/batch_decode_regression.py`. Since
comparing a slot's own before/after decode() call against a later
decode_batch() call on the SAME slot would be confounded by cache-state
mutation between the two calls, it instead prefills TWO independent,
identically-initialized groups of physical slots per test (`2*batch`
logical slots total) -- a "reference" group decoded one-request-at-a-time
via the already-verified single-request `_forward()`, and a "batch" group
of the same size decoded via one `_forward_batch()` call -- then compares
logits bytewise (SHA-256 hash, same rigor as the Stage A-D work) between
corresponding pairs.

**Step 1 (batch=1): 19/20 PASS.** The 1 failure was a `504 Gateway
Timeout` from the `hf-mirror.com` HF Hub mirror during
`ModelConfig.__post_init__`'s pooling-config file-listing lookup --
happened before any model/kernel code ran, unrelated to decode_batch
logic. Worked around for later runs via `HF_HUB_OFFLINE=1` (the 22GB
checkpoint is already fully cached locally, so this lookup is avoidable
network dependency, not a real requirement). At batch=1 the reference and
batch paths use identical M=1 GEMM shapes throughout, so bytewise-identical
output is exactly what a batching change *should* produce -- confirmed.

**Step 2 (batch=2): initial result was 0/2 bytewise match, BUT traced to a
pre-existing effect, not a decode_batch bug.** Both rows of the batch
(same prompt duplicated, to rule out a per-row addressing mixup) produced
an IDENTICAL wrong-vs-reference hash to each other (`5de15ae1adf5bb68` vs
the correct `477f73db20c29485`) -- internally self-consistent, not
request-order corruption, but genuinely different from the single-request
value.

**Root-cause check, using ONLY real vLLM machinery (no code of ours at
all)**: a throwaway diagnostic (`/tmp/qwen_check/diag_real_batch2.py`, not
committed) drove vLLM's real `LLM` class -- real Scheduler, real metadata
builders, real cache -- and compared "The capital of France is" generated
**alone** vs **concurrently with a second real request** ("The largest
planet..."), greedy (`temperature=0.0`), 8 tokens:

```
ALONE      : " Paris.\n\n<think>\nHere's a"
TOGETHER[A]: " Paris.\n\n<think>\n\n</think>\n\nThat"
MATCH: False
```

Reproduced **identically** on a second independent run (fully
deterministic, not a race). **This proves the batch-composition-dependent
numerical difference is a pre-existing property of the real production
stack, not a bug introduced by this round's hand-built
`build_attention_metadata_batch`/`build_gdn_metadata_batch`.** The
mechanism is almost certainly floating-point non-associativity in batched
GEMM (MLP/dense layers processing `[batch, hidden]` matrices pick
different tile/kernel configs depending on the M dimension, changing
summation order and hence exact rounding -- a widely-documented industry
issue for "batch-invariant" LLM inference, not specific to this project's
kernels). Consistent with batch=1's clean pass: at batch=1 there is no
alternate M-dimension path to diverge from.

**Practical implication for the remaining validation ladder (steps
2-8)**: requiring bytewise-identical logits between the single-request
and N-request-batched paths is not an achievable bar -- not even vLLM's
own real, fully-official production path meets it. The `decode_batch`
implementation itself is not shown to be wrong by this test; the test's
acceptance criterion needs to change from "bytewise identical" to
something that tolerates expected batch-composition floating-point noise
while still catching genuine addressing/correctness bugs (e.g., argmax
agreement across many decode steps of a real generation, cosine
similarity above a threshold, or the project's own established
signal-probe/marker-token methodology for cross-request leakage --
distinguishing "slightly different rounding" from "reading the wrong
slot's data entirely"). This is a methodology decision, flagged to the
coordinator rather than decided unilaterally, before continuing to
batch=4/variable-length/slot-reuse/continuous-generation/no-crosstalk.

**Coordinator decision (2026-07-16): new acceptance criteria adopted**,
replacing bytewise-identical-vs-run-alone:
1. Argmax/token-sequence plausibility (fluent, non-garbage output).
2. Signal-probe/marker-token crosstalk detection as the primary
   bug-catching tool -- distinguishes genuine cross-slot data leakage
   from ordinary floating-point noise.
3. Same-batch internal self-consistency (two identical prompts in the
   SAME batch call must produce bytewise-identical results to each
   other) as the correctness reference, replacing "run alone".

## Batch decode ladder, steps 2-6 (2026-07-16): all pass under the new criteria

New harness `benchmarks/batch_decode_signal_probe.py`, replacing
`batch_decode_regression.py`'s now-obsolete run-alone-reference approach.
Each active slot's prompt is `"{filler}The value of X is {number}. The
value of X is"` -- a strong in-context copy cue the model reliably
completes with its own number regardless of instruction-following
ability. For batch >= 3, the last slot duplicates the first slot's number
(and, for the varlen variant, its filler length too -- see below) to give
a same-batch self-consistency pair, while every other slot gets a
distinct number for crosstalk detection. Every step's `_forward_batch()`
call is checked two ways: (a) do slot 0 and the duplicate slot's logits
hash-match exactly (self-consistency), and (b) after `steps` rounds, does
each slot's decoded text contain its OWN number and NONE of the other
slots' numbers (signal-probe crosstalk detection).

**Step 2, batch=3 (re-verified under the new criteria, not re-litigating
bytewise): 1 sanity run + 3/3 repeat, all PASS.** Self-consistency held
every step; each slot recovered exactly its own number with zero leakage.

**Step 3, batch=4: 1 sanity run + 3/3 repeat, all PASS.** Same result
pattern as batch=3.

**Step 4, variable-length requests (batch=4, each slot given a different
filler-sentence-repeat count so `prior_kv_len`/page count differs across
the batch): first attempt FAILED self-consistency 3/3 -- but this was a
TEST HARNESS bug, not a decode_batch bug.** The duplicate-number pair
(slots 0 and 3) had been given *different* filler lengths (`i` per slot
index), so their prompts weren't actually identical -- comparing their
logits was comparing two genuinely different inputs, not testing
self-consistency at all. Diagnostic: `signal_ok` was `True` in all 3
"failing" runs (each slot still recovered its own number correctly, e.g.
slot 2's `' 71053. The weather'` vs slot 0/3's `' 84317. The value'`) --
confirming no real crosstalk/addressing bug, just a broken test premise.
Fixed by adding `_assign_filler_repeats()`: the self-consistency pair
(slots 0 and batch-1) now shares BOTH number and filler length, while
interior slots still get distinct lengths for the varlen coverage this
step is meant to exercise. **After the fix: 3/3 PASS**, self-consistency
held and no crosstalk, confirming `build_attention_metadata_batch`'s
per-request CSR construction (heterogeneous page counts per request in
one batched call) is correct.

**Step 5, slot release + reuse (batch=3, `--reuse`): 3/3 PASS.** After 8
decode rounds, slot 0 is released (`reset_slot`) and immediately
re-prefilled with a brand-new number (91827, disjoint from the batch's
numbers) while slots 1/2 remain untouched; 8 further decode-only steps on
slot 0 alone recover exactly the new number with zero residue from either
its own prior occupant or the still-active other slots -- confirming
`reset_slot`'s "don't zero tensors, rely on kv_len=0 +
`has_initial_state`/`slot_gdn_initialized`=False" convention still holds
correctly under the batched decode path, not just the original
single-request path it was designed for.

**Step 6, continuous 256-token generation (batch=4, `steps=256`): 2/2
PASS.** 256 sequential real `_forward_batch()` calls per run (512 total
across both repeats) with no crash, and the signal-probe/self-consistency
checks (evaluated against the FULL 257-token generated text, not just the
first few tokens) still passed cleanly -- no drift or leakage emerging
only after sustained multi-step generation.

**Signal-probe no-crosstalk (cross-cutting, not a separate scenario)**:
every single run above -- batch=2/3/4, varlen, reuse, and the 256-token
continuous run, 1 initial + repeats -- reported `signal_ok: true` with
zero leaked-other-slot's-number instances. This is the primary evidence
that `decode_batch`'s physical-slot addressing (the exact class of bug
the 2026-07-16 slot-0-reservation root-cause work fixed for the
single-request path) generalizes correctly to the real N-request batched
path.

**Repeat count note**: used 3x (2x for the more expensive 256-step run)
rather than 20x for these deterministic single-process checks -- reasoned
explicitly, not just cost-cut: unlike the earlier cross-process bytewise
hash comparisons (sensitive to legitimate floating-point noise across
independent process launches), a genuine addressing/crosstalk bug here is
a deterministic logic error that would manifest on the very first run,
not a rare hardware race; repeats mainly guard against the kind of rare
CUTLASS-kernel race already investigated and ruled out as this bug's root
cause earlier in this project. 3x was judged sufficient for that residual
risk given every run already passed cleanly.

**Not attempted this round (correctly out of scope, per the ladder's own
"MTP排最后" ordering)**: MTP/speculative-decode batched verification --
needs multi-token-per-request query support in the batch metadata
builders, a substantially larger feature than single-token decode
batching, left for its own dedicated round once requested.
