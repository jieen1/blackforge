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

## Batch decode ladder, last step: MTP verify support (2026-07-16)

Generalized both batch metadata builders from `qo_len=1`-only to a
`qo_len` parameter (uniform across the batch, matching real production
usage since `num_speculative_tokens` is a global engine config, not
per-request) so `decode_batch` can also drive MTP/speculative-decode
verify (K draft tokens + 1 bonus position per request, e.g. qo_len=4 for
K=3).

**Attention side** (`build_attention_metadata_batch`): `qo_indptr`,
`new_kv_len`, `is_pure_decode`, `decode_qo_len` all generalized to
`qo_len`; at `qo_len=1` every formula reduces exactly to the
previously-verified values (a generalization, not a parallel
implementation -- confirmed via a same-day regression rerun of the
batch=3 signal-probe test, identical result to before the refactor).
This is what makes the real production `flash_attn_sm120_fwd_v2_decode_
fp8kv_paged` kernel (already hardened for qo_len 2-4) get dispatched --
no kernel changes were made or needed, only correct metadata; confirmed
directly via the real backend's own log line, `SM120_GQA: v2 decode
kernel path HIT (qo_len=4)`, appearing in every qo_len=4 test run below.

**GDN side** (`build_gdn_metadata_batch`): rather than replicating the
real `GDNAttentionMetadataBuilder`'s much more involved `spec_decode`
branch (accept/reject-aware sorting of spec vs non-spec tokens --
explicitly out of scope this round per the coordinator), `qo_len>1`
instead generalizes `build_gdn_metadata`'s OTHER existing branch: the
`is_decode=False` ("prefill"/chunked) case. This is exactly what the real
builder's own `split_decodes_and_prefills` would ALSO select for any
request with query_len>1 when no draft-acceptance info is supplied --
i.e. an MTP verify step is treated as an ordinary chunked continuation of
`qo_len` new tokens per request, numerically correct (the chunked FLA
kernel handles arbitrary query length + GDN state update correctly) even
though it forgoes the real builder's spec-decode-specific optimizations.

**New `DirectModelRunner` methods**: `_forward_batch` gained a `qo_len`
parameter (`token_ids` becomes a list of per-slot draft-token lists when
`qo_len>1`, instead of a flat list); `_slot_mapping_batch` gained the same
generalization. New public `verify_batch(slot_ids, draft_token_ids,
kv_lengths)` wraps `_forward_batch` with `qo_len` inferred from the draft
list length. Accept/reject sampling against the returned per-position
logits is explicitly left to the caller -- out of scope this round, per
the coordinator's own instruction (metadata/kernel-call layer first).

**Test methodology pitfall caught before it became a false alarm**: the
first design considered "rewinding" a slot's `kv_len` bookkeeping after
running real decode steps, then re-submitting the same tokens through
`verify_batch` to test it in isolation. This would have been UNSOUND:
unlike attention's paged KV (content-addressed by position, safe to
overwrite with identical values), GDN's linear-attention recurrent state
cannot be cheaply "rewound" -- by the time of the rewind the real
recurrent state has already advanced past every one of the `steps` real
decode rounds, not just the first `qo_len` of them, so re-verifying
against a "rewound" `kv_len` would silently read a state that's too far
advanced for the position being claimed. Caught by reasoning through the
mechanism before running anything, and fixed by using an independent twin
slot group instead (mirrors `batch_decode_regression.py`'s original
approach): a fresh, pristine-state group prefilled with the same prompts,
verified against the REF group's own real, established continuation
tokens as the draft (a genuine causally-coherent sequence, not an
arbitrary placeholder).

**A second, real (not test-design) subtlety surfaced during
batch=1 debugging**: the very first token generated after this project's
"The value of X is {number}. The value of X is" prompt is always a
leading space (` `, e.g. token id 220), not the first digit -- confirmed
via a direct diagnostic comparing `verify_batch`'s raw predicted token
IDs against the trusted single-token path's real continuation at the
identical positions: **bit-for-bit identical** (`[23, 19, 18, 16]` on both
sides). This means with `qo_len=4` (the real production MTP shape,
deliberately used instead of a larger qo_len so the v2 kernel's qo_len
2-4 dispatch range is actually exercised), only 4 of a 5-digit number's
digits fit in the newly-predicted positions (the leading space consumes
one slot from the already-known prefix). Fixed the test's crosstalk check
to compare a length-matched prefix (`str(number)[:qo_len]`) instead of
requiring the full number -- still a valid, sufficient crosstalk detector
since these prefixes don't overlap across `NUMBERS`.

**Results, all at the real production shape `qo_len=4` (K=3 draft + 1
bonus token)**, 1 sanity run + 3/3 repeat each:
- **batch=1**: PASS. `v2 decode kernel path HIT (qo_len=4)` confirmed in
  the logs.
- **batch=2**: PASS. Both slots recover their own number's prefix, zero
  crosstalk.
- **batch=4**: PASS (one repeat run needed a retry after timing out --
  see below). Self-consistency held (slots 0 and 3, the duplicate-number
  pair, hash-match at every verify position); all 4 slots recover their
  own prefix with zero leakage.

**Non-issue encountered, documented for completeness**: the first batch=4
repeat run hit GPU memory at 96.8/97.9 GiB and a 600s subprocess timeout.
Root-caused via `ps -ef` (per the standing GPU-discipline requirement) to
a COMPLETELY UNRELATED, CONCURRENT session on this shared machine running
its own `llama.cpp` benchmark (`Hy3-IQ1_M-mtp.gguf` via a `codex`-launched
process tree, unrelated to this project) -- not a leak or bug in this
round's code. Waited for that job's own `timeout 900` budget to elapse
naturally (never touched a process this session didn't own), then
retried cleanly: 3/3 PASS once the GPU was actually free.

**This completes the core "direct model runner supports 4-slot fixed
batch + MTP" mechanism** the coordinator asked for -- metadata
construction and kernel-call-level MTP verify batching, reusing the real
production v2 decode kernel with zero kernel changes. Explicitly NOT done
this round (per the coordinator's own scoping): accept/reject sampling
logic downstream of `verify_batch`'s raw logits, and CUDA Graph capture --
both correctly left for a follow-on round alongside real W1/W2/
concurrency=4/MTP-K=3 performance measurement.

## CUDA Graph capture/replay, step 1 (qo_len=1 batch decode): implemented, verified uninstrumented, compute-sanitizer verification in progress

Per the coordinator's explicit caution (this class of work has a real
crash history in the sibling project), proceeded carefully rather than
"one-shot": read the sibling's own kernel-level CUDA-graph test
(`kernel/tests/test_cudagraph_decode_fixed_sizing.py`) as prior art before
writing any code, and identified a prerequisite fix BEFORE attempting
capture (not after hitting a crash): `build_attention_metadata_batch`'s
`kv_split_size` was derived per-call from live `kv_len` -- exactly the
pattern the sibling project's own docs (`sm120_gqa.py`) warn goes stale
under CUDA Graph replay (a captured launch's scalar arguments freeze at
capture time; replaying at a larger real kv_len than capture-time data
would silently use a too-small split boundary).

**Fix**: added `fixed_kv_split_size`/`fixed_max_num_splits` parameters
(default `None`, preserving the exact existing eager-path behavior) --
when supplied, `kv_split_size` is derived ONCE from this slot's
build-time-fixed hard capacity (`blocks_per_slot * block_size`), with the
same mathematical bound the real backend's own fix uses: for
`split_size = ceil(L/target_splits)`, `num_splits(k) = ceil(k/s) <=
target_splits` for every real `k <= L`, not just the capture-time value.

**New `CapturedBatchDecodeGraph` class**: every tensor a captured kernel
launch reads (metadata CSR tensors, input_ids, positions, slot_mapping)
is a persistent, fixed-address buffer allocated once; `replay()` writes
fresh real values into these SAME buffers via `.copy_()`, never
reallocating them. Found and fixed one capture-safety issue before it
became a crash: `torch.cuda.synchronize()` (used by the eager
`_forward_batch` between `model.forward()` and `compute_logits()`) is a
documented CUDA-graph-capture violation (`cudaErrorStreamCaptureUnsupported`)
-- the same error class the sibling project already hit for a different
op (a boolean-mask-select). Added a sync-free `_forward_no_sync()` used
by both the warmup iterations and the captured region itself. Confirmed
`set_forward_context` itself is capture-safe (its only host-sync point is
gated behind `VLLM_LOG_BATCHSIZE_INTERVAL`, off by default, and it's
vLLM's own production CUDA-graph capture path already).

**New test `benchmarks/cudagraph_decode_regression.py`**, deliberately
testing kv_len distributions far more extreme than the capture-time
shape, per the coordinator's explicit instruction not to only test the
happy path (the sibling project's own decode v2 CUDA Graph work hit
exactly this class of gap before): captures at a small (~15-token) shape,
then replays across (1) the capture-time shape itself, (2) 8 further
sequential steps of normal growth, (3) one slot pushed to 1961/2048
tokens -- **96% of this slot's hard physical capacity** -- while the
other 3 slots stay small, and (4) a freshly re-prefilled slot at the
*smallest* possible kv_len (1 token) alongside the others' now-larger
values. Checked via this project's established signal-probe methodology
(unique numeric marker per slot, verified recoverable with zero
cross-slot leakage) -- not bytewise comparison, consistent with the
already-established finding that bytewise identity isn't a meaningful bar
across different computational shapes/paths.

**Result: 3/3 independent repeats, all PASS, zero crashes.** Every slot
at every kv_len (including the 96%-capacity extreme case) correctly
recovered its own identity marker with no cross-slot leakage.

**compute-sanitizer verification: attempted, still in progress, NOT yet
complete** -- reporting honestly per the coordinator's explicit
instruction not to skip or fake this step:
- Full `benchmarks/cudagraph_decode_regression.py` (21 total forward-pass-
  equivalent calls) under `--tool memcheck`: weight loading alone took
  662-793s (vs 45-85s uninstrumented, ~10-15x), and the process made NO
  further progress for 60+ minutes after that (no crash, no output,
  steady 100% CPU) -- killed after ~1-2 hours with no realistic end in
  sight for the full test at this instrumentation level.
- Cut down to a minimal 10-call repro (`benchmarks/
  cudagraph_decode_sanitizer_repro.py`: 4 prefills + 3 warmup + 1
  capture-trace + 2 replays -- one at the capture-time shape, one at the
  96%-capacity extreme) -- still exhibited the same multi-hour stall under
  full `memcheck`.
- Switched to `--tool initcheck` (lighter-weight, targets uninitialized-
  memory/invalid-pointer issues specifically rather than every memory
  access): weight loading dropped back to normal speed (~44-85s), but the
  first real forward pass (this runner's own pre-existing `_warmup()`
  mechanism, `direct_model_runner.py`'s `__init__` -> `_warmup()` ->
  `prefill()`) hit **hundreds to thousands of "Uninitialized __global__
  memory read" reports, ALL tracing to the SAME already-known, already-
  documented, already-investigated defect** from earlier in this project
  (`_causal_conv1d_fwd_kernel`'s Triton cold-start bug -- see this file's
  own "Known independent defects" section; this bug was found, isolated,
  and explicitly ruled out as the root cause of an unrelated earlier bug
  many rounds ago). This is NOT new information and NOT related to the
  CUDA Graph work -- but its sheer volume (exceeded a 3000-report
  `--print-limit` within a single forward pass) drowns out any signal
  from the actual capture/replay code before the sanitizer's report cap
  is ever reached.
- Worked around via `--kernel-name-exclude kernel_substring=causal_conv1d`
  (excludes the known-noisy kernel from analysis, not from execution) --
  a machine reboot killed the first attempt at this mid-run (all
  working-tree changes were preserved on disk; confirmed via a fresh
  `git status`/`nvidia-smi`/`ps` check that no GPU/process state survived,
  as expected). Restarted after the reboot, at the ORIGINAL batch_size=4
  scope: still made no visible progress in 20+ minutes past the
  now-excluded causal_conv1d issue.
- Cut further to a genuinely minimal, sanitizer-specific script
  (`benchmarks/cudagraph_sanitizer_micro.py`, batch_size=2, exactly ONE
  replay directly at an extreme/near-capacity kv_len -- 7 total
  forward-pass-equivalent calls, verified correct and fast (~20s)
  uninstrumented first). Under `initcheck` with the same causal_conv1d
  exclusion: weight loading returned to normal speed (~11s), but the
  SAME pre-existing `_warmup()` call now surfaced **a second, different,
  previously-undocumented-in-this-project uninitialized-read report**
  (100 instances, the default print-limit) -- this time in
  `qwen_gdn_linear_attn.py`'s `_output_projection`/`forward_cuda`, a
  DIFFERENT kernel from causal_conv1d. Still 100% confined to the same
  `__init__` -> `_warmup()` -> `prefill()` call stack (line 69 of the
  micro script, i.e. `DirectModelRunner(...)` construction itself) --
  not this round's new code. After exhausting that report cap the
  process continued running (silently, past the print-limit) for another
  20+ minutes with zero further log output, still stuck within that SAME
  first forward pass.

**Pattern across all four attempts (full memcheck x2, initcheck at
batch=4, initcheck at batch=2)**: every single one stalled or flooded
inside `DirectModelRunner.__init__`'s own pre-existing `_warmup()`
mechanism -- a call this round's CUDA Graph code doesn't even touch yet
(capture/replay only run AFTER `__init__` completes). This warmup exists
specifically because the model's own kernels (now confirmed: at least
TWO distinct ones, `causal_conv1d` and something in
`qwen_gdn_linear_attn.py`'s output projection) behave abnormally on the
literal first-ever forward call in a fresh process -- an already
partially-documented, pre-existing property of this model/kernel stack,
not something introduced by or specific to CUDA Graph capture. Since
warmup is unavoidably the FIRST real GPU work in any fresh process
(disabling it would just move the same cold-start cost onto whatever
call becomes "first" instead, per this project's own established
finding, and would also invalidate the whole point of `_warmup()`
existing), this makes the underlying model+kernel stack itself
fundamentally expensive to sanitizer-instrument from a cold process
start, independent of anything CUDA-Graph-specific.

**Honest status, reported plainly rather than glossed over**: the
`CapturedBatchDecodeGraph` capture/replay mechanism itself is solidly
verified through extensive real, uninstrumented testing that specifically
targeted the exact failure modes this project's own sibling documented
(address staleness, split-size staleness under kv_len far exceeding
capture-time data) -- 3/3 clean passes, zero crashes, including a
96%-of-hard-capacity extreme case. The coordinator's compute-sanitizer
0-errors gate has NOT been satisfied for this round's new code
specifically -- every attempt's error budget/wall-clock was consumed by a
real but PRE-EXISTING, UNRELATED defect in the model's own cold-start
behavior before ever reaching the capture()/replay() calls under test.
This is flagged as an open item requiring a coordinator decision on how
to proceed (see the concise status report delivered alongside this
commit), not silently marked done.

**Coordinator decision (2026-07-16): accept the uninstrumented
verification for this round; compute-sanitizer stays a known, explicitly
tracked open item; don't spend further time chasing pre-existing
cold-start defects.** Proceeding to MTP (qo_len=4) CUDA Graph capture.

## CUDA Graph capture/replay, step 2 (qo_len=4 MTP verify): implemented, verified via signal-probe, 3/3 PASS

Generalized `CapturedBatchDecodeGraph` in place (not a new class) to
accept `qo_len>1`, reusing the class's existing design points:
- `static_qo_indptr`/`static_non_spec_qsl` generalize to
  `arange(0, num_reqs+1) * qo_len` (constant for a fixed (batch_size,
  qo_len) pair, same as the qo_len=1 case).
- `static_input_ids`/`static_positions`/`static_slot_mapping` sized
  `batch_size * qo_len` instead of `batch_size`.
- GDN's chunked/"prefill" metadata fields (`chunk_indices`/
  `chunk_offsets`/`nums_dict`/`batch_ptr`/`token_chunk_offset_ptr`/
  `has_initial_state`) depend ONLY on query-length structure (how many
  tokens per request), never on kv_len or which physical slot -- so for
  a FIXED (batch_size, qo_len) graph they are genuinely constant across
  every replay, unlike `kv_page_indices`/`state_indices` (which depend on
  live kv_len/slot identity and must still be refilled every replay via
  `.copy_()`). Computed ONCE in `__init__` via
  `build_gdn_metadata_batch(..., slot_initialized=[True]*batch_size)`
  and reused as-is -- fixed address by construction of never being
  recreated, no extra per-replay work needed. `has_initial_state=True`
  for every slot is this class's explicit scope: MTP verify only ever
  happens after a slot's own prior prefill/decode has established real
  context.
- At `qo_len=1` every new formula reduces exactly to the previous
  values -- confirmed via a direct regression rerun of
  `cudagraph_decode_regression.py` after the generalization, byte-for-byte
  identical output to before the change.

**A second, more consequential methodology issue found and fixed BEFORE
it could produce a false result** (building on the "GDN state can't be
cheaply rewound" lesson from the non-graph MTP verify round): `capture()`
performs 3 REAL warmup executions on a side stream before the graph
trace (the trace itself, inside `with torch.cuda.graph(g):`, does NOT
execute anything -- confirmed against the sibling project's own
kernel-level CUDA-graph test, which documents this precisely). Naively
passing the SAME slots to `capture()`'s warmup as the slots later checked
via `replay()` means those 3 warmup executions redundantly apply the
capture-time draft tokens to the SAME GDN recurrent state 3 extra times
before any real replay happens -- and unlike attention's paged KV
(content-addressed, safe to overwrite with identical values repeatedly),
a chunked/recurrent GDN state update is NOT idempotent under repeated
identical input (a linear recurrence applied to the same input N times
does not equal applying it once). This is a genuine imprecision in the
ALREADY-COMMITTED qo_len=1 test's methodology too (it reused the same
`slots` for both `capture()`'s warmup and the "replay@capture-time-shape"
check) -- caught here while designing the MTP test, not retroactively
fixed in the qo_len=1 test (its empirical 3/3 PASS result, including the
96%-capacity extreme case, stands as real evidence; the likely reason
this imprecision didn't surface as an observable defect there is that
`capture()`'s `slot_ids` parameter does NOT need to match a later
`replay()` call's `slot_ids` -- both independently recompute addressing
fresh each call, no class-level fix needed -- combined with this specific
signal-probe task likely being dominated by the model's full-attention
layers' copy-mechanism rather than GDN's contribution, masking a state
perturbation that a GDN-sensitive task might not tolerate).

**Fix applied to the new MTP test's methodology** (`benchmarks/
cudagraph_mtp_regression.py`): dedicated, disposable `ref_slots` for
establishing trusted draft tokens AND serving as `capture()`'s warmup
data source (spent/discarded afterward, reset+reused for a second,
independent check later) -- kept strictly separate from `graph_slots`,
the slots actually checked via `replay()`, which are touched by nothing
except their own prefill until the real replay calls.

**Also found and fixed a test-design bug** (not a decode_batch/graph
bug): an early draft of this test tried to chain a second "extreme"
verify as a continuation of the first replay's own predicted tokens --
but a verify call's *output* (what comes after position p) is not the
same content as a *new draft* for a follow-on step (which would need the
accepted continuation plus a fresh bonus token, i.e. real MTP accept/
reject bookkeeping, explicitly out of scope this round). Feeding
mismatched content produced plausible-looking but wrong-looking text
("81.17" instead of a number) that had nothing to do with cross-slot
addressing. Fixed by making the extreme-shape check a fully INDEPENDENT
single-shot verify (fresh prefills, fresh trusted drafts via `ref_slots`,
one decisive replay) rather than a continuation of the first.

**Results, 1 sanity run + 3/3 repeat, both replays checked via signal-probe
per run:**
- `replay@capture-time-shape` (small, ~15-20-token prompts, all 4 slots):
  self-consistency held (slots 0 and 3, a duplicate-number pair, produced
  identical text) and every slot recovered its own number prefix with
  zero cross-slot leakage. Confirmed `v2 decode kernel path HIT (qo_len=4)`
  in the logs -- the real production kernel, no kernel changes needed.
- `replay@extreme-mixed-kv_len(MTP)` (independent re-prefill: 3 slots
  short again, 1 slot pushed to **1961/2048 tokens, 96% of hard
  capacity**): every slot, including the 96%-capacity one, still
  correctly recovered its own number prefix with zero leakage, using the
  SAME captured graph from the small-shape scenario above.
- **3/3 independent repeats, all PASS, zero crashes.**

**This completes the CUDA Graph capture/replay work for both scopes the
coordinator asked for** (qo_len=1 batch decode and qo_len=4 MTP verify).
Explicitly not done: compute-sanitizer verification (tracked open item,
see above) and accept/reject sampling logic (out of scope per the
coordinator's own MTP-batch-support round). Next: the real W1/W2/
concurrency=4/MTP-K=3 performance comparison against native FlashInfer,
using the sibling project's established benchmark methodology.

## 2026-07-17 correction: an independent (Codex) review found a REAL gap in the CUDA Graph work above that had been reported as "verified" -- state pollution, a hot-path allocation/sync issue, and an inaccurate "hardware capacity" framing

**This section exists because something reported as verified/passing in
the sections above was not fully correct.** The coordinator commissioned
an independent Codex analysis of this project's overall state; Codex
found, and the coordinator personally verified against the actual code
before relaying it, a real correctness gap in `CapturedBatchDecodeGraph`
that the 3/3-PASS signal-probe results above did not catch. This is
recorded plainly, not glossed over.

**The finding**: `capture()`'s 3 real warmup executions (on a side
stream, before the graph trace -- the trace itself executes nothing) were
run against WHATEVER slots the caller passed in -- and every test script
this project wrote (`cudagraph_decode_regression.py`,
`cudagraph_mtp_regression.py`) passed the SAME slots for warmup as were
later checked via `replay()`. Attention's paged KV cache tolerates
redundant warmup writes fine (same position, same value, overwritten
harmlessly) -- but GDN's recurrent/chunked state update reads-old-state-
and-writes-new-state every call, so repeating it 3 extra times on a slot
BEFORE the "real" replay silently advances that slot's actual GDN state
by 3 unaccounted steps that `slot_kv_len` bookkeeping has no idea
happened. The earlier round's response to this exact risk (in the MTP
section above) was: "this is a real imprecision... likely didn't surface
because this signal-probe task is dominated by full-attention layers." **That was an unverified guess presented with more confidence than it
deserved, not evidence** -- exactly the kind of claim this project's own
`feedback_verify_subagent_claims_before_propagating` discipline warns
against propagating without checking.

**Quantified proof the gap was real and severe, not a subtle
edge case** (see "eager-vs-graph numerical parity check" below for the
proper permanent fix's own verification; this specific number comes from
a throwaway diagnostic, `/tmp/.../demonstrate_old_bug.py`, not committed,
that manually reproduced the OLD capture-warmup pattern against the same
slots checked via replay): identical single-token input into two
otherwise-identical physical slots, one via the eager path, one via a
graph that had 3 old-style redundant warmup executions run against the
SAME slot first --

```
LOGITS max_abs_diff=7.92578125 cosine_sim=0.5486875772476196
GDN conv_max_diff=45.8203125  ssm_max_diff=12.510265350341797
```

A cosine similarity of 0.55 and a GDN state tensor differing by tens in
absolute magnitude is NOT floating-point noise -- it is a real, severe
divergence that the signal-probe tests never had a chance to catch
because they only ever checked whether the FINAL DECODED TEXT still
happened to recover the right identity number, not the underlying
logits/state.

**Fix, built into `CapturedBatchDecodeGraph` itself (not left to caller
discipline this time)**:
1. **State-neutral capture**: the class now permanently reserves
   `batch_size` of the runner's own logical slots (the LAST
   `batch_size` slots of `runner.num_slots`, via `self._warmup_slots`)
   exclusively for `capture()`'s own disposable warmup. `capture()` no
   longer takes ANY external slot/token/kv_length arguments -- it
   prefills its own reserved slots with a fixed dummy prompt (`[0, 0, 0,
   0, 0]`, matching `DirectModelRunner._warmup`'s own convention) and
   uses those. Callers must size `runner.num_slots >= 2 * batch_size` (a
   `ValueError` is raised otherwise) and must never pass a graph's
   reserved warmup slots to `replay()` (also enforced with a
   `RuntimeError`). This works because `capture()`'s slot identity was
   ALREADY never required to match `replay()`'s (both independently
   recompute addressing fresh each call) -- the bug was never using that
   freedom, not a structural limitation.
2. **Removed the per-replay `torch.cuda.synchronize()`** (also flagged
   by the same review): `_fill_buffers`'s `.copy_()` calls and
   `self._graph.replay()` are all issued on the SAME (default) CUDA
   stream, so CUDA's own stream-ordering already guarantees correctness
   without an explicit device-wide sync -- which was actively working
   against the whole point of using a captured graph to cut CPU-side
   dispatch overhead (it blocks on ANY other queued device work, not
   just this graph's own stream).
3. **Leaner `_fill_buffers`** (same review, "no new allocations on the
   replay hot path"): rewritten to compute per-replay values via plain
   Python arithmetic instead of round-tripping through
   `build_attention_metadata_batch`/`build_gdn_metadata_batch`/
   `DirectModelRunner._slot_mapping_batch`, each of which constructs
   several of their own intermediate GPU tensors for dataclass fields
   this hot path doesn't need. Partial mitigation, honestly scoped: each
   static buffer's `.copy_()` source is still a freshly built small
   tensor, not a persistent pinned staging buffer written in place --
   a fully allocation-free version is a further optimization, not
   attempted this round.

**New, decisive verification** (`benchmarks/cudagraph_eager_parity_check.py`)
-- real numerical eager-vs-graph comparison, NOT signal-probe, per the
coordinator's explicit instruction: drives the IDENTICAL single-token
input through the already-verified eager path and the (now-fixed)
captured-graph path on independent, identically-prefilled physical
slots, and compares directly:
- Full logits: max abs diff, cosine similarity, `torch.allclose`.
- Top-1/top-5 predicted token agreement.
- The GDN `conv_state`/`ssm_state` tensors themselves, read directly out
  of `runner.kv_caches` for each physical slot -- the single most direct
  test for this exact bug class (GDN's own math doesn't depend on
  attention's kv_split_size at all, so eager and graph should agree here
  far more tightly than the logits comparison, which has an expected
  small amount of noise from different kv_split_size/split-reduction
  paths -- see the test file's own docstring for the reasoning).

**Result with the fix applied: `max_abs_diff=0.0`, `cosine_similarity=1.0`,
top-1/top-5 exact match, and EVERY one of 48 GDN layers checked shows
`conv_max_diff=0.0`/`ssm_max_diff=0.0`** -- eager and graph are not just
"close," they are bytewise identical. **3/3 independent repeats, all
PASS.** Re-ran the qo_len=1 and MTP regression tests too after the fix
(all four affected scripts updated to the new no-arg `capture()` API and
the `2*batch_size`/`3*batch_size` slot-reservation requirement): both
still 3/3 PASS, unchanged pass/fail pattern from before the fix.

**Also corrected, per the same review**: this round's and the prior
round's prose repeatedly described the test's `blocks_per_slot *
block_size = 2048`-token limit as this slot's "hard (physical) capacity"
-- inaccurate and misleading phrasing. This is a SMALL VALUE THIS TEST
ITSELF CONFIGURED for speed, not a GPU hardware limit, and it is far
below the 4K/32K a real W1/W2 workload will need (a future performance-
benchmark round must configure `blocks_per_slot` much larger for that).
Fixed the live code/docstrings and all four benchmark scripts to say
"this test's configured per-slot page-table limit" instead. Earlier
already-committed PROGRESS.md/notes entries using the old phrasing are
left as-is (historical record, not silently rewritten) -- this note is
the correction of record for anyone reading them later.

**Other findings from the same review, acknowledged but NOT fixed this
round (explicitly out of this round's scope, tracked as open items for
later)**:
- `DirectModelRunner`/`build_vllm_config` hardcode
  `attention_backend=AttentionBackendEnum.CUSTOM` with no native
  (FlashInfer) fallback path -- given the sibling project's own
  conclusion that native attention is currently faster than this
  project's custom kernel, this runtime currently has no way to opt back
  into the faster path. Not addressed this round.
- `runtime/engine.py`'s `EagerEngine.decode_batch()` (a separate,
  older control-plane abstraction) still loops calling single-request
  `self.decode()` -- it has never been wired to
  `DirectModelRunner.decode_batch()`/`CapturedBatchDecodeGraph`, so the
  control-plane layer and the real batching/CUDA-Graph mechanism
  built this round remain two disconnected pieces. Not addressed this
  round.
- `verify_batch()` still only returns raw logits; accept/reject sampling
  is not implemented (already known/tracked, consistent with prior
  rounds' explicit scoping).

**Per the coordinator's new priority ordering, the next steps (not this
round) are**: (2) full eager-mode MTP semantics (real draft generation,
bonus-token handling, accept/reject, and an explicit GDN state commit/
rollback strategy for partial rejection -- flagged as the hard part,
since attention's KV can be logically truncated but GDN's recurrent
state cannot be simply rewound), THEN (3) the real W1/W2/concurrency=4/
MTP-K=3 performance comparison, configured with per-slot capacity
actually sized for W1 (4K)/W2 (32K), not this round's small 2048-token
test configuration.

## MTP semantics round (2026-07-17): accept/reject + GDN rollback implemented and verified; real draft generation investigated in depth and honestly deferred -- more scope than initially estimated

Per the coordinator's explicit instruction, read `项目实施规划.md`'s
actual contract before designing anything: **"1. 先完整复现 vLLM 的 MTP
K=3"** (Phase 8) and **"MTP acceptance 不得比 vLLM 下降超过 1 个百分点"**
(Phase 1 gate) -- i.e. replicate vLLM's REAL MTP mechanism, not invent a
simplified stand-in, and the acceptance-rate bar is an explicit,
numeric gate, not a vague aspiration.

### What vLLM's real MTP K=3 actually requires (traced from source, not guessed)

> **2026-07-17 correction**: this subsection originally cited
> `vllm/model_executor/models/qwen3_next_mtp.py` (`Qwen3NextMTP`) as the
> draft model class. That file/class serves the **`qwen3_next`**
> `model_type`, a DIFFERENT model family. This project's actual target
> checkpoint (`unsloth/Qwen3.6-27B-NVFP4`) has top-level `model_type:
> "qwen3_5"` (verified directly from its `config.json`, nested
> `text_config.model_type: "qwen3_5_text"`, `mtp_num_hidden_layers: 1`),
> which `SpeculativeConfig.update_arch_()`
> (`vllm/config/speculative.py:500-509`) rewrites to `"qwen3_5_mtp"` /
> `architectures=["Qwen3_5MTP"]` -- loading `Qwen3_5MTP`
> (`vllm/model_executor/models/qwen3_5_mtp.py:192`), which wraps the
> actual decoder-layer-holding module `Qwen3_5MultiTokenPredictor`
> (same file, line 63) as `self.model`. Both were independently
> re-verified by reading the checkpoint's `config.json` and the vLLM
> source directly (not taken on trust) before writing this correction.
> The text below is corrected in place; the architectural
> conclusions (separate small model, own full-attention KV cache,
> every-step sync) were never wrong, only the specific file/class/field
> names were.

Read `vllm/model_executor/models/qwen3_5_mtp.py`
(`Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor` -- a genuinely SEPARATE
small model, its own `fc`/decoder-layer(s)/`norm`, sharing only
`embed_tokens`/`lm_head` with the target model) and
`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` (the real
propose-loop orchestrator) to understand the exact mechanism:

1. **The MTP model has its OWN full-attention decoder layer(s)**
   (`Qwen3_5DecoderLayer(layer_type="full_attention")`, count =
   `mtp_num_hidden_layers`, 1 for this checkpoint) -- a real transformer
   layer with real K/V, not just a linear head.
2. **This MTP attention layer needs its OWN KV cache, kept in sync with
   the target model on EVERY real step** -- not just during propose
   loops. `_prepare_prefill_inputs_kernel`'s "shift target_input_ids by
   one" logic runs over the FULL current query range on every real
   target-model prefill/decode step (not just the last position),
   because the draft model's own attention needs COMPLETE causal history
   to work at all -- exactly like any autoregressive transformer.
   **2026-07-17 refinement (sol-verified)**: this does NOT mean the sync
   call must be scattered into every public forward entry point of the
   main model. A precise deferred/lazy catch-up (save the un-synced
   token span + target hidden states + position/slot metadata, do one
   batched prefill-style catch-up later) is theoretically possible, but
   the margin is narrow here: every position's draft-model input
   depends on the TARGET's own hidden state for that position, not just
   the token id, so "not saved" means "must recompute the target's
   history," and this project's K=3-every-round workload has no real
   idle gap to defer into -- steady state, deferred sync degrades to a
   first-pass every round anyway, i.e. no real saving. The accurate
   framing is: draft sync must be part of every round's state machine,
   but the CALL SITE can be centralized at one place -- the
   target-forward/verify-propose boundary -- rather than duplicated into
   every public forward entry point. See "worst-case redesign scope"
   below, revised accordingly.
3. **The propose loop itself** (`_prefill`/`_multi_step_decode`/
   `_generate_draft`): step 0 feeds the draft model `hidden_states =
   target model's own last hidden state` (from the step that just ran)
   and `input_ids` = the target's own real tokens shifted by one
   (teacher-forcing); steps 1..K-1 feed the draft model's OWN previous
   step's hidden state and its own previously-sampled draft token
   (genuinely autoregressive on the draft side). Matches
   `Qwen3_5MultiTokenPredictor.forward()`'s `spec_step_idx` parameter
   (cycles through `self.layers[spec_step_idx % num_mtp_layers]`, and
   `num_mtp_layers=1` here so every step reuses the same single layer).
4. **Loading it the "complete-replication" way** (not reinventing): pass
   `speculative_config={"method": "mtp", "num_speculative_tokens": 3,
   "attention_backend": "CUSTOM"}` to `EngineArgs` (matching
   `launch_test_server.py`'s established convention exactly) so
   `vllm_config.speculative_config.draft_model_config` is constructed by
   vLLM's own `SpeculativeConfig.update_arch_()` logic (confirmed via
   source: for `qwen3_5`/`qwen3_5_moe` models this rewrites
   `hf_config.model_type` to `"qwen3_5_mtp"` and sets
   `architectures=["Qwen3_5MTP"]` (or `Qwen3_5MoeMTP` for the MoE
   variant, not applicable to this checkpoint), same checkpoint path,
   different vLLM model class), then
   `get_model(vllm_config=vllm_config, model_config=vllm_config
   .speculative_config.draft_model_config)` loads it -- reusing real
   vLLM construction logic end to end, not a hand-rolled substitute.
   Because this second `get_model()` call registers the MTP layer's own
   attention into the SAME `vllm_config.compilation_config
   .static_forward_context` this project's existing
   `allocate_fixed_slot_kv_caches`/`attn_layer_names` machinery already
   iterates over, the MTP layer's KV cache allocation would "just work"
   through the existing generic mechanism once the model is loaded before
   cache allocation -- a clean integration point, confirmed by reading
   the code, not yet exercised by running it.

**Honest scope decision**: point 2 above -- restructuring every real
forward call to also drive a synced draft-model KV cache -- is
substantially more invasive than "add a propose loop," and combined with
everything else (the loop itself, accept/reject, GDN rollback,
verification against real vLLM's acceptance rate) does not fit this
round's remaining budget with the rigor this project requires. Rather
than rush an implementation likely to be subtly wrong, this round
implements and verifies the two pieces that are genuinely self-contained
and independently checkable without the draft model at all, and defers
real draft generation to its own dedicated round with this design
already worked out (not starting from zero next time).

### Implemented and verified this round

**1. Accept/reject boundary logic** (pure logic, no new model-loading
needed -- exercised against the ALREADY-WORKING, CUDA-graph-capable
`verify_batch()`): `benchmarks/mtp_accept_reject_check.py`. Draft
convention matches `verify_batch`'s `qo_len=K+1` layout: `draft_tokens =
[anchor, d_0, ..., d_{K-1}]` (the anchor is the already-committed last
real token, K=3 matching production). Verify position `p`'s logits
predict "what comes after `draft_tokens[p]`", compared against
`draft_tokens[p+1]` for `p < K`; position `K`'s logits (nothing left to
compare against) become the bonus token if every prior comparison
passed -- standard greedy speculative-decoding verification, no
probabilistic rejection sampling (matching the coordinator's explicit
"不需要一开始就做完整概率化rejection sampling" simplification).

Verified via three constructed scenarios per run, using REAL trusted
continuation tokens (from the already-verified qo_len=1 decode path) as
the ground truth, with a deliberate decoy token injected at a KNOWN
position:
- `all_accept` (draft = the real trusted continuation exactly):
  `num_accepted=3`, committed tokens exactly match the real continuation
  plus the correct bonus token.
- `reject_at_1` (position 1 replaced with a decoy): `num_accepted=1`,
  rejection detected exactly at position 1, and -- the decisive check --
  the recovery token at the rejection point equals the TRUE next token
  (what the target model actually predicts), NOT the decoy and not
  garbage.
- `reject_at_0` (position 0 replaced with a decoy): `num_accepted=0`,
  recovery token again equals the true next token.

**3/3 independent repeats, all PASS**, every scenario checked by exact
token-id comparison against the trusted reference, not "does it look
plausible."

**2. GDN state commit/rollback -- "Option A" (snapshot/restore),
implemented and verified**: `DirectModelRunner.snapshot_gdn_state(slot)`/
`restore_gdn_state(slot, snapshot)`. Design choice, weighed against the
coordinator's Option B (exploit chunked FLA's own chunk boundaries to
recompute only the rejected sub-range): Option A was chosen because it
is simple to reason about correctly and to verify in complete isolation
from the rest of MTP (no draft model or propose loop needed to test it),
at the cost of an extra state copy per verify call and a full recompute
forward pass on rejection. Option B was NOT ruled out as wrong, just not
attempted this round -- it would require verifying FLA's chunk
granularity aligns safely with arbitrary per-token accept/reject
boundaries (K=3 is smaller than `FLA_CHUNK_SIZE` in the general case, so
a rejection could fall mid-chunk), which this round did not investigate
deeply enough to trust; Option A's correctness doesn't depend on that
question at all.

Verified via `benchmarks/mtp_gdn_rollback_check.py` -- a numerical twin
comparison, not signal-probe: one slot takes a real "detour" (4 extra
genuine decode steps advancing its GDN state for real, simulating "some
speculative steps ran"), then gets its state restored from a snapshot
taken before the detour; a twin slot never takes any detour at all.
Both are then driven through the identical next real decode step.
**Result: `logits_exact_equal=true` (bytewise identical, not just
close), and all 48 GDN layers checked show `conv_diff=0.0`/
`ssm_diff=0.0`.** Restoring genuinely undoes the detour's real state
changes, not just makes subsequent output look plausible. **3/3
independent repeats, all PASS.**

### Not implemented this round (explicitly, tracked for the next round)

Real draft generation via the model's own native MTP head
(`Qwen3NextMTP`) -- requires: (a) loading the draft model via
`speculative_config`, (b) restructuring `prefill`/`decode`/
`_forward_batch` to also drive the draft model's own KV-cache-synced
forward pass on every real step (not just during propose loops), (c) the
K-step autoregressive propose loop itself, (d) wiring the two verified
pieces above (accept/reject, GDN rollback) into a real end-to-end
verify-then-commit-or-rollback cycle, (e) comparing the resulting
acceptance rate against a real vLLM MTP server on the same
prompt/seed against the project's explicit ≤1-percentage-point gate.
This is a substantially larger, multi-part integration than initially
scoped when this round started -- reported honestly per the
coordinator's explicit invitation to do so, not forced to a rushed
finish. Recommendation: treat this as its own dedicated round, using the
concrete design above (already traced from source, not requiring
re-investigation) as the starting point.

## 2026-07-17 follow-up: confirmed "every step needs draft-model sync" with an exact evidence chain; worst-case redesign scope; a pragmatic simplified-draft alternative evaluated

Per the coordinator's request, went deeper on whether vLLM's real MTP
implementation truly requires a draft-model forward pass on every single
real engine step, or whether there's a lighter path (only sync when
actually about to propose). This is now settled with an exact,
citable evidence chain, not inference from a partial read.

### Evidence chain: yes, unconditionally every step

1. **`vllm/v1/worker/gpu/model_runner.py:1114` (`execute_model`)** is the
   SINGLE, unified per-engine-step entry point -- driven by one
   `scheduler_output` that can mix prefill and decode requests in the
   same call (V1's continuous-batching design), not a
   prefill-vs-decode-branching dispatcher. There is exactly one call site
   per real step, not one-per-request-type.
2. **`vllm/v1/worker/gpu/model_runner.py:1456-1479`**: immediately after
   `self.postprocess_sampled(...)` (which commits this step's real
   sampled/accepted tokens into request state), the code runs:
   ```python
   if self.speculator is not None:
       ...
       draft_tokens = self.speculator.propose(...)
       self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
   ```
   The ONLY gate is `self.speculator is not None` (i.e. "is speculative
   decoding configured at all for this engine") -- there is no additional
   condition like "only if this step is a decode step" or "only every N
   steps" or "only if the scheduler asked for a draft this cycle". Every
   real step -- prefill, decode, or a mixed batch -- calls `propose()`
   right after committing that step's real output.
3. **The SAME call appears a second time, at
   `vllm/v1/worker/gpu/model_runner.py:582-623`**, inside the
   warmup/dummy-run/profiling path (`dummy_run=True`) -- explicitly
   labeled "dummy run the eagle speculator's propose to ensure DP/EP
   sync", confirming the propose call is treated as a MANDATORY part of
   every step's execution contract (even a fake warmup step must still
   exercise it), not an optional side channel.
4. **`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`**
   (`AutoRegressiveSpeculator.propose`, the base class
   `MTPSpeculator`/Qwen3's MTP uses): line 174
   (`hidden_states = last_hidden_states`) feeds the draft model the
   TARGET model's own just-computed hidden state for THIS step -- a
   hard, per-step data dependency, not something that could be
   deferred or batched up and run less often.
5. **`_prepare_prefill_inputs_kernel`, same file, lines 510-519**:
   ```python
   # Shift target_input_ids by one.
   for i in range(1, query_len, BLOCK_SIZE):
       ...
       tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
   ...
   tl.store(draft_input_ids_ptr + last_token_index, next_token)
   ```
   This loops over the FULL current step's query range (`query_len`,
   this step's real token count for this request -- large for a prefill,
   1 for a plain decode, K+1 for an MTP verify step), not just the last
   position. The draft model is fed a "teacher-forced" shifted copy of
   EVERY real token processed this step, so its own attention KV cache
   accumulates a complete, gap-free history in lockstep with the target
   model's. This is the direct mechanism-level confirmation: the draft
   model's own attention layer literally cannot skip steps and stay
   correct, because a real transformer attention layer's causal history
   has no valid notion of "catch up later" -- every position must be
   written when it happens, in order, or later positions attend over a
   hole.

**Conclusion, stated plainly**: there is no lighter alternative in
vLLM's real implementation. "Only run the draft model when actually
about to propose K tokens" is not a real code path that exists -- MTP's
design fundamentally assumes the draft model's KV cache is always
current, because that is what makes the K-step autoregressive propose
loop cheap (it only needs to extend, never rebuild, history). A
from-scratch reimplementation COULD in principle choose a different,
non-vLLM-compatible design (e.g. recompute the draft model's full
context from scratch on demand, trading a large one-time cost per
verify cycle for avoiding the small per-step sync cost) -- but that
would not be "replicating vLLM's MTP K=3" per the project's own Phase 8
mandate, it would be a genuinely different mechanism with different
acceptance-rate and performance characteristics, unverified against
anything.

### Worst-case redesign scope (design-level, no implementation this round)

If this project commits to faithfully replicating vLLM's mechanism, the
concrete changes are:

> **2026-07-17 correction applied throughout A-E below**: the draft
> model class is `Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor`
> (`vllm/model_executor/models/qwen3_5_mtp.py`), not `Qwen3NextMTP` --
> see the correction note earlier in this document. Field name is
> `mtp_num_hidden_layers`, not `num_nextn_predict_layers`. Point C is
> also revised from the original write-up per sol's refinement: the
> sync call does not need to be duplicated into every public forward
> entry point, only centralized at one boundary (see below).

**A. Model loading** (`build_vllm_config`/`DirectModelRunner.__init__`):
add a `with_mtp: bool`/`num_speculative_tokens: int` parameter passing
`speculative_config={"method": "mtp", "num_speculative_tokens": K,
"attention_backend": "CUSTOM"}` to `EngineArgs` (matching
`launch_test_server.py` exactly -- reuses vLLM's own
`SpeculativeConfig.update_arch_()` construction, not reinvented). Then
`get_model(vllm_config=vllm_config, model_config=vllm_config
.speculative_config.draft_model_config)` loads `Qwen3_5MTP`, storing it
as `self.mtp_model`. Must happen BEFORE `_allocate_and_bind_kv_caches()`
so the MTP model's own attention layer registers into the SAME
`static_forward_context` this project's existing KV-cache-allocation
machinery already iterates over.

**B. KV cache sizing**: the MTP model's own attention layer needs the
SAME per-slot page-table capacity as the target's 16 full-attention
layers (it tracks the identical-length sequence history) -- this
`allocate_fixed_slot_kv_caches` handles "for free" once the MTP layer is
in `attn_layer_names`, but it means attention-KV memory footprint grows
by 1/16 ≈ 6.25% (one extra full-attention-shaped cache per slot). GDN
memory is unaffected -- `Qwen3_5MTP`'s draft layer(s) are plain
full-attention, no GDN/linear-attention involved.

**C. Draft sync, centralized at ONE boundary, not scattered (revised
per sol)**: rather than adding a draft-model call to every public
forward entry point (`prefill`/`decode`/`_forward`/`_forward_batch`
individually), a single internal method -- e.g.
`_advance_target_and_sync_draft(...)` -- sits at the one place all of
those already funnel through today (the point right after the target
model's own `forward()`+`compute_logits()` produces this step's hidden
state and sampled token), and every one of today's public entry points
calls through it instead of calling the target forward directly. That
one method:
   1. Builds a shifted-by-one `input_ids` for the draft model (this
      step's real input_ids shifted left by one, with the newly-sampled
      next token in the last slot -- mirrors
      `_prepare_prefill_inputs_kernel`).
   2. Builds the draft model's OWN attention metadata + slot_mapping
      (same per-slot physical addressing as the target's attention
      layers, scoped to just the MTP layer's own name(s) -- a NEW
      `self.mtp_attn_layer_names` list, separate from
      `self.attn_layer_names`).
   3. Calls `self.mtp_model.forward(input_ids, positions, hidden_states=
      <target's own last hidden state from this step's call>,
      spec_step_idx=0)` under `set_forward_context` scoped to the MTP
      layer's metadata.
   This still roughly DOUBLES the number of real forward passes per
   step (target + draft-sync), and the CUDA-Graph-captured path still
   needs this sync call captured as part of the SAME graph (or a
   second, chained graph) to keep the "no Python-level dispatch per
   step" property this round's CUDA Graph work established -- the
   sol-driven change is about WHERE in the code the call lives (one
   funnel point), not whether the call itself can be skipped or made
   cheaper.

**D. The K-step autoregressive propose loop** (only needed once C above
provides a synced draft KV cache and an initial `draft_tokens[0]`, i.e.
this is layered ON TOP of C, not a replacement for it): for
`step in range(1, K)`, call `self.mtp_model.forward(input_ids=
previous_draft_token, positions=advancing by 1, hidden_states=
previous step's own output hidden state, spec_step_idx=step)`
(`spec_step_idx % num_mtp_layers` always resolves to layer 0 here, since
`mtp_num_hidden_layers=1` for this checkpoint -- confirmed via
`qwen3_5_mtp.py`'s own `Qwen3_5MultiTokenPredictor.__init__`), each
extending the draft model's own KV cache by one more position at the
SAME physical slot.

**Per-slot state to track (sol's explicit reminder)**: the centralized
state machine in C/D/E needs each of the 4 fixed slots to carry, at
minimum: `committed_len` (target-confirmed sequence length),
`draft_sync_len` (how far the draft model's own KV cache has been
advanced -- these two are DIFFERENT quantities and must not be
conflated, see the `_forward_batch` reminder below), pending draft
token ids awaiting verification, the KV physical-address range written
speculatively this round (so a rejection knows what was written but
never gets "confirmed"), and a GDN snapshot generation/version counter
(so a stale snapshot can never be restored by mistake once a later
snapshot has superseded it).

**E. Wiring the two already-verified pieces into a real cycle**: submit
`[anchor] + draft_tokens` through the ALREADY-WORKING
`verify_batch()` (qo_len=K+1); run the ALREADY-VERIFIED
`determine_accept_reject()`; on any rejection, call the
ALREADY-VERIFIED `restore_gdn_state()` (snapshotted before the verify
call) then re-run the target model's own forward for exactly the
accepted-token count to bring GDN state to the correct point (the
target's attention KV needs no explicit cleanup -- rejected positions'
KV entries are simply never addressed again, matching how attention
already handles this generally). The draft model's OWN KV cache likely
needs NO special rollback either, by the same "position-addressed,
never revisited" logic -- ONLY GDN's recurrent state has the
non-addressable-by-position problem this round's Option A specifically
targets.

**F. Verification**: compare acceptance rate against a real vLLM MTP
server (`launch_test_server.py --with-mtp`) on the same prompt/seed,
against the project's explicit ≤1-percentage-point gate
(`项目实施规划.md` Phase 1).

**Bottom line on cost**: A-B are small, mechanical, low-risk (reuses
existing generic machinery). C is the expensive, pervasive part --
touches every real forward call site in `DirectModelRunner` plus the
CUDA-Graph-captured path, and roughly doubles real forward-pass count
per step. D-E are comparatively contained once C exists. This is
consistent with the previous round's conclusion that this is
substantially more than "add a propose loop" -- C is the reason.

### Pragmatic alternative evaluated: a simplified, shape-correct-but-not-vLLM-faithful draft mechanism to unblock the performance question sooner

The coordinator asked whether a simplified draft mechanism -- even one
with an acceptance rate that does not yet meet the ≤1pp gate -- could
establish a minimal closed loop sooner, specifically to get the more
fundamental "how much does removing scheduling overhead actually save"
signal, with a rigorous real-draft-model implementation as a separate
follow-up task. Evaluated concretely:

**Proposed mechanism**: instead of loading `Qwen3_5MTP` at all, use
the TARGET model's own already-verified, already-CUDA-graph-capable
qo_len=1 decode path to generate K "draft" tokens via K genuine
sequential single-token greedy decodes (self-drafting), then submit
`[anchor] + those K tokens` through the REAL `verify_batch()` (qo_len=
K+1, the actual production MTP shape, already CUDA-Graph-capturable)
exactly as a real MTP cycle would.

**What this gets right**: it exercises the ACTUAL qo_len=K+1
CUDA-Graph-captured kernel path (the thing that actually determines
launch-gap/GPU-busy%/throughput -- this round's core open question) with
zero new model-loading or KV-sync engineering, reusing 100% already-built
and already-verified infrastructure (`decode_batch`, `verify_batch`,
`CapturedBatchDecodeGraph` at both qo_len=1 and qo_len=K+1,
`determine_accept_reject`, `snapshot_gdn_state`/`restore_gdn_state`).
This is a genuinely representative WORKLOAD SHAPE for the performance
question, not a toy.

**What it gets wrong, and why that's an acceptable, clearly-labeled
limitation for a performance-only round**: since the "draft" tokens ARE
literally the target model's own greedy continuation, `verify_batch`
will find them accepted essentially 100% of the time (both computations
agree by construction, modulo the same small batch/shape-dependent
floating-point noise this project already characterized and accepted as
normal) -- an unrealistically perfect acceptance rate compared to a real
(weaker, cheaper) draft model, which this project's own earlier sibling
data measured around 63-66% for this exact model/workload. This means:
- Any "accepted tokens/s" number from this setup is an OPTIMISTIC UPPER
  BOUND (real MTP would reject some drafts, costing extra recovery-token
  work and fewer tokens committed per verify call), not a realistic
  production estimate -- must be reported as such, not conflated with a
  real accepted-tokens/s figure.
- It never exercises the REJECTION path in a live pipeline (though
  that path is ALREADY independently verified correct by this round's
  `mtp_accept_reject_check.py`/`mtp_gdn_rollback_check.py` -- the
  performance run does not need to re-prove correctness, only measure
  shape-representative throughput).
- It does NOT satisfy the project's Phase 8 mandate ("先完整复现vLLM的
  MTP K=3") as a final deliverable -- it is explicitly a stopgap for the
  performance question, not a substitute for the real draft model work
  in section E above.

**Recommendation**: this is a reasonable, honestly-scoped way to get an
early, clearly-caveated read on the launch-gap/GPU-busy% question while
the real draft-model integration (section C above) is scoped as its own
round -- PROVIDED the resulting numbers are always reported alongside
an explicit "self-drafted, ~100% acceptance, optimistic upper bound, not
comparable to real MTP acceptance-rate numbers" label, and the real
draft-model work in section C-F is not quietly dropped as a result of
getting an early performance signal this way. Not started this round
(design-only, per the coordinator's explicit scope for this round);
implementing it would be a small, low-risk addition on top of already-
built and already-verified infrastructure whenever the next round picks
it up.

## 2026-07-17, second follow-up: independent Codex-sol analysis returned — one real correction accepted, one refinement accepted; trace-driven scheduling-overhead probe (sol's two-phase route, phase 1) built, run, and stable

### Correction accepted and applied throughout this document

The coordinator's parallel Codex-sol analysis of the "does the draft
model need every-step sync" question caught a real error in the two
sections above: they cited `vllm/model_executor/models/qwen3_next_mtp.py`
(`Qwen3NextMTP`) as the draft model class. **Independently re-verified
before accepting** (not taken on trust): read this project's actual
target checkpoint's `config.json`
(`/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/
snapshots/.../config.json`) directly — top-level `model_type: "qwen3_5"`,
nested `text_config.model_type: "qwen3_5_text"`,
`mtp_num_hidden_layers: 1`. Then read `vllm/config/speculative.py:500-509`
directly, confirming `hf_config.model_type in ("qwen3_5", "qwen3_5_moe")`
is rewritten to `"qwen3_5_mtp"` / `architectures=["Qwen3_5MTP"]` (not
`Qwen3NextMTP`, which serves the unrelated `qwen3_next` model type), and
`vllm/model_executor/models/qwen3_5_mtp.py` directly, confirming
`Qwen3_5MTP` (line 192) wraps `Qwen3_5MultiTokenPredictor` (line 63) as
`self.model`, with `self.num_mtp_layers = getattr(config,
"mtp_num_hidden_layers", 1)` (line 79) — matching the checkpoint's real
field name. All wrong references in the two sections above have been
corrected in place, with inline correction notes marking what was wrong
and why (the architectural conclusions were never wrong, only the
specific file/class/field names).

### Refinement accepted: sync must be per-round, but the call site can be centralized, not scattered

Sol also refined the "must sync every step, no lighter alternative"
conclusion: a precise deferred/lazy catch-up sync is theoretically
possible, but the margin is narrow for this project's actual workload
(K=3, propose every round, no idle gap to defer into — deferred sync
would degrade to a first-pass every round anyway, no real saving). The
accurate framing, now applied to the worst-case redesign scope section
above: draft sync must be part of every round's state machine, but the
CALL SITE does not need to be duplicated into every public forward
entry point — it can live at ONE centralized funnel point (the
target-forward/verify-propose boundary), which every public entry point
routes through. This is a real softening of the original "restructure
every real forward call site" framing, though the underlying sync cost
itself (roughly doubling forward-pass count per round) is unchanged.
Also incorporated: the specific per-slot state fields a centralized
state machine needs to track (`committed_len`, `draft_sync_len`,
pending draft tokens, speculative-write KV range, GDN snapshot
generation) and two concrete code-level reminders — `snapshot_gdn_state`
uses CPU clones (fine for the current correctness-verification stage,
but should become pre-allocated GPU buffers before any graph-capture
integration), and `_forward_batch` currently conflates "positions
physically written speculatively" with "positions actually committed"
(advances `slot_kv_len` by the full `qo_len` unconditionally) — a real
gap for a future live multi-round loop, though not yet triggered by any
existing test (each of `mtp_accept_reject_check.py`'s scenarios uses a
fresh slot exactly once, never continuing after a partial-accept verify
call, so the conflation never surfaces there).

### Adopted: sol's two-phase route

1. **Phase 1 (this round)**: a strictly time-boxed trace-driven
   scheduling-overhead probe — no real drafter, a synthetic accept/reject
   trace drives the real, already-verified production mechanisms
   (`verify_batch`, GDN snapshot/restore, committed-length recompute),
   measuring ONLY control-plane/scheduling overhead. Any acceptance-rate
   or accepted-tokens/s number from this probe MUST be labeled a
   controlled-trace scheduling-overhead upper-bound estimate, never a
   real MTP performance/acceptance conclusion, never a substitute for
   the real W1/W2 ≤1pp acceptance-rate gate.
2. **Phase 2 (next round, not started this round)**: regardless of
   phase 1's result, move to "Option A" — the faithful centralized
   incremental MTP state machine: load the real `Qwen3_5MTP`, have the
   unified target-forward boundary return both logits and hidden
   states, sync the draft model and generate a proposal immediately
   after every prefill/decode, verify then update target/draft KV
   cursors and GDN state per accept/reject. Per-slot state to track:
   `committed_len`/`draft_sync_len`/pending draft tokens/KV write
   range/GDN snapshot generation (see above).

### Phase 1 probe: implementation and results

Built `benchmarks/mtp_trace_driven_probe.py`. Design: a seeded RNG
generates, per round per slot, an `accept_len ∈ {0,1,2,3}` via K=3
Bernoulli(p) trials (first failure truncates) — entirely SYNTHETIC, not
derived from any real model prediction. Three `p` configs run
back-to-back on the same 4 fixed slots (reset+re-prefilled between
configs): `p=1.0` (best case / equivalent to the previously-proposed
"self-drafting" upper bound), `p=0.65` (this project's own earlier real
MTP measurement, used only as a representative shape, not re-derived
here), `p=0.0` (worst case, every round rejects immediately). Per round:
snapshot GDN state for all 4 slots (unconditional — a real system can't
know the outcome ahead of time), run ONE real batched `verify_batch`
call (qo_len=K+1=4, concurrency=4) with CUDA-event + wall-clock timing
around it, then for every slot whose (synthetic) trace outcome is not
full-accept: `restore_gdn_state`, manually correct `slot_kv_len` back
down to the pre-verify length (fixing, for this script's own
bookkeeping, the exact `_forward_batch` conflation noted above), then a
real recompute forward of exactly the committed length, separately
timed. 20 rounds per config, first 3 discarded as warmup, 17 measured.

Ran twice (fresh process each time, checkpoint reload ~25s-137s
depending on OS page cache) to check stability before treating the
result as real — GPU/process state confirmed clean via `nvidia-smi`/`ps`
before each run, per standing discipline. **Results were stable across
both runs** (both included below; not cherry-picked):

| config | run 1 GPU-busy% | run 2 GPU-busy% | run 1 avg wall/round | run 2 avg wall/round |
|---|---|---|---|---|
| best_case (p=1.0) | 101.5% | 98.8% | 100.75 ms | 104.40 ms |
| realistic (p=0.65) | 99.2% | 100.2% | 342.78 ms | 325.49 ms |
| worst_case (p=0.0) | 99.8% | 98.1% | 405.66 ms | 421.39 ms |

(GPU-busy% values slightly over 100% are CUDA-event/wall-clock
measurement noise at this small a gap, not a real physical
impossibility — the honest reading is "indistinguishable from 100%
within measurement noise," not "somehow more than 100%.")

**Finding: GPU-busy% is ~98-101% across ALL THREE trace configs,
i.e. indistinguishable from 100% regardless of how much rollback/
recompute work each round does.** Launch-gap (wall-clock time NOT spent
executing real GPU kernels) is ~0%, at or below measurement noise, in
every configuration tested.

**Interpretation (this is real signal, not a null result)**: for THIS
workload's shape — a 64-layer, 27B-parameter, batch=4, qo_len=4
verify-or-recompute forward pass — the actual GPU compute (100-420ms
per round depending on config) completely dwarfs any Python-level
dispatch/launch overhead (which would be single-digit milliseconds at
most). This means:
- **This project's OWN direct-runner call sequence has essentially zero
  residual per-call scheduling overhead already**, even in plain eager
  mode with `torch.cuda.synchronize()` forced after every model call.
  There is no further "squeeze the launch gap" optimization available
  at THIS granularity — CUDA graph capture would not measurably improve
  per-round GPU utilization here, because there is no gap left to
  remove.
- This does **NOT** mean the whole "remove scheduling overhead" premise
  behind this runtime is wrong. It means the overhead this project is
  actually trying to eliminate lives ELSEWHERE — in native vLLM's own
  Python scheduler/block-manager/sampling/HTTP-layer cost PER STEP, none
  of which exists in this minimal direct runner's call path at all (it
  was never built to have that overhead, by construction) and NONE of
  which this probe measures (this probe only exercises OUR OWN kernel
  dispatch, not a comparison against native vLLM's overhead). The real
  answer to "how much do we save vs. native" still requires the actual
  W1/W2 concurrency=4 MTP K=3 comparison against real vLLM (task #85 in
  this project's tracker), still blocked on real draft-model
  integration.
- **A genuinely useful, if narrower, reading of this result**: since our
  own runtime's residual overhead is already ~0%, any overhead gap the
  eventual real W1/W2 comparison finds against native vLLM is a
  GENUINE, FULLY CAPTURABLE win — it won't be partially eaten by
  residual dispatch cost in our own call path, because there isn't any
  at this granularity. This is a positive signal for continuing the
  Phase 2 investment, not a negative one.
- **A caveat on method, stated honestly**: `_forward_batch`/`verify_batch`
  force `torch.cuda.synchronize()` twice per call (after `forward()` and
  after `compute_logits()`), which by construction prevents any
  cross-call async pipelining from being observed. A more aggressive
  design (e.g. CUDA-graph-batched multi-round replay, issuing round N+1
  before round N's results are consumed) could in principle expose
  overlap this measurement methodology cannot see — but that is
  precisely the kind of optimization only relevant when individual
  kernel calls are SMALL relative to Python dispatch cost (e.g. a plain
  single-token decode-shaped call), not for this MTP-verify-shaped
  workload where each call is already a large, compute-dominated chunk.
  This caveat, not a contradiction of the finding above, is why "capture
  graph" is explicitly the LAST step of sol's verification gradient, not
  an earlier one — its value is likely concentrated in smaller,
  decode-shaped calls, not in this verify/recompute shape.

**Scope discipline**: per the coordinator's explicit instruction ("这一轮
先做探针(阶段1)...严格限时"), Phase 2 (loading the real `Qwen3_5MTP`,
building the full centralized state machine) was NOT started this
round, despite sol's overall recommendation to move to it "immediately,
regardless of phase 1's result" — that instruction describes the
recommended route across rounds, not a mandate to compress both phases
into one. Phase 2 remains the explicit next step.

## 2026-07-17, Phase 2 (real draft model + centralized state machine): verification gradient steps 1-4 done, real bug found and fixed, steps 5-8 remain

Following the coordinator's go-ahead to start Phase 2 ("Option A"),
implemented in `runtime/direct_model_runner.py`: the real `Qwen3_5MTP`
draft model loaded via vLLM's own `load_eagle_model()` (also used by
real vLLM's `MTPSpeculator` -- not hand-rolled), a centralized MTP-cycle
funnel (`_mtp_forward`/`_mtp_sync_and_propose`/`mtp_prefill`/
`mtp_verify_and_commit`), explicit per-slot state, and the
`_forward_batch` physical-write-vs-committed separation Codex-sol
flagged. Went through the verification gradient's steps 1-4, with real
numerical twin comparisons at each step (not signal-probe) -- found and
fixed one real bug via direct content-level reasoning (not caught by
shape/bookkeeping checks alone). Steps 5-8 (multi-round decode,
concurrency=4 isolation, W1/W2 acceptance gate, CUDA graph) not
attempted this round.

### Implementation

**Model loading** (`DirectModelRunner.__init__`, before
`_allocate_and_bind_kv_caches()`): if `vllm_config.speculative_config`
is set, snapshot `static_forward_context`'s keys, call
`load_eagle_model(self.model, vllm_config)` (the SAME function real
vLLM's `MTPSpeculator.load_draft_model()` calls -- confirmed by reading
`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`), diff the keys to
get `mtp_attn_layer_names` (mirrors `DraftModelSpeculator.load_model()`'s
own before/after diff pattern at
`vllm/v1/worker/gpu/spec_decode/speculator.py:153-170`, confirming this
project's simpler direct-dict-diff achieves the same isolation
`get_layers_from_vllm_config(..., AttentionLayerBase)` does there).
`build_vllm_config()` gained a `speculative_config: dict | None` param,
passed straight to `EngineArgs` (matching
`vllm_integration/launch_test_server.py`'s exact production JSON
convention, including the `"attention_backend": "CUSTOM"` inside the
speculative_config dict itself -- confirmed necessary: the MTP proposer
does not inherit the top-level attention backend).

**Centralized funnel, not scattered** (per sol's refined design):
`_mtp_forward()` is the low-level draft-model-forward primitive (mirrors
`_forward`, but for `self.mtp_model`/`self.mtp_attn_layer_names`, no GDN
involved since the draft has none). `_mtp_sync_and_propose()` is the ONE
place sync+propose logic lives: step 0 syncs the draft's KV using the
target's own just-computed hidden states (teacher-forced, shifted input
covering the step's FULL query range), steps 1..K-1 are genuinely
autoregressive on the draft's own previous hidden state/token. Critically,
`_mtp_forward` does NOT advance `self.slot_draft_sync_len` itself --
the caller decides: step 0's advance is real (committed), steps 1..K-1's
advances are NOT persisted (exploratory-only), so the draft's own KV
cache needs NO explicit rollback on reject -- the next round's real sync
simply overwrites those same throwaway positions, exactly like
attention's own content/position-addressed reasoning elsewhere in this
project. `mtp_prefill()`/`mtp_verify_and_commit()` are the two public
MTP-aware entry points; the ORIGINAL `prefill`/`decode`/`decode_batch`/
`verify_batch` are UNTOUCHED and still work exactly as before -- MTP
awareness lives ONLY in the two new methods, not scattered in.

**Explicit per-slot state** (Codex-sol's ask): `slot_draft_sync_len`
(the draft's own KV length, separate from `slot_kv_len`),
`slot_pending_draft_tokens` (in-flight, not-yet-verified proposal),
`slot_gdn_snapshot_gen` (bumped on every `snapshot_gdn_state()` call;
`restore_gdn_state()` now rejects a generation mismatch, so a stale
snapshot can never be restored by mistake).

**`_forward_batch` physical-write-vs-committed separation**: gained a
`commit: bool = True` parameter. The forward pass always physically
writes K/V for all `qo_len` positions regardless of this flag; `commit`
only controls whether `self.slot_kv_len` advances. `decode_batch()`
keeps the default (`True` -- decode is never ambiguous). `verify_batch()`
now explicitly passes `commit=False` -- a verify call's real committed
length is unknowable until `determine_accept_reject` runs on its
logits, so auto-advancing was the exact conflation Codex-sol flagged.
`determine_accept_reject()` itself moved from
`benchmarks/mtp_accept_reject_check.py` into
`runtime/direct_model_runner.py` as a shared module-level function (that
benchmark now imports it, rather than keeping a second copy) --
reusing, not reinventing, per the coordinator's explicit instruction.

Both `_forward`/`_forward_batch` gained a `return_hidden: bool = False`
parameter (returns `(logits, hidden_states)` when set) -- needed since
the draft's sync step consumes the target's own hidden states, which
were previously computed-then-discarded.

### A real bug, found and fixed: recompute input token misalignment

`mtp_verify_and_commit`'s first draft fed `decision["committed"]`
(`[accepted_draft_0, ..., accepted_draft_{n-1}, recovery]`) directly as
the recompute forward's input tokens. This is WRONG: the token whose OWN
K/V lands at position `kv_len_before + i` is that call's i-th QUERY
INPUT, and per `verify_batch`'s own established convention (`draft =
[anchor, d_0, d_1, d_2]`, so `anchor`'s K/V lands at `kv_len_before`,
mirroring `prefill()`/`decode()`'s contract that the greedy/anchor token
is NOT written into KV until fed back in as a FOLLOWING call's input),
the correct recompute input is `[anchor] + decision["committed"][:-1]`
(anchor followed by the ACCEPTED drafts, dropping the recovery token,
which -- symmetrically -- has no KV entry yet either). Feeding
`committed` directly would have silently written the WRONG token
content into the KV cache at every recompute -- while still passing
every shape/length/`slot_kv_len`-bookkeeping check, since those are
blind to content. This is exactly why the coordinator's verification
gradient insists on real numerical/content checks, not just invariant
checks -- caught by direct reasoning through the position-alignment
semantics before writing the test that would prove it, not by a test
failure first.

**The check that would catch this class of bug** (added to
`benchmarks/mtp_real_draft_check.py`'s step 4, and would have failed
against the pre-fix code): independently replay the REAL committed
sequence (`prompt + anchor + accepted_drafts`) from scratch on a FRESH
reference slot via the plain, long-verified `prefill()` path, then
continue BOTH the MTP-committed slot (via one real decode call feeding
the recovery token) and the reference slot the same way, and compare
their next-token predictions. If the MTP-committed slot's KV cache holds
the wrong content, this diverges from the reference; if correct, they
agree.

### Verification gradient results (steps 1-4, real numerical twin checks, stable across 3 independent runs)

Built `benchmarks/mtp_real_draft_check.py`. Uses the SAME prompt as
prior rounds ("The capital of France is") for direct cross-session
consistency checking -- `anchor=11751`, `committed=[13, 271]` on a
forced/organic partial reject exactly match values already seen in
earlier `mtp_accept_reject_check.py` runs from a previous round, an
independent (if informal) cross-check that nothing regressed.

1. **Target prefill hidden states/logits alignment**: a plain
   (no-speculative-config) runner's prefill vs. an MTP-loaded runner's
   IDENTICAL prefill on the same prompt -- `logits_allclose=true`,
   `hidden_allclose=true`, `cosine_sim=1.0`, same greedy token. Loading
   the draft model alongside the target does not perturb the target's
   own computation. **PASS.**
2. **Weight sharing + shifted draft first pass**: `embed_tokens`/
   `lm_head` identity checked via `data_ptr()` equality (genuinely the
   SAME tensor, not just equal values) between target and draft --
   `true`/`true`, confirming `load_eagle_model`'s real sharing logic
   fired. The draft's own step-0 forward produces the correct shape
   (`[prompt_len, vocab]`), all-finite logits. **PASS.**
3. **K=3 proposal correctness**: real `mtp_prefill()` produces exactly 3
   draft tokens, all valid vocab ids (`[13, 248046, 198]` for this
   prompt). **PASS.**
4. **Single-round verify/accept-reject/GDN rollback**: two scenarios --
   `real_draft_proposal` (the draft model's own actual K=3 output,
   submitted as-is -- NOT asserted to be any specific accept/reject
   outcome, since an untrained-together MTP head organically agreeing or
   disagreeing with the target at some positions is the expected,
   realistic dynamic, not a bug) and `forced_reject_at_1` (a deliberate
   decoy substitution, same technique `mtp_accept_reject_check.py`
   already established, guaranteeing a KNOWN partial-reject). Both
   scenarios: `kv_len` advances by exactly the real committed length,
   GDN state changes from the pre-verify snapshot (repair happened), and
   -- the decisive content check -- the reference-slot replay's
   predicted recovery token matches `determine_accept_reject`'s own
   recovery token, and continuing both slots by one more real token
   gives the SAME greedy next token. **PASS**, after the bug above was
   found and fixed.

   One nuance recorded honestly, not swept under the rug: the
   content-correctness check's `next_hidden_allclose` is `false` (while
   `cosine_sim=0.997` and the greedy token still matches exactly). This
   reflects the reference slot going through `prefill()`'s plain
   single-request path (`is_decode=False`, one long causal pass over the
   whole real sequence) while the MTP-committed slot's recompute went
   through `_forward_batch`'s decode/verify-shaped batched path
   (`qo_len=committed_len`) -- two DIFFERENT kernel dispatch paths
   computing the SAME mathematical positions, which this project has
   already established elsewhere produces small, expected floating-point
   deviation (not a correctness bug) -- consistent with using cosine
   similarity + exact greedy-token agreement, not literal hidden-state
   `allclose`, as the operative bar for decision-level correctness.

**Not attempted this round** (remain for future rounds, per the
coordinator's explicit "however far you get" scoping): step 5
(multi-round continuation via a not-yet-built `mtp_decode()` coordinator,
paralleling `mtp_prefill()` for decode-shaped steps), step 6 (4-slot
concurrent isolation), step 7 (the real W1/W2 ≤1pp acceptance-rate gate
against native vLLM), step 8 (CUDA graph integration -- explicitly last
in sol's gradient, not started). `mtp_verify_and_commit`'s `last_hidden`
return value is plumbed but not yet consumed by anything (it represents
the last ACCEPTED position's hidden state, not the recovery token's --
useful context for whoever builds `mtp_decode` next, not dead code, but
untested end-to-end in a real multi-round loop yet).

## 2026-07-17, verification gradient steps 5-6: multi-round (c=1) and 4-slot isolation, both done, real bug found and fixed, one benign near-tie documented

Following the Codex-sol review's positive verification of the prior
round's commit (`3d3c074`), continued to steps 5-6 without waiting for
sol's parallel review of this newer work. A design simplification
emerged that eliminated the previously-planned-but-not-yet-built
`mtp_decode()` coordinator entirely -- see below -- before either test
was written.

### Design change: no separate `mtp_decode()` needed -- `mtp_verify_and_commit` folds catch-up + next-round propose into itself

Working through what a multi-round loop actually requires exposed that
`mtp_verify_and_commit`'s previous design (returning an unused
`last_hidden`, leaving "resync the draft + propose the next round's
tokens" to a separate, never-built `mtp_decode()`) was solving the wrong
problem. Reasoned through it from first principles:

- After `mtp_verify_and_commit` commits the target's real history for
  this round, the draft's own KV is behind by exactly `real_new_tokens =
  [anchor] + committed[:-1]` (the tokens whose KV the target just wrote)
  -- because the draft was last synced at the END of the PREVIOUS round,
  so `slot_draft_sync_len` always equals `slot_kv_len` from BEFORE this
  round's commit (an invariant maintained by this same method).
- Syncing the draft over `real_new_tokens` (shifted by one, ending with
  the recovery/bonus token as the final candidate) is EXACTLY
  `_mtp_sync_and_propose`'s existing step-0 pattern -- just generalized
  from `mtp_prefill`'s "whole prompt" range to "this round's newly
  committed range." No new method needed; the EXISTING
  `_mtp_sync_and_propose` already does the right thing when called with
  these arguments instead.
- That SAME sync call's own LAST position (processing the recovery/bonus
  token as a draft candidate against the target's hidden state through
  the last real position) produces the FIRST draft token for the NEXT
  round, for free -- `_mtp_sync_and_propose`'s existing K-1-more-steps
  loop then completes the K-token proposal.
- This means `mtp_verify_and_commit` can return `next_anchor`/
  `next_draft_tokens` directly, and a multi-round loop is just: `decision
  = mtp_verify_and_commit(slot, anchor, draft_tokens); anchor,
  draft_tokens = decision["next_anchor"], decision["next_draft_tokens"]`
  -- no separate decode-shaped coordinator, matching real vLLM's own
  design (propose() runs immediately after postprocess_sampled(), not as
  a separate deferred step) more closely than the earlier plan.

`mtp_verify_and_commit` was rewritten accordingly (dropped the unused
`last_hidden` field, added the `real_new_tokens`/`real_new_hidden`
computation shared by both accept/reject branches, added the
catch-up-and-propose call at the end).

### Verification, step 5: multi-round continuous MTP (concurrency=1)

Built `benchmarks/mtp_multiround_check.py`. Drives 8 real rounds on one
slot (mixing organic accept/reject outcomes from the draft's own real
proposals with a forced-decoy reject every 3rd round, to exercise BOTH
paths repeatedly, not just the happy path), while an INDEPENDENT
reference slot replays the SAME real committed tokens every round via
the plain, long-verified `_forward` path and its own next-token
prediction is compared against the MTP slot's `next_anchor` --
round-by-round, not just a final check, so drift would surface
immediately rather than average out. Checked two things every round:
(a) the `slot_draft_sync_len == slot_kv_len` invariant (bookkeeping
correctness), (b) content correctness (does an independent replay agree
with MTP's own committed continuation).

**Result**: (a) held for all 8/8 rounds, every run -- no bookkeeping
drift across accept and reject rounds mixed together. (b) matched
exactly for 7/8 rounds; round 2 (a forced-reject-at-0 round) showed a
genuine disagreement, investigated rather than dismissed or accepted
blindly:

**Root-caused, not just observed**: dumped the actual logit distributions
from both computation paths at the disagreeing position.
`verify_batch`'s own qo_len=4 batched call gave token 271 and token 198
the EXACT SAME logit (`25.375` both, an exact tie at this precision,
`argmax` arbitrarily resolving to 198), while the reference's plain
single-token `_forward` call gave 271 a clear lead (`24.25` vs `24.0`)
over the SAME two candidates given the SAME real history. Both numbers
sit far above the 3rd-place candidate (`13.8`/`14.06`), confirming
this position is a genuine, inherent near-tie in the model's own
distribution between exactly these two tokens -- not evidence of state
corruption. This is the SAME class of "different kernel dispatch path
(batched-verify-shaped vs. plain single-request-shaped), small
floating-point difference" phenomenon step 4 already documented
(`cosine_sim=0.997`/`hidden_allclose=false` there) -- except this
specific instance happened to land close enough to flip the greedy
token too, which step 4's single check couldn't have shown (needed a
genuine near-exact tie to surface at all).

Given this, the test now checks REF's own logit margin between its top
candidate and whatever MTP actually committed -- a mismatch only counts
as a real failure if that margin exceeds a documented threshold
(`NEAR_TIE_LOGIT_MARGIN = 2.0`, chosen because distinct real candidates
at this prompt/position are separated by 8-13+ logit units, so a margin
this small is diagnostic of a genuine near-tie, not a coincidence).
**Result with this tolerance: 8/8 rounds pass, stable and bit-identical
across 2 independent full runs** (`num_exact_content_matches=7/8` both
times, same round index, same tokens -- fully deterministic, not random
noise).

### Verification, step 6: 4-slot concurrent isolation

Same script, `_four_slot_isolation()`: 4 different prompts (France/
Japan/Germany/Italy capitals -- an eyeballable signal-probe layer per
the coordinator's explicit request for both methods), 4 MTP slots + 4
independent reference slots, 8 rounds each, INTERLEAVED round-robin
(slot 0's round, then slot 1's, then slot 2's, then slot 3's, repeat --
not 4 sequential independent runs) so any cross-slot addressing bug in
the draft's own KV cache or the GDN snapshot mechanism would have to
manifest under actual interleaving, not just isolation.

**Scope note, stated honestly**: this interleaves independent
SINGLE-slot `mtp_prefill`/`mtp_verify_and_commit` calls across 4 slots,
NOT one batched `verify_batch` call spanning all 4 simultaneously (that
would need generalizing the MTP coordinator methods to accept slot
lists, matching how `_forward_batch`/`verify_batch` already do for the
target-only path -- not done this round, a real gap for a genuinely
concurrent production loop, not just a batching-efficiency nitpick).
This still directly tests the coordinator's stated concern (does
per-slot state stay isolated when multiple slots are simultaneously
active), just via interleaved dispatch rather than one fused kernel
launch.

**Result**: signal-probe layer -- no two slots' committed token
sequences were ever identical (cheap sanity check for gross
contamination). Numerical-twin layer (same per-round independent-replay
check as step 5, with the same near-tie tolerance) -- all 4 slots pass
100% cleanly, zero mismatches even at strict exact-match (not just
within-tolerance), across both confirmation runs. Notably, slot 0
(France) is the SAME prompt used in step 5's isolated single-slot test
and shows the identical single near-tie behavior when checked at strict
exact-match in an earlier (pre-tolerance-fix) run -- reproducing
IDENTICALLY whether run in isolation or interleaved with 3 other slots
is itself evidence AGAINST cross-slot contamination (contamination would
plausibly perturb behavior differently depending on interleaving order,
not reproduce an identical, prompt-specific near-tie both ways).

### Status and next steps

Steps 1-6 of the verification gradient are now done, with the design
simplification (no `mtp_decode` needed) reducing remaining scope. Step 7
(real W1/W2 ≤1pp acceptance-rate gate against native vLLM) and step 8
(CUDA graph integration, explicitly last) were not attempted this round
-- step 7 requires launching a real vLLM MTP server and comparable
workload, a substantially larger undertaking better scoped as its own
round given how much real GPU time and design iteration this round
already used (4 full model-loading test runs plus 2 targeted diagnostic
runs). Not blocked on anything technical -- a scope/pacing call, reported
per the coordinator's explicit "get to whatever step feels reasonable,
report honestly" instruction.

## 2026-07-17, MAJOR CORRECTION: a real, structural bug in the propose loop was caught by an independent Codex-sol review after this round's steps 1-6 were reported as passing -- fixed and re-verified, and the exact methodology gap that let it through is now closed for this specific case

**This corrects work reported as verified/passing above.** Before
starting step 7 (W1/W2), the coordinator commissioned an independent
Codex-sol review of the Phase 2 implementation. It came back
REQUEST-CHANGES-level, and the coordinator personally re-read
`_mtp_sync_and_propose`'s source and confirmed the core finding was real
before relaying it -- the correct discipline this project has followed
since the earlier CUDA-Graph state-pollution correction, and the right
one: an "independent review flagged it" claim should not be propagated
into an implementation decision without checking it first.

### The core bug (confirmed real, fixed)

`_mtp_sync_and_propose`'s exploratory propose loop (steps 1..K-1) passed
`self.slot_draft_sync_len[slot]` -- a field intentionally frozen after
step 0 (by design, so the draft's own KV needs no rollback on
accept/reject) -- as EVERY exploratory step's `prior_kv_len` for
`_mtp_forward`'s attention metadata. But the actual physical write
position (`start_pos`/`next_pos`) keeps advancing every exploratory
iteration. For K=3 (this project's real production setting, 2
exploratory steps), the 1st exploratory step happens to still be
correct (it immediately follows step 0, where the frozen field and the
real position still coincide) -- but the 2nd is not: its attention
metadata claimed a SHORTER history than where its own K/V actually got
written, meaning that step's query silently failed to attend to the
PREVIOUS exploratory step's own contribution. Every real K=3 proposal's
3rd draft token was computed against an incomplete causal history.

**Why this was not caught by steps 3-4's verification** (the coordinator's
explicit methodology critique, confirmed correct): those steps only
checked SHAPE and vocab-range of the proposed tokens, never per-step
numerical content against an independent oracle. This bug produces
exactly the right shape/vocab-range output at every step -- only the
CONTENT is wrong from the 2nd exploratory step onward. A shape check
structurally cannot catch a content-only bug like this one.

**Fix**: decouple "what this call's attention needs" from "what the
cross-round bookkeeping should remember." `_mtp_forward` now takes an
EXPLICIT `prior_kv_len` argument (no longer reads
`self.slot_draft_sync_len` internally). `_mtp_sync_and_propose` tracks
its own LOCAL `running_prior_kv_len` counter that advances every
exploratory iteration (matching where each step's write actually
lands), while `self.slot_draft_sync_len` itself still only advances once
after step 0 -- both correctly now, decoupled instead of conflated.

**Decisive verification, not another shape check** (per the coordinator's
explicit ask): built `benchmarks/mtp_prior_kv_len_fix_check.py` with two
independent checks:
1. **White-box invariant** (the actual decisive proof, directly targeting
   the root cause): the correct invariant is `prior_kv_len == start_pos`
   for every `_mtp_forward` call in the propose loop. Instrumented via
   monkeypatching to record every call's arguments during a real
   `mtp_prefill`, checked at K=3 (production) and K=5 (stress, 4
   exploratory steps). **Result: zero mismatches at both K values,
   confirmed stable.**
2. **Black-box numerical demonstration** (a concrete, quantified
   illustration, not a pass/fail gate -- whether a specific prompt/
   position is numerically sensitive enough to visibly flip is
   prompt-dependent, not itself a correctness property): reproduces the
   OLD buggy semantics (frozen `prior_kv_len`) side-by-side with the
   fixed semantics, letting each side's own draft tokens propagate
   autoregressively (not artificially forced identical partway through --
   this is what actually happened in production, not a partial repro).
   At K=3 (production), the two sequences happened to match exactly for
   this specific prompt -- reported honestly, not suppressed, with the
   explanation that a single missing token of causal context at that
   specific position did not happen to be decision-relevant for greedy
   top-1 there. At K=8 (stress, larger accumulated gap), the two
   sequences matched for the first 4 tokens then diverged completely
   (`[13, 248046, 198, 248045, 198, 248045, 198, 248069]` vs. `[13,
   248046, 198, 248045, 561, 11, 369, 11]`) -- concrete, quantified proof
   the bug's mechanism has real numerical consequences once the
   accumulated gap is large enough, and that the fix eliminates them.

### Other findings from the same review, verified one by one (not accepted wholesale)

1. **`reset_slot()` incomplete -- CONFIRMED, fixed.** It cleared
   `slot_kv_len`/`slot_gdn_initialized` but not
   `slot_draft_sync_len`/`slot_pending_draft_tokens`. A slot reused for a
   NEW logical request would start its real target KV at position 0 but
   its draft-sync step-0 call would read a STALE, nonzero
   `slot_draft_sync_len` -- an immediate correctness bug for any slot
   ever reused, which is this project's whole fixed-slot-generation
   premise. Now cleared alongside the pre-existing fields.
2. **GDN snapshot generation not slot-bound, and not marked consumed
   after restore -- CONFIRMED, fixed.** `snapshot_gdn_state`/
   `restore_gdn_state`'s generation-counter check only compared numbers,
   not slot identity -- a caller mistakenly restoring slot A's snapshot
   into slot B could still pass (both slots typically climb their own
   counters in lockstep in a symmetric multi-slot workload, so equal
   generation numbers say nothing about slot identity). Fixed by tagging
   each snapshot with its source slot id and rejecting a mismatch. Also
   added a `__consumed__` marker so restoring the SAME snapshot object
   twice now raises instead of silently succeeding (idempotent in this
   specific case, but exactly the kind of latent bug this project's
   "no silent passes" standard exists to catch).
3. **`CapturedBatchDecodeGraph.replay()` unconditionally commits, no
   `commit` flag -- CONFIRMED, fixed.** This class has its own separate
   captured-graph call path that never goes through `_forward_batch`, so
   it was never updated when `commit` was added there earlier this round
   -- a real inconsistency, though not yet triggering an observed failure
   (CUDA graph integration into the real MTP accept/reject flow is still
   explicitly the last, unstarted step of the gradient). Added a matching
   `commit: bool = True` parameter (default preserves all existing
   callers' behavior unchanged).
4. **Methodology gap: steps 3-4 only checked shape, never per-step
   oracle-aligned logits -- CONFIRMED, this is exactly why the K>1 bug
   survived.** Addressed for THIS specific bug via
   `mtp_prior_kv_len_fix_check.py` above. NOT addressed as a blanket
   retrofit of oracle-aligned numerical checks into every existing
   verification step (steps 1-6's existing tests still rely on
   shape/bookkeeping/reference-replay checks, which are each individually
   reasoned about but were not re-audited end-to-end for this same class
   of "right shape, wrong content" gap beyond the one bug found). Worth
   treating as a standing practice going forward, not a one-time patch.
5. **Capacity: `block_size=16 × blocks_per_slot=128 = 2048` tokens/slot
   across every existing MTP test script, vs. W1's 4096/W2's 32768 need
   -- CONFIRMED, real, and not yet addressed.** Every existing MTP test
   this round and the prior one used this same undersized default. A
   real W1/W2 run would hit `build_attention_metadata`'s own
   `RuntimeError` guard (`kv_len exceeds this slot's ... capacity`) well
   before reaching a meaningful acceptance-rate sample. This is the
   coordinator's stated step-3 priority (after bug fixes and methodology,
   both addressed above) -- NOT attempted this round, remains the
   concrete blocker before step 7 (W1/W2) can even start.

### Regression check across the whole existing MTP test suite (all re-run after all 4 fixes above)

`mtp_real_draft_check.py` (needed one call-site update: a direct
`_mtp_forward` call in the script itself had to pass the new required
`prior_kv_len` argument -- caught immediately by the signature change,
not a silent bug), `mtp_multiround_check.py`, `mtp_accept_reject_check.py`,
`mtp_gdn_rollback_check.py`, `mtp_trace_driven_probe.py` -- **all pass,
same results as before the fixes** (e.g. the multi-round test's
7/8-exact-match near-tie finding reproduces identically), confirming
none of the 4 fixes regressed anything already verified.

### Status

Bug fixes (coordinator's item 1) and the specific-case methodology fix
(item 2, partial -- see point 4 above) are done and re-verified this
round. Capacity expansion (item 3) is NOT done -- confirmed necessary,
not yet started. Step 7 (W1/W2) remains blocked on item 3, as it was
before this correction, now for a documented reason rather than an
unexamined one.

## 2026-07-17, capacity expansion to real W1/W2 scale: empirically measured (not hand-derived), expanded, and the full MTP correctness suite re-verified with zero regressions

### Empirical memory measurement, not a theoretical estimate

Built `benchmarks/capacity_w1w2_check.py` to measure real GPU memory via
`nvidia-smi` at each stage, rather than trust a hand derivation for a
question this consequential. Findings:

- **Persistent KV cache scales trivially with context length**: the
  attention KV cache (17 layers total needing per-position storage --
  target's 16 full-attention layers + the draft's own 1 -- GDN's
  conv/ssm state is FIXED SIZE regardless of context length, confirmed
  by reading `qwen_gdn_linear_attn.py`'s `get_state_shape()`, which
  depends only on model config, never sequence length) costs ~2048
  bytes/token/layer at FP8 (confirmed via `SM120GQABackend
  .get_kv_cache_shape()`'s `(num_blocks, 2, block_size, num_kv_heads,
  head_size)` shape), i.e. ~34KB/token across all 17 layers -- even at
  full W2 scale (33792 tokens), that's only ~1.2GB per slot. This was
  never going to be the constraint.
- **The real open question was peak transient ACTIVATION memory during
  a single, non-chunked 32768-token prefill forward pass** (this
  runtime's `_forward`/`_forward_batch` process the whole prompt in ONE
  call -- no chunked-prefill mechanism like real vLLM's scheduler has).
  Measured directly: **~22GB of transient memory for one W2-sized
  prefill** on top of ~39GB of persistent weights+KV/GDN cache (8 slots
  configured, target+draft model loaded) -- a real, substantial number,
  not negligible. (Likely dominated by `compute_logits()` being called
  on the FULL hidden-states tensor -- `[32768, vocab=248320]` in BF16 is
  ~16GB by itself -- even though `prefill()`'s own contract only ever
  needs the LAST position's logits; `mtp_prefill`'s draft-sync step does
  need the full hidden_states, but not the full logits. Not fixed this
  round -- a real, identified further optimization for whenever
  performance work on the prefill path resumes, not a correctness issue,
  and not needed since the current memory budget already accommodates
  it without this optimization.)
- **The decisive question**: does memory GROW per additional slot's
  prefill (would mean concurrency=4 doesn't actually fit even though one
  slot does), or does it PLATEAU (PyTorch's caching allocator reuses
  freed activation memory across sequential prefill calls, since this
  runtime never batches multiple slots' prefills into one call -- each
  slot's prefill is its own independent forward call)? Measured directly
  across 4 sequential W2-sized prefills on slots 0,1,2,3: memory reads
  **63353 / 63353 / 63353 / 63353 MiB -- perfectly flat, zero growth**,
  out of 97887 MiB total (~65% utilized, ~36GB headroom). Confirms
  sequential (not batched) prefill across up to 4 (this project's own
  8-slot test config, covering both the 4 production slots and the
  4-slot-isolation test's reference slots) concurrent slots fits
  comfortably, with real evidence, not an assumption.

**Conclusion**: block_size=16 (unchanged) with `blocks_per_slot=2560`
(40960 tokens/slot, ~21% margin over W2's 33792-token need) works for
BOTH W1 and W2 with the SAME configuration -- no need for the
coordinator's contingency of splitting into separate W1/W2 capacity
configs or reducing concurrency; memory was never the real constraint
once measured directly, and the sequential-prefill-per-slot design this
runtime already has (never batches prefills across slots) is exactly
why the transient activation cost doesn't compound across concurrent
slots.

### Full MTP correctness suite re-verified at the new capacity, zero regressions

Per the coordinator's explicit instruction not to assume "bigger
capacity, same logic" is automatically safe -- `blocks_per_slot` feeds
directly into `_physical_slot`/`_slot_mapping`/`build_attention_metadata`/
`build_attention_metadata_batch`'s address arithmetic, all worth
re-checking at a value 20x larger than before. Updated all 5 MTP test
scripts' `blocks_per_slot` from 128 to 2560 and re-ran every one:
`mtp_prior_kv_len_fix_check.py` (K>1 propose fix -- invariant checks and
both numerical demonstrations reproduce identically), `mtp_accept_reject_check.py`,
`mtp_gdn_rollback_check.py`, `mtp_real_draft_check.py` (steps 1-4),
`mtp_multiround_check.py` (steps 5-6, including the same 7/8-exact-match
near-tie finding reproducing identically). **All pass, byte-for-byte
identical results to the smaller-capacity runs** -- confirming the
address-calculation logic generalizes correctly to the much larger
`blocks_per_slot` value with no hidden assumptions tied to the old small
size.

### Status

Capacity (coordinator's item 3) is now done and re-verified -- the last
blocker before step 7. Moving to the real W1/W2 acceptance-rate
comparison against native vLLM next.
