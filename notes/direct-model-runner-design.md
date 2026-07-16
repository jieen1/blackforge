# Direct model runner: design (2026-07-15/16)

Replaces the HTTP bridge (`runtime/vllm_bridge_backend.py`, commit `b28942c`)
with an in-process runner that owns GPU KV/GDN state directly, per the
project's re-prioritized main line (2026-07-16: attention-kernel tuning in
the sibling `sm120-flash-attention` project hit diminishing returns --
decode v2/prefill v2's "beats native" claims were both overturned -- so
development effort moved here).

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
