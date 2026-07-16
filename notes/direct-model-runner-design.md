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

**Concrete, NOT-yet-explained lead**: after prefill, `ssm_state` (the GDN
recurrent state) is mostly non-zero (593152/786432) as expected for a real
computation, but `conv_state` (the GDN causal-conv1d state) is **entirely
zero** (0/30720) for slot 0's layer -- i.e. one of the two GDN state tensors
persists correctly and the other does not, for the same layer, same call,
same metadata object. `qwen_gdn_linear_attn.py`'s `_forward_core` passes
`conv_states=conv_state, cache_indices=non_spec_state_indices_tensor` to
`causal_conv1d_fn` for the prefill (`num_prefills>0`) branch, using the same
`non_spec_state_indices_tensor=[slot]` this runner already sets correctly (verified
by reading the code back). `conv_state` itself is
`self.kv_cache[0]` transposed via `.transpose(-1,-2)` when
`is_conv_state_dim_first()` is `False` (confirmed: `get_conv_state_layout()`
defaults to `"SD"`, so this transpose does happen) -- a non-contiguous view.
Whether `causal_conv1d_fn` correctly writes through that non-contiguous view
in general (it must, since real vLLM serving allocates conv_state the exact
same way and works) or whether something about *this* call's metadata
specifically defeats it is the open question. **This may or may not be the
actual cause of the wrong final output** (conv_state not persisting would
only visibly matter for a *second* forward call needing that history, not
necessarily for prefill's own single-call output correctness) -- it is
reported as a concrete, reproducible anomaly, not a confirmed root cause.

**Next debugging steps** (for whoever picks this up next, session or not):
1. Compare intermediate hidden states layer-by-layer against a known-good
   reference (e.g. hook the real HTTP-bridge-driven server's model at the
   same prompt, if that's still feasible, or run this same model through
   plain `transformers`/eager HF forward as an independent oracle) to
   bisect whether the corruption starts in an attention layer, a GDN layer,
   or earlier (embedding/RoPE).
2. Try isolating GDN entirely: temporarily patch/monkeypatch a `has_initial_state`-independent
   sanity check, or single-step through `_forward_core` with prints, to see
   whether `mixed_qkv`/`b`/`a` (the conv1d's own inputs) are already wrong
   *before* reaching `causal_conv1d_fn` -- would point at the projection
   layers or `has_initial_state`/chunking metadata instead of the conv-state
   write itself.
3. Try running with GDN's `has_initial_state` fields flipped or with
   `chunk_indices`/`chunk_offsets` recomputed a different way, to see if the
   FLA chunked-prefill path (vs. a hypothetical non-chunked path) is where
   the divergence is -- `FLA_CHUNK_SIZE=64` vs. a 5-token prompt means
   exactly one chunk, which should be the simplest possible case to get
   right, so if even *that* is wrong, the chunk metadata plumbing itself
   (not chunk-boundary edge cases) is suspect.
4. Once a signal-probe run finally produces plausible text, re-run the
   *exact* signal-probe prompts from the HTTP-bridge round
   (`benchmarks/real_forward_smoke.py`'s marker-code prompts) for a
   like-for-like comparison, and specifically re-test slot release + reuse
   (`reset_slot`) under this direct ownership model, since that is the
   scenario this whole effort exists to get right (vLLM issue #37554's risk
   class) -- a mechanism that merely "runs" without that guarantee holding
   is not yet done.
