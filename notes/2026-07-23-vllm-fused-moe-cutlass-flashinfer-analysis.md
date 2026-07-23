# vLLM fused_moe — CUTLASS / FlashInfer Backend Technical Analysis

Source: `/home/bot/vllm/vllm/model_executor/layers/fused_moe/` (vLLM, snapshot 2026-07-10)
Focus: the CUTLASS and FlashInfer kernel dispatch paths. All line numbers refer to
that tree.

## 0. Two execution universes

vLLM MoE has two parallel execution paths:

1. **Monolithic / functional Triton path** — `fused_moe.py`. A single
   `fused_experts_impl` function wires quantize → align → GEMM1 → activation →
   GEMM2 → sum directly with Triton kernels. Used by the unquantized / fp8 /
   wna16 Triton backends.
2. **Modular kernel path** — `modular_kernel.py`. A combinator that pairs a
   **PrepareAndFinalize** strategy (quantize + token dispatch/combine, incl.
   all2all) with an **Experts** strategy (the grouped-GEMM compute). This is the
   path used by **CUTLASS** and **FlashInfer** backends.

The CUTLASS/FlashInfer backends live almost entirely in the modular path.

---

## 1. fused_moe.py — Triton data flow (reference baseline)

Entry: `fused_experts` (L1529) → `torch.ops.vllm.fused_experts` →
`fused_experts_impl` (L1592). Complete data flow inside `fused_experts_impl`:

| Stage | Call | Line |
|---|---|---|
| input quantize | `moe_kernel_quantize_input(A=hidden_states, ...)` | ~L1700 |
| expert assignment / block align | `_prepare_expert_assignment(...)` → `moe_align_block_size(...)` | L1480 / L1520 |
| **GEMM1 (gate_up)** | `dispatch_fused_moe_kernel(qhidden_states, w1, intermediate_cache1, ...)` | L853 |
| activation (SiLU) | `apply_moe_activation(activation_enum, intermediate_cache2, cache1.view(-1,N))` | — |
| re-quantize | `moe_kernel_quantize_input(A=intermediate_cache2, ...)` | — |
| **GEMM2 (down)** | `dispatch_fused_moe_kernel(qintermediate_cache2, w2, intermediate_cache3, ..., top_k=1)` | L853 |
| finalize / reduce | `ops.moe_sum(intermediate_cache3, out_hidden_states)` | — |

Kernel dispatch chain:
- `dispatch_fused_moe_kernel` (L853) branches:
  - wna16 (int8/int4 w*a16) + block_shape → `invoke_fused_moe_wna16_cuda_kernel`
    (L588) or `invoke_fused_moe_wna16_triton_kernel` (L646), gated by
    `should_moe_wna16_use_cuda` (L1228).
  - otherwise → `invoke_fused_moe_triton_kernel` (L736).
- `invoke_fused_moe_triton_kernel` computes the grid
  `cdiv(EM, BLOCK_M) * cdiv(N, BLOCK_N)` and launches the Triton kernel
  `fused_moe_kernel[grid](...)` (kernel defined at L295). `EM` comes from
  `sorted_token_ids` (the block-aligned permutation).

Key signatures:
```python
def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                  activation=MoEActivation.SILU, apply_router_weight_on_input=False,
                  global_num_experts=-1, expert_map=None, quant_config=None)  # L1529

def dispatch_fused_moe_kernel(A, B, C, A_scale, B_scale, B_zp, topk_weights,
                  sorted_token_ids, expert_ids, num_tokens_post_padded,
                  mul_routed_weight, top_k, config, compute_type, ...)        # L853
```
Note: GEMM2 is called with `top_k=1` and `mul_routed_weight = not
apply_router_weight_on_input`, i.e. router weights are folded in during the down
projection, then `moe_sum` reduces across the top-k dimension.

---

## 2. modular_kernel.py — the prepare/finalize + experts combinator

### Interfaces
- `FusedMoEPrepareAndFinalize` (L181, ABC). Capability properties:
  `activation_format` (L202), `topk_indices_dtype` (L210),
  `max_num_tokens_per_rank` (L220), `num_dispatchers` (L231),
  `output_is_reduced` (L235), `supports_async` (L242).
- `FusedMoEPrepareAndFinalizeModular` (L258): the two abstract verbs:
  ```python
  def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map,
              apply_router_weight_on_input, quant_config, defer_input_quant
              ) -> PrepareResultType   # L265
     # returns (a1q, a1q_scale, ExpertTokensMetadata|None, topk_ids|None, topk_weights|None)
  def finalize(self, output, fused_expert_output, topk_weights, topk_ids,
              apply_router_weight_on_input, weight_and_reduce_impl) -> None   # L355
  ```
  Async variants `prepare_async` (L302) / `finalize_async` (L379) return
  `(hook, receiver)` pairs for DBO (dual-batch overlap).
- `FusedMoEPrepareAndFinalizeMonolithic` (L422): operates on `router_logits`
  instead of topk ids (routing done inside the kernel).
- `FusedMoEExperts` (L472, ABC) with a battery of static capability predicates
  `_supports_*` (device L586, quant scheme L604, activation L612, parallel config
  L620, routing method L630, shape L657, batch invariance L665) rolled up into
  `is_supported_config` (L537) — this is what the oracle queries.
- `FusedMoEExpertsModular` (L763): `moe_problem_size` (L773),
  `workspace_shapes` (L823), `adjust_N_for_activation` (L864),
  `finalize_weight_and_reduce_impl` (L898), and the abstract compute verb:
  ```python
  def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
            activation, global_num_experts, expert_map, a1q_scale, a2_scale,
            workspace13, workspace2, expert_tokens_meta,
            apply_router_weight_on_input) -> None   # L902
  ```
- `TopKWeightAndReduce` (L118): strategy for applying router weights + reducing
  the (M, topk, K) expert output to (M, K).

### Orchestration
- `FusedMoEKernelModularImpl` (L1025). `apply` (L1371) runs:
  1. `_prepare` (L1118) — wraps `prepare_finalize.prepare`, handles padding-token
     skip (`VLLM_MOE_SKIP_PADDING`) and async/DBO.
  2. `_fused_experts` (L1222) — computes problem size via `moe_problem_size`,
     allocates `workspace13/workspace2/fused_out` via `_allocate_buffers` (L1042,
     workspace13 is shared between GEMM1 and GEMM2 output), then calls
     `self.fused_experts.apply(...)`.
  3. `_finalize` (L1303) — wraps `prepare_finalize.finalize` with the experts'
     `finalize_weight_and_reduce_impl()`.
- `FusedMoEKernel` (L1529) is the public facade: `apply` (L1640, modular) and
  `apply_monolithic` (L1608). It selects `FusedMoEKernelModularImpl` vs
  `FusedMoEKernelMonolithicImpl` (L1468) in `_post_init_setup` (L1590).

---

## 3. prepare_finalize/ — quantize + dispatch / combine

Files: `batched.py`, `deepep_ht.py`, `deepep_ll.py`, `deepep_v2.py`,
`flashinfer_nvlink_one_sided.py`, `flashinfer_nvlink_two_sided.py`, `mori.py`,
`naive_dp_ep.py`, `nixl_ep.py`, `no_dp_ep.py`.

### no_dp_ep.py (single GPU / TP-only — the common FLASHINFER_CUTLASS case)
`MoEPrepareAndFinalizeNoDPEPModular`:
- `prepare`: optionally applies router weight on input (topk=1), then
  `_quantize_input` → `moe_kernel_quantize_input(...)` (or defers quant to the
  expert kernel when `defer_input_quant`). Returns `(a1q, a1q_scale, None, None,
  None)` — **no permutation here**; the CUTLASS/FlashInfer expert kernel does its
  own token routing.
- `finalize`: delegates to `TopKWeightAndReduceContiguous` (i.e. `ops.moe_sum`
  style reduce) unless the experts already reduced (NoOP).
- `activation_format = Standard`, `output_is_reduced = False`, `num_dispatchers = 1`.

### FlashInfer NVLink (EP over NVLink, MNNVL)
`FlashInferNVLinkOneSidedPrepareAndFinalize` (flashinfer_nvlink_one_sided.py):
- ctor grabs `get_ep_group().device_communicator.all2all_manager` and calls
  `all2all_manager.initialize(max_num_tokens, top_k, num_experts, hidden_size,
  dispatch_dtype_bytes_per_elem, dispatch_scale_bytes_per_token)`.
- `prepare`: quantize input (scale swizzle delayed until after comm), pack
  payloads `[a1q, (a1q_scale), topk_ids, topk_weights]`, then
  `all2all_manager.moe_alltoall.dispatch(token_selected_experts=topk_ids,
  input_payloads=..., runtime_max_tokens_per_rank=..., invalid_token_expert_id=-1,
  expert_id_payload_index=...)`. For nvfp4 it applies
  `nvfp4_block_scale_interleave` to the received scales (CUTLASS layout).
- `finalize`: reshape to `(ep_size, max_tokens_per_rank, hidden)` and
  `moe_alltoall.combine(payload=..., runtime_max_tokens_per_rank=...)`,
  `output.copy_(combined)`.
- `output_is_reduced = True`, `topk_indices_dtype = int32`.

`FlashInferNVLinkTwoSidedPrepareAndFinalize` (flashinfer_nvlink_two_sided.py):
same idea via helper functions `flashinfer_alltoall_dispatch` /
`flashinfer_alltoall_combine`; `output_is_reduced = True`.

---

## 4. experts/ — the grouped-GEMM compute

Files (CUTLASS/FlashInfer-relevant): `cutlass_moe.py`, `flashinfer_cutlass_moe.py`,
`flashinfer_b12x_moe.py`, `flashinfer_cutedsl_moe.py`,
`flashinfer_cutedsl_batched_moe.py`, `triton_cutlass_moe.py` (plus triton,
deep_gemm, marlin, trtllm, rocm, cpu, xpu variants).

### FlashInferExperts — the FLASHINFER_CUTLASS backend (flashinfer_cutlass_moe.py)
- Class `FlashInferExperts(mk.FusedMoEExpertsModular)` (L62).
- Device gate `_supports_current_device` (L133): CUDA and (SM90 **or** SM100
  family **or** SM120 family; SM110 excluded — flashinfer#3134) and
  `has_flashinfer_cutlass_fused_moe()`.
- Quant schemes `_supports_quant_scheme` (L149): `(None,None)` and fp8
  static-per-tensor on SM90+; mxfp4 / fp8 128-block on SM90; mxfp4+mxfp8 and
  nvfp4 on SM100+.
- `finalize_weight_and_reduce_impl` → `TopKWeightAndReduceNoOP` (L207): **the
  FlashInfer kernel applies router weights and reduces internally**, so finalize
  is a no-op.
- `apply` (L248): builds `quant_scales` per scheme and makes a **single fused
  call**:
  ```python
  flashinfer_cutlass_fused_moe(
      input=hidden_states,
      token_selected_experts=topk_ids.to(torch.int),
      token_final_scales=topk_weights,
      fc1_expert_weights=fc1_expert_weights,   # w1 (gate_up), .view(torch.long) for nvfp4
      fc2_expert_weights=fc2_expert_weights,   # w2 (down)
      fc1_expert_biases=..., fc2_expert_biases=...,
      swiglu_alpha=..., swiglu_beta=..., swiglu_limit=...,
      output=output, output_dtype=self.out_dtype,
      quant_scales=quant_scales, input_sf=a1q_scale,
      tp_size=..., tp_rank=..., ep_size=..., ep_rank=...,
      activation_type=ActivationType.Swiglu,    # from MoEActivation
      use_deepseek_fp8_block_scale=..., use_mxfp8_act_scaling=...,
      use_w4_group_scaling=...)
  ```
  This is the whole gate_up → SwiGLU → down → weighted-reduce pipeline in one
  library call (topk permute is internal to FlashInfer).
- Dispatch chain: `flashinfer_cutlass_fused_moe` is a lazy wrapper
  (`vllm/utils/flashinfer.py:122`) around `flashinfer.fused_moe.cutlass_fused_moe`.
  Availability via `has_flashinfer_cutlass_fused_moe()` (flashinfer.py:263),
  requiring `cutlass_fused_moe`, `fp4_quantize`, `nvfp4_block_scale_interleave`,
  `trtllm_fp4_block_scale_moe`.
- nvfp4 scale ordering passed as `quant_scales = [a1_gscale, w1_scale(int32),
  g1_alphas, a2_gscale, w2_scale(int32), g2_alphas]`.

### CutlassExperts — vLLM-native CUTLASS grouped GEMM (cutlass_moe.py)
Variants: `CutlassExpertsFp8` / `CutlassBatchedExpertsFp8` (L403/L447),
`CutlassExpertsFp4` (L678), `CutlassExpertsMxfp4` (L990), `CutlassExpertsW4A8Fp8`
(L1250). The FP8 compute is `run_cutlass_moe_fp8` (L53):

Two token layouts:
- **Standard (non-batched)**: input `[total_tokens, hidden]`, must be permuted so
  each expert's tokens are contiguous.
- **Batched**: input `[num_experts, max_tokens_per_expert, hidden]`, already
  contiguous per expert (from a batched dispatch); no permute.

Standard FP8 dispatch chain:
```
moe_permute(a1q, a1q_scale, topk_ids, ...)                       # -> permuted a1q + expert_first_token_offset + inv_perm
ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(        # build per-expert (M,N,K) problem sizes
    expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab)
ops.cutlass_moe_mm(mm1_out, a1q, w1, a1q_scale, w1_scale,        # GEMM1 gate_up (grouped GEMM)
    expert_offsets, problem_sizes1, ab_strides1, ..., per_act_token, per_out_ch)
apply_moe_activation(activation, act_out, mm1_out)               # SiLU/SwiGLU
ops.scaled_fp8_quant(act_out, a2_scale, ..., output=quant_out)   # re-quantize activation
ops.cutlass_moe_mm(mm2_out, a2q, w2, a2q_scale, w2_scale,        # GEMM2 down (grouped GEMM)
    expert_offsets, problem_sizes2, ab_strides2, ...)
moe_unpermute(out, mm2_out, topk_weights, inv_perm, expert_first_token_offset)  # weighted scatter-reduce
```
- `swap_ab = a1q.size(0) <= 64` — a CUTLASS grouped-GEMM optimization that reduces
  padding for small M (decode).
- Batched path uses `ops.get_cutlass_batched_moe_mm_data(...)` to build
  `expert_offsets/problem_sizes` from `expert_num_tokens`, and writes output with
  a plain `output.copy_(mm2_out.reshape(...))` (combine handled by P/F).
- The grouped GEMM itself is `ops.cutlass_moe_mm` (a CUTLASS group GEMM over the
  per-expert problem sizes); FP4 uses `ops.cutlass_fp4_moe_mm` with
  `ops.scaled_fp4_experts_quant` / `ops.silu_and_mul_scaled_fp4_experts_quant`
  (run_cutlass_moe_fp4, L493) and `ops.shuffle_rows` for permute/unpermute.

### FlashInferB12xExperts — SM120/SM12x path (flashinfer_b12x_moe.py)
- Class at L30; `apply` (L248) runs a CuteDSL SM12x fused-MoE wrapper:
  `wrapper.run(x, w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale, w2_weight,
  w2_weight_sf, w2_alpha, token_selected_experts, token_final_scales)`.
- Availability `has_flashinfer_b12x_moe()` requires
  `flashinfer.fused_moe.b12x_fused_moe` + `convert_sf_to_mma_layout`;
  `has_flashinfer_b12x_gemm()` looks for `Sm120B12xBlockScaledDenseGemmKernel`.
- **Excluded from auto-selection** (see §7) — opt in with
  `moe_backend="flashinfer_b12x"`.

---

## 5. router/ — topk routing

Files: `base_router.py`, `fused_moe_router.py` (ABC), `fused_topk_router.py`,
`grouped_topk_router.py`, `fused_topk_bias_router.py`, `gate_linear.py`,
`router_factory.py`, plus aiter/custom/zero/simulator routers.

- Public entry `FusedMoERouter.select_experts(hidden_states, router_logits,
  topk_indices_dtype, input_ids)` (fused_moe_router.py:45) → `_select_experts`
  → `_compute_routing`, then EPLB mapping + dtype conversion (base_router.py
  `_select_experts` L260).
- `FusedTopKRouter._compute_routing` → `fused_topk` (fused_topk_router.py:69):
  allocates `topk_weights[M,topk]`, `topk_ids[M,topk]`, then a **single fused CUDA
  kernel** `ops.topk_softmax(...)` (or `ops.topk_sigmoid(...)`), which fuses
  softmax + top-k selection + optional renormalize over the `[M, num_experts]`
  logits.
- `grouped_topk` (grouped_topk_router.py:80): for grouped expert selection
  (DeepSeek-style). sigmoid → fully fused `ops.grouped_topk`; softmax →
  `torch.softmax` then `ops.grouped_topk`.
- **Where it runs / overhead**: routing happens in the runner *before*
  `forward_modular` (`runner/moe_runner.py:573`). It is one elementwise/reduce
  kernel over `[M, E]` logits — O(M·E) cheap FLOPs and a small top-k sort. It is
  negligible next to the two expert GEMMs (which are O(M·topk·N·K) each) and next
  to all2all/permute data movement. The router cost only becomes visible at very
  small batches / huge E, or when it forces a sync; the fused kernel avoids that.

---

## 6. Token permutation & block alignment

### moe_permute_unpermute.py (used by vLLM-native CUTLASS experts)
- `moe_permute(hidden_states, a1q_scale, topk_ids, n_expert, n_local_expert,
  expert_map, permuted_hidden_states, scratch)` (~L117): expands each token topk
  times and sorts rows so each expert's tokens are contiguous. Backed by
  `torch.ops._moe_C.moe_permute` (no scratch) or `moe_permute_with_scratch`
  (reuses `MoEPermuteScratch` buffers, L10). Returns `(permuted_hidden_states,
  a1q_scale, expert_first_token_offset, inv_permuted_idx, permuted_idx)`.
  `expert_first_token_offset` is what feeds the CUTLASS grouped-GEMM problem-size
  builder.
- `moe_unpermute(out, permuted_hidden_states, topk_weights, inv_permuted_idx,
  expert_first_token_offset)` (~L250): `torch.ops._moe_C.moe_unpermute` — fused
  router-weight multiply + scatter-reduce back to `[M, hidden]` (requires hidden
  dim 16B-aligned).
- `MoEPermuteScratch` preallocates sort workspace via
  `_moe_C.moe_permute_sort_workspace_size`.

### moe_align_block_size.py (used by the Triton path)
- `moe_align_block_size(topk_ids, block_size, num_experts, expert_map,
  pad_sorted_ids, ignore_invalid_experts)` (L12) → `ops.moe_align_block_size`.
  Pads each expert's token count up to a multiple of `block_size` so every Triton
  program processes a full `BLOCK_SIZE_M` tile. Returns `(sorted_token_ids,
  expert_ids, num_tokens_post_padded)`. Padding rows use an out-of-range token id
  that the kernel skips; with EP, out-of-rank experts are marked `-1`.
- `batched_moe_align_block_size(max_tokens_per_batch, block_size,
  expert_num_tokens)` (L104) is the batched-format equivalent.

The two mechanisms are alternatives: Triton uses `moe_align_block_size`
(block-sorted ids), vLLM-CUTLASS uses `moe_permute` (contiguous permute +
offsets), FlashInfer-CUTLASS does its own internal permute.

---

## 7. config.py / oracle — backend selection & FLASHINFER_CUTLASS

- `FusedMoEConfig.moe_backend: MoEBackend = "auto"` (config.py:1285). `MoEBackend`
  is a `Literal` in `vllm/config/kernel.py:122`:
  `auto, triton, deep_gemm, deep_gemm_mega_moe, cutlass, flashinfer_trtllm,
  flashinfer_cutlass, flashinfer_cutedsl, flashinfer_b12x, marlin, humming,
  triton_unfused, aiter, flydsl, hpc, emulation`.
- Selection is delegated to per-quant-scheme **oracles** in `oracle/`
  (`base.py` ABC `MoEKernelOracle`, plus `fp8.py, nvfp4.py, mxfp4.py, mxfp8.py,
  int8.py, int_wna16.py, w4a8*.py, unquantized.py`). The oracle contract:
  `get_priority_backends`, `backend_to_kernel_cls`, `map_backend`,
  `select_backend`, `make_kernel` (base.py L42–108).

### nvfp4 example (oracle/nvfp4.py)
- `NvFp4MoeBackend` enum (L38): FLASHINFER_TRTLLM, FLASHINFER_CUTLASS,
  FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_B12X, VLLM_CUTLASS,
  MARLIN, HUMMING, EMULATION.
- `backend_to_kernel_cls` (L67): `FLASHINFER_CUTLASS → [FlashInferExperts]`,
  `VLLM_CUTLASS → [CutlassExpertsFp4]`, `FLASHINFER_B12X → [FlashInferB12xExperts]`.
- `map_nvfp4_backend` (L145): user string `"flashinfer_cutlass" →
  NvFp4MoeBackend.FLASHINFER_CUTLASS`, `"cutlass" → VLLM_CUTLASS`, etc.
- `select_nvfp4_moe_backend` (L165):
  - **Explicit**: if `moe_backend != "auto"`, map it and validate via
    `experts_cls.is_supported_config(...)`; raise if unsupported. (swiglu_limit
    restricts to TRTLLM/CUTLASS/MARLIN which implement the clamp.)
  - **Auto** priority order: `FLASHINFER_TRTLLM → FLASHINFER_CUTEDSL →
    FLASHINFER_CUTEDSL_BATCHED → FLASHINFER_CUTLASS → VLLM_CUTLASS → MARLIN →
    HUMMING → EMULATION`. First whose `is_supported_config` passes wins.
    **FLASHINFER_B12X is deliberately excluded from auto** (upstream CUTLASS SM121
    MMA op guard) — opt in with `moe_backend="flashinfer_b12x"`.
- `make_nvfp4_moe_kernel` (L525): builds the combinator:
  ```python
  prepare_finalize = maybe_make_prepare_finalize(moe, quant_config, routing_tables,
                       allow_new_interface=True, use_monolithic=...)
  experts = experts_cls(moe_config=..., quant_config=..., [max_num_tokens=..., num_dispatchers=...])
  kernel = mk.FusedMoEKernel(prepare_finalize, experts)
  ```

### Prepare/Finalize selection (all2all_utils.py:117 `maybe_make_prepare_finalize`)
Driven by the all2all backend on `FusedMoEParallelConfig` (config.py:1020):
- no all2all + single GPU → `MoEPrepareAndFinalizeNoDPEP*` (no_dp_ep.py)
- no all2all + DP → naive AllGather+ReduceScatter (`naive_dp_ep.py`)
- `flashinfer_nvlink_two_sided` → `FlashInferNVLinkTwoSidedPrepareAndFinalize`
- `flashinfer_nvlink_one_sided` → `FlashInferNVLinkOneSidedPrepareAndFinalize`
  (dispatch bytes/scale layout chosen per quant_dtype: bf16 / nvfp4 / mxfp8)
- deepep ht/ll/v2, mori, nixl_ep, ag_rs → their respective P/F classes.

So **FLASHINFER_CUTLASS = `FlashInferExperts` (compute) + a P/F chosen by the
all2all backend** (NoDPEP on a single GPU; FlashInfer NVLink one/two-sided under
EP). The FlashInfer kernel does permute + both GEMMs + SwiGLU + weighted reduce in
one call; the P/F only handles (de)quantization and, under EP, the all2all
dispatch/combine.

---

## 8. End-to-end FLASHINFER_CUTLASS dispatch chain (single GPU, nvfp4)

```
FusedMoE layer forward
  -> runner: router.select_experts(...)                     # ops.topk_softmax  (router/)
  -> RoutedExperts.forward_modular(x, topk_weights, topk_ids)
  -> quant_method.apply -> mk.FusedMoEKernel.apply          # modular_kernel.py:1640
       -> FusedMoEKernelModularImpl.apply                   # :1371
            -> _prepare -> NoDPEP.prepare                   # quantize (nvfp4) -> a1q, a1q_scale
            -> _fused_experts -> FlashInferExperts.apply    # flashinfer_cutlass_moe.py:248
                 -> flashinfer_cutlass_fused_moe(...)       # = flashinfer.fused_moe.cutlass_fused_moe
                      # internal: topk permute -> GEMM1(gate_up) -> SwiGLU -> GEMM2(down) -> weighted reduce
            -> _finalize -> TopKWeightAndReduceNoOP         # no-op (kernel already reduced)
```

Under EP, `_prepare/_finalize` are the FlashInfer NVLink one/two-sided P/F which
wrap `moe_alltoall.dispatch` / `.combine` around the same expert kernel.
