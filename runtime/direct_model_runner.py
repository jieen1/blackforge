"""Direct (non-HTTP) model runner: this process owns GPU KV/GDN state itself
and drives ``model.forward()`` directly, replacing the HTTP bridge to a
separate vLLM server (``runtime/vllm_bridge_backend.py``, commit ``b28942c``).

Design and the four reused vLLM primitives this depends on (``EngineArgs
.create_engine_config()``, ``get_model()``, ``bind_kv_cache()``,
``set_forward_context()``) are documented in
``notes/direct-model-runner-design.md`` -- read that first.

Scope this round (see the design doc's "explicitly out of scope" section):
only slot 0 is exercised, no CUDA graph, no real multi-request batching, no
MTP. Metadata is hand-built for exactly one request at a time, not through
the production ``SM120GQAMetadataBuilder``/``GDNAttentionMetadataBuilder``
(those handle concerns -- persistent CUDA-graph-safe buffers, spec-decode,
multi-request batching -- this round's scope does not need).
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.model_loader import get_model
from vllm.utils.network_utils import get_distributed_init_method, get_open_port
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.utils import bind_kv_cache

NUM_SLOTS = 4
_SM120_BACKEND_PATH = "vllm.v1.attention.backends.sm120_gqa.SM120GQABackend"

# Physical index 0 (block index / GDN state index) is never used for real
# request data -- confirmed empirically from a real vLLM SchedulerOutput
# dump (block_ids=([1], [2], [3], [4]) for the first-ever scheduled
# request; see notes/direct-model-runner-design.md's "Stage C field diff"
# section). Root cause of the 100%-deterministic wrong output this round:
# our hand-built metadata hardcoded physical index = logical slot (so slot
# 0 -> physical index 0), which real vLLM's convention never produces --
# something about index 0 (padding/NULL_BLOCK_ID-adjacent) makes the model
# read/write the wrong state. Fix: reserve physical index 0 permanently
# and offset every logical slot by +1 when computing a physical address.
RESERVED_PHYSICAL_SLOTS = 1


def _physical_slot(logical_slot: int) -> int:
    return logical_slot + RESERVED_PHYSICAL_SLOTS


def _ensure_sm120_backend_registered() -> None:
    """register_backend() is a plain dict write (see registry.py's
    _ATTN_OVERRIDES) -- safe to call more than once."""
    register_backend(AttentionBackendEnum.CUSTOM, _SM120_BACKEND_PATH)


def allocate_fixed_slot_kv_caches(
    static_forward_context: dict,
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    num_slots: int,
    block_size: int,
    blocks_per_slot: int,
) -> dict[str, object]:
    """Allocate our own num_slots-fixed-slot KV (attention) and state (GDN)
    tensors and bind them via vLLM's own real ``bind_kv_cache()`` -- shared
    between ``DirectModelRunner`` (hand-built metadata) and
    ``runtime/vllm_stage_b_baseline.py`` (real vLLM metadata/scheduler,
    Stage B of the 2026-07-16 ownership-transfer ladder: this is the ONLY
    thing that differs from vLLM's own tensor allocation -- everything else
    stays real). Returns the same ``dict[str, tensor|tuple]`` bind_kv_cache
    expects, keyed by layer name.
    """
    attn_layer_names = []
    gdn_layer_names = []
    for name, layer in static_forward_context.items():
        if hasattr(layer, "get_state_shape"):
            gdn_layer_names.append(name)
        else:
            attn_layer_names.append(name)

    kv_caches: dict[str, object] = {}

    if attn_layer_names:
        any_attn = static_forward_context[attn_layer_names[0]]
        backend_cls = any_attn.get_attn_backend()
        num_kv_heads = any_attn.num_kv_heads
        head_size = any_attn.head_size
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot
        shape = backend_cls.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )
        torch_dtype = any_attn.kv_cache_torch_dtype
        for name in attn_layer_names:
            kv_caches[name] = torch.zeros(shape, dtype=torch_dtype, device=device)

    for name in gdn_layer_names:
        layer = static_forward_context[name]
        conv_shape, ssm_shape = layer.get_state_shape()
        conv_dtype, ssm_dtype = layer.get_state_dtype()
        total_physical_slots = num_slots + RESERVED_PHYSICAL_SLOTS
        conv_state = torch.zeros((total_physical_slots, *conv_shape), dtype=conv_dtype, device=device)
        ssm_state = torch.zeros((total_physical_slots, *ssm_shape), dtype=ssm_dtype, device=device)
        kv_caches[name] = (conv_state, ssm_state)

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_caches, static_forward_context, runner_kv_caches)
    return kv_caches


def build_attention_metadata(
    *,
    prior_kv_len: int,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for one request in one fixed slot. Shared
    between ``DirectModelRunner`` (which tracks ``prior_kv_len`` itself via
    ``self.slot_kv_len``) and Stage C of the 2026-07-16 ownership-transfer
    ladder (``runtime/vllm_stage_c_baseline.py``, which derives
    ``prior_kv_len`` from vLLM's own real, scheduler-computed
    ``CommonAttentionMetadata`` instead) -- this is deliberately the exact
    same field-construction logic in both cases, so Stage C tests whether
    *this logic* is correct, not a second, independently-written copy of it.
    """
    new_kv_len = prior_kv_len + num_new_tokens
    page_size = block_size
    first_block = _physical_slot(slot) * blocks_per_slot
    num_pages = (new_kv_len + page_size - 1) // page_size
    if num_pages > blocks_per_slot:
        raise RuntimeError(
            f"slot {slot} kv_len {new_kv_len} exceeds this slot's "
            f"{blocks_per_slot * page_size}-token capacity"
        )
    qo_indptr = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(
        first_block, first_block + num_pages, dtype=torch.int32, device=device
    )
    last_page_len = new_kv_len - (num_pages - 1) * page_size
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    return SM120GQAMetadata(
        num_actual_tokens=num_new_tokens,
        num_reqs=1,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=is_decode and num_new_tokens == 1,
        kv_split_size=max(new_kv_len, 1),
        max_num_splits=1,
        decode_qo_len=num_new_tokens if is_decode else 0,
    )


def build_gdn_metadata(
    *,
    slot_initialized: bool,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    device: torch.device,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for one request in one fixed slot --
    see ``build_attention_metadata``'s docstring for why this is a shared
    function, not a second copy, between ``DirectModelRunner`` and Stage C.
    """
    state_indices = torch.tensor([_physical_slot(slot)], dtype=torch.int32, device=device)
    if is_decode:
        assert num_new_tokens == 1
        non_spec_qsl = torch.tensor([0, 1], dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=1,
            num_decode_tokens=1,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=1,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    query_start_loc = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor([slot_initialized], dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    return GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=num_new_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_new_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def build_attention_metadata_batch(
    *,
    slots: list[int],
    prior_kv_lens: list[int],
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
    qo_len: int = 1,
    fixed_kv_split_size: int | None = None,
    fixed_max_num_splits: int | None = None,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for a real batch of requests spanning
    multiple fixed physical slots in a SINGLE metadata object, each
    contributing the SAME ``qo_len`` new query tokens this step (uniform
    across the batch -- the normal production case, since
    ``num_speculative_tokens`` is a global engine config, not
    per-request). ``qo_len=1`` (the default) is the batched analogue of
    ``build_attention_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch=1/2/3/4/varlen/reuse/continuous-generation
    ladder). ``qo_len>1`` is MTP/speculative-decode verify (K draft tokens
    + 1 bonus token per request, e.g. qo_len=4 for K=3) -- the real
    ``SM120GQAImpl.forward()`` dispatches this to
    ``flash_attn_sm120_fwd_v2_decode_fp8kv_paged`` (already
    production-hardened for qo_len in 2..4) whenever
    ``attn_metadata.decode_qo_len`` is in that range; this function only
    needs to construct the metadata correctly, not touch the kernel.

    Field construction (CSR qo_indptr/kv_page_indptr/kv_page_indices/
    kv_last_page_len, is_pure_decode, decode_qo_len) matches the real
    ``SM120GQAMetadataBuilder.build()`` (vllm/v1/attention/backends/
    sm120_gqa.py) generalized from vLLM's dense block-table -> CSR
    conversion to this project's fixed-slot addressing. At ``qo_len=1``
    every formula below reduces exactly to the previously-verified
    qo_len=1-only formulas (same numeric values, same tensors) -- this is
    a generalization, not a parallel implementation, so the existing
    ladder's results remain valid evidence for the qo_len=1 case.

    Kept as a SEPARATE function from ``build_attention_metadata`` (not a
    generalization of it) so this new batch path cannot regress the
    already-verified single-request path (2026-07-16 slot-0-reservation
    fix, Stage C/D 20/20) -- the two are cross-checked instead via the
    batch=1 equivalence test in ``benchmarks/batch_decode_regression.py``.

    ``fixed_kv_split_size``/``fixed_max_num_splits`` (both None by
    default, preserving the existing per-call-derived behavior for the
    eager decode_batch/verify_batch paths): REQUIRED for CUDA-graph
    capture. Per this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real, previously-hit illegal-memory-access crash): a captured kernel
    launch's scalar arguments (kv_split_size/max_num_splits are plain
    Python ints, not device tensors) freeze to whatever value was live at
    capture time. If kv_split_size were still derived per-call from live
    ``new_kv_lens`` (as the default path does), replaying the SAME
    captured launch at a LARGER kv_len than capture time would silently
    use a stale, too-small split boundary -- the real backend's own fix
    (and this project's, when these are supplied) is to derive
    kv_split_size ONCE from a build-time-fixed upper bound L (this
    project's own ``blocks_per_slot * block_size``, i.e. the per-slot
    page-table limit THIS RUNTIME'S CALLER configured when constructing
    ``DirectModelRunner`` -- a software-chosen ceiling, not a GPU hardware
    limit; already enforced by the RuntimeError check below) via
    ``kv_split_size = ceil(L / target_splits)``,
    ``max_num_splits = target_splits``. Proof this stays correct for
    EVERY real kv_len from 1 up to L, not just the capture-time value:
    for split_size s = ceil(L/target_splits) and any real kv_len k <= L,
    num_splits(k) = ceil(k/s) <= ceil(L/s) <= target_splits (s >= L/target_splits
    by construction) -- so a single fixed pair is a valid upper bound for
    the entire decode lifetime of any request in this slot.
    """
    num_reqs = len(slots)
    if len(prior_kv_lens) != num_reqs:
        raise ValueError("slots and prior_kv_lens must have equal length")
    page_size = block_size
    new_kv_lens = [kv_len + qo_len for kv_len in prior_kv_lens]
    num_pages_per_req = [(kv_len + page_size - 1) // page_size for kv_len in new_kv_lens]
    for slot, kv_len, num_pages in zip(slots, new_kv_lens, num_pages_per_req):
        if num_pages > blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {kv_len} exceeds this slot's "
                f"{blocks_per_slot * page_size}-token capacity"
            )

    qo_indptr = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len

    kv_page_indptr_list = [0]
    for num_pages in num_pages_per_req:
        kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)

    page_index_chunks = [
        torch.arange(
            _physical_slot(slot) * blocks_per_slot,
            _physical_slot(slot) * blocks_per_slot + num_pages,
            dtype=torch.int32,
            device=device,
        )
        for slot, num_pages in zip(slots, num_pages_per_req)
    ]
    kv_page_indices = (
        torch.cat(page_index_chunks) if page_index_chunks else torch.empty(0, dtype=torch.int32, device=device)
    )
    kv_last_page_len = torch.tensor(
        [kv_len - (num_pages - 1) * page_size for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)],
        dtype=torch.int32,
        device=device,
    )
    if fixed_kv_split_size is not None:
        # CUDA-graph-safe path: fixed once from a build-time bound, never
        # from this call's live data -- see the docstring's proof.
        kv_split_size = fixed_kv_split_size
        max_num_splits = fixed_max_num_splits if fixed_max_num_splits is not None else 1
    else:
        # Conservative, correctness-first choice matching the single-request
        # function: kv_split_size >= every request's own new_kv_len forces
        # num_splits == 1 for all of them (no cross-request split-size tuning
        # yet -- a performance follow-on, not a correctness concern, exactly
        # the same tradeoff build_attention_metadata already makes). NOT
        # CUDA-graph-safe (see docstring) -- only used by the eager
        # decode_batch/verify_batch paths.
        kv_split_size = max(max(new_kv_lens, default=1), 1)
        max_num_splits = 1
    return SM120GQAMetadata(
        num_actual_tokens=num_reqs * qo_len,
        num_reqs=num_reqs,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=(qo_len == 1),
        kv_split_size=kv_split_size,
        max_num_splits=max_num_splits,
        decode_qo_len=qo_len,
    )


def build_gdn_metadata_batch(
    *,
    slots: list[int],
    device: torch.device,
    qo_len: int = 1,
    slot_initialized: list[bool] | None = None,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for a real batch of requests, each
    contributing the SAME ``qo_len`` new query tokens this step (see
    ``build_attention_metadata_batch``'s docstring for the uniform-qo_len
    scope rationale shared by both functions).

    ``qo_len=1`` (the default) is the batched analogue of
    ``build_gdn_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch ladder) -- matches the real
    ``GDNAttentionMetadataBuilder.build()``'s pure non-spec-decode branch
    (gdn_attn.py): only num_decodes/num_decode_tokens/
    non_spec_query_start_loc/non_spec_state_indices_tensor are populated,
    everything else stays None. At qo_len=1 the formulas below reduce
    exactly to the previously-verified values -- a generalization, not a
    parallel implementation.

    ``qo_len>1`` is MTP/speculative-decode verify. Rather than
    replicating the real builder's much more involved ``spec_decode``
    branch (accept/reject bookkeeping, sorting spec vs non-spec tokens --
    explicitly out of scope this round, see notes/direct-model-runner-
    design.md), this generalizes ``build_gdn_metadata``'s OTHER existing
    branch instead: the ``is_decode=False`` ("prefill"/chunked) case,
    which the real builder's own ``split_decodes_and_prefills`` would also
    select for any request with query_len>1 when no draft-acceptance info
    is supplied -- i.e. this treats an MTP verify step as an ordinary
    chunked continuation of ``qo_len`` new tokens per request, which is
    numerically correct (the chunked FLA kernel handles arbitrary query
    length, GDN state update included) even though it foregoes the
    real builder's spec-decode-specific optimizations.
    """
    num_reqs = len(slots)
    state_indices = torch.tensor(
        [_physical_slot(slot) for slot in slots], dtype=torch.int32, device=device
    )
    if qo_len == 1:
        non_spec_qsl = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=num_reqs,
            num_decode_tokens=num_reqs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_reqs,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    if slot_initialized is None or len(slot_initialized) != num_reqs:
        raise ValueError("slot_initialized (one bool per slot) is required when qo_len > 1")
    query_start_loc = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor(slot_initialized, dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    num_actual_tokens = num_reqs * qo_len
    return GDNAttentionMetadata(
        num_prefills=num_reqs,
        num_prefill_tokens=num_actual_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def build_vllm_config(
    *,
    model: str,
    kv_cache_dtype: str = "fp8_e4m3",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.5,
    speculative_config: dict | None = None,
) -> VllmConfig:
    _ensure_sm120_backend_registered()
    args = EngineArgs(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=AttentionBackendEnum.CUSTOM,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        language_model_only=True,
        async_scheduling=False,
        speculative_config=speculative_config,
    )
    return args.create_engine_config()


def determine_accept_reject(draft_tokens: list[int], verify_logits) -> dict:
    """Greedy MTP accept/reject (2026-07-17, moved here from
    ``benchmarks/mtp_accept_reject_check.py`` so the real
    ``mtp_verify_and_commit`` coordinator and that benchmark's regression
    test share ONE implementation, not two copies). ``draft_tokens`` has
    K+1 entries (anchor + K drafts); ``verify_logits`` is shaped
    ``[K+1, vocab]`` for ONE request. Returns ``num_accepted`` (0..K), the
    committed real token ids (accepted drafts, if any, plus exactly one
    recovery/bonus token), and the rejection position (``None`` if all K
    were accepted)."""
    k = len(draft_tokens) - 1
    committed: list[int] = []
    for p in range(k):
        predicted = int(verify_logits[p].argmax(dim=-1).item())
        if predicted == draft_tokens[p + 1]:
            committed.append(draft_tokens[p + 1])
        else:
            committed.append(predicted)
            return {"num_accepted": p, "committed": committed, "rejected_at": p}
    bonus = int(verify_logits[k].argmax(dim=-1).item())
    committed.append(bonus)
    return {"num_accepted": k, "committed": committed, "rejected_at": None}


class DirectModelRunner:
    """Owns the model, the 4-slot KV/GDN state tensors, and drives forward
    passes directly. This round: single request, slot 0 only."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_slots: int = NUM_SLOTS,
        block_size: int = 16,
        blocks_per_slot: int = 128,
    ) -> None:
        self.vllm_config = vllm_config
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        with set_current_vllm_config(vllm_config):
            init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            init_worker_distributed_environment(
                vllm_config, rank=0, distributed_init_method=init_method, local_rank=0
            )
            self.model = get_model(vllm_config=vllm_config)

        sfc = vllm_config.compilation_config.static_forward_context
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        self.gdn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_state_shape"):
                self.gdn_layer_names.append(name)
            else:
                self.attn_layer_names.append(name)
        if not self.attn_layer_names or not self.gdn_layer_names:
            raise RuntimeError(
                f"expected both attention and GDN layers, got "
                f"{len(self.attn_layer_names)} attn / {len(self.gdn_layer_names)} gdn"
            )

        # Real MTP draft model (2026-07-17, Phase 2 / sol's "Option A"),
        # loaded ONLY if the caller configured speculative decoding via
        # build_vllm_config(speculative_config=...). Uses vLLM's own real
        # loading mechanism (load_eagle_model -- also used by vLLM's real
        # MTPSpeculator, not just EAGLE) so embed_tokens/lm_head sharing
        # matches production exactly, nothing hand-rolled. Must load
        # BEFORE _allocate_and_bind_kv_caches() so the draft's own
        # attention layer registers into the SAME static_forward_context
        # this project's existing generic KV-cache-allocation machinery
        # already iterates over -- confirmed by reading vLLM's own
        # DraftModelSpeculator.load_model() (vllm/v1/worker/gpu/spec_decode
        # /speculator.py:153-170), which snapshots attention layer names
        # before/after loading the draft for the exact same reason (there
        # via get_layers_from_vllm_config(..., AttentionLayerBase); here
        # via a direct before/after diff of static_forward_context, which
        # is equivalent since every layer -- attention or GDN -- is
        # registered into that same dict).
        self.mtp_model = None
        self.mtp_attn_layer_names: list[str] = []
        self.num_speculative_tokens: int | None = None
        if vllm_config.speculative_config is not None:
            from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

            names_before = set(sfc.keys())
            with set_current_vllm_config(vllm_config):
                self.mtp_model = load_eagle_model(self.model, vllm_config)
            names_after = set(sfc.keys())
            self.mtp_attn_layer_names = sorted(names_after - names_before)
            if not self.mtp_attn_layer_names:
                raise RuntimeError("loading the MTP draft model registered no new layers")
            for name in self.mtp_attn_layer_names:
                if hasattr(sfc[name], "get_state_shape"):
                    raise RuntimeError(f"unexpected GDN layer in MTP draft model: {name}")
            self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

        self._allocate_and_bind_kv_caches()

        # Per-slot bookkeeping: attention kv_len (tokens actually written into
        # the paged KV cache) and GDN "has state been initialized" flag.
        self.slot_kv_len = [0] * num_slots
        self.slot_gdn_initialized = [False] * num_slots

        # Per-slot MTP state (explicit fields, not implicit -- 2026-07-17
        # Codex-sol review asked for this precisely so a live multi-round
        # loop can't silently conflate "physically written" with
        # "committed"). ``slot_kv_len``/``slot_gdn_initialized`` above
        # ARE the target's committed_len/init-state -- no separate
        # "committed_len" field is added since that would just be a second
        # name for the same quantity; what's genuinely new is the DRAFT
        # model's own sync length (a different KV cache, tracked
        # separately) and the in-flight pending proposal.
        self.slot_draft_sync_len = [0] * num_slots
        self.slot_pending_draft_tokens: list[list[int] | None] = [None] * num_slots
        self.slot_gdn_snapshot_gen = [0] * num_slots

        self._warmup()

    def _warmup(self) -> None:
        """Real vLLM always runs a profiling/warmup forward before serving
        (see gpu_model_runner.py's warmup pass, and this project's own
        server logs: "Initial profiling/warmup run took N s"). Motivated by
        a real, isolated repro (see notes/direct-model-runner-design.md's
        "deep dive on the conv_state lead" section): causal_conv1d_fn's
        Triton kernel returns an all-zero result on its first-ever call in
        a process, in complete isolation, unrelated to this runtime's code.
        Kept here since it mirrors real vLLM's own behavior and cannot
        hurt, but -- reported honestly -- this alone does NOT fix the real
        model's wrong output (verified: neither a 1-token nor a
        shape-matched 5-token warmup changed the observed wrong completion
        for "The capital of France is"). The cold-start bug is real but
        evidently not the whole story; see the design doc for the
        follow-up isolated tests that show a messier, not-yet-characterized
        pattern (interleaved shapes don't self-correct the way repeating
        one shape does) and the next debugging steps."""
        try:
            self.prefill(0, [0, 0, 0, 0, 0])
        finally:
            self.reset_slot(0)

    def _allocate_and_bind_kv_caches(self) -> None:
        self.kv_caches = allocate_fixed_slot_kv_caches(
            self.static_forward_context,
            self.vllm_config,
            self.device,
            num_slots=self.num_slots,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
        )

    def _attention_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> SM120GQAMetadata:
        return build_attention_metadata(
            prior_kv_len=self.slot_kv_len[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
        )

    def _gdn_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> GDNAttentionMetadata:
        return build_gdn_metadata(
            slot_initialized=self.slot_gdn_initialized[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            device=self.device,
        )

    def _slot_mapping(self, slot: int, start_pos: int, num_new_tokens: int) -> torch.Tensor:
        """Flat per-token KV-cache write index: block_id * block_size + offset
        -- the same convention vLLM's own paged attention backends use (see
        attention.py's do_kv_cache_update, which reads this from
        ``forward_context.slot_mapping[layer_name]``, NOT from
        ``attn_metadata`` -- easy to miss, and missing it means K/V are never
        written into the cache at all)."""
        first_block = _physical_slot(slot) * self.blocks_per_slot
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )
        block_ids = first_block + positions // self.block_size
        offsets = positions % self.block_size
        return (block_ids * self.block_size + offsets).to(torch.long)

    def _forward(
        self,
        slot: int,
        token_ids: list[int],
        start_pos: int,
        *,
        is_decode: bool,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        num_new_tokens = len(token_ids)
        attn_meta = self._attention_metadata(
            slot, num_new_tokens=num_new_tokens, is_decode=is_decode
        )
        gdn_meta = self._gdn_metadata(
            slot, num_new_tokens=num_new_tokens, is_decode=is_decode
        )
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        torch.cuda.synchronize()
        logits = self.model.compute_logits(hidden_states)
        torch.cuda.synchronize()

        self.slot_kv_len[slot] += num_new_tokens
        self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def prefill(self, slot: int, prompt_token_ids: list[int]) -> int:
        """Run the prompt through the model; returns the greedy next token id."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        logits = self._forward(slot, prompt_token_ids, start_pos=0, is_decode=False)
        return int(logits[-1].argmax(dim=-1).item())

    def decode(self, slot: int, token_id: int) -> int:
        """Consume one token, return the greedy next token id."""
        start_pos = self.slot_kv_len[slot]
        logits = self._forward(slot, [token_id], start_pos=start_pos, is_decode=True)
        return int(logits[-1].argmax(dim=-1).item())

    def _slot_mapping_batch(
        self, slots: list[int], kv_lengths: list[int], qo_len: int = 1
    ) -> torch.Tensor:
        """Batched analogue of ``_slot_mapping``: each request contributes
        ``qo_len`` new tokens starting at its own ``kv_lengths[i]``,
        flattened in the SAME per-request-contiguous order ``_forward_batch``
        uses for ``input_ids``/``positions`` (request 0's ``qo_len`` tokens,
        then request 1's, ...). At ``qo_len=1`` this reduces exactly to the
        previously-verified one-position-per-request mapping."""
        positions = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]
        slots_per_token = [slot for slot in slots for _ in range(qo_len)]
        block_ids = torch.tensor(
            [
                _physical_slot(slot) * self.blocks_per_slot + pos // self.block_size
                for slot, pos in zip(slots_per_token, positions)
            ],
            dtype=torch.long,
            device=self.device,
        )
        offsets = torch.tensor(
            [pos % self.block_size for pos in positions], dtype=torch.long, device=self.device
        )
        return block_ids * self.block_size + offsets

    def _forward_batch(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        qo_len: int = 1,
        commit: bool = True,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Real batched decode/verify: ONE batched attention/GDN metadata
        object and ONE ``model.forward()`` call covering every listed slot
        -- not a Python loop calling ``_forward``/``decode`` per slot.
        ``kv_lengths`` is the caller-asserted prior KV length (before this
        step's new tokens) for each slot; cross-checked against this
        runner's own ``self.slot_kv_len`` bookkeeping to catch drift early
        rather than silently addressing the wrong cache rows.

        ``qo_len=1`` (the default, unchanged from the original decode-only
        batch path): ``token_ids`` is a flat list, one token id per slot.
        ``qo_len>1`` (MTP/speculative-decode verify, uniform across the
        batch): ``token_ids`` is a list of per-slot token-id lists, each of
        length ``qo_len`` -- the K draft tokens + 1 bonus-position
        placeholder being verified in one batched call.
        Returns logits shaped ``[num_reqs * qo_len, vocab]``, flattened in
        request-then-position order (request 0's qo_len rows, then request
        1's, ...) -- the same order ``SM120GQAImpl.forward()``'s own
        ``q_decode.reshape(num_reqs, qo_len, ...)`` expects.

        ``commit`` (default ``True``, preserving the original decode_batch
        behavior exactly): whether to advance ``self.slot_kv_len`` by
        ``qo_len`` for every listed slot. The forward pass ALWAYS
        physically writes K/V for all ``qo_len`` positions regardless of
        this flag -- ``commit`` only controls this method's own
        bookkeeping. Real MTP verify calls (``verify_batch``) pass
        ``commit=False``, since the actual committed length is not known
        until the caller's accept/reject decision runs on the returned
        logits (2026-07-17, fixing the exact "physically-written vs.
        committed" conflation Codex-sol's review flagged) -- the caller
        (``mtp_verify_and_commit``) is responsible for advancing
        ``slot_kv_len`` by the REAL committed length afterward. Attention's
        own KV needs no explicit rollback either way (content/position
        addressed -- positions beyond the real committed length are simply
        never read again); only GDN's recurrent state needs the
        snapshot/restore + recompute-forward repair on a non-full-accept
        outcome, exactly as already verified by
        ``benchmarks/mtp_gdn_rollback_check.py``.
        """
        num_reqs = len(slot_ids)
        if qo_len == 1:
            if not (len(token_ids) == num_reqs and len(kv_lengths) == num_reqs):
                raise ValueError("slot_ids/token_ids/kv_lengths must have equal length")
            flat_token_ids = token_ids
        else:
            if not (
                len(token_ids) == num_reqs
                and len(kv_lengths) == num_reqs
                and all(len(t) == qo_len for t in token_ids)
            ):
                raise ValueError(
                    "slot_ids/token_ids/kv_lengths must have equal length, and "
                    f"every token_ids[i] must have exactly qo_len={qo_len} tokens"
                )
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.slot_kv_len[slot]}"
                )
            if not self.slot_gdn_initialized[slot]:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")

        attn_meta = build_attention_metadata_batch(
            slots=slot_ids,
            prior_kv_lens=kv_lengths,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            qo_len=qo_len,
        )
        gdn_meta = build_gdn_metadata_batch(
            slots=slot_ids,
            device=self.device,
            qo_len=qo_len,
            slot_initialized=[self.slot_gdn_initialized[s] for s in slot_ids] if qo_len > 1 else None,
        )
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping_batch(slot_ids, kv_lengths, qo_len=qo_len)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            [kv_len + j for kv_len in kv_lengths for j in range(qo_len)],
            dtype=torch.long,
            device=self.device,
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        torch.cuda.synchronize()
        logits = self.model.compute_logits(hidden_states)
        torch.cuda.synchronize()

        for slot in slot_ids:
            if commit:
                self.slot_kv_len[slot] += qo_len
            self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def decode_batch(
        self, slot_ids: list[int], token_ids: list[int], kv_lengths: list[int]
    ) -> list[int]:
        """Decode one token for each of several active slots via a single
        real batched forward call. Returns the greedy next token id per
        slot, in the same order as ``slot_ids``."""
        logits = self._forward_batch(slot_ids, token_ids, kv_lengths)
        return [int(logits[i].argmax(dim=-1).item()) for i in range(len(slot_ids))]

    def verify_batch(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """MTP/speculative-decode verify: submit ``qo_len`` draft tokens
        (K speculative + 1 bonus position) per active slot and run them all
        through ONE real batched forward call. ``draft_token_ids[i]`` is
        slot ``slot_ids[i]``'s own list of draft tokens (same length for
        every slot this step, since ``num_speculative_tokens`` is a global
        engine config). Returns raw logits shaped
        ``[num_reqs * qo_len, vocab]`` (request-then-position order) --
        accept/reject sampling against these logits is the caller's job
        (``determine_accept_reject``/``mtp_verify_and_commit``).
        ``commit=False`` is passed to ``_forward_batch`` unconditionally --
        a verify call's real committed length is never known until
        accept/reject runs on these logits, so ``slot_kv_len`` is
        deliberately NOT advanced here (2026-07-17 fix; see
        ``_forward_batch``'s docstring)."""
        qo_len = len(draft_token_ids[0]) if draft_token_ids else 0
        return self._forward_batch(
            slot_ids, draft_token_ids, kv_lengths, qo_len=qo_len, commit=False, return_hidden=return_hidden
        )

    def reset_slot(self, slot: int) -> None:
        """Release a slot for reuse by a new logical request. Does not zero
        the underlying tensors -- the next prefill's has_initial_state=False
        and kv_len bookkeeping starting from 0 is what makes reuse correct,
        matching this project's established fixed-slot-generation design."""
        self.slot_kv_len[slot] = 0
        self.slot_gdn_initialized[slot] = False

    def snapshot_gdn_state(self, slot: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Copy out this slot's ``(conv_state, ssm_state)`` for every GDN
        layer, keyed by layer name. Building block for MTP verify's GDN
        state commit/rollback (2026-07-17 round): unlike attention's paged
        KV cache (content-addressed by position, safe to just stop
        advancing ``slot_kv_len`` past a rejected boundary), GDN's
        recurrent/chunked state has no position index to truncate to -- it
        is a single accumulated value per slot that a verify call updates
        in place. Snapshotting before a verify call and restoring here on
        partial rejection (this class's chosen strategy -- "Option A" in
        notes/direct-model-runner-design.md's MTP-semantics design
        section) is the correctness-first approach: simple to reason about
        and to verify independently of the rest of MTP (see
        ``benchmarks/mtp_gdn_rollback_check.py``), at the cost of an extra
        state copy per verify call and a recompute forward pass on
        rejection. Returns CPU-resident clones (not GPU-resident, to avoid
        holding extra persistent GPU memory for a snapshot that is usually
        discarded within one verify step) -- restore moves them back to
        device.

        Tags the snapshot with this slot's current generation counter
        (``self.slot_gdn_snapshot_gen``, bumped on every snapshot) --
        2026-07-17 addition per Codex-sol's explicit ask for explicit
        per-slot state so a STALE snapshot (e.g. a caller accidentally
        holding on to one from two rounds ago) can never be restored by
        mistake; ``restore_gdn_state`` rejects a generation mismatch."""
        physical = _physical_slot(slot)
        self.slot_gdn_snapshot_gen[slot] += 1
        snapshot: dict = {"__generation__": self.slot_gdn_snapshot_gen[slot]}
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            snapshot[name] = (
                conv_state[physical].detach().to("cpu", copy=True),
                ssm_state[physical].detach().to("cpu", copy=True),
            )
        return snapshot

    def restore_gdn_state(
        self, slot: int, snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Restore this slot's GDN state from a prior
        ``snapshot_gdn_state()`` call -- writes IN PLACE into the same
        persistent ``kv_caches`` tensors (never reallocates them), so this
        is safe to call between real forward passes without disturbing any
        other slot or any fixed-address buffer a CUDA-graph-captured call
        might depend on. Rejects a stale snapshot (generation counter
        mismatch -- see ``snapshot_gdn_state``'s docstring)."""
        gen = snapshot.get("__generation__")
        if gen != self.slot_gdn_snapshot_gen[slot]:
            raise RuntimeError(
                f"stale GDN snapshot for slot {slot}: snapshot generation {gen} != "
                f"current {self.slot_gdn_snapshot_gen[slot]}"
            )
        physical = _physical_slot(slot)
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            snap_conv, snap_ssm = snapshot[name]
            conv_state[physical].copy_(snap_conv.to(self.device))
            ssm_state[physical].copy_(snap_ssm.to(self.device))

    def _mtp_forward(
        self,
        slot: int,
        token_ids: list[int],
        hidden_states_in: torch.Tensor,
        start_pos: int,
        *,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Real draft-model (``Qwen3_5MTP``) forward for ONE slot -- the
        low-level primitive the centralized MTP-cycle coordinator methods
        (``mtp_prefill``/``mtp_verify_and_commit``) build on. Deliberately
        does NOT touch ``self.slot_draft_sync_len`` itself (unlike
        ``_forward``/``_forward_batch``'s unconditional-or-commit-flag
        bookkeeping) -- the caller decides whether this call's advance
        represents the real synced history (step 0, teacher-forced with
        the target's own just-computed hidden state) or a throwaway
        exploratory propose step (steps 1..K-1, autoregressive on the
        draft's own previous hidden state/token) that must NOT be counted
        as committed. This is what makes the draft model's own KV cache
        need no explicit rollback on accept/reject, unlike GDN: an
        exploratory step's positions are simply overwritten by the next
        round's real sync call, exactly like attention's own
        content/position-addressed reasoning elsewhere in this file.

        ``hidden_states_in`` must have exactly ``len(token_ids)`` rows
        (``Qwen3_5MultiTokenPredictor.forward()`` concatenates it against
        the embedded ``input_ids`` along the hidden-size dim, so the
        sequence-length dim must already match)."""
        if self.mtp_model is None:
            raise RuntimeError(
                "no MTP draft model loaded -- build_vllm_config(speculative_config=...) first"
            )
        num_new_tokens = len(token_ids)
        attn_meta = build_attention_metadata(
            prior_kv_len=self.slot_draft_sync_len[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
        )
        attn_metadata_dict = {name: attn_meta for name in self.mtp_attn_layer_names}
        slot_mapping = self._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self.mtp_attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = self.mtp_model.forward(input_ids, positions, hidden_states_in)
        torch.cuda.synchronize()
        logits = self.mtp_model.compute_logits(hidden_states_out)
        torch.cuda.synchronize()
        return logits, hidden_states_out

    def _mtp_sync_and_propose(
        self,
        slot: int,
        shifted_input_ids: list[int],
        target_hidden_states: torch.Tensor,
        start_pos: int,
        num_new_tokens: int,
        k: int,
    ) -> list[int]:
        """The centralized sync+propose funnel every MTP-aware entry point
        (``mtp_prefill``/eventual ``mtp_decode``) routes through -- per
        2026-07-17's sol-refined design, this is the ONE place draft-sync
        logic lives, not duplicated per public entry point. Step 0 is the
        real sync (teacher-forced with the target's OWN just-computed
        hidden states, covering this step's FULL real query range --
        matches vLLM's real ``_prepare_prefill_inputs_kernel`` shift-by-one
        mechanism); steps 1..k-1 are genuinely autoregressive on the
        draft's own previous hidden state/token, and are NOT committed to
        ``self.slot_draft_sync_len`` (see ``_mtp_forward``'s docstring)."""
        step0_logits, step0_hidden = self._mtp_forward(
            slot, shifted_input_ids, target_hidden_states, start_pos, is_decode=(num_new_tokens == 1)
        )
        self.slot_draft_sync_len[slot] += num_new_tokens
        draft_tokens = [int(step0_logits[-1].argmax(dim=-1).item())]
        prev_hidden = step0_hidden[-1:]
        prev_token = draft_tokens[0]
        next_pos = start_pos + num_new_tokens
        for _ in range(1, k):
            step_logits, step_hidden = self._mtp_forward(
                slot, [prev_token], prev_hidden, next_pos, is_decode=True
            )
            prev_token = int(step_logits[-1].argmax(dim=-1).item())
            draft_tokens.append(prev_token)
            prev_hidden = step_hidden[-1:]
            next_pos += 1
        return draft_tokens

    def mtp_prefill(self, slot: int, prompt_token_ids: list[int]) -> dict:
        """Unified MTP cycle funnel point for a fresh prefill: real target
        prefill (with hidden states) -> draft KV sync (step 0, teacher-
        forced shift over the WHOLE prompt) -> K-1 more autoregressive
        draft steps. Returns the anchor (target's own greedy next token,
        matching plain ``prefill()``'s contract -- not yet written into
        the target's own KV) and the K proposed draft tokens, ready for
        the caller to submit through ``mtp_verify_and_commit``."""
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if self.slot_kv_len[slot] != 0 or self.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        target_logits, target_hidden = self._forward(
            slot, prompt_token_ids, start_pos=0, is_decode=False, return_hidden=True
        )
        anchor = int(target_logits[-1].argmax(dim=-1).item())
        shifted_input_ids = prompt_token_ids[1:] + [anchor]
        draft_tokens = self._mtp_sync_and_propose(
            slot,
            shifted_input_ids,
            target_hidden,
            start_pos=0,
            num_new_tokens=len(prompt_token_ids),
            k=self.num_speculative_tokens,
        )
        self.slot_pending_draft_tokens[slot] = draft_tokens
        return {"anchor": anchor, "draft_tokens": draft_tokens}

    def mtp_verify_and_commit(self, slot: int, anchor: int, draft_tokens: list[int]) -> dict:
        """Unified MTP cycle funnel point for verify+commit+resync+propose
        -- the ONE method a real multi-round loop calls repeatedly (no
        separate "decode" coordinator needed; see the design note below on
        why). Submits ``[anchor] + draft_tokens`` through the real,
        already-verified ``verify_batch`` (``commit=False`` -- see its
        docstring), applies greedy ``determine_accept_reject``, and on any
        non-full-accept outcome repairs GDN state (``restore_gdn_state`` +
        a real recompute forward for exactly the committed length) and
        corrects ``slot_kv_len``. The draft model's own KV needs no repair
        either way -- see ``_mtp_forward``'s docstring.

        Recompute input alignment (2026-07-17, fixed after a real bug was
        caught by direct KV-content reasoning, not just shape/bookkeeping
        checks -- see notes/direct-model-runner-design.md): the token
        whose OWN K/V gets written at position ``kv_len_before + i`` is
        the i-th QUERY INPUT of that forward call, matching
        ``verify_batch``'s own convention where ``draft[0]=anchor``'s K/V
        lands at ``kv_len_before`` (mirroring ``prefill()``/``decode()``'s
        established contract: the anchor/greedy-next token is NOT written
        into KV until it is fed back in as the FOLLOWING call's input).
        ``decision["committed"]`` is ``[accepted_draft_0, ..., accepted_
        draft_{n-1}, recovery]`` -- the recovery/bonus token is, symmetrically,
        NOT yet written into KV either (it becomes the next round's own
        anchor-equivalent). So the real input tokens for positions
        ``kv_len_before..+committed_len-1`` are ``real_new_tokens =
        [anchor] + committed[:-1]`` (anchor + accepted drafts, dropping the
        not-yet-written recovery token) -- NOT ``committed`` itself, which
        would silently write the WRONG token content into the KV cache
        while still looking correct on every shape/length/bookkeeping
        check (exactly why the verification gradient calls for real
        numerical/content checks, not just invariant checks).

        Draft catch-up + next-round propose, folded into ONE call
        (2026-07-17 multi-round design): after committing, the draft's own
        KV is behind by exactly ``real_new_tokens`` (it was last synced at
        the END of the PREVIOUS round -- ``mtp_prefill``/this same method
        -- so ``slot_draft_sync_len`` always equals ``slot_kv_len`` from
        BEFORE this round's commit). Syncing the draft over
        ``real_new_tokens`` (shifted by one, ending in the recovery/bonus
        token as the final candidate -- exactly ``_mtp_sync_and_propose``'s
        existing step-0 pattern, just generalized from
        ``mtp_prefill``'s "whole prompt" range to "this round's newly
        committed range") both catches the draft's KV up to
        ``slot_kv_len`` again (restoring the invariant) AND, at that same
        call's LAST position (processing the recovery/bonus token as a
        candidate against the target's hidden state up through the last
        real position), produces the FIRST draft token for the NEXT
        round -- for free, no extra forward call. ``_mtp_sync_and_propose``
        then runs the usual K-1 further autoregressive steps on top. This
        mirrors real vLLM's own design (propose() runs immediately after
        postprocess_sampled(), not as a separate deferred step) more
        closely than an earlier draft of this method (which returned an
        unused ``last_hidden`` and left resync/propose to a separate,
        never-built ``mtp_decode``).

        Returns the accept/reject decision plus ``next_anchor`` (the
        recovery/bonus token -- feed this as ``anchor`` to the NEXT
        ``mtp_verify_and_commit`` call) and ``next_draft_tokens`` (K fresh
        proposed tokens for that next call)."""
        k = len(draft_tokens)
        draft = [anchor] + draft_tokens
        kv_len_before = self.slot_kv_len[slot]
        snapshot = self.snapshot_gdn_state(slot)
        verify_logits, verify_hidden = self.verify_batch(
            [slot], [draft], [kv_len_before], return_hidden=True
        )
        decision = determine_accept_reject(draft, verify_logits)
        committed_len = decision["num_accepted"] + 1
        # Real input tokens for positions kv_len_before..+committed_len-1:
        # anchor followed by the accepted drafts (NOT the recovery token --
        # see the docstring above). Valid for EITHER branch below.
        real_new_tokens = [anchor] + decision["committed"][:-1]

        if decision["num_accepted"] == k:
            self.slot_kv_len[slot] = kv_len_before + k + 1
            real_new_hidden = verify_hidden
        else:
            self.restore_gdn_state(slot, snapshot)
            self.slot_kv_len[slot] = kv_len_before
            _, real_new_hidden = self._forward_batch(
                [slot],
                [real_new_tokens] if committed_len > 1 else real_new_tokens,
                [kv_len_before],
                qo_len=committed_len,
                commit=True,
                return_hidden=True,
            )

        next_anchor = decision["committed"][-1]
        next_draft_tokens = self._mtp_sync_and_propose(
            slot,
            real_new_tokens[1:] + [next_anchor],
            real_new_hidden,
            start_pos=self.slot_draft_sync_len[slot],
            num_new_tokens=committed_len,
            k=k,
        )
        self.slot_pending_draft_tokens[slot] = next_draft_tokens
        return {**decision, "next_anchor": next_anchor, "next_draft_tokens": next_draft_tokens}


class CapturedBatchDecodeGraph:
    """CUDA-graph-captured batch decode/verify for a FIXED batch size and
    FIXED ``qo_len`` (1 = pure decode, >1 = MTP/speculative-decode verify,
    e.g. 4 for K=3 draft + 1 bonus token), replayable at ANY per-slot
    kv_len up to this runtime's per-slot capacity (``blocks_per_slot *
    block_size``) -- not just whatever dummy shape was used at capture
    time.

    2026-07-16, CUDA Graph round: this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real illegal-memory-access crash, root-caused to metadata tensors
    without fixed addresses) plus its OTHER documented lesson
    (``kv_split_size``/``max_num_splits`` frozen at capture time going
    stale under a later, larger real kv_len) directly motivate this
    class's two central design points:

    1. Every tensor a captured kernel launch reads (metadata CSR tensors,
       input_ids, positions, slot_mapping) is a PERSISTENT, fixed-address
       buffer, allocated once in ``__init__``. ``replay()`` writes freshly
       computed REAL values into these SAME buffers via ``.copy_()`` --
       it never reallocates them. This is what makes replaying at a
       kv_len the buffers were never filled with at capture time safe.
    2. ``kv_split_size``/``max_num_splits`` are derived ONCE from this
       runtime's configured per-slot page-table limit (``blocks_per_slot *
       block_size`` -- a software ceiling the caller chose when
       constructing ``DirectModelRunner``, NOT a GPU hardware limit), via
       ``build_attention_metadata_batch``'s ``fixed_kv_split_size``/
       ``fixed_max_num_splits`` parameters -- see that function's
       docstring for the correctness proof that this bounds every real
       kv_len up to that configured limit, not just the capture-time
       value.

    2026-07-16, MTP extension (qo_len>1): GDN's chunked/"prefill" metadata
    fields (``chunk_indices``/``chunk_offsets``/``nums_dict``/
    ``batch_ptr``/``token_chunk_offset_ptr``/``has_initial_state``) depend
    ONLY on the query-length structure (how many tokens per request), not
    on which physical slot each request maps to or on kv_len -- so for a
    FIXED (batch_size, qo_len) graph they are genuinely CONSTANT across
    every replay (unlike ``kv_page_indices``/``state_indices``, which
    depend on live kv_len/slot identity and must be refilled every
    replay). Computed once in ``__init__`` via
    ``build_gdn_metadata_batch(..., slot_initialized=[True]*batch_size)``
    and reused as-is -- fixed address by construction of never being
    recreated, no ``.copy_()`` needed. ``has_initial_state=True`` for
    every slot is this class's scope: MTP verify only ever happens after
    a slot's own prior prefill/decode has established real context.

    A replayed CUDA graph is a pre-recorded sequence of GPU kernel
    launches, NOT a re-execution of Python control flow -- ``model
    .forward()`` (the Python function) is only ever actually called
    during ``capture()`` (plus its warmup iterations), never during
    ``replay()``. This means whatever kernel-dispatch branch
    ``SM120GQAImpl.forward()`` takes (decode-kernel vs general, FP8 vs
    NVFP4, MMA vs v2 vs scalar, ...) must be identical for every real
    kv_len this graph will ever replay at -- true here because dispatch
    depends only on ``qo_len``/kv-cache dtype/model config, all fixed for
    a given (batch_size, qo_len) graph, never on the live kv_len itself.

    2026-07-17, state-neutral capture (correctness fix, found via an
    independent review this project's coordinator commissioned and
    personally verified): ``capture()``'s warmup runs 3 REAL executions
    on a side stream before the graph trace (the trace itself, inside
    ``with torch.cuda.graph(g):``, executes nothing -- confirmed against
    the sibling project's own kernel-level CUDA-graph test). Attention's
    paged KV cache tolerates redundant warmup writes fine (same position,
    same value, overwritten harmlessly) -- but GDN's recurrent/chunked
    state update reads-old-state-and-writes-new-state each call, so it is
    NOT idempotent under repeated identical input; running warmup against
    slots a caller will later actually replay against silently advances
    those slots' real GDN state by 3 extra (unaccounted, un-bookkept)
    applications before any real replay happens. This was a genuine gap
    in this project's own initial qo_len=1/MTP test scripts (both reused
    real/twin-established slots for warmup) -- their empirical PASS
    results are not proof this doesn't matter, only evidence it didn't
    surface for that specific signal-probe task (plausibly because
    full-attention layers dominate identity recall, masking a GDN
    perturbation a GDN-sensitive task might not tolerate).

    Fix: this class now reserves ``batch_size`` of the runner's logical
    slots PERMANENTLY for its own exclusive, disposable warmup use (the
    LAST ``batch_size`` slots of ``runner.num_slots`` -- see
    ``self._warmup_slots``) -- ``capture()`` takes no external slot/token/
    kv_length arguments at all anymore, so this can no longer depend on
    caller discipline the way the original design implicitly did. Callers
    must size ``runner.num_slots >= 2 * batch_size`` and never pass this
    graph's reserved warmup slots to ``replay()`` or any other runner
    method.

    Also fixed the same round: ``replay()`` no longer calls
    ``torch.cuda.synchronize()`` (see that method's docstring for why this
    is safe and why the removed blanket device-wide sync worked against
    the whole point of using a captured graph to cut CPU-side dispatch
    overhead), and ``_fill_buffers`` now computes per-replay values via
    plain Python arithmetic instead of round-tripping through
    ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
    ``DirectModelRunner._slot_mapping_batch`` (which each construct several
    of their own intermediate GPU tensors -- real, avoidable per-replay
    allocation overhead on what should be a lean hot path). This is a
    partial mitigation (each static buffer's ``.copy_()`` source is still
    a freshly constructed small tensor, not a persistent pinned staging
    buffer written in place) -- a fully allocation-free version is a
    further optimization, not attempted this round.
    """

    TARGET_SPLITS = 16

    def __init__(self, runner: "DirectModelRunner", batch_size: int, qo_len: int = 1) -> None:
        if runner.num_slots < 2 * batch_size:
            raise ValueError(
                f"runner.num_slots={runner.num_slots} must be >= 2*batch_size "
                f"({2 * batch_size}): {batch_size} logical slots for real "
                f"replay() traffic plus {batch_size} PERMANENTLY RESERVED for "
                "capture()'s own disposable warmup (never exposed to real "
                "callers) -- see the class docstring's 'state-neutral "
                "capture' section for why this is required, not optional."
            )
        self.runner = runner
        self.batch_size = batch_size
        self.qo_len = qo_len
        device = runner.device
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot
        capacity = blocks_per_slot * block_size  # configured per-slot page-table limit (software, not GPU hardware)
        self.fixed_kv_split_size = max(1, -(-capacity // self.TARGET_SPLITS))
        self.fixed_max_num_splits = self.TARGET_SPLITS

        # Permanently reserved for THIS graph object's own capture()
        # warmup -- the last batch_size logical slots of the runner.
        # Callers must never pass these to replay() or any other runner
        # method; doing so would defeat the whole point of reserving them.
        self._warmup_slots = list(range(runner.num_slots - batch_size, runner.num_slots))

        num_reqs = batch_size
        n_tokens = num_reqs * qo_len

        # Attention metadata static buffers -- worst-case sized (a request
        # could in principle use this slot's entire page capacity).
        # qo_indptr is CONSTANT for a fixed (batch_size, qo_len) pair
        # ([0, qo_len, 2*qo_len, ..., num_reqs*qo_len]) -- computed once,
        # never refilled.
        self.static_qo_indptr = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
        self.static_kv_page_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.static_kv_page_indices = torch.zeros(num_reqs * blocks_per_slot, dtype=torch.int32, device=device)
        self.static_kv_last_page_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # GDN metadata static buffers. non_spec_query_start_loc is
        # likewise constant; state_indices is per-replay-filled (depends
        # on slot_ids, not just batch_size/qo_len).
        self.static_state_indices = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        self.static_non_spec_qsl = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len

        # Model I/O static buffers.
        self.static_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=device)

        # MTP-only (qo_len>1): the chunked/"prefill" GDN fields that
        # depend only on query-length structure -- computed once, see the
        # class docstring's "MTP extension" section.
        self._const_gdn_extra: GDNAttentionMetadata | None = None
        if qo_len > 1:
            self._const_gdn_extra = build_gdn_metadata_batch(
                slots=list(range(num_reqs)),
                device=device,
                qo_len=qo_len,
                slot_initialized=[True] * num_reqs,
            )

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None

    def _fill_buffers(self, slot_ids: list[int], token_ids, kv_lengths: list[int]) -> None:
        """Write real, per-replay-varying values into the persistent static
        buffers. Computes everything via plain Python arithmetic (CPU-only,
        no GPU allocation) instead of calling
        ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
        ``DirectModelRunner._slot_mapping_batch`` -- those each construct
        several of their own intermediate GPU tensors (dataclass fields the
        caller doesn't need here), real avoidable overhead on a hot path
        meant to be lean. Each static buffer's ``.copy_()`` source below is
        still a freshly built small tensor (a partial mitigation, not a
        fully allocation-free design -- see the class docstring)."""
        runner = self.runner
        device = runner.device
        qo_len = self.qo_len
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot

        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        for slot, kv_len, num_pages in zip(slot_ids, new_kv_lens, num_pages_per_req):
            if num_pages > blocks_per_slot:
                raise RuntimeError(
                    f"slot {slot} kv_len {kv_len} exceeds this slot's "
                    f"{blocks_per_slot * block_size}-token capacity"
                )

        kv_page_indptr_list = [0]
        for num_pages in num_pages_per_req:
            kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)

        page_indices_list: list[int] = []
        for slot, num_pages in zip(slot_ids, num_pages_per_req):
            first_block = _physical_slot(slot) * blocks_per_slot
            page_indices_list.extend(range(first_block, first_block + num_pages))

        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        state_indices_list = [_physical_slot(slot) for slot in slot_ids]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]

        slot_mapping_list: list[int] = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            first_block = _physical_slot(slot) * blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = first_block + pos // block_size
                offset = pos % block_size
                slot_mapping_list.append(block_id * block_size + offset)

        self.static_kv_page_indptr.copy_(torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device))
        self.static_kv_page_indices.zero_()
        if page_indices_list:
            self.static_kv_page_indices[: len(page_indices_list)].copy_(
                torch.tensor(page_indices_list, dtype=torch.int32, device=device)
            )
        self.static_kv_last_page_len.copy_(torch.tensor(last_page_len_list, dtype=torch.int32, device=device))
        self.static_state_indices.copy_(torch.tensor(state_indices_list, dtype=torch.int32, device=device))
        self.static_input_ids.copy_(torch.tensor(flat_token_ids, dtype=torch.long, device=device))
        self.static_positions.copy_(torch.tensor(positions_list, dtype=torch.long, device=device))
        self.static_slot_mapping.copy_(torch.tensor(slot_mapping_list, dtype=torch.long, device=device))

    def _static_metadata_dicts(self) -> tuple[dict, dict]:
        runner = self.runner
        n_tokens = self.batch_size * self.qo_len
        attn_meta = SM120GQAMetadata(
            num_actual_tokens=n_tokens,
            num_reqs=self.batch_size,
            qo_indptr=self.static_qo_indptr,
            kv_page_indptr=self.static_kv_page_indptr,
            kv_page_indices=self.static_kv_page_indices,
            kv_last_page_len=self.static_kv_last_page_len,
            page_size=runner.block_size,
            is_pure_decode=(self.qo_len == 1),
            kv_split_size=self.fixed_kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=self.qo_len,
        )
        if self.qo_len == 1:
            gdn_meta = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=self.batch_size,
                num_decode_tokens=self.batch_size,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=self.batch_size,
                non_spec_query_start_loc=self.static_non_spec_qsl,
                non_spec_state_indices_tensor=self.static_state_indices,
            )
        else:
            extra = self._const_gdn_extra
            assert extra is not None
            gdn_meta = GDNAttentionMetadata(
                num_prefills=self.batch_size,
                num_prefill_tokens=n_tokens,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=n_tokens,
                has_initial_state=extra.has_initial_state,
                non_spec_query_start_loc=self.static_non_spec_qsl,
                non_spec_state_indices_tensor=self.static_state_indices,
                chunk_indices=extra.chunk_indices,
                chunk_offsets=extra.chunk_offsets,
                prefill_query_start_loc=self.static_non_spec_qsl,
                prefill_state_indices=self.static_state_indices,
                prefill_has_initial_state=extra.prefill_has_initial_state,
                nums_dict=extra.nums_dict,
                batch_ptr=extra.batch_ptr,
                token_chunk_offset_ptr=extra.token_chunk_offset_ptr,
            )
        attn_metadata_dict = {name: attn_meta for name in runner.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in runner.gdn_layer_names})
        slot_mapping_dict = {name: self.static_slot_mapping for name in runner.attn_layer_names}
        return attn_metadata_dict, slot_mapping_dict

    def _forward_no_sync(self) -> torch.Tensor:
        """Same op sequence as ``DirectModelRunner._forward_batch``, minus
        the ``torch.cuda.synchronize()`` calls -- calling those DURING
        capture is a documented CUDA-graph-capture violation (raises
        ``cudaErrorStreamCaptureUnsupported``), the same error class the
        sibling project already hit and documented for a different op (a
        boolean-mask-select) during its own CUDA Graph work."""
        runner = self.runner
        attn_metadata_dict, slot_mapping_dict = self._static_metadata_dicts()
        with set_forward_context(attn_metadata_dict, runner.vllm_config, slot_mapping=slot_mapping_dict):
            hidden_states = runner.model.forward(self.static_input_ids, self.static_positions)
        return runner.model.compute_logits(hidden_states)

    def capture(self) -> None:
        """Warm up (uncaptured, on a side stream -- required by
        ``torch.cuda.graph`` before capture) then capture the graph, using
        this object's OWN permanently reserved, disposable warmup slots
        (``self._warmup_slots``) -- NEVER any slot a caller will later pass
        to ``replay()``. This is what makes capture state-neutral for real
        traffic: the only slots capture()'s 3 real warmup executions (plus
        the graph-trace call, which itself executes nothing -- see the
        class docstring) can touch are these reserved slots.

        Warmup content is disposable and never checked for correctness --
        any valid token id works, so a fixed dummy prompt is used (``[0,
        0, 0, 0, 0]``, matching ``DirectModelRunner._warmup``'s own
        convention). This is also why fixed-sizing kv_split_size/
        max_num_splits is required in the first place: the real kv_len
        distribution ``replay()`` sees is expected to differ, often
        drastically, from this disposable warmup shape."""
        if self._graph is not None:
            raise RuntimeError("already captured")
        runner = self.runner
        warmup_slots = self._warmup_slots
        for slot in warmup_slots:
            if runner.slot_kv_len[slot] != 0:
                raise RuntimeError(
                    f"reserved warmup slot {slot} is not fresh -- capture() "
                    "must run before anything else touches this graph's "
                    "own warmup slots, and exactly once per graph object"
                )
            runner.prefill(slot, [0, 0, 0, 0, 0])
        warmup_kv_lengths = [runner.slot_kv_len[s] for s in warmup_slots]
        if self.qo_len == 1:
            warmup_token_ids = [0] * self.batch_size
        else:
            warmup_token_ids = [[0] * self.qo_len for _ in range(self.batch_size)]
        self._fill_buffers(warmup_slots, warmup_token_ids, warmup_kv_lengths)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward_no_sync()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_logits = self._forward_no_sync()
        self._graph = g

    def replay(self, slot_ids: list[int], token_ids, kv_lengths: list[int]) -> torch.Tensor:
        """Replay the captured graph at REAL (slot_ids, token_ids,
        kv_lengths) data -- may (and, per this round's explicit test
        scope, deliberately does) differ drastically from capture()'s
        warmup data, including kv_len values much larger or smaller than
        whatever was used at capture time. Returns logits shaped
        ``[batch_size * qo_len, vocab]`` (request-then-position order).

        No ``torch.cuda.synchronize()`` here (removed 2026-07-17, a
        correctness-review finding): ``_fill_buffers``'s ``.copy_()`` calls
        and ``self._graph.replay()`` are all issued on the SAME (default)
        CUDA stream, so CUDA's own stream-ordering already guarantees the
        graph's kernels observe the freshly-copied buffer contents, and
        that the NEXT call's ``_fill_buffers`` won't overwrite data this
        replay is still reading -- no explicit device-wide sync is needed
        for that. The caller gets an implicit, narrowly-scoped sync for
        free the moment it actually reads back a value (e.g.
        ``.argmax(dim=-1).item()`` on the returned logits). A blanket
        ``torch.cuda.synchronize()`` here would additionally block on any
        OTHER unrelated work queued on the device -- directly working
        against the whole point of using a captured graph to cut CPU-side
        launch/dispatch overhead."""
        if slot_ids == self._warmup_slots or set(slot_ids) & set(self._warmup_slots):
            raise RuntimeError(
                f"slot(s) {set(slot_ids) & set(self._warmup_slots)} are this "
                "graph's own reserved warmup slots -- never replay() against "
                "them, they exist solely for capture()'s internal use"
            )
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        if not (len(slot_ids) == self.batch_size == len(token_ids) == len(kv_lengths)):
            raise ValueError("slot_ids/token_ids/kv_lengths must match batch_size")
        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.runner.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.runner.slot_kv_len[slot]}"
                )
            if not self.runner.slot_gdn_initialized[slot]:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")
        self._fill_buffers(slot_ids, token_ids, kv_lengths)
        self._graph.replay()
        for slot in slot_ids:
            self.runner.slot_kv_len[slot] += self.qo_len
            self.runner.slot_gdn_initialized[slot] = True
        return self._static_logits
