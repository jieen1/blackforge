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


def _ensure_sm120_backend_registered() -> None:
    """register_backend() is a plain dict write (see registry.py's
    _ATTN_OVERRIDES) -- safe to call more than once."""
    register_backend(AttentionBackendEnum.CUSTOM, _SM120_BACKEND_PATH)


def build_vllm_config(
    *,
    model: str,
    kv_cache_dtype: str = "fp8_e4m3",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.5,
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
    )
    return args.create_engine_config()


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

        self._allocate_and_bind_kv_caches()

        # Per-slot bookkeeping: attention kv_len (tokens actually written into
        # the paged KV cache) and GDN "has state been initialized" flag.
        self.slot_kv_len = [0] * num_slots
        self.slot_gdn_initialized = [False] * num_slots

    def _allocate_and_bind_kv_caches(self) -> None:
        kv_caches: dict[str, object] = {}

        any_attn = self.static_forward_context[self.attn_layer_names[0]]
        backend_cls = any_attn.get_attn_backend()
        num_kv_heads = any_attn.num_kv_heads
        head_size = any_attn.head_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype
        num_blocks = self.num_slots * self.blocks_per_slot
        shape = backend_cls.get_kv_cache_shape(
            num_blocks, self.block_size, num_kv_heads, head_size, cache_dtype_str
        )
        torch_dtype = any_attn.kv_cache_torch_dtype
        for name in self.attn_layer_names:
            kv_caches[name] = torch.zeros(shape, dtype=torch_dtype, device=self.device)

        for name in self.gdn_layer_names:
            layer = self.static_forward_context[name]
            conv_shape, ssm_shape = layer.get_state_shape()
            conv_dtype, ssm_dtype = layer.get_state_dtype()
            conv_state = torch.zeros(
                (self.num_slots, *conv_shape), dtype=conv_dtype, device=self.device
            )
            ssm_state = torch.zeros(
                (self.num_slots, *ssm_shape), dtype=ssm_dtype, device=self.device
            )
            kv_caches[name] = (conv_state, ssm_state)

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.static_forward_context, runner_kv_caches)
        self.kv_caches = kv_caches

    def _attention_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> SM120GQAMetadata:
        prior_kv_len = self.slot_kv_len[slot]
        new_kv_len = prior_kv_len + num_new_tokens
        page_size = self.block_size
        first_block = slot * self.blocks_per_slot
        num_pages = (new_kv_len + page_size - 1) // page_size
        if num_pages > self.blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {new_kv_len} exceeds this slot's "
                f"{self.blocks_per_slot * page_size}-token capacity"
            )
        device = self.device
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

    def _gdn_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> GDNAttentionMetadata:
        device = self.device
        state_indices = torch.tensor([slot], dtype=torch.int32, device=device)
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
        has_initial_state = torch.tensor(
            [self.slot_gdn_initialized[slot]], dtype=torch.bool, device=device
        )
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

    def _slot_mapping(self, slot: int, start_pos: int, num_new_tokens: int) -> torch.Tensor:
        """Flat per-token KV-cache write index: block_id * block_size + offset
        -- the same convention vLLM's own paged attention backends use (see
        attention.py's do_kv_cache_update, which reads this from
        ``forward_context.slot_mapping[layer_name]``, NOT from
        ``attn_metadata`` -- easy to miss, and missing it means K/V are never
        written into the cache at all)."""
        first_block = slot * self.blocks_per_slot
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )
        block_ids = first_block + positions // self.block_size
        offsets = positions % self.block_size
        return (block_ids * self.block_size + offsets).to(torch.long)

    def _forward(
        self, slot: int, token_ids: list[int], start_pos: int, *, is_decode: bool
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
        logits = self.model.compute_logits(hidden_states)

        self.slot_kv_len[slot] += num_new_tokens
        self.slot_gdn_initialized[slot] = True
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

    def reset_slot(self, slot: int) -> None:
        """Release a slot for reuse by a new logical request. Does not zero
        the underlying tensors -- the next prefill's has_initial_state=False
        and kv_len bookkeeping starting from 0 is what makes reuse correct,
        matching this project's established fixed-slot-generation design."""
        self.slot_kv_len[slot] = 0
        self.slot_gdn_initialized[slot] = False
