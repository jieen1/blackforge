"""E3: Laguna-S-2.1 Backend — direct model.forward() without vLLM's LLM engine.

Loads the model via compat_vllm.get_model(), allocates KV caches, builds
FlashInfer attention metadata via vLLM's own FlashInferMetadataBuilder, and
drives prefill/decode forward passes directly.

Architecture: 48 layers (12 full attn 48-head + 36 SWA 72-head window=512),
47 MoE layers, 8 KV heads / head_dim=128, NVFP4 quantized.

Roadmap ref: E3 Laguna L2 = "LagunaBackend — 过质量链"
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from runtime.block_pool import ChunkedPrefillState
from runtime.compat_vllm import (
    VllmConfig,
    bind_kv_cache,
    get_distributed_init_method,
    get_model,
    get_open_port,
    init_worker_distributed_environment,
    set_current_vllm_config,
    set_forward_context,
)
from runtime.logprobs import compute_logprobs
from runtime.model_spec import ModelSpec
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

logger = logging.getLogger("qwen_sm120_runtime.laguna_backend")

RESERVED_PHYSICAL_SLOTS = 1

# Ring KV for SWA layers: parameterized for DFlash verify qo_max=16
# Formula: cdiv(window - 1 + qo_max, block_size) + 1
# qo_max=1 → 33, qo_max=16 → 34 (审查阻断①)
SWA_QO_MAX = 16


def _ring_blocks_for_window(window: int, block_size: int, qo_max: int = SWA_QO_MAX) -> int:
    return -(-( window - 1 + qo_max) // block_size) + 1  # cdiv + 1


def _physical_slot(slot: int) -> int:
    return slot + RESERVED_PHYSICAL_SLOTS


class LagunaBackend:
    """Direct model runner for Laguna-S-2.1-NVFP4.

    Uses vLLM's own FlashInferMetadataBuilder for correct attention metadata.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_slots: int = 4,
        block_size: int = 16,
        blocks_per_slot: int = 512,
    ) -> None:
        import os as _os
        import sys as _sys

        _venv_bin = _os.path.dirname(_sys.executable)
        if _venv_bin not in _os.environ.get("PATH", ""):
            _os.environ["PATH"] = _venv_bin + ":" + _os.environ.get("PATH", "")

        torch.set_grad_enabled(False)

        self.vllm_config = vllm_config
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # Apply A2 patches before loading
        from runtime.nvfp4_custom_gemm import patch_nvfp4_custom_gemm
        from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct

        patch_nvfp4_prefer_cutlass_direct()
        patch_nvfp4_custom_gemm()

        # Load model
        with set_current_vllm_config(vllm_config):
            init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            init_worker_distributed_environment(
                vllm_config, rank=0, distributed_init_method=init_method, local_rank=0
            )
            self.model = get_model(vllm_config=vllm_config)

        # Initialize workspace manager for MoE layers
        from runtime.compat_vllm import init_flashinfer_workspace

        init_flashinfer_workspace(self.device)

        # Discover attention layers from static_forward_context
        sfc = vllm_config.compilation_config.static_forward_context
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_attn_backend"):
                self.attn_layer_names.append(name)
        logger.info("Laguna: %d attention layers discovered", len(self.attn_layer_names))

        # Group layers by (num_qo_heads, num_kv_heads, window_left)
        # Each group gets its own FlashInferMetadataBuilder
        hf_config = vllm_config.model_config.hf_config
        layer_types = getattr(hf_config, "layer_types", None)
        sliding_window = getattr(hf_config, "sliding_window", None)

        self._layer_groups: dict[tuple, list[str]] = {}
        for name in self.attn_layer_names:
            layer = sfc[name]
            nqh = layer.num_heads
            nkvh = layer.num_kv_heads
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            if layer_types is not None and sliding_window is not None:
                if layer_idx is not None and layer_idx < len(layer_types):
                    is_sliding = layer_types[layer_idx] == "sliding_attention"
                    wl = (sliding_window - 1) if is_sliding else -1
                else:
                    wl = -1
            else:
                wl = -1
            key = (wl, nqh, nkvh)
            self._layer_groups.setdefault(key, []).append(name)

        logger.info(
            "Laguna: layer groups: %s",
            {f"wl={k[0]},qh={k[1]},kvh={k[2]}": len(v) for k, v in self._layer_groups.items()},
        )

        # Create FlashInferMetadataBuilder for each group
        from runtime.compat_vllm import get_flashinfer_metadata_builder
        FlashInferMetadataBuilder = get_flashinfer_metadata_builder()

        self._metadata_builders: dict[tuple, FlashInferMetadataBuilder] = {}
        with set_current_vllm_config(vllm_config):
            for group_key, layer_names in self._layer_groups.items():
                first_layer = sfc[layer_names[0]]
                kv_cache_spec = first_layer.get_kv_cache_spec(vllm_config)
                builder = FlashInferMetadataBuilder(
                    kv_cache_spec=kv_cache_spec,
                    layer_names=layer_names,
                    vllm_config=vllm_config,
                    device=self.device,
                )
                self._metadata_builders[group_key] = builder

        # ── Classify layers: full attention vs SWA ──
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        self._cache_dtype_str = cache_dtype_str
        self._full_layer_names: list[str] = []
        self._swa_layer_names: list[str] = []
        self._swa_window: int = 0
        for name in self.attn_layer_names:
            layer = sfc[name]
            spec = layer.get_kv_cache_spec(vllm_config)
            spec_cls = type(spec).__name__
            if spec_cls == "SlidingWindowSpec":
                self._swa_layer_names.append(name)
                self._swa_window = spec.sliding_window
            else:
                self._full_layer_names.append(name)

        self._ring_blocks_per_slot = (
            _ring_blocks_for_window(self._swa_window, block_size)
            if self._swa_window > 0
            else 0
        )
        self._ring_slots_per_slot = self._ring_blocks_per_slot * block_size
        logger.info(
            "Laguna: %d full layers, %d SWA layers (window=%d, ring_blocks=%d/slot)",
            len(self._full_layer_names),
            len(self._swa_layer_names),
            self._swa_window,
            self._ring_blocks_per_slot,
        )

        # ── Allocate KV caches: per-group ──
        num_phys = num_slots + RESERVED_PHYSICAL_SLOTS
        full_num_blocks = num_phys * blocks_per_slot
        ring_num_blocks = num_phys * self._ring_blocks_per_slot
        self.kv_caches: dict[str, torch.Tensor] = {}
        for name in self.attn_layer_names:
            layer = sfc[name]
            backend_cls = layer.get_attn_backend()
            is_swa = name in self._swa_layer_names
            n_blocks = ring_num_blocks if is_swa else full_num_blocks
            shape = backend_cls.get_kv_cache_shape(
                n_blocks, block_size, layer.num_kv_heads, layer.head_size, cache_dtype_str
            )
            self.kv_caches[name] = torch.zeros(
                shape, dtype=layer.kv_cache_torch_dtype, device=self.device
            )
        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(self.kv_caches, sfc, runner_kv_caches)

        # ── Persistent prefill scratch for SWA layers (审查非阻断③) ──
        # Allocated once, reused across slots. Not zeroed (causal mask
        # guarantees no read-before-write within the window).
        self._swa_scratch: dict[str, torch.Tensor] = {}
        if self._swa_layer_names:
            for name in self._swa_layer_names:
                layer = sfc[name]
                backend_cls = layer.get_attn_backend()
                shape = backend_cls.get_kv_cache_shape(
                    blocks_per_slot, block_size, layer.num_kv_heads,
                    layer.head_size, cache_dtype_str,
                )
                self._swa_scratch[name] = torch.empty(
                    shape, dtype=layer.kv_cache_torch_dtype, device=self.device
                )

        # Per-slot state
        self.slot_kv_len: list[int] = [0] * num_slots
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]
        # E1: mirrors DirectModelRunner.block_table's role as a per-slot
        # "has this slot ever been touched" dirty flag for admission. Laguna
        # has no block-table indirection (physical slot is a direct
        # arithmetic mapping, see _physical_slot) -- this list is never
        # populated, only kept empty/falsy so ServerEngine's shared admission
        # check (`slot_kv_len[slot] != 0 or block_table[slot]`) works
        # unmodified against either backend.
        self.block_table: list[list[int]] = [[] for _ in range(num_slots)]

        # Pre-allocated decode buffers (avoid per-step tensor allocation)
        max_batch = num_slots
        self._decode_input_ids = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self._decode_positions = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self._decode_seq_lens = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self._decode_block_table = torch.zeros(
            max_batch, blocks_per_slot, dtype=torch.int32, device=self.device
        )
        self._decode_slot_mapping = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        # query_start_loc: [0, 1, 2, ..., batch_size] for decode (qo_len=1)
        self._decode_qsl_gpu = torch.arange(max_batch + 1, dtype=torch.int32, device=self.device)
        self._decode_qsl_cpu = torch.arange(max_batch + 1, dtype=torch.int32, pin_memory=True)

        # SWA ring decode buffers (separate from full-attention buffers)
        if self._ring_blocks_per_slot > 0:
            self._swa_decode_block_table = torch.zeros(
                max_batch, self._ring_blocks_per_slot, dtype=torch.int32, device=self.device
            )
            self._swa_decode_slot_mapping = torch.zeros(
                max_batch, dtype=torch.long, device=self.device
            )
            self._swa_decode_seq_lens = torch.zeros(
                max_batch, dtype=torch.int32, device=self.device
            )

        # Expose for engine compatibility
        self.num_speculative_tokens = 0
        model_id = getattr(vllm_config.model_config, "model", "poolside/Laguna-S-2.1-NVFP4")
        architecture = getattr(hf_config, "architectures", ["LagunaForCausalLM"])[0]
        # E1: no MTP draft model and no GDN layers -- Laguna has neither
        # speculative decoding (DFlash is planned, roadmap L3, not wired
        # into this backend yet) nor a GDN/SSM recursive state; every
        # discovered layer is a (full or sliding-window) attention layer.
        self.spec = ModelSpec.from_runner_init(
            model_id=model_id,
            architecture=architecture,
            attn_layer_names=self.attn_layer_names,
            gdn_layer_names=[],
            mtp_model_id=None,
            num_speculative_tokens=0,
            kv_dtype=cache_dtype_str,
            block_size=block_size,
        )
        self.num_qo_heads = sfc[self.attn_layer_names[0]].num_heads
        self.num_kv_heads = sfc[self.attn_layer_names[0]].num_kv_heads
        self.head_dim = sfc[self.attn_layer_names[0]].head_size

        logger.info(
            "LagunaBackend initialized: %d slots, block_size=%d",
            num_slots,
            block_size,
        )

    def _fill_decode_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Fill pre-allocated buffers for decode (avoids per-step tensor allocation).

        Full-attention layers: standard contiguous block_table.
        SWA layers: ring block_table (block-aligned window) + ring slot_mapping.
        """
        batch_size = len(slot_ids)
        bs = self.block_size
        for i in range(batch_size):
            self._decode_input_ids[i] = token_ids[i]
            self._decode_positions[i] = kv_lengths[i]
            self._decode_seq_lens[i] = kv_lengths[i] + 1

            phys = _physical_slot(slot_ids[i])
            pos = kv_lengths[i]
            new_kv_len = kv_lengths[i] + 1

            # ── Full-attention block_table / slot_mapping ──
            full_base = phys * self.blocks_per_slot
            n_blocks = (new_kv_len + bs - 1) // bs
            self._decode_block_table[i, :n_blocks] = torch.arange(
                full_base, full_base + n_blocks, dtype=torch.int32, device=self.device
            )
            if n_blocks < self.blocks_per_slot:
                self._decode_block_table[i, n_blocks:] = 0
            self._decode_slot_mapping[i] = (
                (full_base + pos // bs) * bs + pos % bs
            )

            # ── SWA ring block_table / slot_mapping ──
            if self._ring_blocks_per_slot > 0:
                ring_base = phys * self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window = self._swa_window

                # Block-aligned window start
                window_start = max(0, pos - window + 1)
                aligned_start = (window_start // bs) * bs
                aligned_len = pos + 1 - aligned_start
                n_ring = (aligned_len + bs - 1) // bs

                for j in range(n_ring):
                    actual_pos = aligned_start + j * bs
                    ring_block = (actual_pos % ring_slots) // bs
                    self._swa_decode_block_table[i, j] = ring_base + ring_block
                if n_ring < self._ring_blocks_per_slot:
                    self._swa_decode_block_table[i, n_ring:] = 0

                self._swa_decode_seq_lens[i] = aligned_len

                # Ring slot_mapping for the new decode token
                ring_block = (pos % ring_slots) // bs
                ring_off = pos % bs
                self._swa_decode_slot_mapping[i] = (
                    (ring_base + ring_block) * bs + ring_off
                )

    def _build_common_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
    ):
        """Build CommonAttentionMetadata for full-attention layers."""
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        page_size = self.block_size
        new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]

        if is_decode and max(qo_lens) == 1:
            query_start_loc = self._decode_qsl_gpu[:num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[:num_reqs + 1]
        else:
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

        if is_decode and max(qo_lens) == 1:
            seq_lens = self._decode_seq_lens[:num_reqs]
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = self._decode_block_table[:num_reqs, :max_blocks]
            slot_mapping = self._decode_slot_mapping[:num_reqs]
        else:
            seq_lens_np = np.array(new_kv_lens, dtype=np.int32)
            seq_lens = torch.from_numpy(seq_lens_np).to(self.device)
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, n_blocks) in enumerate(
                zip(slot_ids, [(kvl + page_size - 1) // page_size for kvl in new_kv_lens])
            ):
                phys = _physical_slot(slot)
                base = phys * self.blocks_per_slot
                for j in range(n_blocks):
                    block_table[i, j] = base + j
            mappings = []
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                phys = _physical_slot(slot)
                for j in range(qo):
                    pos = kv_len + j
                    bid = phys * self.blocks_per_slot + pos // self.block_size
                    off = pos % self.block_size
                    mappings.append(bid * self.block_size + off)
            slot_mapping = torch.tensor(mappings, dtype=torch.long, device=self.device)

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max(qo_lens),
            max_seq_len=max(new_kv_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

    def _build_swa_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
    ):
        """Build CommonAttentionMetadata for SWA layers (ring block_table)."""
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot

        if is_decode and max(qo_lens) == 1:
            query_start_loc = self._decode_qsl_gpu[:num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[:num_reqs + 1]
            seq_lens = self._swa_decode_seq_lens[:num_reqs]
            max_blocks = max(
                int(self._swa_decode_seq_lens[i].item()) for i in range(num_reqs)
            )
            max_blocks = (max_blocks + bs - 1) // bs
            block_table = self._swa_decode_block_table[:num_reqs, :max_blocks]
            slot_mapping = self._swa_decode_slot_mapping[:num_reqs]
            max_seq = int(seq_lens.max().item())
        else:
            # Prefill: use full block_table (scratch KV is full-size)
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

            new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]
            seq_lens_np = np.array(new_kv_lens, dtype=np.int32)
            seq_lens = torch.from_numpy(seq_lens_np).to(self.device)
            max_blocks = max((kvl + bs - 1) // bs for kvl in new_kv_lens)
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, n_blocks) in enumerate(
                zip(slot_ids, [(kvl + bs - 1) // bs for kvl in new_kv_lens])
            ):
                # Prefill scratch uses contiguous blocks [0, blocks_per_slot)
                for j in range(n_blocks):
                    block_table[i, j] = j
            mappings = []
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                for j in range(qo):
                    pos = kv_len + j
                    bid = pos // bs
                    off = pos % bs
                    mappings.append(bid * bs + off)
            slot_mapping = torch.tensor(mappings, dtype=torch.long, device=self.device)
            max_seq = max(new_kv_lens)

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max(qo_lens),
            max_seq_len=max_seq,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

    def _forward(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        qo_len: int = 1,
        is_decode: bool = True,
    ) -> torch.Tensor:
        """Run one forward pass for a batch of slots."""
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs

        if is_decode and qo_len == 1:
            self._fill_decode_buffers(slot_ids, token_ids, kv_lengths)

        # Build CommonAttentionMetadata
        common_meta = self._build_common_attn_metadata(
            slot_ids, kv_lengths, qo_lens, is_decode
        )

        # Build per-group FlashInferMetadata using vLLM's builder
        # SWA groups use ring metadata; full groups use standard metadata
        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        swa_meta = None
        if self._ring_blocks_per_slot > 0 and self._swa_layer_names:
            swa_meta = self._build_swa_attn_metadata(
                slot_ids, kv_lengths, qo_lens, is_decode
            )

        for group_key, builder in self._metadata_builders.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_meta if (is_swa_group and swa_meta is not None) else common_meta
            with set_current_vllm_config(self.vllm_config):
                metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=meta,
                )
            for name in self._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = meta.slot_mapping

        # Build input tensors (use pre-allocated buffers for decode)
        if is_decode and qo_len == 1:
            input_ids = self._decode_input_ids[:num_reqs]
            positions = self._decode_positions[:num_reqs]
        else:
            if qo_len == 1:
                flat_token_ids = token_ids
            elif num_reqs == 1:
                flat_token_ids = token_ids
            else:
                flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

            input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
            positions_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                positions_list.extend(range(kv_len, kv_len + qo))
            positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            result = self.model.forward(input_ids, positions)

        # Handle tuple return when aux_hidden_state_layers is set (DFlash)
        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result

        logits = self.model.compute_logits(hidden_states)
        return logits

    def _prefill_with_swa_scratch(
        self, slot: int, prompt_ids: list[int]
    ) -> torch.Tensor:
        """Run prefill with SWA layers rebound to scratch, then copy to ring."""
        sfc = self.static_forward_context
        bs = self.block_size

        # Rebind SWA layers to scratch KV
        if self._swa_scratch:
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]

        logits = self._forward(
            [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
        )

        # Copy last window from scratch to ring — slab copy (审查非阻断④)
        # At most 3 contiguous slabs due to ring wrap-around.
        if self._swa_scratch:
            prompt_len = len(prompt_ids)
            window = self._swa_window
            ring_slots = self._ring_slots_per_slot
            phys = _physical_slot(slot)
            ring_base = phys * self._ring_blocks_per_slot
            window_start = max(0, prompt_len - window)
            n_copy = prompt_len - window_start

            # Build slab list: [(src_start_pos, dst_ring_slot, count), ...]
            slabs: list[tuple[int, int, int]] = []
            pos = window_start
            while pos < prompt_len:
                ring_slot = pos % ring_slots
                # How many consecutive positions fit before ring wraps?
                until_wrap = ring_slots - ring_slot
                # How many fit in the current scratch block?
                src_off = pos % bs
                until_block_end = bs - src_off
                count = min(until_wrap, until_block_end, prompt_len - pos)
                slabs.append((pos, ring_slot, count))
                pos += count

            for name in self._swa_layer_names:
                scratch = self._swa_scratch[name]
                ring = self.kv_caches[name]
                for src_pos, dst_ring_slot, count in slabs:
                    sb = src_pos // bs
                    so = src_pos % bs
                    db = dst_ring_slot // bs + ring_base
                    do = dst_ring_slot % bs
                    # scratch[sb, :, so:so+count] → ring[db, :, do:do+count]
                    ring[db, :, do:do + count] = scratch[sb, :, so:so + count]

            # Rebind SWA layers back to ring KV
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self.kv_caches[name]

        return logits

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Prefill prompt and return the greedy first token."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
        if self._swa_scratch:
            logits = self._prefill_with_swa_scratch(slot, prompt_ids)
        else:
            logits = self._forward(
                [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
            )
        first_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token

    def prefill_sampled(
        self, slot: int, prompt_ids: list[int], params: SamplingParams
    ) -> int:
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
        if self._swa_scratch:
            logits = self._prefill_with_swa_scratch(slot, prompt_ids)
        else:
            logits = self._forward(
                [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
            )
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        first_token = int(
            sample_from_logits(last_logits, params, generator=gen).item()
        )
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token

    def decode(self, slot: int, token_id: int) -> int:
        kv_len = self.slot_kv_len[slot]
        logits = self._forward([slot], [token_id], [kv_len], qo_len=1, is_decode=True)
        next_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] += 1
        self.slot_committed_tokens[slot].append(token_id)
        return next_token

    def decode_sampled(
        self, slot: int, token_id: int, params: SamplingParams
    ) -> int:
        kv_len = self.slot_kv_len[slot]
        logits = self._forward([slot], [token_id], [kv_len], qo_len=1, is_decode=True)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        next_token = int(
            sample_from_logits(last_logits, params, generator=gen).item()
        )
        self.slot_kv_len[slot] += 1
        self.slot_committed_tokens[slot].append(token_id)
        return next_token

    def decode_batch(self, slot_ids: list[int], token_ids: list[int]) -> list[int]:
        kv_lengths = [self.slot_kv_len[s] for s in slot_ids]
        logits = self._forward(
            slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True
        )
        next_tokens = []
        for i, slot in enumerate(slot_ids):
            next_token = int(logits[i].argmax(dim=-1).item())
            next_tokens.append(next_token)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        return next_tokens

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """Decode one token per slot with per-request sampling params.

        Signature matches ``DirectModelRunner.decode_batch_sampled`` (E1: the
        two backends share ServerEngine's calling convention) -- greedy is
        temperature=0, a plain special case of sampling, per B1.
        """
        logits = self._forward(
            slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True
        )
        next_tokens: list[int] = []
        for i, (slot, params) in enumerate(zip(slot_ids, params_list)):
            if params.is_greedy:
                tok = int(logits[i].argmax(dim=-1).item())
            else:
                row = logits[i].unsqueeze(0)
                gen = make_generator(params.seed)
                tok = int(sample_from_logits(row, params, generator=gen).item())
            next_tokens.append(tok)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        if return_logprobs:
            lp_list = [
                compute_logprobs(logits[i].unsqueeze(0), [next_tokens[i]], top_k=top_logprobs)[0]
                for i in range(len(next_tokens))
            ]
            return next_tokens, lp_list
        return next_tokens

    def reset_slot(self, slot: int) -> None:
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []
        phys = _physical_slot(slot)
        # Full-attention layers: clear blocks_per_slot blocks
        full_start = phys * self.blocks_per_slot
        full_end = full_start + self.blocks_per_slot
        for name in self._full_layer_names:
            self.kv_caches[name][full_start:full_end].zero_()
        # SWA layers: clear only ring_blocks_per_slot blocks
        if self._ring_blocks_per_slot > 0:
            ring_start = phys * self._ring_blocks_per_slot
            ring_end = ring_start + self._ring_blocks_per_slot
            for name in self._swa_layer_names:
                self.kv_caches[name][ring_start:ring_end].zero_()

    def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
        """E1 stub: Laguna has no persistent content-addressed prefix cache
        yet (roadmap L2/L3 TODO) -- every admission is a cold miss."""
        return 0

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
    ) -> ChunkedPrefillState:
        """E1: one-shot prefill wrapper matching DirectModelRunner's chunked-
        prefill contract so ServerEngine's admission path is backend-neutral.

        Laguna has no incremental chunking yet (TODO, tracked for roadmap
        L2/L3): this processes each slot's WHOLE prompt in one call and
        always returns ``done=True`` immediately. A single very long prompt
        will therefore block the engine thread for its entire prefill
        instead of interleaving with other slots' decode rounds -- unlike
        the Qwen path's true incremental chunking (A5/B4).
        """
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not slots:
            return ChunkedPrefillState(done=True, result={})
        result: dict[int, dict] = {}
        for slot, prompt in zip(slots, prompts_per_slot):
            first_token = self.prefill(slot, prompt)
            result[slot] = {"anchor": first_token, "draft_tokens": []}
        return ChunkedPrefillState(done=True, result=result)

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """Laguna prefill is never incremental; state is always already done."""
        return state.done

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[int]:
        slot = 0
        self.reset_slot(slot)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p if top_p < 1.0 else 1.0,
            top_k=top_k if top_k > 0 else 0,
        )
        if temperature == 0:
            first = self.prefill(slot, prompt_ids)
        else:
            first = self.prefill_sampled(slot, prompt_ids, params)
        tokens = [first]
        for _ in range(max_tokens - 1):
            if temperature == 0:
                tok = self.decode(slot, tokens[-1])
            else:
                tok = self.decode_sampled(slot, tokens[-1], params)
            tokens.append(tok)
            if tok in (2, 24):  # Laguna EOS (generation_config.json)
                break
        self.reset_slot(slot)
        return tokens
