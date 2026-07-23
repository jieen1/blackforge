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
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

logger = logging.getLogger("qwen_sm120_runtime.laguna_backend")

RESERVED_PHYSICAL_SLOTS = 1


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

        # Allocate KV caches
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        self._cache_dtype_str = cache_dtype_str
        num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot
        self.kv_caches: dict[str, torch.Tensor] = {}
        for name in self.attn_layer_names:
            layer = sfc[name]
            backend_cls = layer.get_attn_backend()
            shape = backend_cls.get_kv_cache_shape(
                num_blocks, block_size, layer.num_kv_heads, layer.head_size, cache_dtype_str
            )
            self.kv_caches[name] = torch.zeros(
                shape, dtype=layer.kv_cache_torch_dtype, device=self.device
            )
        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(self.kv_caches, sfc, runner_kv_caches)

        # Per-slot state
        self.slot_kv_len: list[int] = [0] * num_slots
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]

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

        # Expose for engine compatibility
        self.num_speculative_tokens = 0
        self.spec = None
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
        """Fill pre-allocated buffers for decode (avoids per-step tensor allocation)."""
        batch_size = len(slot_ids)
        for i in range(batch_size):
            self._decode_input_ids[i] = token_ids[i]
            self._decode_positions[i] = kv_lengths[i]
            self._decode_seq_lens[i] = kv_lengths[i] + 1

            phys = _physical_slot(slot_ids[i])
            base = phys * self.blocks_per_slot
            new_kv_len = kv_lengths[i] + 1
            n_blocks = (new_kv_len + self.block_size - 1) // self.block_size
            self._decode_block_table[i, :n_blocks] = torch.arange(
                base, base + n_blocks, dtype=torch.int32, device=self.device
            )
            if n_blocks < self.blocks_per_slot:
                self._decode_block_table[i, n_blocks:] = 0

            pos = kv_lengths[i]
            bid = base + pos // self.block_size
            off = pos % self.block_size
            self._decode_slot_mapping[i] = bid * self.block_size + off

    def _build_common_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
    ):
        """Build CommonAttentionMetadata for vLLM's FlashInferMetadataBuilder."""
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        page_size = self.block_size

        # New KV lengths (after this forward)
        new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]

        # query_start_loc (use pre-allocated for decode)
        if is_decode and max(qo_lens) == 1:
            query_start_loc = self._decode_qsl_gpu[:num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[:num_reqs + 1]
        else:
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

        # Use pre-allocated buffers for decode, allocate for prefill
        if is_decode and max(qo_lens) == 1:
            seq_lens = self._decode_seq_lens[:num_reqs]
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = self._decode_block_table[:num_reqs, :max_blocks]
            slot_mapping = self._decode_slot_mapping[:num_reqs]
        else:
            # seq_lens (new KV lengths)
            seq_lens_np = np.array(new_kv_lens, dtype=np.int32)
            seq_lens = torch.from_numpy(seq_lens_np).to(self.device)

            # block_table_tensor: [num_reqs, max_blocks_per_req]
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, n_blocks) in enumerate(
                zip(slot_ids, [(kvl + page_size - 1) // page_size for kvl in new_kv_lens])
            ):
                phys = _physical_slot(slot)
                base = phys * self.blocks_per_slot
                for j in range(n_blocks):
                    block_table[i, j] = base + j

            # slot_mapping
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
        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, builder in self._metadata_builders.items():
            with set_current_vllm_config(self.vllm_config):
                metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_meta,
                )
            for name in self._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = common_meta.slot_mapping

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
            hidden_states = self.model.forward(input_ids, positions)

        logits = self.model.compute_logits(hidden_states)
        return logits

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Prefill prompt and return the greedy first token."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
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
        sampling_params_list: list[SamplingParams],
        **kwargs: Any,
    ) -> list[int]:
        kv_lengths = [self.slot_kv_len[s] for s in slot_ids]
        logits = self._forward(
            slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True
        )
        next_tokens = []
        for i, slot in enumerate(slot_ids):
            params = sampling_params_list[i]
            row = logits[i].unsqueeze(0)
            gen = make_generator(params.seed)
            tok = int(sample_from_logits(row, params, generator=gen).item())
            next_tokens.append(tok)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        return next_tokens

    def reset_slot(self, slot: int) -> None:
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []

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
            max_tokens=max_tokens,
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
