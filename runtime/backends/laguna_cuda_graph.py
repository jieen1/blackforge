"""Laguna CUDA Graph decode — FlashInfer cudagraph wrapper + fast_decode_plan。

利用 FlashInfer 原生 cudagraph 模式（BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)）
和 vLLM vendor 的 fast_decode_plan（每步只做 indptr/last_page_len 的小 H2D copy）。

简化红利：block table 每槽连续（base = phys * blocks_per_slot + j），
paged_kv_indices 在槽位生命周期内不变，每步真正变的只有 last_page_len 和
跨页时的 indptr。

性能目标：消除 51ms Python dispatch overhead，ITL 从 66ms → ~15ms。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from runtime.backends.laguna import LagunaBackend

logger = logging.getLogger("qwen_sm120_runtime.laguna_cuda_graph")


class LagunaCudaGraphDecode:
    """CUDA-graph-captured decode for a FIXED batch size.

    Per (batch_size) instance. Captures the full decode forward
    (metadata build + model.forward + compute_logits) using FlashInfer's
    cudagraph-enabled decode wrappers.

    Usage:
        cg = LagunaCudaGraphDecode(backend, batch_size=1)
        cg.capture()  # warmup + plan + capture

        # In decode loop:
        next_tokens = cg.replay(slot_ids, token_ids, kv_lengths)
    """

    def __init__(self, backend: LagunaBackend, batch_size: int) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.device = backend.device
        self.block_size = backend.block_size
        self.blocks_per_slot = backend.blocks_per_slot
        self.max_kv_len = backend.blocks_per_slot * backend.block_size

        self._graph: torch.cuda.CUDAGraph | None = None
        self._captured = False
        self._logits: torch.Tensor | None = None

        # ── Pre-allocated input buffers (fixed address) ──
        self._input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # ── FlashInfer cudagraph buffers per layer group ──
        # indptr: [batch_size + 1], indices: [batch_size * blocks_per_slot],
        # last_page_len: [batch_size]
        max_pages = batch_size * self.blocks_per_slot
        self._fi_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._fi_indptr_gpu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        self._fi_indices_gpu = torch.zeros(max_pages, dtype=torch.int32, device=self.device)
        self._fi_last_page_len_cpu = torch.zeros(batch_size, dtype=torch.int32, pin_memory=True)
        self._fi_last_page_len_gpu = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        # Staging buffer: write here first, then copy to pinned → GPU.
        # Prevents race if .item() sync is removed in future optimization.
        self._fi_last_page_len_staging = torch.zeros(batch_size, dtype=torch.int32)

        # ── CommonAttentionMetadata pre-allocated fields ──
        self._qsl_gpu = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
        self._qsl_cpu = torch.arange(batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._seq_lens_gpu = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self._block_table = torch.zeros(
            batch_size, self.blocks_per_slot, dtype=torch.int32, device=self.device
        )

        # ── FlashInfer cudagraph decode wrappers (one per layer group) ──
        self._decode_wrappers: dict[tuple, Any] = {}
        self._fi_metadata: dict[tuple, Any] = {}
        self._workspaces: list[torch.Tensor] = []

        # ── Per-slot page-crossing tracker ──
        self._prev_n_blocks: list[int] = [0] * batch_size

    def _init_wrappers(self) -> None:
        """Create FlashInfer cudagraph-enabled decode wrappers per layer group.

        Each wrapper gets its OWN workspace buffer (not shared with the eager
        builder) to prevent prefill from polluting decode's scheduling area.
        """
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        backend = self.backend
        bs = self.batch_size

        for group_key, builder in backend._metadata_builders.items():
            workspace = torch.empty(
                builder._get_workspace_buffer().numel(),
                dtype=torch.uint8,
                device=self.device,
            )
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",  # kv_layout
                use_cuda_graph=True,
                paged_kv_indptr_buffer=self._fi_indptr_gpu[:bs + 1],
                paged_kv_indices_buffer=self._fi_indices_gpu,
                paged_kv_last_page_len_buffer=self._fi_last_page_len_gpu[:bs],
                use_tensor_cores=True,
            )
            self._decode_wrappers[group_key] = wrapper
            self._workspaces.append(workspace)

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Update pre-allocated buffers for replay (vectorized)."""
        from runtime.backends.laguna import _physical_slot

        bs = len(slot_ids)
        ps = self.block_size

        # Vectorized: input_ids, positions, seq_lens
        self._input_ids[:bs] = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        kv_t = torch.tensor(kv_lengths, dtype=torch.long, device=self.device)
        self._positions[:bs] = kv_t
        new_kv = kv_t.int() + 1
        self._seq_lens_gpu[:bs] = new_kv

        # Per-slot: block_table, slot_mapping, last_page_len, indices
        n_blocks_t = (new_kv + ps - 1) // ps
        self._fi_indptr_cpu[0] = 0
        for i in range(bs):
            phys = _physical_slot(slot_ids[i])
            base = phys * self.blocks_per_slot
            nb = int(n_blocks_t[i].item())

            self._block_table[i, :nb] = torch.arange(
                base, base + nb, dtype=torch.int32, device=self.device
            )

            pos = kv_lengths[i]
            self._slot_mapping[i] = (base + pos // ps) * ps + pos % ps

            lpl = int(new_kv[i].item()) % ps
            self._fi_last_page_len_cpu[i] = lpl if lpl != 0 else ps

            self._fi_indptr_cpu[i + 1] = self._fi_indptr_cpu[i] + nb

            start = int(self._fi_indptr_cpu[i].item())
            self._fi_indices_gpu[start:start + nb] = torch.arange(
                base, base + nb, dtype=torch.int32, device=self.device
            )

        self._fi_indptr_gpu[:bs + 1].copy_(self._fi_indptr_cpu[:bs + 1], non_blocking=True)
        self._fi_last_page_len_gpu[:bs].copy_(self._fi_last_page_len_cpu[:bs], non_blocking=True)

    def _run_plan(self, slot_ids: list[int], kv_lengths: list[int]) -> None:
        """Run fast_decode_plan on all layer group wrappers."""
        from vllm.v1.attention.backends.flashinfer import fast_plan_decode


        backend = self.backend
        bs = len(slot_ids)

        for group_key, wrapper in self._decode_wrappers.items():
            wl, nqh, nkvh = group_key
            head_dim = backend.head_dim
            page_size = self.block_size

            # Determine kv_cache_dtype
            kv_dtype = torch.float8_e4m3fn if "fp8" in backend._cache_dtype_str else torch.bfloat16

            builder_sm_scale = backend._metadata_builders[group_key].sm_scale
            fast_plan_decode(
                wrapper,
                indptr_cpu=self._fi_indptr_cpu[:bs + 1],
                indices=self._fi_indices_gpu,
                last_page_len_cpu=self._fi_last_page_len_cpu[:bs],
                num_qo_heads=nqh,
                num_kv_heads=nkvh,
                head_dim=head_dim,
                page_size=page_size,
                pos_encoding_mode="NONE",
                window_left=wl,
                logits_soft_cap=None,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_dtype,
                sm_scale=builder_sm_scale,
                non_blocking=True,
                fixed_split_size=2048,
                disable_split_kv=False,
            )
            wrapper._sm_scale = builder_sm_scale

    def _build_metadata_and_forward(self) -> torch.Tensor:
        """Build FlashInferMetadata from pre-allocated buffers and run forward."""
        from runtime.compat_vllm import (
            set_current_vllm_config,
            set_forward_context,
        )

        backend = self.backend
        bs = self.batch_size

        # Build FlashInferMetadata using cudagraph wrappers
        from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, wrapper in self._decode_wrappers.items():
            metadata = FlashInferMetadata(
                num_actual_tokens=bs,
                slot_mapping=self._slot_mapping[:bs],
                q_data_type_prefill=torch.bfloat16,
                q_data_type_decode=torch.bfloat16,
                num_decodes=bs,
                num_decode_tokens=bs,
                num_prefills=0,
                num_prefill_tokens=0,
                causal=True,
                use_cascade=False,
                prefill=None,
                decode=FIDecode(wrapper=wrapper),
                cascade_wrapper=None,
            )
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = self._slot_mapping[:bs]

        with set_current_vllm_config(backend.vllm_config):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict
            ):
                hidden_states = backend.model.forward(
                    self._input_ids[:bs], self._positions[:bs]
                )

        return backend.model.compute_logits(hidden_states)

    def capture(self) -> None:
        """Warmup → plan → capture the decode forward."""
        if self._captured:
            return

        backend = self.backend
        bs = self.batch_size

        # Reserve warmup slots (last bs slots)
        warmup_slots = list(range(backend.num_slots - bs, backend.num_slots))
        dummy_tokens = [1] * bs
        dummy_kv_lens = [32] * bs  # 2 pages

        logger.info("Capturing Laguna CUDA Graph: batch_size=%d", bs)

        # Initialize wrappers
        self._init_wrappers()

        # Fill buffers with dummy data
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)

        # Warmup: run plan + forward on side stream (3x)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)
                self._run_plan(warmup_slots, dummy_kv_lens)
                self._build_metadata_and_forward()
        side_stream.synchronize()

        # Final fill + plan before capture
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)
        self._run_plan(warmup_slots, dummy_kv_lens)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._logits = self._build_metadata_and_forward()

        self._graph = graph
        self._captured = True

        logger.info("Laguna CUDA Graph captured: batch_size=%d", bs)

    def replay(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> list[int]:
        """Replay the captured graph with real data.

        FlashInfer contract: plan→run must be paired every step.
        fast_plan_decode is cheap (small H2D copies), never skip it.
        """
        if not self._captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        bs = len(slot_ids)
        ps = self.block_size
        bps = self.blocks_per_slot

        for i in range(bs):
            kvl = kv_lengths[i]
            new_kv = kvl + 1

            self._input_ids[i] = token_ids[i]
            self._positions[i] = kvl

            phys = slot_ids[i] + 1  # inlined _physical_slot
            base = phys * bps
            self._slot_mapping[i] = base * ps + kvl

            lpl = new_kv % ps
            self._fi_last_page_len_staging[i] = lpl if lpl != 0 else ps

            n_blocks = (new_kv + ps - 1) // ps
            if n_blocks != self._prev_n_blocks[i]:
                self._prev_n_blocks[i] = n_blocks
                self._block_table[i, :n_blocks] = torch.arange(
                    base, base + n_blocks, dtype=torch.int32, device=self.device
                )
                self._fi_indptr_cpu[0] = 0
                for j in range(bs):
                    nb = self._prev_n_blocks[j]
                    self._fi_indptr_cpu[j + 1] = self._fi_indptr_cpu[j] + nb
                    p2 = slot_ids[j] + 1
                    b2 = p2 * bps
                    start = int(self._fi_indptr_cpu[j].item())
                    self._fi_indices_gpu[start:start + nb] = torch.arange(
                        b2, b2 + nb, dtype=torch.int32, device=self.device
                    )

        self._fi_last_page_len_cpu[:bs].copy_(self._fi_last_page_len_staging[:bs])
        # fast_decode_plan (cudagraph mode) does NOT copy CPU→GPU; caller must.
        # These GPU buffers are the wrapper's fixed-address plan inputs.
        self._fi_indptr_gpu[:bs + 1].copy_(self._fi_indptr_cpu[:bs + 1], non_blocking=True)
        self._fi_last_page_len_gpu[:bs].copy_(self._fi_last_page_len_cpu[:bs], non_blocking=True)
        self._run_plan(slot_ids, kv_lengths)
        self._graph.replay()

        if bs == 1:
            return [int(self._logits[0].argmax(dim=-1).item())]
        return [int(self._logits[i].argmax(dim=-1).item()) for i in range(bs)]

    def reset(self) -> None:
        """Reset per-slot tracking state for a fresh generation.

        Zeros workspace buffers to prevent capture-warmup residue from
        affecting the first replay (run1 vs run2 divergence).
        """
        self._prev_n_blocks = [0] * self.batch_size
        for ws in self._workspaces:
            ws.zero_()

    @property
    def is_captured(self) -> bool:
        return self._captured
