"""Laguna CUDA Graph capture for decode — 消除 Python overhead。

Laguna 是纯 transformer + MoE（无 GDN/SSM），CUDA Graph 比 Qwen 简单：
- 无 GDN 状态管理
- 无 MTP verify
- FlashInfer attention（非 SM120GQA）

设计要点（沿用 CapturedBatchDecodeGraph 的核心原则）：
1. 所有 tensor 预分配固定地址，replay 用 .copy_() 更新
2. FlashInfer plan() 在 capture 时执行，replay 前更新底层 tensor
3. 每个 (batch_size) 一个 graph 实例

性能目标：消除 _forward() 中的 Python overhead（metadata 构建 + tensor 分配），
将 decode ITL 从 ~70ms 降到接近 vLLM 的 ~47ms。

状态：骨架代码，待 GPU 验证。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from runtime.backends.laguna import LagunaBackend

logger = logging.getLogger("qwen_sm120_runtime.laguna_cuda_graph")


class LagunaDecodeGraph:
    """CUDA-graph-captured decode for a FIXED batch size.

    Usage:
        graph = LagunaDecodeGraph(backend, batch_size=4)
        graph.capture()

        # In decode loop:
        next_tokens = graph.replay(slot_ids, token_ids, kv_lengths)
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

        # ── Pre-allocated fixed-address buffers ──
        # Input tensors
        self._input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Metadata tensors (max size for batch_size)
        max_blocks_per_req = self.blocks_per_slot
        self._block_table = torch.zeros(
            batch_size, max_blocks_per_req, dtype=torch.int32, device=self.device
        )
        self._slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        # Output buffer
        self._logits: torch.Tensor | None = None

        # Persistent slot mapping for replay
        self._replay_slot_ids: list[int] = []

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Fill pre-allocated buffers with real values for replay."""
        from runtime.backends.laguna import _physical_slot

        batch_size = len(slot_ids)
        assert batch_size <= self.batch_size

        # Input IDs and positions
        for i in range(batch_size):
            self._input_ids[i] = token_ids[i]
            self._positions[i] = kv_lengths[i]

        # Pad unused slots (if batch_size < self.batch_size)
        for i in range(batch_size, self.batch_size):
            self._input_ids[i] = 0
            self._positions[i] = 0

        # Seq lens (new KV lengths after this forward)
        for i in range(batch_size):
            self._seq_lens[i] = kv_lengths[i] + 1

        # Block table
        self._block_table.zero_()
        for i, slot in enumerate(slot_ids):
            phys = _physical_slot(slot)
            base = phys * self.blocks_per_slot
            new_kv_len = kv_lengths[i] + 1
            n_blocks = (new_kv_len + self.block_size - 1) // self.block_size
            for j in range(n_blocks):
                self._block_table[i, j] = base + j

        # Slot mapping
        for i, slot in enumerate(slot_ids):
            phys = _physical_slot(slot)
            pos = kv_lengths[i]
            bid = phys * self.blocks_per_slot + pos // self.block_size
            off = pos % self.block_size
            self._slot_mapping[i] = bid * self.block_size + off

    def capture(self) -> None:
        """Capture the decode forward pass as a CUDA Graph.

        Uses dummy data for warmup, then captures the actual graph.
        """
        if self._captured:
            return

        backend = self.backend
        batch_size = self.batch_size

        logger.info(
            "Capturing Laguna decode graph: batch_size=%d, max_kv_len=%d",
            batch_size,
            self.max_kv_len,
        )

        # Use dummy slot IDs for warmup (reserve last batch_size slots)
        dummy_slots = list(range(backend.num_slots - batch_size, backend.num_slots))
        dummy_tokens = [1] * batch_size  # dummy token IDs
        dummy_kv_lens = [16] * batch_size  # dummy KV lengths

        # Fill buffers with dummy data
        self._fill_buffers(dummy_slots, dummy_tokens, dummy_kv_lens)

        # Warmup runs (3x, on side stream)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(dummy_slots, dummy_tokens, dummy_kv_lens)
                self._run_forward(dummy_slots, dummy_kv_lens)
        side_stream.synchronize()

        # Capture
        self._fill_buffers(dummy_slots, dummy_tokens, dummy_kv_lens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._logits = self._run_forward(dummy_slots, dummy_kv_lens)

        self._graph = graph
        self._captured = True
        logger.info("Laguna decode graph captured: batch_size=%d", batch_size)

    def _run_forward(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
    ) -> torch.Tensor:
        """Run forward using pre-allocated buffers (for capture/replay)."""
        backend = self.backend
        batch_size = len(slot_ids)
        qo_lens = [1] * batch_size

        # Build CommonAttentionMetadata using pre-allocated tensors
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        qo_indptr = np.zeros(batch_size + 1, dtype=np.int32)
        np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
        query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
        query_start_loc_cpu = torch.from_numpy(qo_indptr)

        common_meta = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self._seq_lens[:batch_size],
            num_reqs=batch_size,
            num_actual_tokens=batch_size,
            max_query_len=1,
            max_seq_len=int(self._seq_lens[:batch_size].max().item()),
            block_table_tensor=self._block_table[:batch_size],
            slot_mapping=self._slot_mapping[:batch_size],
            causal=True,
        )

        # Build per-group FlashInferMetadata
        from runtime.compat_vllm import set_current_vllm_config, set_forward_context

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, builder in backend._metadata_builders.items():
            with set_current_vllm_config(backend.vllm_config):
                metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_meta,
                )
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = common_meta.slot_mapping

        with set_forward_context(
            attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = backend.model.forward(
                self._input_ids[:batch_size], self._positions[:batch_size]
            )

        return backend.model.compute_logits(hidden_states)

    def replay(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> list[int]:
        """Replay the captured graph with real data."""
        if not self._captured:
            raise RuntimeError("Graph not captured yet. Call capture() first.")

        assert len(slot_ids) <= self.batch_size

        # Fill buffers with real data
        self._fill_buffers(slot_ids, token_ids, kv_lengths)

        # Replay
        self._graph.replay()

        # Extract results
        batch_size = len(slot_ids)
        next_tokens = []
        for i in range(batch_size):
            next_token = int(self._logits[i].argmax(dim=-1).item())
            next_tokens.append(next_token)

        return next_tokens

    @property
    def is_captured(self) -> bool:
        return self._captured
