"""CUDA Graph wrapper for DFlash speculative decoding.

Captures the verify (M=16) and draft (M=16) forward passes as CUDA Graphs
for zero-overhead replay during decode. The main model's decode (M=1) uses
the existing LagunaCudaGraphDecode.

Architecture:
- Verify graph: main model forward with qo_len=16 (parallel verify)
- Draft graph: draft model forward with 16 tokens (1 bonus + 15 mask)
- Decode graph: main model forward with qo_len=1 (existing, from laguna_cuda_graph.py)

Per speculative step with CUDA Graphs:
1. Decode graph replay (M=1) → bonus token + aux hidden states
2. combine + precompute_context_kv (eager, small ops)
3. Draft graph replay (M=16) → 15 draft tokens
4. Verify graph replay (M=16) → accept/reject
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from runtime.backends.dflash_constants import (
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
    NUM_SPECULATIVE_TOKENS,
)
from runtime.backends.laguna import _physical_slot

logger = logging.getLogger("qwen_sm120_runtime.dflash_cudagraph")


class DFlashCudaGraphDecode:
    """CUDA Graph wrapper for DFlash speculative decode (single slot).

    Captures three graphs:
    1. Main decode (M=1): existing LagunaCudaGraphDecode handles this
    2. Draft forward (M=16): draft model with 16 query tokens
    3. Verify forward (M=16): main model with 16 query tokens

    The draft and verify graphs are captured at fixed batch_size=1 (single slot)
    with padded token count=16.
    """

    def __init__(self, engine, batch_size: int = 1) -> None:
        """Initialize DFlash CUDA Graph wrapper.

        Args:
            engine: DFlashEngine instance
            batch_size: number of slots (currently only 1 supported)
        """
        self.engine = engine
        self.backend = engine.backend
        self.device = engine.device
        self.batch_size = batch_size
        self.block_size = engine.block_size

        # Graph state
        self._verify_graph: torch.cuda.CUDAGraph | None = None
        self._draft_graph: torch.cuda.CUDAGraph | None = None
        self._captured = False

        # Pre-allocated buffers for verify graph
        self._verify_input_ids = torch.zeros(
            NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        self._verify_positions = torch.zeros(
            NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        self._verify_logits: torch.Tensor | None = None

        # Pre-allocated buffers for draft graph
        self._draft_input_ids = torch.zeros(
            NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        self._draft_positions = torch.zeros(
            NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        self._draft_logits: torch.Tensor | None = None

    def capture(self) -> None:
        """Capture verify and draft CUDA Graphs.

        Must be called after the engine is initialized and KV caches are allocated.
        Uses dummy data for warmup and capture.
        """
        if self._captured:
            return

        logger.info("Capturing DFlash CUDA Graphs (verify M=16, draft M=16)...")

        # Warmup: run verify and draft forward 3x on side stream
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._warmup_verify()
                self._warmup_draft()
        side_stream.synchronize()

        # Capture verify graph
        self._capture_verify()

        # Capture draft graph
        self._capture_draft()

        self._captured = True
        logger.info("DFlash CUDA Graphs captured successfully")

    def _warmup_verify(self) -> None:
        """Warmup run for verify forward (no graph capture)."""
        # Fill with dummy data
        self._verify_input_ids[:NUM_QUERY_PER_REQ] = 1
        self._verify_positions[:NUM_QUERY_PER_REQ] = torch.arange(
            32, 32 + NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        # Run verify forward (eager)
        self.engine._forward_verify(0, [1] * NUM_QUERY_PER_REQ, 32, NUM_QUERY_PER_REQ)

    def _warmup_draft(self) -> None:
        """Warmup run for draft forward (no graph capture)."""
        self._draft_input_ids[0] = 1
        self._draft_input_ids[1:NUM_QUERY_PER_REQ] = MASK_TOKEN_ID
        self._draft_positions[:NUM_QUERY_PER_REQ] = torch.arange(
            32, 32 + NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )
        # Run draft forward (eager)
        self.engine._draft_forward(0, 1, 32)

    def _capture_verify(self) -> None:
        """Capture the verify forward as a CUDA Graph."""
        # Final fill before capture
        self._verify_input_ids[:NUM_QUERY_PER_REQ] = 1
        self._verify_positions[:NUM_QUERY_PER_REQ] = torch.arange(
            32, 32 + NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, _ = self.engine._forward_verify(
                0, [1] * NUM_QUERY_PER_REQ, 32, NUM_QUERY_PER_REQ
            )
            self._verify_logits = logits

        self._verify_graph = graph

    def _capture_draft(self) -> None:
        """Capture the draft forward as a CUDA Graph."""
        self._draft_input_ids[0] = 1
        self._draft_input_ids[1:NUM_QUERY_PER_REQ] = MASK_TOKEN_ID
        self._draft_positions[:NUM_QUERY_PER_REQ] = torch.arange(
            32, 32 + NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Draft forward returns token list (involves .tolist() sync)
            # For CUDA Graph, we capture the logits computation only
            draft_model = self.engine.draft_model
            num_tokens = NUM_QUERY_PER_REQ

            # Build metadata (must be done outside graph for FlashInfer plan)
            # Actually, FlashInfer plan cannot be inside CUDA Graph
            # So we only capture the model forward, not the metadata build
            pass

        # Note: Draft graph capture is complex because FlashInfer plan
        # must run outside the graph. For now, draft stays eager.
        # TODO: Implement draft graph with pre-planned FlashInfer wrappers
        self._draft_graph = None
        logger.info("Draft graph: deferred (FlashInfer plan incompatible with capture)")

    def replay_verify(
        self,
        slot: int,
        tokens: list[int],
        kv_len: int,
    ) -> torch.Tensor:
        """Replay the verify CUDA Graph with real data.

        Args:
            slot: slot index
            tokens: [bonus_token] + draft_tokens (16 tokens)
            kv_len: current KV length

        Returns:
            logits tensor [16, vocab_size]
        """
        if self._verify_graph is None:
            # Fallback to eager
            logits, _ = self.engine._forward_verify(slot, tokens, kv_len, len(tokens))
            return logits

        # Update input buffers (GPU→GPU copy, no sync)
        num_tokens = len(tokens)
        self._verify_input_ids[:num_tokens] = torch.tensor(
            tokens, dtype=torch.long, device=self.device
        )
        self._verify_positions[:num_tokens] = torch.arange(
            kv_len, kv_len + num_tokens, dtype=torch.long, device=self.device
        )

        # Update attention metadata (FlashInfer plan - must be outside graph)
        # This is the bottleneck: plan must run every step
        self.engine._update_verify_plan(slot, kv_len, num_tokens)

        # Replay graph
        self._verify_graph.replay()
        return self._verify_logits

    @property
    def is_captured(self) -> bool:
        return self._captured
