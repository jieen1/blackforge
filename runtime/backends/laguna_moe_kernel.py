"""Laguna MoE: self-researched direct-FlashInfer kernel, zero vLLM dependency.

Background: Laguna's MoE FFN currently runs through vLLM's ``FusedMoE``
layer, which dispatches (via ``select_nvfp4_moe_backend`` in
``vllm/model_executor/layers/fused_moe/oracle/nvfp4.py``) to one of several
expert-kernel classes -- by default ``FlashInferExperts`` (FLASHINFER_CUTLASS,
the confirmed-working baseline per ``notes/2026-07-23-laguna-moe-node-trace.md``
and ``benchmarks/laguna_moe_backend_scan.py``). That whole dispatch path
(``modular_kernel``, ``FusedMoEConfig``, per-backend expert classes) is vLLM
machinery -- exactly the kind of "厚依赖" B7 targets for removal (see
architecture.md sec 3, roadmap.md sec 5 B7 dependency tiers).

vLLM's OWN ``FlashInferB12xExperts`` class (the SM120-native backend,
deliberately excluded from vLLM's auto-selection oracle pending an upstream
CUTLASS SM121 MMA op guard -- see oracle/nvfp4.py:176-178, the exact same
reason ``runtime/nvfp4_b12x_patch.py`` exists for the linear NVFP4 kernel) is
itself just a ~250-line adapter around ``flashinfer.fused_moe.B12xMoEWrapper``
-- a CuTe-DSL kernel (source: local checkout
``/home/bot/project/flashinfer/flashinfer/fused_moe/cute_dsl/b12x_moe.py`` +
``.../blackwell_sm12x/``) that fuses routing + FC1 + SiLU + FC2 + scatter
into ONE kernel launch, vs. FLASHINFER_CUTLASS's 8-9 separate kernel launches
per MoE layer (see notes/2026-07-23-cutlass-fp4-moe-fusion-design.md sec 4.1).

This module is the B7-style "self-researched" replacement: it depends on
FlashInfer directly (an accepted upstream pip dependency per B7's tiering,
same as FLA/causal-conv1d) and NOT on vLLM's fused_moe dispatch machinery at
all -- no ``modular_kernel``, no ``FusedMoEConfig``, no oracle backend
selection. The only vLLM-shaped surface this touches is READING the already-
loaded NVFP4 weight tensor attribute names vLLM's checkpoint loader
populates on the model's ``FusedMoE`` module instances (``w13_weight`` etc.)
-- the same minimal, unavoidable surface ``laguna_cuda_graph.py`` already
accepts for reading ``backend._metadata_builders``/``backend.model``.

Scale-factor bake-in and MMA-layout conversion mirror vLLM's own
``FlashInferB12xExperts.process_weights_after_loading`` (verified correct by
that class's test suite: 24/25 parametrized GPU cases passed on this exact
GPU 2026-07-23, tests/kernels/moe/test_flashinfer_b12x_moe.py; the 1 failure
is an unrelated pre-existing test-harness bug in vLLM's ReLU2 variant, a
code path Laguna doesn't use) -- reimplemented here standalone rather than
imported, so this module has no vLLM import at all.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.laguna_moe_kernel")


class LagunaMoEB12x:
    """Direct FlashInfer B12x MoE kernel for one Laguna MoE layer, for a
    FIXED batch size (mirrors ``LagunaCudaGraphDecode``'s one-instance-per-
    batch-size design in ``runtime/backends/laguna_cuda_graph.py`` -- use
    ``LagunaMoEB12xMultiBatch`` below to serve a range of batch sizes).

    Usage (eager, any batch size up to ``max_num_tokens``)::

        moe = LagunaMoEB12x(
            num_experts=256, top_k=10, hidden_size=3072,
            intermediate_size=1024, device="cuda",
        )
        moe.load_weights(w13_weight=..., w13_weight_scale=..., ...)
        output = moe.forward(hidden_states, topk_ids, topk_weights)

    Usage (CUDA Graph, decode's real hot path -- verified correct + ~80%
    faster than eager at M=1, see notes/2026-07-23-laguna-moe-b12x-direct-
    kernel.md): construct with ``use_cuda_graph=True, max_num_tokens=BS``
    for the EXACT batch size this instance will always be called with,
    call ``capture()`` once after ``load_weights()``, then ``forward()``
    every round -- capture/replay is handled internally, the caller just
    passes fresh tensors each time like the eager path::

        moe = LagunaMoEB12x(..., use_cuda_graph=True, max_num_tokens=BS)
        moe.load_weights(w13_weight=..., ...)
        moe.capture()
        output = moe.forward(hidden_states, topk_ids, topk_weights)  # replay

    The weight tensors this reads (``w13_weight`` / ``w13_weight_scale`` /
    ``w13_weight_scale_2`` / ``w2_weight`` / ``w2_weight_scale`` /
    ``w2_weight_scale_2``) match vLLM's standard NVFP4 MoE checkpoint-loader
    attribute names on a loaded ``FusedMoE`` module, but nothing about
    routing, dispatch, or the forward computation itself goes through vLLM.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device | str = "cuda",
        use_cuda_graph: bool = False,
        max_num_tokens: int = 4096,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = torch.device(device)
        self.use_cuda_graph = use_cuda_graph
        self.max_num_tokens = max_num_tokens

        self._wrapper: Any | None = None
        self.w1_weight: torch.Tensor | None = None
        self.w2_weight: torch.Tensor | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None
        self.w1_alpha: torch.Tensor | None = None
        self.w2_alpha: torch.Tensor | None = None
        self.fc2_input_scale: torch.Tensor | None = None

        # -- CUDA Graph state (populated by capture()) --
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_hidden_states: torch.Tensor | None = None
        self._graph_topk_ids: torch.Tensor | None = None
        self._graph_topk_weights: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    def _ensure_wrapper(self) -> None:
        if self._wrapper is not None:
            return
        from flashinfer.fused_moe import B12xMoEWrapper

        self._wrapper = B12xMoEWrapper(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            use_cuda_graph=self.use_cuda_graph,
            max_num_tokens=self.max_num_tokens,
            activation="silu",
        )

    def load_weights(
        self,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_weight_scale_2: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_weight_scale_2: torch.Tensor,
        a2_gscale: torch.Tensor | None = None,
    ) -> None:
        """Ingest a loaded NVFP4 MoE layer's raw weight tensors.

        Bakes the per-expert global scale (w_gs) into the block scales so
        the kernel's w1_alpha/w2_alpha are 1.0 -- mirrors
        FlashInferB12xExperts.process_weights_after_loading exactly (same
        convention, reimplemented so this module has zero vLLM import).
        """
        self._ensure_wrapper()
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

        w1_scale = (w13_weight_scale.float() * w13_weight_scale_2.view(-1, 1, 1)).to(
            w13_weight_scale.dtype
        )
        w2_scale = (w2_weight_scale.float() * w2_weight_scale_2.view(-1, 1, 1)).to(
            w2_weight_scale.dtype
        )

        self.w1_weight = w13_weight
        self.w2_weight = w2_weight
        self.w1_alpha = torch.ones(
            w13_weight.size(0), device=self.device, dtype=torch.float32
        )
        self.w2_alpha = torch.ones(
            w2_weight.size(0), device=self.device, dtype=torch.float32
        )

        if a2_gscale is not None:
            a2_gscale.fill_(1.0)
            self.fc2_input_scale = a2_gscale
        else:
            self.fc2_input_scale = torch.ones(
                w13_weight.size(0), device=self.device, dtype=torch.float32
            )

        e1, m1, k1_sf = w1_scale.shape
        self.w1_sf_mma = convert_sf_to_mma_layout(
            w1_scale.reshape(e1 * m1, k1_sf), m=m1, k=k1_sf * 16, num_groups=e1
        )
        e2, m2, k2_sf = w2_scale.shape
        self.w2_sf_mma = convert_sf_to_mma_layout(
            w2_scale.reshape(e2 * m2, k2_sf), m=m2, k=k2_sf * 16, num_groups=e2
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """The real ``wrapper.run()`` call -- shared by eager calls, graph
        warmup, and graph capture. Never call directly during graph capture
        setup with attention to argument identity: whatever tensors are
        passed here become the addresses baked into the graph if called
        inside ``torch.cuda.graph()``."""
        assert self._wrapper is not None
        return self._wrapper.run(
            x=hidden_states,
            w1_weight=self.w1_weight,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.w1_alpha,
            fc2_input_scale=self.fc2_input_scale,
            w2_weight=self.w2_weight,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.w2_alpha,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )

    def capture(self) -> None:
        """Capture the CUDA Graph for this instance's (fixed) batch size.

        Must be called after ``load_weights()``. Dummy-but-valid routing
        data (each row routes to experts ``[0..top_k-1]``, uniform weights)
        is used for warmup + capture -- the kernel's control flow depends
        only on shapes (fixed at construction), not routing values, so any
        in-range expert assignment is fine for warmup/capture; real routing
        is written in on every ``forward()`` call before replay.

        Warmup runs on the current/default stream, NOT a side stream --
        an earlier investigation found a side-stream warmup left this
        wrapper incorrectly primed for capture on the default stream (see
        notes/2026-07-23-laguna-moe-b12x-direct-kernel.md). Capture itself
        does not execute (FlashInfer's own CUDA Graph test notes this:
        "capture doesn't execute - output may be zeros here"), so this
        replays once immediately after capture before returning.
        """
        if not self.use_cuda_graph:
            raise RuntimeError("capture() requires use_cuda_graph=True")
        if self._graph is not None:
            return
        if self.w1_weight is None:
            raise RuntimeError("load_weights() must be called before capture()")

        bs = self.max_num_tokens
        self._graph_hidden_states = torch.zeros(
            bs, self.hidden_size, device=self.device, dtype=torch.bfloat16
        )
        self._graph_topk_ids = (
            torch.arange(self.top_k, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, -1)
            .contiguous()
        )
        self._graph_topk_weights = torch.full(
            (bs, self.top_k), 1.0 / self.top_k, device=self.device, dtype=torch.float32
        )

        for _ in range(5):
            self._graph_output = self._forward_impl(
                self._graph_hidden_states, self._graph_topk_ids, self._graph_topk_weights
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._graph_output = self._forward_impl(
                self._graph_hidden_states, self._graph_topk_ids, self._graph_topk_weights
            )
        graph.replay()
        torch.cuda.synchronize()
        self._graph = graph
        logger.info(
            "LagunaMoEB12x CUDA Graph captured: batch_size=%d experts=%d top_k=%d",
            bs, self.num_experts, self.top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused MoE forward pass. hidden_states: [num_tokens, hidden_size], bf16.

        If ``capture()`` has been called, this replays the captured graph
        (fast path: copies fresh inputs into the fixed-address graph
        buffers, then ``graph.replay()``) instead of dispatching eagerly.
        ``num_tokens`` must exactly equal the batch size passed to
        ``capture()`` -- this class serves ONE fixed batch size; see
        ``LagunaMoEB12xMultiBatch`` for a range of batch sizes.
        """
        if self.w1_weight is None:
            raise RuntimeError("load_weights() must be called before forward()")
        if self._graph is None:
            return self._forward_impl(hidden_states, topk_ids, topk_weights)

        bs = hidden_states.shape[0]
        if bs != self.max_num_tokens:
            raise ValueError(
                f"LagunaMoEB12x was captured for batch_size={self.max_num_tokens}, "
                f"got {bs} tokens. Use LagunaMoEB12xMultiBatch to serve a range "
                f"of batch sizes."
            )
        assert self._graph_hidden_states is not None
        assert self._graph_topk_ids is not None
        assert self._graph_topk_weights is not None
        assert self._graph_output is not None
        self._graph_hidden_states.copy_(hidden_states)
        self._graph_topk_ids.copy_(topk_ids.to(torch.int32))
        self._graph_topk_weights.copy_(topk_weights)
        self._graph.replay()
        return self._graph_output


class LagunaMoEB12xMultiBatch:
    """Manages one CUDA-Graph-captured ``LagunaMoEB12x`` per batch_size in
    ``1..max_batch_size``, dispatching by actual batch size -- mirrors
    ``MultiBatchGraphManager`` in ``runtime/backends/laguna_cuda_graph.py``
    (the same pattern already established there for the attention/decode
    operator), applied to the MoE operator.

    One captured graph per exact batch size (rather than a single graph
    padded to the max and replayed with unused rows) means each replay only
    computes exactly as much work as that round's real batch size --
    consistent with why ``LagunaCudaGraphDecode`` does the same.

    Usage::

        moe = LagunaMoEB12xMultiBatch(
            num_experts=256, top_k=10, hidden_size=3072,
            intermediate_size=1024, max_batch_size=4,
        )
        moe.load_weights(w13_weight=..., w13_weight_scale=..., ...)
        moe.capture_all()
        output = moe.forward(hidden_states, topk_ids, topk_weights)  # dispatches by shape[0]
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device | str = "cuda",
        max_batch_size: int = 4,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self._instances: dict[int, LagunaMoEB12x] = {}
        self._weight_kwargs: dict[str, Any] | None = None

    def load_weights(
        self,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_weight_scale_2: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_weight_scale_2: torch.Tensor,
        a2_gscale: torch.Tensor | None = None,
    ) -> None:
        """Store weight kwargs; applied to every batch-size instance in
        ``capture_all()``. The same weight tensors are shared (read-only)
        across all instances -- each instance only owns its own graph
        buffers and workspace, never the weights themselves."""
        self._weight_kwargs = {
            "w13_weight": w13_weight,
            "w13_weight_scale": w13_weight_scale,
            "w13_weight_scale_2": w13_weight_scale_2,
            "w2_weight": w2_weight,
            "w2_weight_scale": w2_weight_scale,
            "w2_weight_scale_2": w2_weight_scale_2,
            "a2_gscale": a2_gscale,
        }

    def capture_all(self) -> None:
        """Capture graphs for all batch sizes 1..max_batch_size."""
        if self._weight_kwargs is None:
            raise RuntimeError("load_weights() must be called before capture_all()")
        for bs in range(1, self.max_batch_size + 1):
            instance = LagunaMoEB12x(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                device=self.device,
                use_cuda_graph=True,
                max_num_tokens=bs,
            )
            instance.load_weights(**self._weight_kwargs)
            instance.capture()
            self._instances[bs] = instance

    def get(self, batch_size: int) -> LagunaMoEB12x:
        """Get the captured instance for a specific batch size."""
        if batch_size not in self._instances:
            raise RuntimeError(
                f"No graph captured for batch_size={batch_size}. "
                f"Available: {sorted(self._instances.keys())}"
            )
        return self._instances[batch_size]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to the correct batch-size instance's replay."""
        bs = hidden_states.shape[0]
        return self.get(bs).forward(hidden_states, topk_ids, topk_weights)

    @property
    def captured_sizes(self) -> list[int]:
        return sorted(self._instances.keys())
