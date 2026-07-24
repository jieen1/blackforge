"""B12x MoE kernel monkey-patch: correct runtime alpha computation.

Root cause (verified against sparkinfer reference):
  vLLM's FlashInferB12xExperts.process_weights_after_loading passes
  w1_alpha = g1_alphas = 1/w_gs (~7.4e-05) to the kernel.
  The kernel expects w1_alpha = w_gs * input_scale = 1/(g1_alphas * a1_gscale)
  (~1.3-6.7). Off by ~235,000x.

  The old code also baked 1/w_gs into fp8 block scales (causing underflow)
  and set alpha=1.0, compounding the error.

Fix:
  1. Keep original fp8 block scales (no bake-in)
  2. Compute correct runtime alpha: 1/(g1_alphas * a1_gscale)
  3. Set fc2_input_scale = a2_gscale (sparkinfer convention)

Status: Alpha fix is correct but the FlashInfer B12x wrapper itself produces
  wrong output on SM120 (6.7% acceptance vs MARLIN's 47%). The wrapper is
  excluded from vLLM auto-selection ("CUTLASS SM121 MMA op guard").
  Full fix requires sparkinfer integration — see notes/b12x_investigation.md.

Usage:
  import runtime.backends.laguna_moe_patch  # auto-patches on import
"""
from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from flashinfer.cute_dsl.utils import (
    convert_sf_to_mma_layout as _convert_sf,
)

_orig_process = FlashInferB12xExperts.process_weights_after_loading


def _correct_process_weights(self: FlashInferB12xExperts, layer: torch.nn.Module) -> None:
    """Fix alpha computation per sparkinfer reference implementation.

    sparkinfer: w1_runtime = w1_global_scale / a1_gscale
    vLLM:       g1_alphas = 1/w1_global_scale, a1_gscale = 1/a1_input_scale
    Therefore:  w1_runtime = 1 / (g1_alphas * a1_gscale)
    """
    qc = self.quant_config

    # FC1 alpha: w_gs * a1_scale = 1/(g1_alphas * a1_gscale)
    a1_gscale = qc.a1_gscale
    if a1_gscale is not None and self.g1_alphas is not None:
        self.g1_alphas.copy_(1.0 / (self.g1_alphas * a1_gscale))

    # FC2 alpha: w_gs * a2_scale = 1/(g2_alphas * a2_gscale)
    a2_gscale = qc.a2_gscale
    if a2_gscale is not None and self.g2_alphas is not None:
        self.g2_alphas.copy_(1.0 / (self.g2_alphas * a2_gscale))

    # FC2 input scale: use a2_gscale (sparkinfer convention)
    if a2_gscale is not None:
        self._fc2_input_scale = a2_gscale.to(torch.float32).contiguous()
    else:
        self._fc2_input_scale = torch.ones(
            self.num_local_experts,
            device=layer.w13_weight.device,
            dtype=torch.float32,
        )

    # Block scales: keep ORIGINAL fp8 values (no bake-in), convert to MMA layout
    assert self.w1_scale is not None
    num_experts_w1, m1, k1_sf = self.w1_scale.shape
    self.w1_sf_mma = _convert_sf(
        self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
        m=m1,
        k=k1_sf * 16,
        num_groups=num_experts_w1,
    )

    assert self.w2_scale is not None
    num_experts_w2, m2, k2_sf = self.w2_scale.shape
    self.w2_sf_mma = _convert_sf(
        self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
        m=m2,
        k=k2_sf * 16,
        num_groups=num_experts_w2,
    )


FlashInferB12xExperts.process_weights_after_loading = _correct_process_weights
