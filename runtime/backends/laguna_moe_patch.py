"""Patch Laguna MoE layers to use LagunaMoEB12x kernel.

Replaces vLLM's RoutedExperts.forward_modular with our B12x fused kernel
on every MoE layer.  The model must be loaded WITHOUT moe_backend="marlin"
so weights stay in original NVFP4 format.

Usage:
    backend = LagunaBackend(vllm_config, ...)  # no moe_backend="marlin"
    patch_moe_b12x(backend.model)
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.moe_patch")


def patch_moe_b12x(model: Any, max_num_tokens: int = 16) -> int:
    """Replace all MoE routed-expert forwards with B12x kernel.

    Returns the number of patched layers.
    """
    from runtime.backends.laguna_moe_kernel import LagunaMoEB12x

    patched = 0
    for name, module in model.named_modules():
        # Find RoutedExperts modules (have w13_weight + w2_weight + quant_method)
        if not (hasattr(module, "w13_weight") and hasattr(module, "w2_weight")):
            continue
        if not hasattr(module, "forward_modular"):
            continue

        w13 = module.w13_weight
        num_experts = w13.shape[0]
        intermediate_size = w13.shape[1] // 2  # w13 is fused gate+up
        hidden_size = w13.shape[2] * 2  # packed uint8 → 2× int4

        # Detect top_k from RoutedExperts.moe_config or direct attribute
        top_k = getattr(module, "top_k", None)
        if top_k is None and hasattr(module, "moe_config"):
            top_k = getattr(module.moe_config, "experts_per_token", None)
        if top_k is None:
            top_k = 10  # Laguna-S-2.1 actual value
            logger.warning("Could not detect top_k for %s, using default %d", name, top_k)

        b12x = LagunaMoEB12x(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=w13.device,
            use_cuda_graph=False,
            max_num_tokens=max_num_tokens,
        )

        b12x.load_weights(
            w13_weight=module.w13_weight,
            w13_weight_scale=module.w13_weight_scale,
            w13_weight_scale_2=module.w13_weight_scale_2,
            w2_weight=module.w2_weight,
            w2_weight_scale=module.w2_weight_scale,
            w2_weight_scale_2=module.w2_weight_scale_2,
        )

        # Monkey-patch forward_modular
        original_forward = module.forward_modular

        def make_b12x_forward(b12x_inst: LagunaMoEB12x):
            def b12x_forward_modular(
                x: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                shared_experts: Any = None,
                shared_experts_input: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return b12x_inst.forward(
                    x, topk_ids.to(torch.int32), topk_weights.to(torch.float32)
                )
            return b12x_forward_modular

        module.forward_modular = make_b12x_forward(b12x)
        module._b12x_kernel = b12x  # prevent GC
        patched += 1

    logger.info("B12x MoE patch: %d layers patched (max_num_tokens=%d)", patched, max_num_tokens)
    return patched
