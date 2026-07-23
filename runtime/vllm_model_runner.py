"""Model-agnostic runner backed by vLLM's LLMEngine.

Provides the same slot-based interface as DirectModelRunner but delegates
all inference to vLLM. Used for models that don't have a dedicated
backend yet (e.g. Laguna-S-2.1 MoE).

No MTP/speculative decoding — simple autoregressive prefill + decode.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from runtime.sampling import SamplingParams

logger = logging.getLogger("qwen_sm120_runtime.vllm_model_runner")


class VllmModelRunner:
    """Slot-based runner backed by vLLM's LLMEngine.

    Implements the minimal interface needed by ServerEngine:
    - prefill(slot, prompt_ids) -> first token id
    - decode_batch_sampled(slot_ids, token_ids, params) -> next token ids
    - reset_slot(slot)
    - slot_kv_len[slot]
    - slot_committed_tokens[slot]
    """

    def __init__(
        self,
        model: str,
        num_slots: int = 4,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        **kwargs: Any,
    ) -> None:
        self.model_id = model
        self.num_slots = num_slots
        self._max_model_len = max_model_len

        # Apply A2 patches before loading
        from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct
        from runtime.nvfp4_custom_gemm import patch_nvfp4_custom_gemm
        patch_nvfp4_prefer_cutlass_direct()
        patch_nvfp4_custom_gemm()

        from vllm import LLM

        self._llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            dtype="bfloat16",
            disable_log_stats=True,
        )

        # Per-slot state
        self.slot_kv_len: list[int] = [0] * num_slots
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]
        self._slot_prompts: list[list[int] | None] = [None] * num_slots

        # Expose for engine compatibility
        self.num_speculative_tokens = 0
        self.spec = None

        logger.info("VllmModelRunner loaded: %s (max_len=%d)", model, max_model_len)

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Prefill prompt and return the first generated token."""
        from vllm import SamplingParams as VllmSamplingParams

        self._slot_prompts[slot] = list(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids)

        params = VllmSamplingParams(temperature=0, max_tokens=1)
        result = self._llm.generate([prompt_ids], params, use_tqdm=False)[0]
        first_token = result.outputs[0].token_ids[0] if result.outputs[0].token_ids else 0

        self.slot_kv_len[slot] = len(prompt_ids) + 1
        self.slot_committed_tokens[slot].append(first_token)
        return first_token

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        sampling_params_list: list[SamplingParams],
        **kwargs: Any,
    ) -> list[int]:
        """Decode one token for each slot in the batch."""
        from vllm import SamplingParams as VllmSamplingParams

        prompts = []
        for i, slot in enumerate(slot_ids):
            self.slot_committed_tokens[slot].append(token_ids[i])
            prompts.append(list(self.slot_committed_tokens[slot]))

        vllm_params = VllmSamplingParams(temperature=0, max_tokens=1)
        results = self._llm.generate(prompts, vllm_params, use_tqdm=False)

        next_tokens = []
        for i, slot in enumerate(slot_ids):
            r = results[i]
            tok = r.outputs[0].token_ids[0] if r.outputs[0].token_ids else 0
            next_tokens.append(tok)
            self.slot_kv_len[slot] = len(self.slot_committed_tokens[slot]) + 1

        return next_tokens

    def reset_slot(self, slot: int) -> None:
        """Reset a slot to empty state."""
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []
        self._slot_prompts[slot] = None

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[int]:
        """Full generation (prefill + decode loop). Convenience method."""
        from vllm import SamplingParams as VllmSamplingParams

        params = VllmSamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p if top_p < 1.0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
        )
        result = self._llm.generate([prompt_ids], params, use_tqdm=False)[0]
        return list(result.outputs[0].token_ids)
