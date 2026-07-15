"""Small, deterministic fixture cases required before kernel optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    batch_size: int
    prompt_tokens: int
    decode_steps: int
    phase: str
    checks: tuple[str, ...]


def golden_cases() -> tuple[GoldenCase, ...]:
    """Return the Phase 1 fixture matrix, independent of model data location."""
    common = ("layer_outputs", "gdn_state", "final_norm", "logits", "mtp_logits")
    return (
        GoldenCase("prefill-16-b1", 1, 16, 0, "prefill", common),
        GoldenCase("prefill-128-b1", 1, 128, 0, "prefill", common),
        GoldenCase("prefill-4096-b1", 1, 4096, 0, "prefill", common),
        GoldenCase("prefill-128-b4", 4, 128, 0, "prefill", common),
        GoldenCase("decode-256-b1", 1, 128, 256, "decode", common),
        GoldenCase("decode-256-b4", 4, 128, 256, "decode", common),
        GoldenCase("slot-reuse-b4", 4, 128, 32, "slot_reuse", common),
    )
