"""Read and validate the narrow Qwen3.6 runtime contract from HF config."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen36Config:
    architecture: str
    hidden_size: int
    num_layers: int
    full_attention_layers: int
    gdn_layers: int
    mtp_hidden_layers: int
    kv_heads: int
    head_dim: int
    layer_types: tuple[str, ...]

    @property
    def is_hybrid(self) -> bool:
        return self.full_attention_layers > 0 and self.gdn_layers > 0

    def layer_type(self, layer_id: int) -> str:
        return self.layer_types[layer_id]


def _text_config(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Qwen3.6 config must contain a text_config object")
    return text


def parse_qwen36_config(raw: dict[str, Any]) -> Qwen36Config:
    """Validate the 64-layer hybrid topology supported by this repository."""
    architectures = raw.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config must declare an architecture")
    architecture = str(architectures[0])
    if architecture != "Qwen3_5ForConditionalGeneration":
        raise ValueError(f"unsupported architecture: {architecture}")

    text = _text_config(raw)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        raise ValueError("text_config.layer_types must be a list")
    full_attention_layers = layer_types.count("full_attention")
    gdn_layers = layer_types.count("linear_attention")
    num_layers = int(text["num_hidden_layers"])
    if len(layer_types) != num_layers:
        raise ValueError("layer_types length must match num_hidden_layers")
    if full_attention_layers != 16 or gdn_layers != 48:
        raise ValueError("this runtime requires exactly 16 full-attention and 48 GDN layers")
    return Qwen36Config(
        architecture=architecture,
        hidden_size=int(text["hidden_size"]),
        num_layers=num_layers,
        full_attention_layers=full_attention_layers,
        gdn_layers=gdn_layers,
        mtp_hidden_layers=int(text["mtp_num_hidden_layers"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        layer_types=tuple(str(layer_type) for layer_type in layer_types),
    )


def load_qwen36_config(model_dir: Path) -> Qwen36Config:
    """Load `config.json` from a local Hugging Face snapshot or model directory."""
    with (model_dir / "config.json").open(encoding="utf-8") as config_file:
        return parse_qwen36_config(json.load(config_file))
