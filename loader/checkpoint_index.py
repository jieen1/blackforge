"""Read a sharded safetensors index without materializing model weights."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loader.safetensors_header import TensorMetadata, read_safetensors_header
from model.qwen36_config import Qwen36Config


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot satisfy the narrow runtime contract."""


@dataclass(frozen=True)
class CheckpointIndex:
    model_dir: Path
    total_size: int
    weight_map: dict[str, str]

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    @property
    def tensor_count(self) -> int:
        return len(self.weight_map)

    @property
    def nvfp4_tensor_count(self) -> int:
        return sum(name.endswith(".weight_packed") for name in self.weight_map)

    def tensors_for_layer(self, layer_id: int) -> tuple[str, ...]:
        prefix = f"model.language_model.layers.{layer_id}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(prefix)))

    def validate_files(self) -> None:
        missing = [name for name in self.shard_names if not (self.model_dir / name).is_file()]
        if missing:
            raise CheckpointError(f"checkpoint is missing shards: {', '.join(missing)}")

    def validate_headers(self) -> dict[str, TensorMetadata]:
        """Verify every indexed tensor is present in the claimed shard header."""
        actual_by_shard = {
            shard_name: read_safetensors_header(self.model_dir / shard_name)
            for shard_name in self.shard_names
        }
        metadata: dict[str, TensorMetadata] = {}
        for tensor_name, shard_name in self.weight_map.items():
            try:
                metadata[tensor_name] = actual_by_shard[shard_name][tensor_name]
            except KeyError as error:
                raise CheckpointError(
                    f"index tensor {tensor_name} is absent from shard {shard_name}"
                ) from error
        unexpected = {}
        for shard_name, header in actual_by_shard.items():
            indexed_names = {
                name for name, shard in self.weight_map.items() if shard == shard_name
            }
            unexpected[shard_name] = sorted(set(header) - indexed_names)
        extras = [
            f"{shard}: {', '.join(names[:3])}"
            for shard, names in unexpected.items()
            if names
        ]
        if extras:
            raise CheckpointError(f"shard headers contain unindexed tensors: {'; '.join(extras)}")
        return metadata

    def validate_nvfp4_companions(self) -> None:
        """NVFP4 packed weights must retain their associated block-scale tensor."""
        missing = []
        for packed_name in self.weight_map:
            if packed_name.endswith(".weight_packed"):
                scale_name = packed_name.removesuffix(".weight_packed") + ".weight_scale"
                if scale_name not in self.weight_map:
                    missing.append(scale_name)
        if missing:
            raise CheckpointError(f"NVFP4 packed weights lack scales: {', '.join(missing[:8])}")

    def validate_qwen36(self, config: Qwen36Config) -> None:
        """Check names needed by loader work before any 23 GB shard is mapped."""
        required_global = {
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
        }
        missing_global = sorted(required_global - self.weight_map.keys())
        if missing_global:
            raise CheckpointError(
                f"checkpoint is missing global tensors: {', '.join(missing_global)}"
            )

        missing: list[str] = []
        for layer_id in range(config.num_layers):
            prefix = f"model.language_model.layers.{layer_id}."
            common = {f"{prefix}input_layernorm.weight", f"{prefix}post_attention_layernorm.weight"}
            if config.layer_type(layer_id) == "linear_attention":
                layer_required = {
                    f"{prefix}linear_attn.in_proj_qkv.weight",
                    f"{prefix}linear_attn.in_proj_z.weight",
                    f"{prefix}linear_attn.out_proj.weight",
                    f"{prefix}linear_attn.conv1d.weight",
                    f"{prefix}linear_attn.A_log",
                }
            else:
                layer_required = {
                    f"{prefix}self_attn.q_proj.weight",
                    f"{prefix}self_attn.k_proj.weight",
                    f"{prefix}self_attn.v_proj.weight",
                    f"{prefix}self_attn.o_proj.weight",
                }
            missing.extend(sorted((common | layer_required) - self.weight_map.keys()))
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise CheckpointError(f"checkpoint misses required layer tensors: {preview}{suffix}")

    def summary(self) -> dict[str, int]:
        shard_counts = Counter(self.weight_map.values())
        return {
            "total_size": self.total_size,
            "tensor_count": self.tensor_count,
            "shard_count": len(shard_counts),
            "nvfp4_tensor_count": self.nvfp4_tensor_count,
        }


def load_checkpoint_index(model_dir: Path) -> CheckpointIndex:
    index_path = model_dir / "model.safetensors.index.json"
    try:
        raw: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = raw["metadata"]
        weight_map = raw["weight_map"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid safetensors index: {index_path}") from error
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise CheckpointError("safetensors index must contain object metadata and weight_map")
    valid_weight_map = all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    )
    if not valid_weight_map:
        raise CheckpointError("safetensors weight_map must map tensor names to shard names")
    return CheckpointIndex(
        model_dir=model_dir,
        total_size=int(metadata.get("total_size", 0)),
        weight_map=dict(weight_map),
    )
