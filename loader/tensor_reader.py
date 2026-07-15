"""Indexed, on-demand safetensors reads for the eager model loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loader.checkpoint_index import CheckpointIndex
from loader.safetensors_header import TensorMetadata

if TYPE_CHECKING:
    import torch


class TensorReader:
    """Load one validated tensor at a time; never eagerly materialize all shards."""

    def __init__(self, index: CheckpointIndex) -> None:
        self.index = index
        self.metadata = index.validate_headers()

    def tensor_metadata(self, name: str) -> TensorMetadata:
        try:
            return self.metadata[name]
        except KeyError as error:
            raise KeyError(f"tensor is not present in the checkpoint index: {name}") from error

    def load(self, name: str, *, device: str = "cpu") -> torch.Tensor:
        """Load one tensor through safetensors using the shard declared by the index."""
        try:
            shard_name = self.index.weight_map[name]
        except KeyError as error:
            raise KeyError(f"tensor is not present in the checkpoint index: {name}") from error
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("install the runtime extra to read safetensors payloads") from error
        with safe_open(
            str(self.index.model_dir / shard_name), framework="pt", device=device
        ) as shard:
            tensor = shard.get_tensor(name)
        expected = self.metadata[name]
        if tuple(tensor.shape) != expected.shape:
            raise RuntimeError(f"tensor shape differs from safetensors header: {name}")
        return tensor
