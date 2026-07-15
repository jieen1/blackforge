"""Versioned metadata for an offline SM120 weight-packing result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from loader.checkpoint_index import CheckpointIndex
from model.qwen36_config import Qwen36Config


PACK_FORMAT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedCheckpointManifest:
    format_version: int
    source_index_sha256: str
    source_total_bytes: int
    source_shards: tuple[str, ...]
    architecture: str
    num_layers: int
    packed_layout: str


def create_manifest(index: CheckpointIndex, config: Qwen36Config) -> PackedCheckpointManifest:
    """Fingerprint the index; shard checksums belong in the packer's final output."""
    index_path = index.model_dir / "model.safetensors.index.json"
    return PackedCheckpointManifest(
        format_version=PACK_FORMAT_VERSION,
        source_index_sha256=_sha256_file(index_path),
        source_total_bytes=index.total_size,
        source_shards=index.shard_names,
        architecture=config.architecture,
        num_layers=config.num_layers,
        packed_layout="sm120-nvfp4-v1",
    )


def write_manifest(path: Path, manifest: PackedCheckpointManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
