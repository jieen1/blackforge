import json
from pathlib import Path
import struct

import pytest

from loader.checkpoint_index import CheckpointError, load_checkpoint_index
from loader.safetensors_header import SafetensorHeaderError, read_safetensors_header


def _write_header(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    header = json.dumps(tensors).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_read_safetensors_header_reads_tensor_metadata(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_header(
        shard,
        {"tensor": {"dtype": "F8_E4M3FN", "shape": [4, 8], "data_offsets": [0, 32]}},
    )

    header = read_safetensors_header(shard)

    assert header["tensor"].shape == (4, 8)
    assert header["tensor"].dtype == "F8_E4M3FN"


def test_header_rejects_truncated_file(tmp_path: Path) -> None:
    shard = tmp_path / "broken.safetensors"
    shard.write_bytes(b"tiny")

    with pytest.raises(SafetensorHeaderError, match="truncated"):
        read_safetensors_header(shard)


def test_checkpoint_index_validates_header_against_index(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_header(shard, {"lm_head.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {"lm_head.weight": shard.name}}),
        encoding="utf-8",
    )

    metadata = load_checkpoint_index(tmp_path).validate_headers()

    assert metadata["lm_head.weight"].shape == (2,)


def test_checkpoint_index_rejects_unindexed_header_tensor(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_header(shard, {"unexpected": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {"lm_head.weight": shard.name}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="absent"):
        load_checkpoint_index(tmp_path).validate_headers()
