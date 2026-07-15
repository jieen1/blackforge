import json
from pathlib import Path

import pytest

from loader.checkpoint_index import load_checkpoint_index
from loader.tensor_reader import TensorReader

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_reader_loads_only_indexed_tensor(tmp_path: Path) -> None:
    shard_name = "model-00001.safetensors"
    safetensors_torch.save_file(
        {
            "first": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "second": torch.ones(2, dtype=torch.bfloat16),
        },
        tmp_path / shard_name,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {"first": shard_name, "second": shard_name},
            }
        ),
        encoding="utf-8",
    )

    reader = TensorReader(load_checkpoint_index(tmp_path))

    loaded = reader.load("first")

    assert loaded.dtype == torch.float32
    assert tuple(loaded.shape) == (2, 3)
    assert torch.equal(loaded, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_reader_rejects_unknown_tensor(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}), encoding="utf-8"
    )
    reader = TensorReader(load_checkpoint_index(tmp_path))

    with pytest.raises(KeyError, match="not present"):
        reader.load("unknown")
