import json
from pathlib import Path

import pytest

from loader.checkpoint_index import CheckpointError, load_checkpoint_index
from model.qwen36_config import parse_qwen36_config


def _config() -> dict[str, object]:
    layer_types = ["linear_attention"] * 48 + ["full_attention"] * 16
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "layer_types": layer_types,
            "mtp_num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
    }


def _weight_map() -> dict[str, str]:
    weights = {
        "lm_head.weight": "model-00001.safetensors",
        "model.language_model.embed_tokens.weight": "model-00001.safetensors",
    }
    for layer_id in range(64):
        prefix = f"model.language_model.layers.{layer_id}."
        weights[f"{prefix}input_layernorm.weight"] = "model-00001.safetensors"
        weights[f"{prefix}post_attention_layernorm.weight"] = "model-00001.safetensors"
        if layer_id < 48:
            for name in ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight", "conv1d.weight", "A_log"):
                weights[f"{prefix}linear_attn.{name}"] = "model-00001.safetensors"
        else:
            for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
                weights[f"{prefix}self_attn.{name}"] = "model-00001.safetensors"
    weights["model.language_model.layers.0.mlp.gate_proj.weight_packed"] = "model-00001.safetensors"
    return weights


def _write_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / "model-00001.safetensors").touch()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map}), encoding="utf-8"
    )


def test_index_validates_expected_qwen_topology(tmp_path: Path) -> None:
    _write_index(tmp_path, _weight_map())

    index = load_checkpoint_index(tmp_path)
    index.validate_files()
    index.validate_qwen36(parse_qwen36_config(_config()))

    assert index.summary() == {
        "total_size": 123,
        "tensor_count": len(_weight_map()),
        "shard_count": 1,
        "nvfp4_tensor_count": 1,
    }


def test_index_reports_missing_shards(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {"lm_head.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="missing shards"):
        load_checkpoint_index(tmp_path).validate_files()
