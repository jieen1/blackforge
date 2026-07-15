import pytest

from model.qwen36_config import parse_qwen36_config


def _config(layer_types: list[str]) -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
            "mtp_num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
    }


def test_qwen36_config_accepts_expected_hybrid_topology() -> None:
    layer_types = ["linear_attention"] * 48 + ["full_attention"] * 16

    config = parse_qwen36_config(_config(layer_types))

    assert config.is_hybrid
    assert config.num_layers == 64
    assert (config.full_attention_layers, config.gdn_layers) == (16, 48)


def test_qwen36_config_rejects_other_topologies() -> None:
    with pytest.raises(ValueError, match="16 full-attention"):
        parse_qwen36_config(_config(["linear_attention"] * 64))
