"""C3: tests for runtime/structured_output.py — structured output / JSON mode."""
import pytest
import torch

from runtime.structured_output import ResponseFormat, _unpack_bitmask_to_mask


class TestResponseFormat:
    def test_none_format_not_constrained(self):
        rf = ResponseFormat.from_api(None)
        assert not rf.is_constrained

    def test_text_format_not_constrained(self):
        rf = ResponseFormat.from_api({"type": "text"})
        assert not rf.is_constrained

    def test_json_object_constrained(self):
        rf = ResponseFormat.from_api({"type": "json_object"})
        assert rf.is_constrained
        assert rf.type == "json_object"

    def test_json_schema_constrained(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        rf = ResponseFormat.from_api({
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema},
        })
        assert rf.is_constrained
        assert rf.type == "json_schema"
        assert rf.json_schema is not None

    def test_json_schema_without_schema_field(self):
        rf = ResponseFormat.from_api({"type": "json_schema"})
        assert rf.is_constrained
        assert rf.json_schema == {}

    def test_unknown_type_defaults_to_text(self):
        rf = ResponseFormat.from_api({"type": "unknown"})
        assert not rf.is_constrained
        assert rf.type == "text"


class TestUnpackBitmask:
    def test_all_ones(self):
        bitmask = torch.tensor([-1], dtype=torch.int32)
        mask = _unpack_bitmask_to_mask(bitmask, 8)
        assert mask.shape == (8,)
        assert mask.all()

    def test_all_zeros(self):
        bitmask = torch.tensor([0], dtype=torch.int32)
        mask = _unpack_bitmask_to_mask(bitmask, 8)
        assert not mask.any()

    def test_first_bit_only(self):
        bitmask = torch.tensor([1], dtype=torch.int32)
        mask = _unpack_bitmask_to_mask(bitmask, 8)
        assert mask[0].item() is True
        assert not mask[1:].any()

    def test_vocab_larger_than_bits(self):
        bitmask = torch.tensor([1, 1], dtype=torch.int32)
        mask = _unpack_bitmask_to_mask(bitmask, 64)
        assert mask[0].item() is True
        assert mask[32].item() is True
        assert mask[1].item() is False
