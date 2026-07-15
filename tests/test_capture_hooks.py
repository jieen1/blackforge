from pathlib import Path

import pytest

from oracle.capture_hooks import CaptureError, ForwardCapture

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


class _TupleBlock(torch.nn.Module):
    def forward(self, value):
        return value + 1, {"state": value * 2}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.block = _TupleBlock()

    def forward(self, token_ids):
        return self.block(self.embed(token_ids))


def test_forward_capture_records_nested_tensor_outputs(tmp_path: Path) -> None:
    model = _Model()
    capture = ForwardCapture(model, ("embed", "block"))

    model(torch.tensor([1, 2], dtype=torch.int64))
    fixture = tmp_path / "capture.safetensors"
    capture.write_safetensors(fixture)

    captured = {item.name: item for item in capture.tensors()}
    persisted = safetensors_torch.load_file(fixture)
    assert set(captured) == {"embed", "block.0", "block.1.state"}
    assert set(persisted) == set(captured)
    assert captured["block.1.state"].shape == (2, 4)
    capture.close()


def test_forward_capture_rejects_unknown_module() -> None:
    with pytest.raises(CaptureError, match="do not exist"):
        ForwardCapture(_Model(), ("does_not_exist",))
