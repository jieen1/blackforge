"""Read-only forward hooks for creating vLLM oracle fixtures.

The hooks attach to an already-instantiated torch model and therefore never
modify the vLLM checkout. Captured tensors are detached, cloned to CPU, and
can be written as a small safetensors fixture after a selected forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class CaptureError(RuntimeError):
    """Raised when a requested oracle capture point is unavailable."""


@dataclass(frozen=True)
class CapturedTensor:
    name: str
    tensor: torch.Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> str:
        return str(self.tensor.dtype)


class ForwardCapture:
    """Collect selected module outputs from one or more model forward passes."""

    def __init__(self, model: Any, module_names: tuple[str, ...]) -> None:
        if not module_names:
            raise ValueError("at least one module name must be captured")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the cuda extra to use oracle capture hooks") from error

        modules = dict(model.named_modules())
        missing = sorted(set(module_names) - modules.keys())
        if missing:
            raise CaptureError(f"capture modules do not exist: {', '.join(missing)}")
        self._torch = torch
        self._tensors: dict[str, torch.Tensor] = {}
        self._handles = [
            modules[name].register_forward_hook(self._hook_for(name)) for name in module_names
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._tensors.clear()

    def tensors(self) -> tuple[CapturedTensor, ...]:
        return tuple(
            CapturedTensor(name=name, tensor=tensor)
            for name, tensor in sorted(self._tensors.items())
        )

    def write_safetensors(self, path: Path) -> None:
        """Persist captured values using names stable enough for the comparator."""
        if not self._tensors:
            raise CaptureError("cannot write an empty capture")
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError("install the cuda extra to write safetensors fixtures") from error
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self._tensors, str(path))

    def _hook_for(self, module_name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            for suffix, tensor in _tensor_leaves(output):
                name = module_name if not suffix else f"{module_name}.{suffix}"
                self._tensors[name] = tensor.detach().to("cpu").clone().contiguous()

        return hook


def _tensor_leaves(value: Any, prefix: str = ""):
    if hasattr(value, "detach") and hasattr(value, "shape"):
        yield prefix, value
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            child = str(index) if not prefix else f"{prefix}.{index}"
            yield from _tensor_leaves(item, child)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            yield from _tensor_leaves(item, child)
