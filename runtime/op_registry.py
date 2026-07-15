"""Explicit replacement points for model operations.

The eager implementation may initially register vLLM, FlashInfer, or torch
operations. Kernel work replaces one registered operation at a time while the
model graph stays unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class OpRegistry:
    """A deliberately small registry for runtime operations."""

    def __init__(self) -> None:
        self._operations: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, operation: Callable[..., Any], *, replace: bool = False) -> None:
        if not name:
            raise ValueError("operation name must not be empty")
        if name in self._operations and not replace:
            raise KeyError(f"operation already registered: {name}")
        self._operations[name] = operation

    def resolve(self, name: str) -> Callable[..., Any]:
        try:
            return self._operations[name]
        except KeyError as error:
            raise KeyError(f"operation is not registered: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))
