import pytest

from runtime.op_registry import OpRegistry


def test_registry_requires_explicit_replacement() -> None:
    registry = OpRegistry()
    registry.register("attention", lambda value: value + 1)

    assert registry.resolve("attention")(2) == 3
    with pytest.raises(KeyError):
        registry.register("attention", lambda value: value)

    registry.register("attention", lambda value: value * 2, replace=True)
    assert registry.resolve("attention")(2) == 4
