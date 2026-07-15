"""Dependency-light numerical checks for captured oracle tensors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Any


def _as_values(value: Any) -> list[float]:
    """Accept lists and common tensor/array objects without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _top_indices(values: list[float], count: int) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count])


@dataclass(frozen=True)
class ComparisonResult:
    count: int
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    top_k_agreement: float

    def passes(self, *, max_abs_error: float, min_cosine: float, min_top_k: float) -> bool:
        return (
            self.max_abs_error <= max_abs_error
            and self.cosine_similarity >= min_cosine
            and self.top_k_agreement >= min_top_k
        )


def compare_values(
    reference: Iterable[float] | Any,
    candidate: Iterable[float] | Any,
    *,
    top_k: int = 10,
) -> ComparisonResult:
    """Measure error and logit-rank agreement for one captured activation."""
    expected = _as_values(reference)
    actual = _as_values(candidate)
    if not expected:
        raise ValueError("cannot compare an empty activation")
    if len(expected) != len(actual):
        raise ValueError("reference and candidate sizes differ")
    if not all(isfinite(value) for value in [*expected, *actual]):
        raise ValueError("comparison inputs must be finite")

    errors = [abs(left - right) for left, right in zip(expected, actual, strict=True)]
    dot = fsum(left * right for left, right in zip(expected, actual, strict=True))
    expected_norm = sqrt(fsum(value * value for value in expected))
    actual_norm = sqrt(fsum(value * value for value in actual))
    cosine = dot / (expected_norm * actual_norm) if expected_norm and actual_norm else 0.0
    k = min(top_k, len(expected))
    expected_top = set(_top_indices(expected, k))
    actual_top = set(_top_indices(actual, k))
    return ComparisonResult(
        count=len(expected),
        max_abs_error=max(errors),
        mean_abs_error=fsum(errors) / len(errors),
        cosine_similarity=cosine,
        top_k_agreement=len(expected_top & actual_top) / k,
    )
