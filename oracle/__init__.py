"""vLLM oracle fixture definitions and numerical comparison helpers."""

from oracle.comparator import ComparisonResult, compare_values
from oracle.fixtures import golden_cases

__all__ = ["ComparisonResult", "compare_values", "golden_cases"]
