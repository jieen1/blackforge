"""vLLM oracle fixture definitions and numerical comparison helpers."""

from oracle.capture_hooks import CapturedTensor, CaptureError, ForwardCapture
from oracle.comparator import ComparisonResult, compare_values
from oracle.fixtures import golden_cases

__all__ = [
    "CaptureError",
    "CapturedTensor",
    "ComparisonResult",
    "ForwardCapture",
    "compare_values",
    "golden_cases",
]
