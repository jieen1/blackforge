"""Execution contracts for the fixed-slot Qwen SM120 runtime."""

from runtime.engine import (
    EagerEngine,
    EngineError,
    ExecutionRequest,
    RequestSnapshot,
    RequestState,
    StepResult,
)
from runtime.hybrid_cache import CacheGeometry, HybridCache
from runtime.op_registry import OpRegistry
from runtime.slot_manager import FixedSlotManager, SlotError

__all__ = [
    "CacheGeometry",
    "EagerEngine",
    "EngineError",
    "ExecutionRequest",
    "FixedSlotManager",
    "HybridCache",
    "OpRegistry",
    "RequestSnapshot",
    "RequestState",
    "SlotError",
    "StepResult",
]
