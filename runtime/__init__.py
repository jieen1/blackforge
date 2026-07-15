"""Execution contracts for the fixed-slot Qwen SM120 runtime."""

from runtime.hybrid_cache import CacheGeometry, HybridCache
from runtime.op_registry import OpRegistry
from runtime.slot_manager import FixedSlotManager, SlotError

__all__ = ["CacheGeometry", "FixedSlotManager", "HybridCache", "OpRegistry", "SlotError"]
