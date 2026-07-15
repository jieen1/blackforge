"""Logical fixed-address metadata for attention KV and GDN recurrent state.

This module owns no CUDA tensors yet. It establishes the address and lifecycle
contract that GPU allocations must preserve when the eager executor arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.slot_manager import FixedSlotManager, SlotAssignment


@dataclass(frozen=True)
class CacheGeometry:
    block_size: int
    max_blocks_per_slot: int
    attention_layers: int = 16
    gdn_layers: int = 48

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.max_blocks_per_slot <= 0:
            raise ValueError("block size and blocks per slot must be positive")
        if self.attention_layers != 16 or self.gdn_layers != 48:
            raise ValueError("this runtime requires 16 attention and 48 GDN layers")


@dataclass(frozen=True)
class CacheView:
    slot_id: int
    request_id: str
    generation: int
    token_count: int
    block_table: tuple[int, ...]

    @property
    def gdn_state_slot(self) -> int:
        """The stable state address index used by all 48 GDN layers."""
        return self.slot_id


class HybridCache:
    """Keep KV block and GDN state ownership isolated across four request slots."""

    def __init__(self, geometry: CacheGeometry, *, capacity: int = 4) -> None:
        self.geometry = geometry
        self._slots = FixedSlotManager(capacity)
        self._token_counts = [0] * capacity

    def acquire(self, request_id: str) -> CacheView:
        assignment = self._slots.acquire(request_id)
        self._token_counts[assignment.slot_id] = 0
        return self._view(assignment)

    def append(self, assignment: SlotAssignment, token_count: int = 1) -> CacheView:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        current = self._assignment_for(assignment)
        next_count = self._token_counts[current.slot_id] + token_count
        if next_count > self.geometry.block_size * self.geometry.max_blocks_per_slot:
            raise RuntimeError("request exceeds the fixed cache capacity for its slot")
        self._token_counts[current.slot_id] = next_count
        return self._view(current)

    def release(self, assignment: SlotAssignment) -> None:
        current = self._assignment_for(assignment)
        self._token_counts[current.slot_id] = 0
        self._slots.release(current)

    def active(self) -> tuple[CacheView, ...]:
        return tuple(self._view(assignment) for assignment in self._slots.active())

    def _assignment_for(self, assignment: SlotAssignment) -> SlotAssignment:
        for active in self._slots.active():
            if active.slot_id == assignment.slot_id:
                if active != assignment:
                    raise RuntimeError("stale or foreign slot assignment")
                return active
        raise RuntimeError("slot is not active")

    def _view(self, assignment: SlotAssignment) -> CacheView:
        token_count = self._token_counts[assignment.slot_id]
        block_count = (token_count + self.geometry.block_size - 1) // self.geometry.block_size
        first_block = assignment.slot_id * self.geometry.max_blocks_per_slot
        block_table = tuple(first_block + offset for offset in range(block_count))
        return CacheView(
            slot_id=assignment.slot_id,
            request_id=assignment.request_id,
            generation=assignment.generation,
            token_count=token_count,
            block_table=block_table,
        )
