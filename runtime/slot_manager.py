"""Stable physical-slot ownership for the batch-1--4 decode hot path."""

from __future__ import annotations

from dataclasses import dataclass


class SlotError(RuntimeError):
    """Raised when a request violates fixed-slot ownership."""


@dataclass(frozen=True)
class SlotAssignment:
    slot_id: int
    request_id: str
    generation: int


class FixedSlotManager:
    """Assign up to four requests without moving physical cache addresses."""

    def __init__(self, capacity: int = 4) -> None:
        if not 1 <= capacity <= 4:
            raise ValueError("capacity must be between 1 and 4")
        self._owners: list[str | None] = [None] * capacity
        self._generations: list[int] = [0] * capacity

    @property
    def capacity(self) -> int:
        return len(self._owners)

    def acquire(self, request_id: str) -> SlotAssignment:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._owners:
            raise SlotError(f"request already owns a slot: {request_id}")
        try:
            slot_id = self._owners.index(None)
        except ValueError as error:
            raise SlotError("no free physical slots") from error
        self._owners[slot_id] = request_id
        return SlotAssignment(slot_id, request_id, self._generations[slot_id])

    def release(self, assignment: SlotAssignment) -> None:
        if self._owners[assignment.slot_id] != assignment.request_id:
            raise SlotError("slot owner does not match release request")
        if self._generations[assignment.slot_id] != assignment.generation:
            raise SlotError("stale slot assignment")
        self._owners[assignment.slot_id] = None
        self._generations[assignment.slot_id] += 1

    def active(self) -> tuple[SlotAssignment, ...]:
        return tuple(
            SlotAssignment(slot_id, request_id, self._generations[slot_id])
            for slot_id, request_id in enumerate(self._owners)
            if request_id is not None
        )
