import pytest

from runtime.slot_manager import FixedSlotManager, SlotError


def test_slots_are_stable_and_reuse_increments_generation() -> None:
    slots = FixedSlotManager()
    assignments = [slots.acquire(f"request-{index}") for index in range(4)]

    with pytest.raises(SlotError, match="no free"):
        slots.acquire("overflow")

    slots.release(assignments[1])
    replacement = slots.acquire("replacement")

    assert replacement.slot_id == assignments[1].slot_id
    assert replacement.generation == assignments[1].generation + 1
    assert len(slots.active()) == 4
