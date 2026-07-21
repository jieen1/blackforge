"""Extended regression tests for FixedSlotManager.

Covers historical bugs:
- 1295637: physical slot/state index 0 must be reserved (tested via
  direct_model_runner._physical_slot, but the slot manager's own
  invariants must hold for the reservation to work)
- a83bc5a: reset_slot didn't clear draft state (slot manager must
  support clean release/re-acquire cycles)
- cd7a9c7: CUDA Graph capture state pollution (slot isolation)

Also covers edge cases not in the original test_slot_manager.py:
- Capacity validation (1-4 only)
- Empty request_id rejection
- Duplicate request_id rejection
- Stale assignment detection on release
- Double-release prevention
- Generation monotonicity across many cycles
- Full capacity exhaustion and recovery
"""

import pytest

from runtime.slot_manager import FixedSlotManager, SlotAssignment, SlotError


class TestCapacityValidation:
    """Capacity must be 1-4 per the fixed-slot contract."""

    def test_capacity_zero_rejected(self):
        with pytest.raises(ValueError, match="between 1 and 4"):
            FixedSlotManager(capacity=0)

    def test_capacity_five_rejected(self):
        with pytest.raises(ValueError, match="between 1 and 4"):
            FixedSlotManager(capacity=5)

    def test_capacity_negative_rejected(self):
        with pytest.raises(ValueError, match="between 1 and 4"):
            FixedSlotManager(capacity=-1)

    @pytest.mark.parametrize("cap", [1, 2, 3, 4])
    def test_valid_capacities_accepted(self, cap):
        mgr = FixedSlotManager(capacity=cap)
        assert mgr.capacity == cap


class TestAcquireEdgeCases:
    """Acquire must reject invalid inputs and enforce ownership."""

    def test_empty_request_id_rejected(self):
        mgr = FixedSlotManager()
        with pytest.raises(ValueError, match="must not be empty"):
            mgr.acquire("")

    def test_duplicate_request_id_rejected(self):
        mgr = FixedSlotManager()
        mgr.acquire("req-1")
        with pytest.raises(SlotError, match="already owns"):
            mgr.acquire("req-1")

    def test_all_slots_exhausted(self):
        mgr = FixedSlotManager(capacity=4)
        for i in range(4):
            mgr.acquire(f"req-{i}")
        with pytest.raises(SlotError, match="no free"):
            mgr.acquire("overflow")

    def test_slot_ids_are_sequential_from_zero(self):
        mgr = FixedSlotManager(capacity=4)
        assignments = [mgr.acquire(f"req-{i}") for i in range(4)]
        assert [a.slot_id for a in assignments] == [0, 1, 2, 3]

    def test_released_slot_is_reused_in_order(self):
        mgr = FixedSlotManager(capacity=2)
        a0 = mgr.acquire("first")
        a1 = mgr.acquire("second")
        mgr.release(a0)
        a2 = mgr.acquire("third")
        assert a2.slot_id == 0
        assert a2.generation == a0.generation + 1


class TestReleaseEdgeCases:
    """Release must detect stale and foreign assignments."""

    def test_release_with_wrong_owner_rejected(self):
        mgr = FixedSlotManager(capacity=2)
        a0 = mgr.acquire("real-owner")
        foreign = SlotAssignment(a0.slot_id, "impostor", a0.generation)
        with pytest.raises(SlotError, match="owner does not match"):
            mgr.release(foreign)

    def test_release_with_stale_generation_rejected(self):
        mgr = FixedSlotManager(capacity=1)
        a0 = mgr.acquire("first")
        mgr.release(a0)
        a1 = mgr.acquire("second")
        stale = SlotAssignment(a0.slot_id, "second", a0.generation)
        with pytest.raises(SlotError, match="stale"):
            mgr.release(stale)

    def test_double_release_rejected(self):
        mgr = FixedSlotManager(capacity=1)
        a0 = mgr.acquire("req")
        mgr.release(a0)
        with pytest.raises(SlotError, match="owner does not match"):
            mgr.release(a0)

    def test_release_unacquired_slot_rejected(self):
        mgr = FixedSlotManager(capacity=2)
        mgr.acquire("req-0")
        fake = SlotAssignment(1, "ghost", 0)
        with pytest.raises(SlotError, match="owner does not match"):
            mgr.release(fake)


class TestGenerationMonotonicity:
    """Generation must monotonically increase across release/acquire cycles."""

    def test_generation_increments_on_reuse(self):
        mgr = FixedSlotManager(capacity=1)
        generations = []
        for i in range(10):
            a = mgr.acquire(f"req-{i}")
            generations.append(a.generation)
            mgr.release(a)
        assert generations == list(range(10))

    def test_independent_slots_have_independent_generations(self):
        mgr = FixedSlotManager(capacity=2)
        a0 = mgr.acquire("slot0-req")
        a1 = mgr.acquire("slot1-req")
        mgr.release(a0)
        a0_next = mgr.acquire("slot0-next")
        assert a0_next.generation == 1
        assert a1.generation == 0


class TestActiveTracking:
    """active() must reflect current ownership accurately."""

    def test_active_empty_initially(self):
        mgr = FixedSlotManager(capacity=4)
        assert mgr.active() == ()

    def test_active_reflects_acquisitions(self):
        mgr = FixedSlotManager(capacity=4)
        mgr.acquire("a")
        mgr.acquire("b")
        active = mgr.active()
        assert len(active) == 2
        assert {a.request_id for a in active} == {"a", "b"}

    def test_active_reflects_releases(self):
        mgr = FixedSlotManager(capacity=4)
        a0 = mgr.acquire("a")
        mgr.acquire("b")
        mgr.release(a0)
        active = mgr.active()
        assert len(active) == 1
        assert active[0].request_id == "b"

    def test_active_returns_stable_slot_order(self):
        mgr = FixedSlotManager(capacity=4)
        mgr.acquire("c")
        mgr.acquire("a")
        mgr.acquire("b")
        active = mgr.active()
        assert [a.slot_id for a in active] == [0, 1, 2]


class TestSlotIsolation:
    """Slots must be fully isolated (regression: cd7a9c7 state pollution)."""

    def test_release_one_slot_does_not_affect_others(self):
        mgr = FixedSlotManager(capacity=4)
        assignments = [mgr.acquire(f"req-{i}") for i in range(4)]
        mgr.release(assignments[2])
        active = mgr.active()
        assert len(active) == 3
        assert all(a.request_id != "req-2" for a in active)
        assert {a.slot_id for a in active} == {0, 1, 3}

    def test_reacquire_after_release_gets_clean_state(self):
        mgr = FixedSlotManager(capacity=1)
        a0 = mgr.acquire("old")
        mgr.release(a0)
        a1 = mgr.acquire("new")
        assert a1.slot_id == a0.slot_id
        assert a1.generation == a0.generation + 1
        assert a1.request_id == "new"
