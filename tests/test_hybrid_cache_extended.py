"""Extended regression tests for HybridCache.

Covers historical bugs:
- a83bc5a: reset_slot didn't clear draft state -- cache release must
  fully reset token_count and block_table for the reused slot
- cd7a9c7: CUDA Graph capture state pollution -- cache isolation across
  slots must be airtight
- 1295637: physical slot 0 reservation -- block_table computation must
  use correct physical addressing

Also covers edge cases:
- Block table computation at exact boundaries
- Multi-append accumulation
- Stale/foreign assignment rejection
- Capacity overflow at exact limit
- Release and re-acquire isolation
- GDN state slot stability
"""

import pytest

from runtime.hybrid_cache import CacheGeometry, CacheView, HybridCache
from runtime.slot_manager import SlotAssignment


def _assignment(view: CacheView) -> SlotAssignment:
    return SlotAssignment(view.slot_id, view.request_id, view.generation)


class TestGeometryValidation:
    """CacheGeometry must enforce the fixed model topology."""

    def test_rejects_zero_block_size(self):
        with pytest.raises(ValueError, match="positive"):
            CacheGeometry(block_size=0, max_blocks_per_slot=4)

    def test_rejects_zero_blocks_per_slot(self):
        with pytest.raises(ValueError, match="positive"):
            CacheGeometry(block_size=16, max_blocks_per_slot=0)

    def test_rejects_wrong_attention_layers(self):
        with pytest.raises(ValueError, match="16 attention"):
            CacheGeometry(block_size=16, max_blocks_per_slot=4, attention_layers=32)

    def test_rejects_wrong_gdn_layers(self):
        with pytest.raises(ValueError, match="48 GDN"):
            CacheGeometry(block_size=16, max_blocks_per_slot=4, gdn_layers=16)

    def test_accepts_correct_topology(self):
        geo = CacheGeometry(block_size=16, max_blocks_per_slot=4)
        assert geo.attention_layers == 16
        assert geo.gdn_layers == 48


class TestBlockTableComputation:
    """Block table must be computed correctly from token counts.

    Regression: 1295637 -- physical addressing must offset by slot_id
    so that different slots never share physical blocks.
    """

    def test_empty_cache_has_empty_block_table(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        assert view.block_table == ()
        assert view.token_count == 0

    def test_single_token_gets_one_block(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=1)
        assert view.block_table == (0,)
        assert view.token_count == 1

    def test_exact_block_boundary(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=16)
        assert view.block_table == (0,)
        assert view.token_count == 16

    def test_one_past_block_boundary(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=17)
        assert view.block_table == (0, 1)
        assert view.token_count == 17

    def test_full_capacity_block_table(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=64)
        assert view.block_table == (0, 1, 2, 3)
        assert view.token_count == 64

    def test_multi_append_accumulates(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=4))
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=5)
        assert view.token_count == 5
        assert view.block_table == (0,)
        view = cache.append(_assignment(view), token_count=5)
        assert view.token_count == 10
        assert view.block_table == (0, 1)

    def test_block_table_uses_slot_offset(self):
        """Different slots must use different physical block ranges."""
        cache = HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=2), capacity=4)
        v0 = cache.acquire("slot0")
        v1 = cache.acquire("slot1")
        v0 = cache.append(_assignment(v0), token_count=1)
        v1 = cache.append(_assignment(v1), token_count=1)
        assert v0.block_table[0] != v1.block_table[0]
        assert v0.block_table == (0,)
        assert v1.block_table == (2,)


class TestCapacityEnforcement:
    """Cache must reject requests that exceed per-slot capacity."""

    def test_exact_capacity_accepted(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=16)
        assert view.token_count == 16

    def test_one_past_capacity_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        view = cache.acquire("req")
        with pytest.raises(RuntimeError, match="exceeds"):
            cache.append(_assignment(view), token_count=17)

    def test_accumulated_overflow_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        view = cache.acquire("req")
        view = cache.append(_assignment(view), token_count=10)
        with pytest.raises(RuntimeError, match="exceeds"):
            cache.append(_assignment(view), token_count=7)

    def test_zero_token_append_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2))
        view = cache.acquire("req")
        with pytest.raises(ValueError, match="positive"):
            cache.append(_assignment(view), token_count=0)

    def test_negative_token_append_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2))
        view = cache.acquire("req")
        with pytest.raises(ValueError, match="positive"):
            cache.append(_assignment(view), token_count=-1)


class TestReleaseAndReuse:
    """Release must fully reset slot state (regression: a83bc5a)."""

    def test_release_resets_token_count(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=4), capacity=1)
        view = cache.acquire("first")
        cache.append(_assignment(view), token_count=20)
        cache.release(_assignment(view))
        replacement = cache.acquire("second")
        assert replacement.token_count == 0
        assert replacement.block_table == ()

    def test_release_resets_block_table(self):
        cache = HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=1)
        view = cache.acquire("first")
        view = cache.append(_assignment(view), token_count=10)
        assert len(view.block_table) == 3
        cache.release(_assignment(view))
        replacement = cache.acquire("second")
        assert replacement.block_table == ()

    def test_reuse_increments_generation(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        v1 = cache.acquire("first")
        gen1 = v1.generation
        cache.release(_assignment(v1))
        v2 = cache.acquire("second")
        assert v2.generation == gen1 + 1

    def test_reuse_gets_same_slot_id(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        v1 = cache.acquire("first")
        slot_id = v1.slot_id
        cache.release(_assignment(v1))
        v2 = cache.acquire("second")
        assert v2.slot_id == slot_id


class TestSlotIsolation:
    """Slots must be fully isolated (regression: cd7a9c7 state pollution)."""

    def test_append_to_one_slot_does_not_affect_another(self):
        cache = HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=2)
        v0 = cache.acquire("slot0")
        cache.acquire("slot1")
        cache.append(_assignment(v0), token_count=10)
        active = cache.active()
        slot1_view = next(v for v in active if v.request_id == "slot1")
        assert slot1_view.token_count == 0
        assert slot1_view.block_table == ()

    def test_release_one_slot_preserves_other(self):
        cache = HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=2)
        v0 = cache.acquire("slot0")
        v1 = cache.acquire("slot1")
        cache.append(_assignment(v0), token_count=8)
        cache.append(_assignment(v1), token_count=4)
        cache.release(_assignment(v0))
        active = cache.active()
        assert len(active) == 1
        assert active[0].request_id == "slot1"
        assert active[0].token_count == 4


class TestStaleAssignmentDetection:
    """Cache must reject stale or foreign slot assignments."""

    def test_stale_assignment_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
        v1 = cache.acquire("first")
        stale = _assignment(v1)
        cache.release(stale)
        cache.acquire("second")
        with pytest.raises(RuntimeError, match="stale|foreign"):
            cache.append(stale, token_count=1)

    def test_foreign_assignment_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=2)
        v0 = cache.acquire("slot0")
        foreign = SlotAssignment(v0.slot_id, "impostor", v0.generation)
        with pytest.raises(RuntimeError, match="stale|foreign"):
            cache.append(foreign, token_count=1)

    def test_inactive_slot_rejected(self):
        cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=2)
        v0 = cache.acquire("slot0")
        assignment = _assignment(v0)
        cache.release(assignment)
        with pytest.raises(RuntimeError, match="not active"):
            cache.append(assignment, token_count=1)


class TestGDNStateSlot:
    """GDN state slot must be stable and equal to slot_id."""

    def test_gdn_state_slot_equals_slot_id(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4), capacity=4)
        for i in range(4):
            view = cache.acquire(f"req-{i}")
            assert view.gdn_state_slot == view.slot_id

    def test_gdn_state_slot_stable_across_appends(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
        view = cache.acquire("req")
        initial_gdn = view.gdn_state_slot
        view = cache.append(_assignment(view), token_count=10)
        assert view.gdn_state_slot == initial_gdn

    def test_gdn_state_slot_stable_across_reuse(self):
        cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4), capacity=1)
        v1 = cache.acquire("first")
        gdn1 = v1.gdn_state_slot
        cache.release(_assignment(v1))
        v2 = cache.acquire("second")
        assert v2.gdn_state_slot == gdn1
