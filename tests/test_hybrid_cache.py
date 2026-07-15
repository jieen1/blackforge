import pytest

from runtime.hybrid_cache import CacheGeometry, HybridCache


def test_cache_blocks_and_gdn_slot_are_stable_until_release() -> None:
    cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=4))
    first = cache.acquire("first")
    advanced = cache.append(_assignment(first), token_count=17)

    assert advanced.block_table == (0, 1)
    assert advanced.gdn_state_slot == 0


def test_cache_reset_and_reuse_does_not_inherit_blocks() -> None:
    cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=2), capacity=1)
    first = cache.acquire("first")
    first_assignment = _assignment(first)
    cache.append(first_assignment, token_count=9)
    cache.release(first_assignment)

    replacement = cache.acquire("replacement")

    assert replacement.slot_id == first.slot_id
    assert replacement.generation == first.generation + 1
    assert replacement.token_count == 0
    assert replacement.block_table == ()


def test_cache_rejects_requests_larger_than_fixed_slot() -> None:
    cache = HybridCache(CacheGeometry(block_size=8, max_blocks_per_slot=1), capacity=1)
    view = cache.acquire("request")

    with pytest.raises(RuntimeError, match="exceeds"):
        cache.append(_assignment(view), token_count=9)


def _assignment(view):
    from runtime.slot_manager import SlotAssignment

    return SlotAssignment(view.slot_id, view.request_id, view.generation)
