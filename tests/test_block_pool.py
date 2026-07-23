"""Unit tests for runtime/block_pool.py — CPU-only."""

from __future__ import annotations

import pytest

from runtime.block_pool import (
    RESERVED_PHYSICAL_SLOTS,
    Block,
    BlockPool,
    FreeBlockQueue,
    _initial_block_table,
    _physical_slot,
    _ssm_spec_row,
    hash_block_tokens,
)


class TestPhysicalSlot:
    def test_offset_by_reserved(self):
        assert _physical_slot(0) == RESERVED_PHYSICAL_SLOTS
        assert _physical_slot(1) == 1 + RESERVED_PHYSICAL_SLOTS
        assert _physical_slot(3) == 3 + RESERVED_PHYSICAL_SLOTS


class TestInitialBlockTable:
    def test_contiguous_blocks(self):
        table = _initial_block_table(0, 16)
        assert len(table) == 16
        first = _physical_slot(0) * 16
        assert table == list(range(first, first + 16))

    def test_different_slots_dont_overlap(self):
        t0 = set(_initial_block_table(0, 16))
        t1 = set(_initial_block_table(1, 16))
        assert t0.isdisjoint(t1)


class TestSsmSpecRow:
    def test_col0_is_physical_slot(self):
        assert _ssm_spec_row(0, 0, 10, 3) == _physical_slot(0)
        assert _ssm_spec_row(2, 0, 10, 3) == _physical_slot(2)

    def test_col_nonzero_in_separate_range(self):
        total = 10
        num_spec = 3
        row_col1 = _ssm_spec_row(0, 1, total, num_spec)
        assert row_col1 >= total

    def test_different_cols_different_rows(self):
        rows = [_ssm_spec_row(0, c, 10, 3) for c in range(4)]
        assert len(set(rows)) == 4


class TestHashBlockTokens:
    def test_deterministic(self):
        h1 = hash_block_tokens(None, [1, 2, 3], ("fp8",))
        h2 = hash_block_tokens(None, [1, 2, 3], ("fp8",))
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        h1 = hash_block_tokens(None, [1, 2, 3], ("fp8",))
        h2 = hash_block_tokens(None, [1, 2, 4], ("fp8",))
        assert h1 != h2

    def test_different_parent_different_hash(self):
        h1 = hash_block_tokens(123, [1, 2, 3], ("fp8",))
        h2 = hash_block_tokens(456, [1, 2, 3], ("fp8",))
        assert h1 != h2

    def test_different_extra_keys_different_hash(self):
        h1 = hash_block_tokens(None, [1, 2, 3], ("fp8",))
        h2 = hash_block_tokens(None, [1, 2, 3], ("nvfp4",))
        assert h1 != h2

    def test_128_bit_range(self):
        h = hash_block_tokens(None, [1, 2, 3], ("fp8",))
        assert 0 < h < 2**128


class TestFreeBlockQueue:
    def test_fifo_order(self):
        blocks = [Block(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        q.append(blocks[0])
        q.append(blocks[1])
        q.append(blocks[2])
        assert len(q) == 3
        assert q.popleft() is blocks[0]
        assert q.popleft() is blocks[1]
        assert q.popleft() is blocks[2]
        assert len(q) == 0

    def test_remove(self):
        blocks = [Block(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        q.append(blocks[0])
        q.append(blocks[1])
        q.append(blocks[2])
        q.remove(blocks[1])
        assert len(q) == 2
        assert q.popleft() is blocks[0]
        assert q.popleft() is blocks[2]

    def test_appendleft(self):
        blocks = [Block(block_id=i) for i in range(5)]
        q = FreeBlockQueue(blocks)
        q.append(blocks[0])
        q.appendleft(blocks[1])
        assert q.popleft() is blocks[1]
        assert q.popleft() is blocks[0]


class TestBlockPool:
    def test_init(self):
        pool = BlockPool(num_blocks=16)
        assert pool.num_blocks == 16
        assert len(pool._free_queue) == 16 - RESERVED_PHYSICAL_SLOTS

    def test_reserved_blocks_not_in_free_queue(self):
        pool = BlockPool(num_blocks=8, reserved=2)
        assert len(pool._free_queue) == 6  # 8 - 2 reserved

    def test_invalid_num_blocks(self):
        with pytest.raises(ValueError):
            BlockPool(num_blocks=1, reserved=1)
