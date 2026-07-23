"""CPU tests for SWA ring KV index math and slab copy logic.

No GPU or model weights required.
"""
import math

import pytest

# Import ring helpers (pure functions, no GPU dependency)
import sys, types

# Stub out runtime.compat_vllm to avoid vllm import
_compat = types.ModuleType("runtime.compat_vllm")
for attr in [
    "VllmConfig", "bind_kv_cache", "get_distributed_init_method",
    "get_model", "get_open_port", "init_worker_distributed_environment",
    "set_current_vllm_config", "set_forward_context",
    "get_flashinfer_metadata_builder", "get_common_attn_metadata_cls",
    "init_flashinfer_workspace",
]:
    setattr(_compat, attr, None)
sys.modules.setdefault("runtime.compat_vllm", _compat)

# Stub other runtime modules that import vllm
for mod_name in [
    "runtime.block_pool", "runtime.logprobs", "runtime.model_spec",
    "runtime.sampling", "runtime.nvfp4_custom_gemm",
    "runtime.nvfp4_cutlass_direct_patch",
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name == "runtime.block_pool":
            m.ChunkedPrefillState = type("ChunkedPrefillState", (), {})
        if mod_name == "runtime.model_spec":
            m.ModelSpec = type("ModelSpec", (), {"from_runner_init": staticmethod(lambda **kw: None)})
        if mod_name == "runtime.sampling":
            m.SamplingParams = type("SamplingParams", (), {})
            m.make_generator = lambda seed: None
            m.sample_from_logits = lambda *a, **kw: None
        if mod_name == "runtime.logprobs":
            m.compute_logprobs = lambda *a, **kw: []
        if mod_name == "runtime.nvfp4_custom_gemm":
            m.patch_nvfp4_custom_gemm = lambda: None
        if mod_name == "runtime.nvfp4_cutlass_direct_patch":
            m.patch_nvfp4_prefer_cutlass_direct = lambda: None
        sys.modules[mod_name] = m

from runtime.backends.laguna import (
    _ring_blocks_for_window,
    _physical_slot,
    RESERVED_PHYSICAL_SLOTS,
    SWA_QO_MAX,
)


class TestRingBlocksFormula:
    """Verify ring_blocks = cdiv(window - 1 + qo_max, block_size) + 1."""

    def test_qo1_window512_bs16(self):
        assert _ring_blocks_for_window(512, 16, qo_max=1) == 33

    def test_qo16_window512_bs16(self):
        # 审查阻断①: DFlash verify qo=16
        assert _ring_blocks_for_window(512, 16, qo_max=16) == 34

    def test_default_qo_max_is_16(self):
        assert SWA_QO_MAX == 16
        assert _ring_blocks_for_window(512, 16) == 34

    def test_ring_slots_cover_max_span(self):
        window, bs, qo_max = 512, 16, 16
        rb = _ring_blocks_for_window(window, bs, qo_max)
        ring_slots = rb * bs
        max_span = (window - 1) + (bs - 1) + (qo_max - 1) + 1
        assert ring_slots >= max_span

    def test_various_windows(self):
        for window in [128, 256, 512, 1024, 2048]:
            for qo in [1, 16]:
                rb = _ring_blocks_for_window(window, 16, qo)
                assert rb * 16 >= window - 1 + 15 + qo


class TestRingIndexMath:
    """Verify ring slot uniqueness within window."""

    WINDOW = 512
    BS = 16
    RING_BLOCKS = 34  # qo_max=16
    RING_SLOTS = 34 * 16  # 544

    def test_no_collision_in_window(self):
        for pos in [0, 100, 511, 512, 1000, 4096, 131071]:
            start = max(0, pos - self.WINDOW + 1)
            slots = [p % self.RING_SLOTS for p in range(start, pos + 1)]
            assert len(set(slots)) == len(slots), f"collision at pos={pos}"

    def test_block_aligned_window_covers_decode(self):
        for pos in [0, 100, 512, 1000, 65536]:
            window_start = max(0, pos - self.WINDOW + 1)
            aligned_start = (window_start // self.BS) * self.BS
            aligned_len = pos + 1 - aligned_start
            n_blocks = math.ceil(aligned_len / self.BS)
            assert n_blocks <= self.RING_BLOCKS

    def test_ring_block_table_construction(self):
        pos = 1000
        window_start = max(0, pos - self.WINDOW + 1)
        aligned_start = (window_start // self.BS) * self.BS
        aligned_len = pos + 1 - aligned_start
        n_blocks = math.ceil(aligned_len / self.BS)
        ring_base = 5 * self.RING_BLOCKS  # phys=5

        bt = []
        for j in range(n_blocks):
            actual = aligned_start + j * self.BS
            rb = (actual % self.RING_SLOTS) // self.BS
            bt.append(ring_base + rb)

        assert len(bt) == n_blocks
        assert all(ring_base <= b < ring_base + self.RING_BLOCKS for b in bt)

    def test_slot_mapping_formula(self):
        pos = 12345
        ring_base = 3 * self.RING_BLOCKS
        rb = (pos % self.RING_SLOTS) // self.BS
        ro = pos % self.BS
        sm = (ring_base + rb) * self.BS + ro
        # Verify: sm points to the correct ring slot
        assert sm // self.BS == ring_base + rb
        assert sm % self.BS == ro


class TestSlabCopyLogic:
    """Verify the slab decomposition for scratch→ring copy."""

    WINDOW = 512
    BS = 16
    RING_SLOTS = 34 * 16

    def _compute_slabs(self, window_start, prompt_len):
        slabs = []
        pos = window_start
        while pos < prompt_len:
            ring_slot = pos % self.RING_SLOTS
            until_wrap = self.RING_SLOTS - ring_slot
            src_off = pos % self.BS
            until_block_end = self.BS - src_off
            count = min(until_wrap, until_block_end, prompt_len - pos)
            slabs.append((pos, ring_slot, count))
            pos += count
        return slabs

    def test_slabs_cover_all_positions(self):
        for prompt_len in [100, 512, 1000, 4096]:
            window_start = max(0, prompt_len - self.WINDOW)
            slabs = self._compute_slabs(window_start, prompt_len)
            covered = sum(s[2] for s in slabs)
            assert covered == prompt_len - window_start

    def test_slabs_no_overlap(self):
        for prompt_len in [100, 512, 1000, 4096]:
            window_start = max(0, prompt_len - self.WINDOW)
            slabs = self._compute_slabs(window_start, prompt_len)
            positions = []
            for src_pos, _, count in slabs:
                positions.extend(range(src_pos, src_pos + count))
            assert len(positions) == len(set(positions))

    def test_slabs_within_block(self):
        """Each slab must not cross a block boundary (for contiguous copy)."""
        for prompt_len in [100, 512, 1000, 4096]:
            window_start = max(0, prompt_len - self.WINDOW)
            slabs = self._compute_slabs(window_start, prompt_len)
            for src_pos, ring_slot, count in slabs:
                src_block_start = src_pos // self.BS
                src_block_end = (src_pos + count - 1) // self.BS
                assert src_block_start == src_block_end, (
                    f"slab crosses block boundary: pos={src_pos} count={count}"
                )
                dst_block_start = ring_slot // self.BS
                dst_block_end = (ring_slot + count - 1) // self.BS
                assert dst_block_start == dst_block_end


class TestPhysicalSlot:
    def test_offset(self):
        assert _physical_slot(0) == RESERVED_PHYSICAL_SLOTS
        assert _physical_slot(3) == 3 + RESERVED_PHYSICAL_SLOTS
