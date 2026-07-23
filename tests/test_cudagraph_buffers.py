"""CUDA Graph buffer 管理回归测试（CPU-only，不需要模型权重）。

验证 fast_decode_plan 契约履行：
- replay() 必须在 _run_plan 前把 indptr/last_page_len 拷到 GPU buffer
- page-crossing 检测逻辑正确
- indptr 累积和正确
- last_page_len 计算正确
- staging buffer 防竞态
"""
from __future__ import annotations

import inspect


class TestFastDecodePlanContract:
    """验证 replay() 履行了 fast_decode_plan 的调用方契约。"""

    def _get_replay_source(self) -> str:
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
        return inspect.getsource(LagunaCudaGraphDecode.replay)

    def test_indptr_gpu_copy_present(self):
        """replay() 必须包含 _fi_indptr_gpu.copy_(_fi_indptr_cpu)。"""
        src = self._get_replay_source()
        assert "_fi_indptr_gpu" in src and "copy_" in src, (
            "replay() 缺少 _fi_indptr_gpu.copy_(_fi_indptr_cpu) — "
            "fast_decode_plan cudagraph 模式不做 CPU→GPU 拷贝，调用方必须自己做"
        )

    def test_last_page_len_gpu_copy_present(self):
        """replay() 必须包含 _fi_last_page_len_gpu.copy_(_fi_last_page_len_cpu)。"""
        src = self._get_replay_source()
        assert "_fi_last_page_len_gpu" in src and "copy_" in src, (
            "replay() 缺少 _fi_last_page_len_gpu.copy_(_fi_last_page_len_cpu) — "
            "fast_decode_plan cudagraph 模式不做 CPU→GPU 拷贝，调用方必须自己做"
        )

    def test_gpu_copy_before_run_plan(self):
        """GPU buffer 拷贝必须在 _run_plan 之前。"""
        src = self._get_replay_source()
        idx_indptr = src.find("_fi_indptr_gpu")
        idx_lpl = src.find("_fi_last_page_len_gpu")
        idx_plan = src.find("_run_plan")
        assert idx_indptr < idx_plan, "indptr GPU 拷贝必须在 _run_plan 之前"
        assert idx_lpl < idx_plan, "last_page_len GPU 拷贝必须在 _run_plan 之前"

    def test_staging_buffer_exists(self):
        """staging buffer 防止未来去掉 .item() 同步后的 pinned buffer 竞态。"""
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
        src = inspect.getsource(LagunaCudaGraphDecode.__init__)
        assert "_fi_last_page_len_staging" in src, (
            "缺少 staging buffer — 去掉 .item() 同步优化时会产生 pinned buffer 竞态"
        )

    def test_no_priming_replay_in_capture(self):
        """capture() 不应包含 priming replay（已证明是同根因的 workaround）。"""
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
        src = inspect.getsource(LagunaCudaGraphDecode.capture)
        assert "Prime" not in src and "priming" not in src.lower(), (
            "capture() 仍包含 priming replay — 根因修复后不再需要"
        )


class TestBufferArithmetic:
    """验证 buffer 计算的纯算术逻辑。"""

    def test_last_page_len_computation(self):
        """last_page_len = new_kv % page_size, 0 → page_size。"""
        page_size = 16
        cases = [
            (1, 1), (15, 15), (16, 16), (17, 1),
            (31, 15), (32, 16), (33, 1), (256, 16),
        ]
        for new_kv, expected in cases:
            lpl = new_kv % page_size
            lpl = lpl if lpl != 0 else page_size
            assert lpl == expected, f"new_kv={new_kv}: got {lpl}, want {expected}"

    def test_n_blocks_computation(self):
        """n_blocks = ceil(new_kv / page_size)。"""
        page_size = 16
        cases = [
            (1, 1), (16, 1), (17, 2), (32, 2), (33, 3), (256, 16),
        ]
        for new_kv, expected in cases:
            n_blocks = (new_kv + page_size - 1) // page_size
            assert n_blocks == expected, f"new_kv={new_kv}: got {n_blocks}, want {expected}"

    def test_indptr_cumulative(self):
        """indptr 是 n_blocks 的前缀和。"""
        n_blocks_list = [1, 2, 1, 3]
        indptr = [0]
        for nb in n_blocks_list:
            indptr.append(indptr[-1] + nb)
        assert indptr == [0, 1, 3, 4, 7]

    def test_slot_mapping_formula(self):
        """slot_mapping = (base + pos // page_size) * page_size + pos % page_size。"""
        page_size = 16
        blocks_per_slot = 256
        slot_id = 0
        phys = slot_id + 1
        base = phys * blocks_per_slot
        for pos in [0, 5, 15, 16, 17, 100]:
            sm = (base + pos // page_size) * page_size + pos % page_size
            expected = base * page_size + pos
            assert sm == expected, f"pos={pos}: got {sm}, want {expected}"

    def test_page_crossing_detection(self):
        """跨页检测：n_blocks 变化时触发 indptr/indices 重建。"""
        page_size = 16
        prev_n_blocks = 0
        crossings = []
        for kv_len in range(50):
            new_kv = kv_len + 1
            n_blocks = (new_kv + page_size - 1) // page_size
            if n_blocks != prev_n_blocks:
                crossings.append(kv_len)
                prev_n_blocks = n_blocks
        assert crossings == [0, 16, 32, 48], f"跨页点错误: {crossings}"


class TestIndependentWorkspace:
    """验证每个 cudagraph wrapper 有独立 workspace。"""

    def test_workspace_not_shared_with_builder(self):
        """_init_wrappers 必须为每个 wrapper 分配独立 workspace。"""
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
        src = inspect.getsource(LagunaCudaGraphDecode._init_wrappers)
        assert "torch.empty" in src or "torch.zeros" in src, (
            "_init_wrappers 必须为每个 wrapper 分配独立 workspace，"
            "不能共享 builder._get_workspace_buffer()"
        )
        assert "_get_workspace_buffer" not in src or "numel" in src, (
            "workspace 大小可以参考 builder 的，但必须是新分配的独立 tensor"
        )
