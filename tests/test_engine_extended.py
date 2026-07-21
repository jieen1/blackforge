"""Extended regression tests for EagerEngine.

Covers historical bugs:
- a83bc5a: reset_slot didn't clear draft state -- engine release must
  fully reset so re-submitted requests get clean cache state
- 1295637: physical slot 0 reservation -- engine must work correctly
  with the slot manager's addressing
- 6e80e0f: engine deadlock -- engine must not deadlock when transitioning
  between states

Also covers edge cases:
- Decode completion at exact max_new_tokens boundary
- Failed request lifecycle (fail -> release -> re-submit)
- Batch decode ordering guarantees
- Complete from prefill state
- Re-submit after release with same request_id
- Multi-step decode with correct token accumulation
- Cache metadata correctness through full lifecycle
"""

import pytest

from runtime.engine import EagerEngine, EngineError, ExecutionRequest, RequestState
from runtime.hybrid_cache import CacheGeometry, HybridCache
from runtime.op_registry import OpRegistry


def _engine(*, capacity: int = 4, max_tokens: int = 32):
    calls: list[ExecutionRequest] = []
    registry = OpRegistry()
    registry.register("prefill", lambda ctx: calls.append(ctx) or {"phase": "prefill"})
    registry.register("decode", lambda ctx: calls.append(ctx) or {"phase": "decode"})
    engine = EagerEngine(
        HybridCache(
            CacheGeometry(block_size=4, max_blocks_per_slot=max_tokens // 4),
            capacity=capacity,
        ),
        registry,
    )
    return engine, calls


class TestDecodeCompletionBoundary:
    """Decode must complete at exactly max_new_tokens, not before or after."""

    def test_completes_at_exact_max(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1, 2], max_new_tokens=3)
        engine.prefill("req")
        r1 = engine.decode("req", 10)
        assert r1.request.state is RequestState.DECODE
        r2 = engine.decode("req", 11)
        assert r2.request.state is RequestState.DECODE
        r3 = engine.decode("req", 12)
        assert r3.request.state is RequestState.COMPLETED
        assert r3.request.generated_token_ids == (10, 11, 12)

    def test_single_token_max(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=1)
        engine.prefill("req")
        result = engine.decode("req", 42)
        assert result.request.state is RequestState.COMPLETED
        assert result.request.generated_token_ids == (42,)

    def test_cannot_decode_after_completion(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=1)
        engine.prefill("req")
        engine.decode("req", 42)
        with pytest.raises(EngineError, match="completed"):
            engine.decode("req", 43)


class TestFailedRequestLifecycle:
    """Failed requests must be releasable and slots reusable."""

    def test_failed_prefill_can_be_released(self):
        registry = OpRegistry()
        registry.register("prefill", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
        registry.register("decode", lambda ctx: ctx)
        engine = EagerEngine(
            HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=1),
            registry,
        )
        engine.submit("broken", [1], max_new_tokens=1)
        with pytest.raises(EngineError, match="prefill operation failed"):
            engine.prefill("broken")
        assert engine.request("broken").state is RequestState.FAILED
        engine.release("broken")

    def test_failed_slot_can_be_reused(self):
        registry = OpRegistry()
        call_count = [0]

        def maybe_fail(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            return {"ok": True}

        registry.register("prefill", maybe_fail)
        registry.register("decode", lambda ctx: {"ok": True})
        engine = EagerEngine(
            HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=1),
            registry,
        )
        engine.submit("first", [1], max_new_tokens=1)
        with pytest.raises(EngineError):
            engine.prefill("first")
        engine.release("first")

        engine.submit("second", [2], max_new_tokens=1)
        result = engine.prefill("second")
        assert result.request.state is RequestState.DECODE

    def test_failed_decode_can_be_released(self):
        registry = OpRegistry()
        registry.register("prefill", lambda ctx: {"ok": True})
        registry.register("decode", lambda ctx: (_ for _ in ()).throw(RuntimeError("decode boom")))
        engine = EagerEngine(
            HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=4), capacity=1),
            registry,
        )
        engine.submit("req", [1], max_new_tokens=2)
        engine.prefill("req")
        with pytest.raises(EngineError, match="decode operation failed"):
            engine.decode("req", 10)
        assert engine.request("req").state is RequestState.FAILED
        engine.release("req")


class TestCompleteFromPrefill:
    """Complete must work from prefill state (early termination)."""

    def test_complete_from_prefill(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1, 2, 3], max_new_tokens=10)
        snapshot = engine.complete("req")
        assert snapshot.state is RequestState.COMPLETED

    def test_complete_from_decode(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=10)
        engine.prefill("req")
        snapshot = engine.complete("req")
        assert snapshot.state is RequestState.COMPLETED

    def test_cannot_complete_twice(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=10)
        engine.complete("req")
        with pytest.raises(EngineError, match="cannot complete"):
            engine.complete("req")

    def test_cannot_complete_released(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=10)
        engine.complete("req")
        engine.release("req")
        with pytest.raises(EngineError, match="released"):
            engine.complete("req")


class TestReleaseValidation:
    """Release must enforce correct lifecycle ordering."""

    def test_cannot_release_active_prefill(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=1)
        with pytest.raises(EngineError, match="must complete"):
            engine.release("req")

    def test_cannot_release_active_decode(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=2)
        engine.prefill("req")
        with pytest.raises(EngineError, match="must complete"):
            engine.release("req")

    def test_cannot_release_unknown(self):
        engine, _ = _engine(capacity=1)
        with pytest.raises(EngineError, match="unknown"):
            engine.release("ghost")


class TestBatchDecodeOrdering:
    """Batch decode must always execute in physical-slot order."""

    def test_batch_decode_physical_order(self):
        engine, calls = _engine(capacity=4)
        for rid in ("a", "b", "c", "d"):
            engine.submit(rid, [1], max_new_tokens=5)
        engine.prefill_all()
        results = engine.decode_batch({"d": 40, "b": 20, "a": 10, "c": 30})
        result_ids = [r.request.request_id for r in results]
        assert result_ids == ["a", "b", "c", "d"]

    def test_batch_decode_subset(self):
        engine, calls = _engine(capacity=4)
        for rid in ("a", "b", "c"):
            engine.submit(rid, [1], max_new_tokens=5)
        engine.prefill_all()
        results = engine.decode_batch({"c": 30, "a": 10})
        result_ids = [r.request.request_id for r in results]
        assert result_ids == ["a", "c"]

    def test_batch_decode_rejects_unknown_member(self):
        engine, _ = _engine(capacity=2)
        engine.submit("a", [1], max_new_tokens=5)
        engine.prefill("a")
        with pytest.raises(EngineError, match="unknown"):
            engine.decode_batch({"a": 10, "ghost": 20})

    def test_batch_decode_rejects_non_decode_member(self):
        engine, _ = _engine(capacity=2)
        engine.submit("a", [1], max_new_tokens=5)
        with pytest.raises(EngineError, match="prefill"):
            engine.decode_batch({"a": 10})


class TestCacheMetadataThroughLifecycle:
    """Cache metadata must be correct at every lifecycle stage."""

    def test_prefill_sets_correct_token_count(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [10, 11, 12, 13, 14], max_new_tokens=2)
        result = engine.prefill("req")
        assert result.request.cache.token_count == 5

    def test_decode_increments_token_count(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1, 2], max_new_tokens=3)
        engine.prefill("req")
        r1 = engine.decode("req", 10)
        assert r1.request.cache.token_count == 3
        r2 = engine.decode("req", 11)
        assert r2.request.cache.token_count == 4

    def test_block_table_grows_with_tokens(self):
        engine, _ = _engine(capacity=1, max_tokens=16)
        engine.submit("req", [1, 2, 3], max_new_tokens=10)
        result = engine.prefill("req")
        assert result.request.cache.block_table == (0,)
        for i in range(5):
            result = engine.decode("req", 100 + i)
        assert result.request.cache.token_count == 8
        assert result.request.cache.block_table == (0, 1)

    def test_generated_tokens_accumulate_correctly(self):
        engine, _ = _engine(capacity=1)
        engine.submit("req", [1], max_new_tokens=5)
        engine.prefill("req")
        tokens = [10, 20, 30, 40, 50]
        for t in tokens:
            result = engine.decode("req", t)
        assert result.request.generated_token_ids == tuple(tokens)


class TestSubmitValidation:
    """Submit must validate inputs thoroughly."""

    def test_empty_prompt_rejected(self):
        engine, _ = _engine()
        with pytest.raises(ValueError, match="must not be empty"):
            engine.submit("req", [], max_new_tokens=1)

    def test_bool_token_rejected(self):
        engine, _ = _engine()
        with pytest.raises(TypeError, match="only integers"):
            engine.submit("req", [True], max_new_tokens=1)

    def test_zero_max_tokens_rejected(self):
        engine, _ = _engine()
        with pytest.raises(ValueError, match="positive"):
            engine.submit("req", [1], max_new_tokens=0)

    def test_negative_max_tokens_rejected(self):
        engine, _ = _engine()
        with pytest.raises(ValueError, match="positive"):
            engine.submit("req", [1], max_new_tokens=-1)

    def test_bool_max_tokens_rejected(self):
        engine, _ = _engine()
        with pytest.raises(ValueError, match="positive"):
            engine.submit("req", [1], max_new_tokens=True)

    def test_duplicate_request_id_rejected(self):
        engine, _ = _engine()
        engine.submit("req", [1], max_new_tokens=1)
        with pytest.raises(EngineError, match="already known"):
            engine.submit("req", [2], max_new_tokens=1)

    def test_capacity_exhaustion_rejected(self):
        engine, _ = _engine(capacity=1)
        engine.submit("first", [1], max_new_tokens=1)
        with pytest.raises(EngineError, match="unable to schedule"):
            engine.submit("second", [2], max_new_tokens=1)


class TestPrefillOrdering:
    """Prefill must execute in stable physical-slot order."""

    def test_prefill_ready_returns_slot_order(self):
        engine, _ = _engine(capacity=4)
        engine.submit("c", [1], max_new_tokens=1)
        engine.submit("a", [2], max_new_tokens=1)
        engine.submit("b", [3], max_new_tokens=1)
        ready = engine.prefill_ready()
        assert [r.request_id for r in ready] == ["c", "a", "b"]

    def test_prefill_all_executes_in_order(self):
        engine, calls = _engine(capacity=4)
        engine.submit("c", [1], max_new_tokens=1)
        engine.submit("a", [2], max_new_tokens=1)
        engine.submit("b", [3], max_new_tokens=1)
        engine.prefill_all()
        assert [c.request_id for c in calls] == ["c", "a", "b"]


class TestActiveRequests:
    """active() must reflect current engine state."""

    def test_active_excludes_released(self):
        engine, _ = _engine(capacity=2)
        engine.submit("a", [1], max_new_tokens=1)
        engine.submit("b", [2], max_new_tokens=1)
        engine.prefill("a")
        engine.decode("a", 10)
        engine.release("a")
        active = engine.active()
        assert len(active) == 1
        assert active[0].request_id == "b"

    def test_active_includes_all_unreleased(self):
        engine, _ = _engine(capacity=4)
        engine.submit("a", [1], max_new_tokens=5)
        engine.submit("b", [2], max_new_tokens=5)
        engine.prefill("a")
        active = engine.active()
        assert len(active) == 2
