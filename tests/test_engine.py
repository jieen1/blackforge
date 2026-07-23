import pytest

from runtime.engine import EagerEngine, EngineError, ExecutionRequest, RequestState
from runtime.hybrid_cache import CacheGeometry, HybridCache
from runtime.op_registry import OpRegistry


def _engine(*, capacity: int = 4, max_tokens: int = 32):
    calls: list[ExecutionRequest] = []
    registry = OpRegistry()
    registry.register("prefill", lambda context: calls.append(context) or {"phase": "prefill"})
    registry.register("decode", lambda context: calls.append(context) or {"phase": "decode"})
    engine = EagerEngine(
        HybridCache(
            CacheGeometry(block_size=4, max_blocks_per_slot=max_tokens // 4), capacity=capacity
        ),
        registry,
    )
    return engine, calls


def test_submit_and_prefill_expose_stable_cache_metadata() -> None:
    engine, calls = _engine()
    submitted = engine.submit("alpha", [10, 11, 12, 13, 14], max_new_tokens=2)

    assert submitted.state is RequestState.PREFILL
    result = engine.prefill("alpha")

    assert result.output == {"phase": "prefill"}
    assert result.request.state is RequestState.DECODE
    assert result.request.cache.token_count == 5
    assert result.request.cache.block_table == (0, 1)
    assert calls[0].phase is RequestState.PREFILL
    assert calls[0].token_ids == (10, 11, 12, 13, 14)
    assert calls[0].cache.gdn_state_slot == 0


def test_decode_completes_at_maximum_and_release_resets_slot() -> None:
    engine, calls = _engine(capacity=1)
    first = engine.submit("first", [1, 2], max_new_tokens=2)
    engine.prefill("first")

    first_decode = engine.decode("first", 20)
    final_decode = engine.decode("first", 21)

    assert first_decode.request.state is RequestState.DECODE
    assert final_decode.request.state is RequestState.COMPLETED
    assert final_decode.request.generated_token_ids == (20, 21)
    assert final_decode.request.cache.token_count == 4
    assert [call.token_ids for call in calls] == [(1, 2), (20,), (21,)]

    engine.release("first")
    second = engine.submit("second", [9], max_new_tokens=1)
    assert second.cache.slot_id == first.cache.slot_id
    assert second.cache.generation == first.cache.generation + 1
    assert second.cache.token_count == 0


def test_prefill_all_and_decode_batch_use_physical_slot_order() -> None:
    engine, calls = _engine(capacity=3)
    for request_id in ("zero", "one", "two"):
        engine.submit(request_id, [0], max_new_tokens=2)

    assert [request.request_id for request in engine.prefill_ready()] == ["zero", "one", "two"]
    engine.prefill_all()
    results = engine.decode_batch({"two": 22, "zero": 20})

    assert [result.request.request_id for result in results] == ["zero", "two"]
    assert [call.request_id for call in calls] == ["zero", "one", "two", "zero", "two"]
    assert [request.request_id for request in engine.decode_ready()] == ["zero", "one", "two"]


def test_invalid_transitions_and_capacity_do_not_release_live_cache() -> None:
    engine, _ = _engine(capacity=1)
    engine.submit("one", [1], max_new_tokens=1)

    with pytest.raises(EngineError, match="expected decode"):
        engine.decode("one", 2)
    with pytest.raises(EngineError, match="must complete"):
        engine.release("one")
    with pytest.raises(EngineError, match="unable to schedule"):
        engine.submit("two", [2], max_new_tokens=1)

    engine.complete("one")
    engine.release("one")
    with pytest.raises(EngineError, match="released"):
        engine.request("one")
    with pytest.raises(EngineError, match="released"):
        engine.prefill("one")


def test_executor_failure_marks_request_for_release() -> None:
    registry = OpRegistry()

    def fail(_: ExecutionRequest) -> None:
        raise RuntimeError("backend error")

    registry.register("prefill", fail)
    registry.register("decode", lambda context: context)
    engine = EagerEngine(
        HybridCache(CacheGeometry(block_size=4, max_blocks_per_slot=2), capacity=1), registry
    )
    engine.submit("broken", [1], max_new_tokens=1)

    with pytest.raises(EngineError, match="prefill operation failed"):
        engine.prefill("broken")
    assert engine.request("broken").state is RequestState.FAILED
    engine.release("broken")


def test_input_validation_and_unknown_batch_members_are_rejected() -> None:
    engine, _ = _engine()

    with pytest.raises(ValueError, match="must not be empty"):
        engine.submit("empty", [], max_new_tokens=1)
    with pytest.raises(TypeError, match="only integers"):
        engine.submit("bad", [True], max_new_tokens=1)
    with pytest.raises(ValueError, match="positive"):
        engine.submit("zero", [1], max_new_tokens=0)

    engine.submit("good", [1], max_new_tokens=1)
    with pytest.raises(EngineError, match="unknown request"):
        engine.decode_batch({"missing": 1})


class TestWatchdog:
    """Unit tests for the engine watchdog (D1) via extracted pure functions."""

    def test_find_stale_slots_detects_wedged_slot(self):
        from server.engine import find_stale_slots

        active = {
            0: {"last_progress_round": 450},
            1: {"last_progress_round": 100},
            2: {"last_progress_round": 499},
        }
        stale = find_stale_slots(active, current_round=500, max_stale_rounds=200)
        assert stale == [1]

    def test_find_stale_slots_no_stale(self):
        from server.engine import find_stale_slots

        active = {
            0: {"last_progress_round": 490},
            1: {"last_progress_round": 495},
        }
        assert find_stale_slots(active, current_round=500, max_stale_rounds=200) == []

    def test_find_stale_slots_disabled_at_zero(self):
        from server.engine import find_stale_slots

        active = {0: {"last_progress_round": 0}}
        assert find_stale_slots(active, current_round=1000, max_stale_rounds=0) == []

    def test_find_stale_slots_missing_key_defaults_to_zero(self):
        from server.engine import find_stale_slots

        active = {0: {}}  # no last_progress_round key
        stale = find_stale_slots(active, current_round=300, max_stale_rounds=200)
        assert stale == [0]

    def test_find_stale_slots_empty_active(self):
        from server.engine import find_stale_slots

        assert find_stale_slots({}, current_round=100, max_stale_rounds=10) == []

    def test_find_stale_slots_boundary_not_stale(self):
        """Exactly max_stale_rounds behind is NOT stale (must exceed)."""
        from server.engine import find_stale_slots

        active = {0: {"last_progress_round": 300}}
        assert find_stale_slots(active, current_round=500, max_stale_rounds=200) == []

    def test_watchdog_config_default(self):
        from server.engine import ServerEngine

        assert "watchdog_max_stale_rounds" in ServerEngine.__init__.__code__.co_varnames


class TestRequestTimeout:
    """Unit tests for request-level timeout (C5) via extracted pure functions."""

    def test_find_timed_out_slots_detects_expired(self):
        from server.engine import find_timed_out_slots

        now = 1000.0
        active = {
            0: {"start_time": now - 100},
            1: {"start_time": now - 700},
            2: {"start_time": now - 50},
        }
        timed_out = find_timed_out_slots(active, now=now, timeout_s=600.0)
        assert timed_out == [1]

    def test_find_timed_out_slots_none_expired(self):
        from server.engine import find_timed_out_slots

        now = 1000.0
        active = {
            0: {"start_time": now - 10},
            1: {"start_time": now - 20},
        }
        assert find_timed_out_slots(active, now=now, timeout_s=600.0) == []

    def test_find_timed_out_slots_disabled_at_zero(self):
        from server.engine import find_timed_out_slots

        active = {0: {"start_time": 0.0}}
        assert find_timed_out_slots(active, now=9999.0, timeout_s=0) == []

    def test_find_timed_out_slots_missing_key_defaults_to_now(self):
        """Missing start_time defaults to now (never times out)."""
        from server.engine import find_timed_out_slots

        active = {0: {}}
        assert find_timed_out_slots(active, now=1000.0, timeout_s=600.0) == []

    def test_find_timed_out_slots_empty_active(self):
        from server.engine import find_timed_out_slots

        assert find_timed_out_slots({}, now=1000.0, timeout_s=60.0) == []

    def test_find_timed_out_slots_boundary_not_expired(self):
        """Exactly at timeout boundary is NOT expired (must exceed)."""
        from server.engine import find_timed_out_slots

        now = 1000.0
        active = {0: {"start_time": now - 600.0}}
        assert find_timed_out_slots(active, now=now, timeout_s=600.0) == []

    def test_timeout_config_exists(self):
        import inspect

        from server.engine import ServerEngine

        sig = inspect.signature(ServerEngine.__init__)
        assert "request_timeout_s" in sig.parameters
