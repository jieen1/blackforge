"""Real (non-mock) smoke test: drive the eager engine's control plane
against a live vLLM server running the actual unsloth/Qwen3.6-27B-NVFP4
checkpoint (16 full-attention + 48 GDN layers), through
``runtime.vllm_bridge_backend.VLLMBridgeBackend``.

Requires an already-running isolated test server -- see
``/home/bot/project/sm120-flash-attention/vllm_integration/launch_test_server.py``.
Not part of the default ``pytest -q`` suite: AGENTS.md's testing guideline
says unit tests must not require downloading model weights / a live server.

Usage:
    ./.venv/bin/python -m benchmarks.real_forward_smoke \\
        --url http://127.0.0.1:8100/v1 --model qwen3.6-sm120-test
"""

from __future__ import annotations

import argparse
import sys

from runtime.engine import EagerEngine, RequestState
from runtime.hybrid_cache import CacheGeometry, HybridCache
from runtime.op_registry import OpRegistry
from runtime.vllm_bridge_backend import VLLMBridgeBackend


def _build_engine(backend: VLLMBridgeBackend, *, capacity: int) -> EagerEngine:
    registry = OpRegistry()
    registry.register("prefill", backend.prefill)
    registry.register("decode", backend.decode)
    cache = HybridCache(CacheGeometry(block_size=16, max_blocks_per_slot=256), capacity=capacity)
    return EagerEngine(cache, registry)


def _finish(engine: EagerEngine, backend: VLLMBridgeBackend, request_id: str) -> None:
    if engine.request(request_id).state is RequestState.DECODE:
        engine.complete(request_id)
    engine.release(request_id)
    backend.forget(request_id)


def _still_going(engine: EagerEngine, request_id: str, step) -> bool:
    """True while the request has budget left and the backend produced a
    real new token (as opposed to a natural stop with no token)."""
    still_decoding = engine.request(request_id).state is RequestState.DECODE
    return still_decoding and step.output["token_id"] is not None


def single_prefill_and_decode(
    engine: EagerEngine, backend: VLLMBridgeBackend, prompt: str, tokenizer, *, max_new: int
) -> str:
    request_id = "smoke-single"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    engine.submit(request_id, prompt_ids, max_new_tokens=max_new)

    step = engine.prefill(request_id)
    text = step.output["text"]
    while _still_going(engine, request_id, step):
        step = engine.decode(request_id, step.output["token_id"])
        text += step.output["text"]

    snapshot = engine.request(request_id)
    assert snapshot.cache.token_count == len(prompt_ids) + len(snapshot.generated_token_ids), (
        f"cache token_count {snapshot.cache.token_count} != "
        f"prompt {len(prompt_ids)} + generated {len(snapshot.generated_token_ids)}"
    )
    _finish(engine, backend, request_id)
    return text


def continuous_generation(
    engine: EagerEngine, backend: VLLMBridgeBackend, tokenizer, prompt: str, target_tokens: int
) -> tuple[str, int]:
    request_id = "smoke-continuous"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    engine.submit(request_id, prompt_ids, max_new_tokens=target_tokens)

    step = engine.prefill(request_id)
    text = step.output["text"]
    while _still_going(engine, request_id, step):
        step = engine.decode(request_id, step.output["token_id"])
        text += step.output["text"]

    snapshot = engine.request(request_id)
    n_generated = len(snapshot.generated_token_ids)
    _finish(engine, backend, request_id)
    return text, n_generated


def four_slot_isolation(
    engine: EagerEngine, backend: VLLMBridgeBackend, tokenizer, *, steps_per_slot: int = 24
) -> dict[str, bool]:
    codes = ["falcon-9182", "harbor-3305", "cinder-7716", "meridian-6640"]
    request_ids = [f"slot-{i}" for i in range(4)]
    prompts = [
        f"Remember this exact code and nothing else: {code}. "
        f"When asked, respond only with the code. The code is:"
        for code in codes
    ]

    for request_id, prompt in zip(request_ids, prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        engine.submit(request_id, prompt_ids, max_new_tokens=steps_per_slot)

    steps = {rid: engine.prefill(rid) for rid in request_ids}
    texts = {rid: step.output["text"] for rid, step in steps.items()}
    stopped = {rid: steps[rid].output["token_id"] is None for rid in request_ids}

    for _ in range(steps_per_slot - 1):
        ready = [
            snap.request_id for snap in engine.decode_ready() if not stopped[snap.request_id]
        ]
        if not ready:
            break
        for request_id in ready:
            step = engine.decode(request_id, steps[request_id].output["token_id"])
            steps[request_id] = step
            texts[request_id] += step.output["text"]
            if step.output["token_id"] is None:
                stopped[request_id] = True

    verdicts = {}
    for request_id, code in zip(request_ids, codes):
        own_ok = code in texts[request_id]
        others_leaked = any(
            other_code in texts[request_id]
            for other_code, other_id in zip(codes, request_ids)
            if other_id != request_id
        )
        verdicts[request_id] = own_ok and not others_leaked
        print(
            f"  [{request_id}] own_code_present={own_ok} other_code_leaked={others_leaked} "
            f"text={texts[request_id]!r}"
        )
        _finish(engine, backend, request_id)

    return verdicts


def slot_reuse_after_release(engine: EagerEngine, backend: VLLMBridgeBackend, tokenizer) -> bool:
    """Fill all 4 slots, release one, submit a new request, confirm the
    freed physical slot is reused with a bumped generation and produces a
    correct, uncorrupted real completion. This targets the vLLM issue
    #37554 class of risk this CLAUDE.md flags: stale GDN dummy-forward
    state leaking into a reused slot."""
    codes = ["opal-1123", "ridge-4471", "vellum-8890", "quartz-2299"]
    request_ids = [f"reuse-{i}" for i in range(4)]
    prompts = [
        f"Remember this exact code and nothing else: {code}. "
        f"When asked, respond only with the code. The code is:"
        for code in codes
    ]
    for request_id, prompt in zip(request_ids, prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        engine.submit(request_id, prompt_ids, max_new_tokens=1)

    for request_id in request_ids:
        engine.prefill(request_id)

    freed_id = request_ids[0]
    freed_snapshot = engine.request(freed_id)
    freed_slot_id = freed_snapshot.cache.slot_id
    _finish(engine, backend, freed_id)

    new_code = "thistle-5540"
    new_prompt = (
        f"Remember this exact code and nothing else: {new_code}. "
        f"When asked, respond only with the code. The code is:"
    )
    new_prompt_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
    submitted = engine.submit("reuse-new", new_prompt_ids, max_new_tokens=12)
    reused_ok = (
        submitted.cache.slot_id == freed_slot_id
        and submitted.cache.generation == freed_snapshot.cache.generation + 1
    )
    step = engine.prefill("reuse-new")
    new_text = step.output["text"]
    while _still_going(engine, "reuse-new", step):
        step = engine.decode("reuse-new", step.output["token_id"])
        new_text += step.output["text"]

    correct_ok = new_code in new_text
    no_leak_ok = not any(c in new_text for c in codes)

    expected_generation = freed_snapshot.cache.generation + 1
    print(
        f"  reused slot_id={submitted.cache.slot_id} (expected {freed_slot_id}), "
        f"generation={submitted.cache.generation} (expected {expected_generation}), "
        f"new_text={new_text!r}, correct_ok={correct_ok}, no_leak_ok={no_leak_ok}"
    )

    for request_id in request_ids[1:] + ["reuse-new"]:
        _finish(engine, backend, request_id)

    return reused_ok and correct_ok and no_leak_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen3.6-sm120-test")
    parser.add_argument("--tokenizer", default="unsloth/Qwen3.6-27B-NVFP4")
    parser.add_argument("--continuous-tokens", type=int, default=256)
    parser.add_argument(
        "--tests", default="1,2,3,4", help="comma-separated subset of test numbers to run"
    )
    args = parser.parse_args()
    selected = {int(t) for t in args.tests.split(",")}

    backend = VLLMBridgeBackend(base_url=args.url, model=args.model, tokenizer_id=args.tokenizer)
    tokenizer = backend._tokenizer  # reuse the already-loaded tokenizer

    all_ok = True

    if 1 in selected:
        print("=== 1) single prefill + short decode ===")
        engine = _build_engine(backend, capacity=1)
        text = single_prefill_and_decode(
            engine, backend, "The capital of France is", tokenizer, max_new=16
        )
        ok = "Paris" in text
        print(f"  generated: {text!r}\n  PASS={ok}")
        all_ok &= ok

    if 2 in selected:
        print("=== 2) continuous generation ===")
        engine = _build_engine(backend, capacity=1)
        text, n = continuous_generation(
            engine, backend, tokenizer,
            "Write a short paragraph describing how lighthouses work.",
            args.continuous_tokens,
        )
        ok = n >= 8 and len(text.strip()) > 0
        print(f"  generated {n} tokens, text[:200]={text[:200]!r}\n  PASS={ok}")
        all_ok &= ok

    if 3 in selected:
        print("=== 3) four-slot concurrent isolation ===")
        engine = _build_engine(backend, capacity=4)
        verdicts = four_slot_isolation(engine, backend, tokenizer)
        ok = all(verdicts.values())
        print(f"  verdicts={verdicts}\n  PASS={ok}")
        all_ok &= ok

    if 4 in selected:
        print("=== 4) slot release + reuse, no state leak ===")
        engine = _build_engine(backend, capacity=4)
        ok = slot_reuse_after_release(engine, backend, tokenizer)
        print(f"  PASS={ok}")
        all_ok &= ok

    print(f"\n=== overall PASS={bool(all_ok)} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
