"""Real end-to-end HTTP validation for ``server/app.py`` (2026-07-19).

An independent audit found this repository's runtime "has never been
deployed anywhere -- server/ is an empty README, 100% of validated work
is benchmark-harness-driven." This script is the closing evidence for the
new ``server/`` code this round added: it actually starts the FastAPI/
uvicorn app (a real ASGI HTTP server bound to a real TCP port on
loopback), drives it with genuine HTTP requests (``httpx`` over a real
socket, not an in-process ASGI test client), and independently verifies
correctness using this project's own established methodology -- single-
slot independent-reference replay (``_ref_check``/``_near_tie_margin_diag``,
imported verbatim from ``mtp_async_arrival_check.py``, NOT reimplemented)
with ``NEAR_TIE_LOGIT_MARGIN=2.0`` near-tie tolerance.

Design notes (why this is one process, one event loop, one model load):

- Only ONE ``DirectModelRunner``/model instance is ever constructed in
  this whole run -- this machine has 23GB host RAM and the model load
  path is RAM-heavy; a second concurrent load (e.g. a genuinely separate
  server subprocess PLUS a separate reference-runner process) risks
  exactly the resource contention this project's own environment notes
  warn about. Instead, the uvicorn ``Server`` is run via
  ``asyncio.create_task(server.serve())`` on THIS SCRIPT'S OWN event
  loop, and the independent reference-replay calls
  (``engine.runner.prefill``/``engine.runner._forward``, direct Python
  calls, not HTTP) are made from coroutines on that SAME event loop --
  single-threaded cooperative scheduling guarantees these never execute
  concurrently with the engine's own admission/verify round (avoiding any
  cross-thread CUDA-context question entirely), while genuine HTTP
  requests (``httpx.AsyncClient`` against ``http://127.0.0.1:<port>``)
  still travel through a real socket/ASGI stack, not a bypassed in-process
  call.
- The engine is constructed with 5*capacity slots (not this project's
  usual 4*capacity production default) so this script gets its OWN
  dedicated validation slots (the top ``capacity`` of THOSE, i.e.
  [4*capacity, 5*capacity)) that are provably disjoint from every slot
  the engine's own production/ref/diag/cudagraph-warmup machinery ever
  touches (see ``server/engine.py``'s slot-layout comment) -- confirmed
  by construction, not merely assumed.
- ``debug_committed_token_ids``/``debug_prompt_token_ids`` (non-standard
  extra fields ``server/app.py`` documents as existing solely for this
  script) are what let this script get the EXACT real committed tokens
  the production batched path generated for a real HTTP request, without
  needing a second model to reproduce them from scratch.

Usage:
    python -m benchmarks.server_e2e_check
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
CAPACITY = 4
# prod(0..3) / ref(4..7) / diag(8..11) / cudagraph-warmup(12..15) --
# engine's own reservation, 4*CAPACITY -- plus 2*CAPACITY more reserved
# here for THIS SCRIPT's own independent-replay slots (16..19) and their
# paired margin-diagnostic slots (20..23, only touched on an actual
# divergence -- mirrors ref_slot_for/diag_slot_for's own two-slots-per-check
# pattern in server/engine.py/mtp_async_arrival_check.py exactly, since
# _near_tie_margin_diag must run on a slot DIFFERENT from the one prefill()
# just used, never the same one -- see that function's call sites).
NUM_SLOTS = 6 * CAPACITY
PORT = 8391

VALIDATION_SLOTS = list(range(4 * CAPACITY, 5 * CAPACITY))
VALIDATION_DIAG_SLOTS = list(range(5 * CAPACITY, 6 * CAPACITY))

os.environ["QSR_SERVER_CAPACITY"] = str(CAPACITY)
os.environ["QSR_SERVER_NUM_SLOTS"] = str(NUM_SLOTS)
os.environ.setdefault("QSR_SERVER_ENABLE_CUDAGRAPH", "1")

sys.path.insert(0, SM120_VLLM_INTEGRATION)


async def _wait_ready(client, timeout_s: float = 900.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = await client.get("/health", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(1.0)
    raise RuntimeError("server did not become ready in time")


async def _independent_reference_replay(
    engine, val_slot: int, diag_slot: int, tok, prompt_text: str, committed_ids: list[int]
) -> dict:
    """Re-derive the exact prompt token ids the server used (by re-encoding
    the SAME text with the SAME tokenizer -- guaranteed identical to what
    ``/v1/completions`` did internally, sidestepping any decode/encode
    round-trip ambiguity), then replay this real HTTP response's ACTUAL
    committed tokens through an INDEPENDENT single slot, one token at a
    time, using this project's own established ``_ref_check``/
    ``_near_tie_margin_diag`` (imported, not reimplemented). This is
    exactly what proves the batched/HTTP path's real output is faithfully
    reproducible from an independent pass through the identical model --
    the same technique this project's own async-arrival/sustained-
    workload benchmarks use to certify no cross-request state leakage."""
    from benchmarks.mtp_async_arrival_check import NEAR_TIE_LOGIT_MARGIN, _near_tie_margin_diag, _ref_check

    runner = engine.runner
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    if runner.slot_kv_len[val_slot] != 0:
        runner.reset_slot(val_slot)
    ref_first = runner.prefill(val_slot, prompt_ids)

    rounds: list[dict] = []
    ok = True
    if not committed_ids:
        return {"ok": True, "rounds": [], "note": "empty completion, nothing to replay"}

    if ref_first != committed_ids[0]:
        # MUST use a slot DIFFERENT from val_slot (which prefill() just
        # advanced) -- _near_tie_margin_diag does its own start_pos=0
        # forward, which would double-count val_slot's kv_len if reused
        # on the same physical slot (the exact reason the established
        # ref_slot_for/diag_slot_for convention uses two distinct slots).
        if runner.slot_kv_len[diag_slot] != 0:
            runner.reset_slot(diag_slot)
        diag = _near_tie_margin_diag(runner, diag_slot, prompt_ids, committed_ids[0])
        rounds.append({"stage": "first_token", "ref_first": ref_first, "served_first": committed_ids[0], "diag": diag})
        if not diag["within_tolerance"]:
            ok = False

    for i in range(len(committed_ids) - 1):
        real_token = committed_ids[i]
        served_next = committed_ids[i + 1]
        report = _ref_check(runner, val_slot, [real_token], served_next, tok=tok)
        rounds.append({"stage": f"round_{i}", **report})
        if not report["content_ok_within_near_tie_tolerance"]:
            ok = False

    return {
        "ok": ok,
        "rounds": rounds,
        "near_tie_logit_margin": NEAR_TIE_LOGIT_MARGIN,
        "num_rounds_checked": len(committed_ids) - 1,
    }


async def _run() -> dict:
    import httpx
    import uvicorn
    from transformers import AutoTokenizer

    import server.app as app_module

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=PORT, log_level="warning")
    uv_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(uv_server.serve())

    result: dict = {"passed": False}
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
            await _wait_ready(client)
            engine = app_module.engine
            assert engine is not None
            tok = engine.tok
            result["engine_config"] = {
                "capacity": engine.capacity,
                "num_slots": engine.num_slots,
                "capacity_tokens_per_slot": engine.capacity_tokens_per_slot,
            }

            # -- 1. Basic real chat-completion round trip: coherent output. --
            basic = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What does this function return? def f(x): return x + 1"}],
                    "max_tokens": 48,
                },
                timeout=120.0,
            )
            basic_ok = basic.status_code == 200 and len(basic.json()["choices"][0]["message"]["content"]) > 0
            result["basic_chat_completion"] = {
                "status_code": basic.status_code,
                "finish_reason": basic.json().get("choices", [{}])[0].get("finish_reason") if basic.status_code == 200 else None,
                "content_preview": basic.json()["choices"][0]["message"]["content"][:200] if basic_ok else None,
                "ok": basic_ok,
            }

            # -- 2. Correctness: real coding-agent-shaped prompts via
            # /v1/completions (raw prompt, avoids chat-template ambiguity
            # for the reference replay), independently verified. Reuses
            # this project's own realistic-prompt generator verbatim. --
            from benchmarks.mtp_sustained_realistic_workload_check import (
                REQUEST_CLASSES,
                _build_coding_prompt,
                _build_filler_ids,
            )
            import random

            rng = random.Random(20260719)
            filler_ids = _build_filler_ids(tok)
            correctness_cases = []
            for req_id, cls_name in enumerate(["short_chat", "medium_explain", "long_context"]):
                prompt_ids_built, max_tokens, _ = _build_coding_prompt(tok, filler_ids, rng, req_id, cls_name)
                prompt_text = tok.decode(prompt_ids_built)
                resp = await client.post(
                    "/v1/completions",
                    json={"prompt": prompt_text, "max_tokens": min(max_tokens, 64), "temperature": 0},
                    timeout=180.0,
                )
                case_result = {"cls": cls_name, "prompt_len_approx": len(prompt_ids_built), "status_code": resp.status_code}
                if resp.status_code == 200:
                    body = resp.json()
                    committed_ids = body["debug_committed_token_ids"]
                    val_slot = VALIDATION_SLOTS[req_id % len(VALIDATION_SLOTS)]
                    diag_slot = VALIDATION_DIAG_SLOTS[req_id % len(VALIDATION_DIAG_SLOTS)]
                    replay = await _independent_reference_replay(
                        engine, val_slot, diag_slot, tok, prompt_text, committed_ids
                    )
                    case_result["finish_reason"] = body["choices"][0]["finish_reason"]
                    case_result["completion_preview"] = body["choices"][0]["text"][:200]
                    case_result["independent_reference_replay"] = replay
                    case_result["ok"] = replay["ok"]
                else:
                    case_result["ok"] = False
                    case_result["body"] = resp.text[:500]
                correctness_cases.append(case_result)
            result["correctness_cases"] = correctness_cases
            correctness_ok = all(c["ok"] for c in correctness_cases)

            # -- 3. Concurrent-batching proof: fire CAPACITY requests at
            # once, then check the engine's own admission-batch-size
            # counters for a real multi-request admission (not inferred
            # from timing alone, though timing is recorded too). --
            stats_before = (await client.get("/debug/stats")).json()
            concurrent_prompts = [
                f"# request-marker: CONC{i:02d}\ndef f{i}(x):\n    return x * {i}\n\nWhat does f{i} return for x=2?"
                for i in range(CAPACITY)
            ]
            t0 = time.perf_counter()
            responses = await asyncio.gather(
                *[
                    client.post("/v1/completions", json={"prompt": p, "max_tokens": 24, "temperature": 0}, timeout=120.0)
                    for p in concurrent_prompts
                ]
            )
            concurrent_wall_s = time.perf_counter() - t0
            stats_after = (await client.get("/debug/stats")).json()
            new_admission_sizes = stats_after["admission_batch_sizes"][len(stats_before["admission_batch_sizes"]):]
            max_joint_admission = max(new_admission_sizes) if new_admission_sizes else 0
            result["concurrent_batching_proof"] = {
                "num_concurrent_requests_sent": CAPACITY,
                "all_status_200": all(r.status_code == 200 for r in responses),
                "wall_s": concurrent_wall_s,
                "admission_batch_sizes_this_test": new_admission_sizes,
                "max_joint_admission_batch_size": max_joint_admission,
                "genuinely_batched": max_joint_admission >= 2,
            }

            # -- 4. Defensive rejections: clean 4xx, not a crash, server
            # stays healthy afterward. --
            defensive_results = {}

            bad_temp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7},
                timeout=30.0,
            )
            defensive_results["non_default_temperature"] = {
                "status_code": bad_temp.status_code,
                "is_clean_4xx": 400 <= bad_temp.status_code < 500,
                "body": bad_temp.json() if bad_temp.status_code < 500 else bad_temp.text[:300],
            }

            oversized_len = engine.capacity_tokens_per_slot + 500
            oversized_prompt = "word " * oversized_len
            oversized = await client.post(
                "/v1/completions",
                json={"prompt": oversized_prompt, "max_tokens": 16, "temperature": 0},
                timeout=30.0,
            )
            defensive_results["oversized_prompt"] = {
                "status_code": oversized.status_code,
                "is_clean_4xx": 400 <= oversized.status_code < 500,
                "body": oversized.json() if oversized.status_code < 500 else oversized.text[:300],
            }

            stream_req = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                timeout=30.0,
            )
            defensive_results["stream_true"] = {
                "status_code": stream_req.status_code,
                "is_clean_4xx": 400 <= stream_req.status_code < 500,
            }

            n_req = await client.post(
                "/v1/completions",
                json={"prompt": "hi", "n": 2, "temperature": 0},
                timeout=30.0,
            )
            defensive_results["n_greater_than_1"] = {
                "status_code": n_req.status_code,
                "is_clean_4xx": 400 <= n_req.status_code < 500,
            }

            health_after = await client.get("/health", timeout=10.0)
            defensive_results["server_healthy_after_rejections"] = health_after.status_code == 200 and health_after.json()["status"] == "ok"

            result["defensive_rejections"] = defensive_results
            defensive_ok = (
                defensive_results["non_default_temperature"]["is_clean_4xx"]
                and defensive_results["oversized_prompt"]["is_clean_4xx"]
                and defensive_results["stream_true"]["is_clean_4xx"]
                and defensive_results["n_greater_than_1"]["is_clean_4xx"]
                and defensive_results["server_healthy_after_rejections"]
            )

            # -- 5. One real, working follow-up request AFTER the
            # defensive-rejection barrage, proving the server is not just
            # "still returning /health" but actually still able to serve. --
            post_defensive = await client.post(
                "/v1/completions",
                json={"prompt": "def add(a, b):\n    return a + b\n\nWhat does add(2, 3) return?", "max_tokens": 24, "temperature": 0},
                timeout=60.0,
            )
            result["post_defensive_real_request_ok"] = (
                post_defensive.status_code == 200 and len(post_defensive.json()["choices"][0]["text"]) > 0
            )

            result["final_stats"] = (await client.get("/debug/stats")).json()

            result["passed"] = bool(
                basic_ok
                and correctness_ok
                and result["concurrent_batching_proof"]["genuinely_batched"]
                and result["concurrent_batching_proof"]["all_status_200"]
                and defensive_ok
                and result["post_defensive_real_request_ok"]
                and result["final_stats"]["bootstrap_checks_failed"] == 0
            )
    finally:
        uv_server.should_exit = True
        await serve_task

    return result


def main() -> int:
    result = asyncio.run(_run())
    summary = {k: v for k, v in result.items() if k != "correctness_cases"}
    print(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"correctness_cases": result.get("correctness_cases", [])}, indent=2, default=str))
    print(f"\n=== {'PASS' if result['passed'] else 'FAIL'} ===")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
