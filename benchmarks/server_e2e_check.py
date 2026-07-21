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
# P4a: the e2e sets its OWN moderate blocks_per_slot (its prompts are a few
# thousand tokens, NOT 64K), so it does NOT pay for the full long-context KV
# pool the server's raised default (4200) allocates up front. NUM_SLOTS here
# is 24 (6*CAPACITY: the engine's 16 + this script's 8 validation slots), and
# (24+1)*4200 blocks of KV would be far more than this moderate-prompt test
# needs. 512 (the pre-P4a default) gives an 8192-token per-slot ceiling --
# ample for the turn-1/turn-2 hit prompts below. The persistent prefix cache
# stays ON (the default): proving the server SERVES a warm hit is the whole
# point of the P4a subtest added below.
os.environ["QSR_SERVER_BLOCKS_PER_SLOT"] = "512"
os.environ.setdefault("QSR_SERVER_ENABLE_PREFIX_CACHE", "1")
# P4b (notes/2026-07-20-p4b-session-affinity-plan.md §4.1): turn ON the opt-in
# session-affinity warm-slot retention for the P4b subtest below, with a generous
# 120s TTL so the retained warm slot survives the sequential turn-1/turn-2 HTTP
# calls deterministically (the default 30s is plenty too, but 120s removes any
# timing sensitivity). The no-session_id P4a subtest still runs unchanged with the
# flag on (warm counters stay 0 without a session_id -- proves no default-path leak).
os.environ["QSR_SERVER_ENABLE_SESSION_AFFINITY"] = "1"
os.environ["QSR_SERVER_SESSION_TTL_S"] = "120"

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

            # -- 6. P4a prefix-cache turn-1/turn-2 hit over real HTTP (the
            # P4 product-value test). Turn 1 POSTs a few-thousand-token prompt
            # (multi-block prefix + measurable prefill, but NOT 64K -- P3.4's
            # longctx check already proved the 64K case at the runtime level)
            # via /v1/completions, temperature 0 => COLD, populates the
            # persistent content-addressed cache. Reset nothing (the cache
            # persists across requests by design -- R10: a slot's published
            # blocks stay hit-able at ref_cnt==0 across reset_slot). Turn 2
            # POSTs the SAME prompt + a short appended turn => should HIT
            # turn-1's cached prefix at L ~= turn-1's prompt boundary. --
            stats_before_turn1 = (await client.get("/debug/stats")).json()
            p4a_block = (
                "def reconcile_prefix_hit(token_ids):\n"
                "    # Walk chained block hashes left-to-right; stop at the first miss.\n"
                "    matched_blocks = 0\n"
                "    for block_hash in compute_prompt_block_hashes(token_ids):\n"
                "        if block_pool.get_cached_block(block_hash) is None:\n"
                "            break\n"
                "        matched_blocks += 1\n"
                "    return matched_blocks * block_size\n\n"
            )
            turn1_text = (
                "# P4A_PREFIX_CACHE_TURN1_MARKER_20260720\n"
                "You are a careful code reviewer. Read the following reference implementation "
                "of a persistent prefix-cache reconciliation probe, repeated for context, then "
                "answer the question at the end precisely and briefly.\n\n"
                + (p4a_block * 32)
                + "\nQuestion: In one short sentence, what does reconcile_prefix_hit return?\n"
            )
            turn2_text = (
                turn1_text
                + "\nFollow-up: now state the time complexity of that hash walk in one short sentence.\n"
            )
            p4a_max_tokens = 32

            t1_t0 = time.perf_counter()
            turn1 = await client.post(
                "/v1/completions",
                json={"prompt": turn1_text, "max_tokens": p4a_max_tokens, "temperature": 0},
                timeout=180.0,
            )
            turn1_wall_s = time.perf_counter() - t1_t0
            t2_t0 = time.perf_counter()
            turn2 = await client.post(
                "/v1/completions",
                json={"prompt": turn2_text, "max_tokens": p4a_max_tokens, "temperature": 0},
                timeout=180.0,
            )
            turn2_wall_s = time.perf_counter() - t2_t0
            stats_after_turn2 = (await client.get("/debug/stats")).json()

            p4a: dict = {
                "turn1_status": turn1.status_code,
                "turn2_status": turn2.status_code,
                "turn1_wall_s": turn1_wall_s,
                "turn2_wall_s": turn2_wall_s,
            }
            if turn1.status_code == 200 and turn2.status_code == 200:
                t1_body = turn1.json()
                t2_body = turn2.json()
                prompt_ids_1 = t1_body["debug_prompt_token_ids"]
                turn2_ids = t2_body["debug_prompt_token_ids"]
                committed_1 = t1_body["debug_committed_token_ids"]
                committed_2 = t2_body["debug_committed_token_ids"]
                # Precondition (what makes the hit deterministic + a hash-
                # collision guard): turn-2 must reproduce turn-1's prompt tokens
                # through the CACHED block-aligned depth G1 = block_align_down(
                # len(prompt_1) - 1) -- the cold completion-checkpoint depth, i.e.
                # exactly the prefix the hit serves. The check is at G1, NOT the
                # full prompt length: the cache holds only FULL prefix blocks [0,
                # G1), and the partial tail block past G1 is not cached -- and the
                # tokenizer can re-segment that tail when the follow-up turn is
                # appended (so a full-length prefix match is neither required nor
                # always true). The server's own debug_prompt_token_ids are the
                # authoritative tokenization, so there is no decode/encode
                # ambiguity in this comparison. prefix_clean_full is reported too,
                # purely as an informational note on that junction re-segmentation.
                cached_prefix_len = ((len(prompt_ids_1) - 1) // 16) * 16
                prefix_clean = turn2_ids[:cached_prefix_len] == prompt_ids_1[:cached_prefix_len]
                prefix_clean_full = turn2_ids[: len(prompt_ids_1)] == prompt_ids_1
                # (a) turn-2 is a HIT: hits advanced during the turn-1/turn-2
                # window (turn-1 is cold by construction -- a unique marker no
                # earlier request populated -- so the new hit is turn-2), and a
                # hit_L sample keyed to turn-2's prompt length has L > 0 at ~= the
                # cold completion checkpoint G = block_align_down(len(prompt_1)-1).
                hits_delta = stats_after_turn2["prefix_cache_hits"] - stats_before_turn1["prefix_cache_hits"]
                misses_delta = stats_after_turn2["prefix_cache_misses"] - stats_before_turn1["prefix_cache_misses"]
                turn2_hit_samples = [
                    samp
                    for samp in stats_after_turn2["prefix_cache_hit_L_samples"]
                    if samp["prompt_tokens"] == len(turn2_ids)
                ]
                hit_L = turn2_hit_samples[-1]["hit_L"] if turn2_hit_samples else 0
                hit_L_ok = (
                    hit_L > 0
                    and hit_L % 16 == 0
                    and len(prompt_ids_1) - 16 <= hit_L <= len(prompt_ids_1)
                )
                # (b) INV1 end-to-end: turn-2's HIT-served committed tokens match
                # an independent COLD single-slot reference replay of the SAME
                # turn-2 prompt (the existing _independent_reference_replay
                # methodology; runner.prefill is a cold prefill that never
                # restores from the cache, so the reference is genuinely
                # independent of the hit path). Proves the warm hit served the
                # SAME tokens a cold prefill would have.
                inv1_replay = await _independent_reference_replay(
                    engine,
                    VALIDATION_SLOTS[3],
                    VALIDATION_DIAG_SLOTS[3],
                    tok,
                    turn2_text,
                    committed_2,
                )
                # (c) user-facing win: turn-2 (hit, prefills only the short
                # appended suffix) is materially faster than turn-1 (cold full
                # prefill). Non-streaming, so wall time is the TTFT proxy; the
                # always-on admission bootstrap check adds a cold reference
                # prefill to BOTH turns, so the differential is the production
                # prefill going cold->hit (still a clear, lenient drop).
                speedup = (turn1_wall_s / turn2_wall_s) if turn2_wall_s > 0 else float("inf")
                p4a.update(
                    {
                        "turn1_prompt_tokens": len(prompt_ids_1),
                        "turn2_prompt_tokens": len(turn2_ids),
                        "turn1_committed_tokens": len(committed_1),
                        "cached_prefix_len": cached_prefix_len,
                        "prefix_clean": prefix_clean,
                        "prefix_clean_full": prefix_clean_full,
                        "hits_delta": hits_delta,
                        "misses_delta": misses_delta,
                        "turn2_hit_L": hit_L,
                        "turn2_hit_L_over_prompt_len": (hit_L / len(prompt_ids_1)) if prompt_ids_1 else 0.0,
                        "hit_L_ok": hit_L_ok,
                        "turn2_is_hit": hits_delta >= 1 and hit_L > 0,
                        "turn1_is_miss": misses_delta >= 1,
                        "inv1_e2e_replay": inv1_replay,
                        "inv1_e2e_ok": inv1_replay["ok"],
                        "ttft_speedup": speedup,
                        "ttft_drop": turn2_wall_s < turn1_wall_s,
                        "turn1_completion_preview": t1_body["choices"][0]["text"][:120],
                        "turn2_completion_preview": t2_body["choices"][0]["text"][:120],
                    }
                )
                p4a["ok"] = bool(
                    prefix_clean
                    and p4a["turn2_is_hit"]
                    and p4a["turn1_is_miss"]
                    and hit_L_ok
                    and p4a["inv1_e2e_ok"]
                    and p4a["ttft_drop"]
                )
            else:
                p4a["ok"] = False
                p4a["turn1_body"] = turn1.text[:300]
                p4a["turn2_body"] = turn2.text[:300]
            result["prefix_cache_turn_hit"] = p4a

            # -- 7. P4b session-affinity warm-slot continuation over real HTTP
            # (the P4b zero-restore test). Turn 1 POSTs a few-thousand-token
            # coding prompt + session_id, temperature 0 => COLD (unique marker,
            # no earlier request populated it), populates the cache, and on finish
            # the slot is RETAINED warm (not reset). Turn 2 POSTs the SAME
            # session_id with a prompt that EXTENDS turn-1's P1+C1 exactly =>
            # continues the warm slot in place with ZERO restore
            # (mtp_prefill_warm_continue). Zero-restore is PROVEN by
            # session_warm_continuations advancing by exactly 1 while
            # prefix_cache_hits does NOT advance for turn-2 (the warm path bypasses
            # reconcile_prefix_hit entirely) -- distinct from the content-hash hit. --
            stats_before_p4b = (await client.get("/debug/stats")).json()
            # Feature must not leak into the default path: the no-session_id P4a
            # subtest above (flag ON, no session_id) left warm continuations at 0.
            p4b_no_leak = stats_before_p4b["session_warm_continuations"] == 0

            p4b_turn1_text = (
                "# P4B_SESSION_AFFINITY_TURN1_MARKER_20260720\n"
                "You are a careful code reviewer. Read the following reference implementation "
                "of a persistent prefix-cache reconciliation probe, repeated for context, then "
                "answer the question at the end precisely and briefly.\n\n"
                + (p4a_block * 32)
                + "\nQuestion: In one short sentence, what does reconcile_prefix_hit return?\n"
            )
            p4b_max_tokens = 32
            p4b_sess_id = "P4B_SESS_1"

            p4b: dict = {"no_leak_default_path": p4b_no_leak}
            p4b_t1_t0 = time.perf_counter()
            p4b_turn1 = await client.post(
                "/v1/completions",
                json={"prompt": p4b_turn1_text, "max_tokens": p4b_max_tokens, "temperature": 0, "session_id": p4b_sess_id},
                timeout=180.0,
            )
            p4b_turn1_wall_s = time.perf_counter() - p4b_t1_t0
            p4b["turn1_status"] = p4b_turn1.status_code
            p4b["turn1_wall_s"] = p4b_turn1_wall_s

            if p4b_turn1.status_code == 200:
                p4b_t1_body = p4b_turn1.json()
                p4b_p1 = p4b_t1_body["debug_prompt_token_ids"]
                p4b_c1 = p4b_t1_body["debug_committed_token_ids"]  # server's max_tokens-truncated view
                # The warm-continue boundary is the runtime's AUTHORITATIVE committed
                # state (slot_kv_len / slot_committed_tokens), which may extend a few
                # tokens past the server's max_tokens-truncated response (MTP final-
                # round overshoot). Read it directly (the e2e already drives
                # engine.runner for the independent reference replay) so turn-2
                # reproduces the EXACT boundary and the warm path fires
                # deterministically. A real HTTP client sees only p4b_c1, so for real
                # traffic the warm path is opportunistic (plan §5 risk #1); the e2e
                # proves the zero-restore MECHANISM by constructing the exact turn-2.
                await asyncio.sleep(0.1)  # let the engine loop finish the retention
                p4b_retained = engine.retained.get(p4b_sess_id)
                assert p4b_retained is not None, "turn-1 slot was not retained warm"
                p4b_retained_slot = p4b_retained["slot"]
                p4b_l1 = engine.runner.slot_kv_len[p4b_retained_slot]
                p4b_target = list(engine.runner.slot_committed_tokens[p4b_retained_slot][:p4b_l1])
                # C1_full = committed tokens AFTER the prompt (incl. any overshoot).
                p4b_c1_full = p4b_target[len(p4b_p1):p4b_l1]
                p4b_c1_text = tok.decode(p4b_c1_full)
                p4b["server_c1_tokens"] = len(p4b_c1)
                p4b["runtime_c1_full_tokens"] = len(p4b_c1_full)
                p4b["mtp_overshoot_tokens"] = p4b_l1 - (len(p4b_p1) + len(p4b_c1))
                # Warm-continue exact-prefix precondition (plan §5 risk #1): turn-2's
                # prompt tokens must equal P1+C1 through L1. The tokenizer re-segments
                # at the turn1|C1 and C1|follow-up junctions, so try a small set of
                # follow-up boundary styles and pick the first whose re-tokenization
                # reproduces P1+C1 EXACTLY -- verified locally with the SAME tokenizer
                # the server uses, then re-asserted against the server's authoritative
                # debug_prompt_token_ids after the POST. (turn1_text ends with "\n"; a
                # C1 starting with "\n" would make "\n\n" which re-segments P1 at idx
                # 2533 -- verified -- so such a C1 cannot satisfy the precondition by
                # simple concatenation and is reported honestly below.)
                p4b_followup_variants = [
                    "\nFollow-up: now state the time complexity of that hash walk in one short sentence.\n",
                    " Follow-up: now state the time complexity of that hash walk in one short sentence.\n",
                    "\n\nFollow-up: now state the time complexity of that hash walk in one short sentence.\n",
                ]
                p4b_turn2_text = None
                p4b_chosen_followup = None
                for fu in p4b_followup_variants:
                    candidate = p4b_turn1_text + p4b_c1_text + fu
                    if tok.encode(candidate, add_special_tokens=False)[:p4b_l1] == p4b_target:
                        p4b_turn2_text = candidate
                        p4b_chosen_followup = fu
                        break
                p4b["turn1_prompt_tokens"] = len(p4b_p1)
                p4b["turn1_committed_tokens"] = len(p4b_c1)
                p4b["c1_text_preview"] = p4b_c1_text[:60]
                p4b["c1_starts_with_newline"] = p4b_c1_text.startswith("\n")
                p4b["chosen_followup"] = repr(p4b_chosen_followup)
                p4b["local_precondition_found"] = p4b_turn2_text is not None

                if p4b_turn2_text is not None:
                    p4b_t2_t0 = time.perf_counter()
                    p4b_turn2 = await client.post(
                        "/v1/completions",
                        json={"prompt": p4b_turn2_text, "max_tokens": p4b_max_tokens, "temperature": 0, "session_id": p4b_sess_id},
                        timeout=180.0,
                    )
                    p4b_turn2_wall_s = time.perf_counter() - p4b_t2_t0
                    p4b["turn2_status"] = p4b_turn2.status_code
                    p4b["turn2_wall_s"] = p4b_turn2_wall_s
                    stats_after_p4b = (await client.get("/debug/stats")).json()

                    if p4b_turn2.status_code == 200:
                        p4b_t2_body = p4b_turn2.json()
                        p4b_turn2_ids = p4b_t2_body["debug_prompt_token_ids"]
                        p4b_committed_2 = p4b_t2_body["debug_committed_token_ids"]
                        # Authoritative exact-prefix precondition (server's own
                        # tokenization): turn-2 reproduces P1+C1 through L1.
                        p4b_precond = p4b_turn2_ids[:p4b_l1] == p4b_target
                        # Zero-restore proof: session_warm_continuations advanced by
                        # exactly 1 AND prefix_cache_hits did NOT advance for turn-2.
                        warm_delta = (
                            stats_after_p4b["session_warm_continuations"]
                            - stats_before_p4b["session_warm_continuations"]
                        )
                        hits_delta_p4b = (
                            stats_after_p4b["prefix_cache_hits"]
                            - stats_before_p4b["prefix_cache_hits"]
                        )
                        retentions_delta = (
                            stats_after_p4b["session_retentions"]
                            - stats_before_p4b["session_retentions"]
                        )
                        fallbacks_delta = (
                            stats_after_p4b["session_warm_fallbacks"]
                            - stats_before_p4b["session_warm_fallbacks"]
                        )
                        # INV1 end-to-end: turn-2's warm-served committed tokens match
                        # an independent COLD single-slot reference replay of the SAME
                        # turn-2 prompt (runner.prefill never restores from the cache,
                        # so the reference is genuinely independent of the warm path).
                        p4b_inv1 = await _independent_reference_replay(
                            engine, VALIDATION_SLOTS[3], VALIDATION_DIAG_SLOTS[3], tok, p4b_turn2_text, p4b_committed_2
                        )
                        speedup_p4b = (p4b_turn1_wall_s / p4b_turn2_wall_s) if p4b_turn2_wall_s > 0 else float("inf")
                        p4b.update(
                            {
                                "turn2_prompt_tokens": len(p4b_turn2_ids),
                                "prior_len_L1": p4b_l1,
                                "exact_prefix_precondition": p4b_precond,
                                "warm_continuations_delta": warm_delta,
                                "prefix_cache_hits_delta": hits_delta_p4b,
                                "session_retentions_delta": retentions_delta,
                                "session_warm_fallbacks_delta": fallbacks_delta,
                                "warm_fired": warm_delta == 1 and hits_delta_p4b == 0,
                                "inv1_e2e_replay": p4b_inv1,
                                "inv1_e2e_ok": p4b_inv1["ok"],
                                "ttft_speedup": speedup_p4b,
                                "ttft_drop": p4b_turn2_wall_s < p4b_turn1_wall_s,
                                "turn1_completion_preview": p4b_t1_body["choices"][0]["text"][:120],
                                "turn2_completion_preview": p4b_t2_body["choices"][0]["text"][:120],
                                "warm_continuation_samples": stats_after_p4b["session_warm_continuation_samples"][-2:],
                            }
                        )
                        p4b["ok"] = bool(
                            p4b_no_leak
                            and p4b_precond
                            and warm_delta == 1
                            and hits_delta_p4b == 0
                            and p4b_inv1["ok"]
                            and p4b["ttft_drop"]
                        )
                    else:
                        p4b["ok"] = False
                        p4b["turn2_body"] = p4b_turn2.text[:300]
                else:
                    p4b["ok"] = False
                    p4b["note"] = "no follow-up construction satisfied the exact-prefix precondition locally"
            else:
                p4b["ok"] = False
                p4b["turn1_body"] = p4b_turn1.text[:300]
            result["session_affinity_warm_continue"] = p4b

            result["final_stats"] = (await client.get("/debug/stats")).json()

            result["passed"] = bool(
                basic_ok
                and correctness_ok
                and result["concurrent_batching_proof"]["genuinely_batched"]
                and result["concurrent_batching_proof"]["all_status_200"]
                and defensive_ok
                and result["post_defensive_real_request_ok"]
                and result["prefix_cache_turn_hit"]["ok"]
                and result["session_affinity_warm_continue"]["ok"]
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
