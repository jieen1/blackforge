"""Correctness verification for Phase 3's CUDA-graph wiring into the REAL
``mtp_verify_and_commit_batch`` entry point (``runtime/direct_model_runner.py``),
per ``notes/2026-07-17-post-ragged-round-next-steps.md``'s Phase 3 (both the
initial round -- verify forward + K-1 draft steps -- and the fast-iteration
follow-up round -- recompute-forward graph reuse, full precapture, and
step-0 draft-resync generalization).

Unlike ``cudagraph_eager_parity_check.py``/``cudagraph_mtp_regression.py``
(which drive ``CapturedBatchDecodeGraph`` directly via ``.capture()``/
``.replay()`` in isolation), this test constructs a runner with
``enable_cudagraph=True`` and drives the REAL production call path
(``mtp_prefill_batch`` -> repeated ``mtp_verify_and_commit_batch``) --
exercising every graph-or-eager dispatch decision inside that method and
``_mtp_sync_and_propose_batch``, not just the underlying primitives.

Methodology: the project's established "independent-reference-replay"
pattern (``mtp_batch_verify_check.py``/``mtp_ragged_recompute_verify_check.py``)
-- feed each round's REAL committed tokens into an independent single-slot
``_forward`` reference call every round, comparing its own greedy argmax
against the batched path's ``next_anchor`` within a near-tie tolerance.
This decouples one round's possible near-tie tie-break flip from
corrupting the next round's check, and needs only ONE model load (one
runner; ``ref_slots`` on the always-eager single-slot path serve as the
oracle for ``mtp_slots`` on the graph-enabled batched path).

Coverage, and which code path each round is designed to exercise:

0-1. Organic rounds -- whatever the model actually does (usually full
     accept given this project's ~72% measured per-position acceptance
     rate). Exercises the common-case verify-graph + step-0-graph +
     draft-step-graph path.
2.   Mixed forced-reject (2 of 4 slots) -- full-accept group's step 0/
     draft-step graphs coexist with the recompute group's ragged-fallback
     eager path in the SAME round.
3.   Fully ragged forced-reject (every slot rejects at a DIFFERENT
     position) -- the recompute forward's eager fallback, exercised at
     its most demanding (no two slots share a committed_len).
4.   Organic again, immediately after a ragged-recompute round -- confirms
     the verify graph's static buffers are unaffected by an intervening
     eager call.
5.   Single-slot forced reject.
6.   UNIFORM forced-reject, committed_len=2 for all 4 slots -- the
     recompute-forward graph-REUSE special case (shares the verify-graph
     cache at (batch_size, qo_len=2) instead of falling back to eager).
7.   UNIFORM forced-reject, committed_len=1 for all 4 slots -- same reuse
     path at qo_len=1, which caught a REAL bug during development: the
     recompute path always builds list-of-lists token_ids, but
     CapturedBatchDecodeGraph's qo_len==1 branch expects a FLAT list --
     fixed by flattening before replay (see direct_model_runner.py's
     mtp_verify_and_commit_batch, "tokens_for_graph" comment).
Shrink.  Drop from 4 active slots to 2 (simulating some requests finishing
     early, the real W1-S tail pattern) -- confirms a SECOND (batch_size=2)
     verify graph and draft-step graph get captured/used correctly.

Also confirms, via ``replay_count`` (not just cache-key presence, which
precapture alone would already satisfy regardless of whether anything
ever replays that shape):
- the verify graph is actually replayed at both batch_size=4 and 2;
- the recompute-forward graph-reuse path is actually replayed at both
  qo_len=2 and qo_len=1 (rounds 6 and 7 above);
- the draft-model step-0 graph is actually replayed (the generalization
  that, during development, needed an explicit ``_MAX_DECODE_QO_LEN``
  guard -- mtp_prefill_batch's own step-0 call uses num_new_tokens=
  prompt_len, e.g. 4096, which is also "uniform" and would otherwise have
  been forced through the decode-kernel dispatch it was never validated
  for; this test's own prefill call is the direct regression check for
  that guard actually firing);
- the draft-model qo_len=1 continuation-step graph is actually replayed.

Usage:
    python -m benchmarks.mtp_verify_cudagraph_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0

FOUR_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]


def _reset_if_needed(runner, slots):
    for slot in slots:
        if runner.slot_kv_len[slot] != 0 or runner.slot_draft_sync_len[slot] != 0:
            runner.reset_slot(slot)


def _decoy_at(drafts: list[int], position: int) -> list[int]:
    """Corrupt drafts[position] with a value guaranteed to differ from the
    real draft (and from itself across positions) so
    determine_accept_reject_batch rejects at EXACTLY this position,
    forcing committed_len = position + 1."""
    out = list(drafts)
    decoy = 100 + position
    while decoy in drafts:
        decoy += 1
    out[position] = decoy
    return out


def _ref_check(runner, ref_slot, real_new_tokens, mtp_next_anchor) -> dict:
    ref_logits = runner._forward(
        ref_slot,
        real_new_tokens,
        start_pos=runner.slot_kv_len[ref_slot],
        is_decode=(len(real_new_tokens) == 1),
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    ref_top1_logit = float(ref_last.max().item())
    ref_logit_for_mtp_choice = float(ref_last[mtp_next_anchor].item())
    near_tie_margin = ref_top1_logit - ref_logit_for_mtp_choice
    exact_match = ref_predicted_next == mtp_next_anchor
    return {
        "ref_predicted_next_matches_mtp_next_anchor": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok_within_near_tie_tolerance": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _run_once() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    tok = AutoTokenizer.from_pretrained(MODEL)
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # mtp_slots (real, graph-enabled batched path) + ref_slots (independent
    # eager single-slot reference oracle) + CapturedBatchDecodeGraph's/
    # CapturedMTPDraftStepGraph's own internally reserved warmup slots
    # (batch_size up to 4 -> 4 slots) = 3*4 = 12. block_size/blocks_per_slot
    # kept small (matches cudagraph_mtp_regression.py's own convention) --
    # this test only needs correctness, not production capacity.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=12,
        block_size=16,
        blocks_per_slot=128,
        enable_cudagraph=True,
    )

    mtp_slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    _reset_if_needed(runner, mtp_slots + ref_slots)

    # mtp_prefill_batch's own step-0 resync call uses num_new_tokens=
    # prompt_len (a large, uniform value) -- the direct regression check
    # for the _MAX_DECODE_QO_LEN guard that keeps this from being
    # mistakenly routed through the draft-step decode-kernel graph.
    prefill_result = runner.mtp_prefill_batch(mtp_slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in mtp_slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in mtp_slots}
    for i, s in enumerate(mtp_slots):
        ref_first = runner.prefill(ref_slots[i], prompt_ids_per_slot[i])
        if ref_first != anchors[s]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot {s}"}

    per_slot_rounds: dict[int, list[dict]] = {s: [] for s in mtp_slots}

    # Force patterns per round: None = organic (whatever the model does),
    # dict = {slot: reject_position} for slots to force-reject this round.
    force_patterns = [
        None,  # round 0: organic
        None,  # round 1: organic
        {mtp_slots[0]: 0, mtp_slots[2]: 1},  # round 2: mixed
        {mtp_slots[0]: 0, mtp_slots[1]: 1, mtp_slots[2]: 2, mtp_slots[3]: 0},  # round 3: fully ragged
        None,  # round 4: organic again, right after a ragged-recompute round
        {mtp_slots[1]: 2},  # round 5: single-slot forced reject
        # round 6: uniform reject, committed_len=2 for all 4 -- recompute
        # forward reuses the verify-graph cache instead of eager.
        {mtp_slots[0]: 1, mtp_slots[1]: 1, mtp_slots[2]: 1, mtp_slots[3]: 1},
        # round 7: uniform reject, committed_len=1 for all 4 -- same reuse
        # path at qo_len=1 (caught a real flatten bug during development).
        {mtp_slots[0]: 0, mtp_slots[1]: 0, mtp_slots[2]: 0, mtp_slots[3]: 0},
    ]

    for r, pattern in enumerate(force_patterns):
        submitted = {}
        for s in mtp_slots:
            if pattern and s in pattern:
                submitted[s] = _decoy_at(drafts[s], pattern[s])
            else:
                submitted[s] = drafts[s]

        decisions = runner.mtp_verify_and_commit_batch(mtp_slots, anchors, submitted)

        for i, s in enumerate(mtp_slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": r,
                    "pattern": pattern.get(s) if pattern else None,
                    "num_accepted": decision["num_accepted"],
                    "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    verify_graph_b4 = runner._verify_graphs.get((4, K + 1))
    # Round 6 forces ALL 4 slots into the recompute group with a UNIFORM
    # committed_len=2 -- this deterministically drives
    # _mtp_sync_and_propose_batch's step-0 resync for that group through
    # _get_draft_step_graph(4, 2) (the qo_len>1 generalization), unlike the
    # full-accept group's own qo_len=k+1 step-0 shape, which depends on ALL
    # 4 slots organically fully-accepting in the SAME round simultaneously
    # (~2% per round given this project's own measured ~72% per-position
    # acceptance rate) -- too unreliable to gate a deterministic test on.
    draft_step0_qo2_graph = runner._draft_step_graphs.get((4, 2))
    # (4, 1) is hit by BOTH the K-1 continuation loop (every round, every
    # group, regardless of step-0's own qo_len) and round 7's step-0
    # (committed_len=1 uniform) -- guaranteed to be exercised every round.
    draft_cont_graph_b4 = runner._draft_step_graphs.get((4, 1))
    recompute_reuse_qo2 = runner._verify_graphs.get((4, 2))
    recompute_reuse_qo1 = runner._verify_graphs.get((4, 1))

    # --- Shrinking-batch check: drop to 2 active slots (simulating slots
    # 2/3 finishing early), continue for a few more rounds -- exercises a
    # SECOND (batch_size=2) graph via _get_verify_graph/_get_draft_step_graph. ---
    shrunk_slots = mtp_slots[:2]
    shrunk_ref = ref_slots[:2]
    for r in range(3):
        decisions = runner.mtp_verify_and_commit_batch(
            shrunk_slots, {s: anchors[s] for s in shrunk_slots}, {s: drafts[s] for s in shrunk_slots}
        )
        for i, s in enumerate(shrunk_slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, shrunk_ref[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": f"shrunk-{r}",
                    "num_accepted": decision["num_accepted"],
                    "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    verify_graph_b2 = runner._verify_graphs.get((2, K + 1))

    per_slot_ok = {}
    for s in mtp_slots:
        rounds = per_slot_rounds[s]
        per_slot_ok[s] = all(rd["kv_len_matches_draft_sync_len"] for rd in rounds) and all(
            rd["content_ok_within_near_tie_tolerance"] for rd in rounds
        )

    # Coverage gates: not just "was this shape captured" (precapture alone
    # satisfies that regardless of use) but "was it actually REPLAYED"
    # (replay_count > 0), i.e. the real round loop actually took this code
    # path at least once, not silently falling back to eager throughout.
    coverage = {
        "verify_graph_batch4_replayed": bool(verify_graph_b4 and verify_graph_b4.replay_count > 0),
        "verify_graph_batch2_replayed": bool(verify_graph_b2 and verify_graph_b2.replay_count > 0),
        "recompute_reuse_qo2_replayed": bool(recompute_reuse_qo2 and recompute_reuse_qo2.replay_count > 0),
        "recompute_reuse_qo1_replayed": bool(recompute_reuse_qo1 and recompute_reuse_qo1.replay_count > 0),
        "draft_step0_qo2_graph_replayed": bool(draft_step0_qo2_graph and draft_step0_qo2_graph.replay_count > 0),
        "draft_continuation_graph_replayed": bool(draft_cont_graph_b4 and draft_cont_graph_b4.replay_count > 0),
    }

    passed = bool(all(per_slot_ok.values()) and all(coverage.values()))
    return {
        "passed": passed,
        "per_slot_ok": per_slot_ok,
        "coverage": coverage,
        "verify_graph_cache_keys": sorted(str(k) for k in runner._verify_graphs.keys()),
        "draft_step_graph_cache_keys": sorted(str(k) for k in runner._draft_step_graphs.keys()),
        "per_slot_rounds": per_slot_rounds,
    }


def main() -> int:
    import json

    result = _run_once()
    summary = {
        "passed": result["passed"],
        "per_slot_ok": result.get("per_slot_ok"),
        "coverage": result.get("coverage"),
        "verify_graph_cache_keys": result.get("verify_graph_cache_keys"),
        "draft_step_graph_cache_keys": result.get("draft_step_graph_cache_keys"),
    }
    print(json.dumps(summary, indent=2, default=str))
    if not result["passed"]:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
