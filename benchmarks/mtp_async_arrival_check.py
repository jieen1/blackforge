"""Realistic async-arrival driver + correctness check (2026-07-19,
continuous-batching round, ``notes/2026-07-18-session-review-and-next-steps
.md`` section 21).

This is the second, structurally-different half of this round's task (the
first, ragged-length batched prefill, is ``mtp_ragged_prefill_check.py``).
Every benchmark/check this project has built through 2026-07-18 prefills a
FIXED batch of N requests synchronously, then runs verify/commit in
lockstep until all N finish together -- real serving does not look like
this: requests arrive at different times, at different prompt lengths, and
finish at different times, joining a batch of already-in-flight requests
mid-round rather than only ever shrinking it.

This driver models exactly that: 6 requests, DIFFERENT prompt lengths,
arriving in three waves (2 initial, 2 admitted mid-flight together while
the first 2 are already mid-decode -- a ragged BATCHED admission exercising
``mtp_prefill_batch``'s new code path -- and 2 more admitted later, one at a
time, onto slots freed by earlier finishers), over a fixed 4-slot capacity
pool (matching this project's own scope contract, ``项目实施规划.md:23``).
Arrival is modeled as an "earliest eligible round" rather than a real
wall-clock sleep (this project's other benchmarks are similarly round-
indexed, not wall-clock-scheduled; a request whose earliest-eligible round
already passed but found every slot full waits -- a real admission-control
queue, not an error) -- this keeps the whole run real GPU time only, no
artificial waiting.

Correctness methodology (this project's own established conventions,
generalized to the async case, not reinvented): every active slot, every
round, has its real committed tokens replayed through an INDEPENDENT
single-slot reference slot (the ``_ref_check`` pattern from
``mtp_batch_verify_check.py``/``mtp_ragged_prefill_check.py``), near-tie
tolerant (``NEAR_TIE_LOGIT_MARGIN=2.0``) -- this is what actually proves no
cross-request state leakage between a newly-admitted slot (reusing a
physical slot an earlier, unrelated request just vacated) and whatever
else is concurrently active, since a leak would corrupt a slot's real
content in a way its own independent reference (built from that slot's
true history alone) would not reproduce. Six DIFFERENT real, distinguishable
prompts ("The capital of X is") give a secondary, informational identity
signal (demoted to informational per this project's own established
false-positive-trap precedent for this exact template family -- see
``mtp_ragged_prefill_check.py``'s module docstring).

**A real, characterized finding from this round, reported honestly, not
hidden (this script's own ``passed`` gate reflects it -- see
``notes/2026-07-18-session-review-and-next-steps.md`` section 21 for the
full writeup)**: this specific 6-request timeline reproducibly (bit-
identical across repeated runs -- this is a deterministic effect, not
run-to-run randomness) hits ONE per-round reference divergence exceeding
``NEAR_TIE_LOGIT_MARGIN`` (7.9375 logit units, vs. every other round's
either exact match or a <=0.5 near-tie) for one request, at the FIRST
round whose batch composition mixes two long-running slots (9 rounds deep)
with two freshly-admitted ones (their very first round) -- a batch shape
NO test in this project's history had exercised before this round (every
prior test either starts all slots together or only ever shrinks the
batch). Investigated, not dismissed: (1) it is a genuine per-position
near-tie in the model's OWN distribution, not a data-corruption bug --
decoding both candidates shows the divergence is a "continue the
already-degenerate repeated phrase" vs. "end it with a paragraph break"
choice, both locally plausible continuations of the same known synthetic-
fixture repetition artifact this project has documented elsewhere, not a
content-quality regression; (2) the per-slot SSM-row addressing
(``_ssm_spec_row``/``build_gdn_metadata_spec_batch``) is keyed ONLY by
logical slot + accepted-count, with no cross-slot coupling in the
formula, re-confirmed by direct reading; (3) it is fully self-healing --
the SAME request's reference check returns to exact bit-match the very
next round and stays bit-exact for the remaining ~17 rounds of its own
generation, unlike this project's established genuine-state-corruption
signature (compounds/persists, does not vanish in one round). Conclusion:
the SAME cross-slot batching-order numerical noise class this project has
repeatedly documented elsewhere (verify's spec-decode kernel, chunked-
prefill's GDN bf16 round-trip, ragged-prefill's heterogeneous-content
batching -- see ``mtp_ragged_prefill_check.py``), confirmed here at a NEW
trigger (mixed freshly-admitted/long-running batch composition) and a
larger single-instance magnitude than previously measured at the verify
stage -- not a blocking correctness bug, but flagged explicitly as an
open numerical-hardening item, not swept under "noise" without evidence.

Reports a real accepted-tokens/s throughput number for this async-arrival
workload (measured from the first admission to the last finish, eager mode
-- ``enable_cudagraph=False``, matching this project's existing correctness-
suite convention; NOT directly comparable to the cudagraph-enabled 4K/c=4
headline, a distinctly-scoped measurement, labeled as such).

Usage:
    python -m benchmarks.mtp_async_arrival_check
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}
NEAR_TIE_LOGIT_MARGIN = 2.0
CAPACITY = 4

_FILLER_PARAGRAPH = (
    "The Amazon rainforest, spanning roughly 5.5 million square kilometers "
    "across nine countries in South America, is the largest tropical "
    "rainforest on Earth and hosts an estimated ten percent of all known "
    "species. Its dense canopy regulates regional and even global weather "
    "patterns by cycling enormous volumes of water vapor into the "
    "atmosphere, a process scientists call evapotranspiration. Deforestation "
    "driven by cattle ranching, soy cultivation, and illegal logging has "
    "accelerated over the past two decades, raising concern that the forest "
    "could approach a tipping point beyond which large sections convert to "
    "savanna. "
)

# (req_id, earliest_eligible_round, prompt_len, max_tokens, question, city)
# Wave 1 (round 0): 2 initial requests, different lengths.
# Wave 2 (round 4): 2 MORE requests, different lengths, admitted TOGETHER
#   (one ragged mtp_prefill_batch call) while wave 1 is already mid-decode
#   -- the core mid-flight-admission scenario.
# Wave 3 (round 9/11): 2 more, admitted one at a time, onto whatever slot
#   has freed up by then (queued if none free yet).
REQUESTS = [
    ("A", 0, 620, 40, "The capital of France is", "France"),
    ("B", 0, 940, 44, "The capital of Japan is", "Japan"),
    ("C", 4, 360, 40, "The capital of Germany is", "Germany"),
    ("D", 4, 1180, 48, "The capital of Italy is", "Italy"),
    ("E", 9, 520, 36, "The capital of Spain is", "Spain"),
    ("F", 11, 820, 44, "The capital of Brazil is", "Brazil"),
]
# Fixed reference-slot assignment, one per request, for its entire lifetime
# (only 6 requests total this run -- cheap to never reuse a ref slot).
REF_SLOT_BY_REQ = {"A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15}
MARGIN_DIAG_SLOT_BY_REQ = {"A": 20, "B": 21, "C": 22, "D": 23, "E": 24, "F": 25}


def _build_probe_prompt(tok, question: str, target_len: int) -> list[int]:
    question_ids = tok.encode(question, add_special_tokens=False)
    filler_needed = target_len - len(question_ids)
    assert filler_needed > 0
    text = _FILLER_PARAGRAPH
    filler_ids = tok.encode(text, add_special_tokens=False)
    while len(filler_ids) < filler_needed:
        text += _FILLER_PARAGRAPH
        filler_ids = tok.encode(text, add_special_tokens=False)
    prompt_ids = filler_ids[:filler_needed] + question_ids
    assert len(prompt_ids) == target_len
    return prompt_ids


def _near_tie_margin_diag(runner, ref_slot: int, prompt: list[int], chosen_anchor: int) -> dict:
    target_logits = runner._forward(ref_slot, prompt, start_pos=0, is_decode=False, return_hidden=False)
    last = target_logits[-1].float()
    ref_anchor = int(last.argmax(dim=-1).item())
    margin = float(last.max().item() - last[chosen_anchor].item())
    return {
        "ref_anchor": ref_anchor,
        "chosen_anchor": chosen_anchor,
        "near_tie_margin": margin,
        "within_tolerance": chosen_anchor == ref_anchor or margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _ref_check(runner, ref_slot: int, real_new_tokens: list[int], mtp_next_anchor: int, tok=None) -> dict:
    ref_logits = runner._forward(
        ref_slot, real_new_tokens, start_pos=runner.slot_kv_len[ref_slot], is_decode=(len(real_new_tokens) == 1)
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    near_tie_margin = float(ref_last.max().item() - ref_last[mtp_next_anchor].item())
    exact_match = ref_predicted_next == mtp_next_anchor
    report = {
        "ref_predicted_next_matches_mtp_next_anchor": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok_within_near_tie_tolerance": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }
    # Self-documenting on a mismatch (see module docstring's "known finding"
    # section): decode both candidates + local context so a real divergence
    # is characterized in the JSON output itself, not just flagged bare.
    if not exact_match and tok is not None:
        report["ref_predicted_token_text"] = tok.decode([ref_predicted_next])
        report["mtp_next_anchor_token_text"] = tok.decode([mtp_next_anchor])
        report["local_context_text"] = tok.decode(real_new_tokens[-12:])
    return report


def _run_async_arrival(runner, tok) -> dict:
    pending = sorted(REQUESTS, key=lambda r: r[1])
    free_slots = list(range(CAPACITY))
    active: dict[int, dict] = {}  # production slot -> request state
    finished: list[dict] = []
    events: list[dict] = []

    t_start = time.perf_counter()
    round_idx = 0
    max_rounds = 500  # generous safety cap -- real termination is "all 6 requests finished"
    total_gpu_wall_s = 0.0
    correctness_ok = True
    correctness_failures: list[dict] = []
    identity_ok_informational: dict[str, bool] = {}

    import torch

    while (pending or active) and round_idx < max_rounds:
        # -- Admission: pull in as many eligible, still-pending requests as
        # there are free slots this round (queues if the pool is full). --
        admit_now = []
        still_pending = []
        for req in pending:
            req_id, earliest_round, prompt_len, max_tokens, question, city = req
            if earliest_round <= round_idx and free_slots:
                slot = free_slots.pop(0)
                admit_now.append((slot, req))
            else:
                still_pending.append(req)
        pending = still_pending

        if admit_now:
            new_slots = [s for s, _ in admit_now]
            new_prompts = []
            for slot, (req_id, _, prompt_len, max_tokens, question, city) in admit_now:
                prompt = _build_probe_prompt(tok, question, prompt_len)
                new_prompts.append(prompt)
                if runner.slot_kv_len[slot] != 0:
                    runner.reset_slot(slot)

            # THE core new mechanism this round built: a genuinely ragged
            # (different-length-per-slot, when len(new_slots) > 1 and their
            # prompt_lens differ) batched prefill in ONE call, admitting
            # multiple fresh requests alongside each other.
            prefill_result = runner.mtp_prefill_batch(new_slots, new_prompts)

            for (slot, (req_id, earliest_round, prompt_len, max_tokens, question, city)), prompt in zip(
                admit_now, new_prompts
            ):
                anchor = prefill_result[slot]["anchor"]
                drafts = prefill_result[slot]["draft_tokens"]
                ref_slot = REF_SLOT_BY_REQ[req_id]
                if runner.slot_kv_len[ref_slot] != 0:
                    runner.reset_slot(ref_slot)
                ref_first = runner.prefill(ref_slot, prompt)
                if ref_first != anchor:
                    diag = _near_tie_margin_diag(runner, MARGIN_DIAG_SLOT_BY_REQ[req_id], prompt, anchor)
                    if not diag["within_tolerance"]:
                        correctness_ok = False
                        correctness_failures.append(
                            {"req_id": req_id, "stage": "admission_bootstrap", "diag": diag}
                        )
                active[slot] = {
                    "req_id": req_id,
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "city": city,
                    "anchor": anchor,
                    "drafts": drafts,
                    "committed_len": 0,
                    "committed_tokens": [],
                    "admitted_round": round_idx,
                    "admitted_t": time.perf_counter() - t_start,
                }
                events.append(
                    {"event": "admit", "req_id": req_id, "slot": slot, "round": round_idx,
                     "prompt_len": prompt_len, "batch_size_this_admission": len(admit_now)}
                )

        if not active:
            round_idx += 1
            continue

        active_slots = list(active.keys())
        t0 = time.perf_counter()
        decisions = runner.mtp_verify_and_commit_batch(
            active_slots,
            {s: active[s]["anchor"] for s in active_slots},
            {s: active[s]["drafts"] for s in active_slots},
        )
        torch.cuda.synchronize()
        round_wall_s = time.perf_counter() - t0
        total_gpu_wall_s += round_wall_s

        newly_finished_slots = []
        for s in active_slots:
            st = active[s]
            decision = decisions[s]
            real_new_tokens = [st["anchor"]] + decision["committed"][:-1]
            ref_report = _ref_check(
                runner, REF_SLOT_BY_REQ[st["req_id"]], real_new_tokens, decision["next_anchor"], tok=tok
            )
            if not ref_report["content_ok_within_near_tie_tolerance"]:
                correctness_ok = False
                correctness_failures.append(
                    {
                        "req_id": st["req_id"],
                        "stage": f"round {round_idx}",
                        "batch_composition": list(active_slots),
                        "ref_report": ref_report,
                    }
                )
            st["committed_tokens"].extend(decision["committed"])
            st["committed_len"] += decision["num_accepted"] + 1
            st["anchor"], st["drafts"] = decision["next_anchor"], decision["next_draft_tokens"]

            if st["committed_len"] >= st["max_tokens"]:
                st["finished_round"] = round_idx
                st["finished_t"] = time.perf_counter() - t_start
                decoded = tok.decode(st["committed_tokens"])
                other_cities = [r[5] for r in REQUESTS if r[0] != st["req_id"]]
                has_own = st["city"] in decoded
                has_other = any(c in decoded for c in other_cities)
                identity_ok_informational[st["req_id"]] = bool(has_own and not has_other)
                finished.append(
                    {
                        "req_id": st["req_id"],
                        "slot": s,
                        "prompt_len": st["prompt_len"],
                        "committed_len": st["committed_len"],
                        "admitted_round": st["admitted_round"],
                        "finished_round": round_idx,
                        "rounds_active": round_idx - st["admitted_round"] + 1,
                        "admitted_t": st["admitted_t"],
                        "finished_t": st["finished_t"],
                        "decoded_completion": decoded,
                    }
                )
                events.append({"event": "finish", "req_id": st["req_id"], "slot": s, "round": round_idx})
                newly_finished_slots.append(s)

        for s in newly_finished_slots:
            del active[s]
            runner.reset_slot(s)
            free_slots.append(s)

        round_idx += 1

    wall_s_e2e = time.perf_counter() - t_start
    total_committed_tokens = sum(f["committed_len"] for f in finished)

    return {
        "passed": bool(correctness_ok and len(finished) == len(REQUESTS) and round_idx < max_rounds),
        "correctness_ok": correctness_ok,
        "correctness_failures": correctness_failures,
        "num_requests": len(REQUESTS),
        "num_finished": len(finished),
        "rounds_used": round_idx,
        "identity_ok_informational": identity_ok_informational,
        "events": events,
        "finished": finished,
        "wall_s_e2e": wall_s_e2e,
        "total_committed_tokens": total_committed_tokens,
        "accepted_tokens_per_sec": total_committed_tokens / wall_s_e2e if wall_s_e2e > 0 else float("nan"),
        "gpu_round_wall_s_summed": total_gpu_wall_s,
    }


def _run_once() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    max_prompt_len = max(r[2] for r in REQUESTS)
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(8192, max_prompt_len + 256),
        gpu_memory_utilization=0.85,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # 4 production + 6 dedicated reference + 6 spare margin-diag (lazy) slots.
    runner = DirectModelRunner(
        vllm_config, num_slots=26, block_size=16, blocks_per_slot=2560, enable_cudagraph=False
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    return _run_async_arrival(runner, tok)


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    print(f"\n=== {'PASS' if result['passed'] else 'FAIL'} ===")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
