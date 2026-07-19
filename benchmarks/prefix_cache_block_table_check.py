"""Correctness gate for prefix-cache P0 -- block-table indirection
substrate (``notes/prefix-cache-design.md`` sec 5, "P0 -- Block-table
indirection substrate").

P0 is a BEHAVIOR-IDENTICAL refactor: ``DirectModelRunner.block_table[slot]``
is introduced, initialized to today's contiguous physical-block range, and
the design doc's four named touch-points (``build_attention_metadata``/
``_batch`` -- read path, ``_slot_mapping``/``_batch`` -- write path,
``CapturedBatchDecodeGraph._fill_buffers`` -- CUDA-graph path, and the new
``_initial_block_table`` "thin allocator") are routed through it, gated by
a new ``enable_block_table`` constructor flag (default ``False``, this
project's established feature-flag convention -- see ``enable_cudagraph``).
GDN, ``_physical_slot``, and reserved-physical-slot-0 handling are
untouched (design doc is explicit that only attention addressing changes
in P0).

This script has two checks:

1. ``_check_arange_equivalence`` (pure Python, no GPU/model): the design
   doc's own explicitly-called-for unit test -- ``block_table[slot]`` (via
   the new ``_initial_block_table`` allocator) must equal the OLD
   arange-derived physical block range for every slot, at several
   (num_slots, blocks_per_slot) shapes.

2. ``_run_numeric_equivalence`` (real GPU, real model): loads ONE real
   ``DirectModelRunner`` and, for the SAME prompt/slot, runs an identical
   prefill+decode sequence twice -- once with ``enable_block_table=False``
   (today's arange addressing) and once with ``enable_block_table=True``
   (routed through ``block_table``) -- and asserts the resulting logits are
   BYTEWISE IDENTICAL (not near-tie: this is a pure addressing-equivalence
   check within one process/one slot at a time, no cross-request batching/
   reduction-order confound at all, so any difference at all would be a
   real bug, not legitimate fp8/batch non-associativity noise). Covers, in
   order: the single-request path (``build_attention_metadata``/
   ``_slot_mapping``), the batched path (``build_attention_metadata_batch``/
   ``_slot_mapping_batch``, both prefill- and decode-shaped calls), and
   ``CapturedBatchDecodeGraph._fill_buffers`` (exercised directly, without
   a full CUDA-graph capture, by calling it twice with the flag toggled and
   diffing the resulting static buffers -- ``_fill_buffers`` is plain
   Python/tensor-copy code, not itself inside ``torch.cuda.graph()``, so
   this is a real, direct exercise of that method, not a reimplementation
   of its logic).

Not covered by this script (explicitly, not an oversight): the MTP
draft-model call sites (``_mtp_forward``/``_mtp_forward_batch``) and
``CapturedMTPDraftStepGraph._fill_buffers``. Both reuse the EXACT SAME
``build_attention_metadata``/``_batch``/``_slot_mapping``/``_batch``
functions this script already exercises via the target model (not a
second, independently-written copy), and the graph class's ``_fill_buffers``
was fixed with the identical code pattern as
``CapturedBatchDecodeGraph._fill_buffers`` (also exercised here) -- loading
the MTP draft model just for this narrow addressing check was judged not
worth the extra weight-load time for this round; the full regression
battery (which does load MTP, with the flag at its default ``False``)
still confirms zero-regression production behavior.

Usage:
    python -m benchmarks.prefix_cache_block_table_check
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"


def _check_arange_equivalence() -> dict:
    from runtime.direct_model_runner import _initial_block_table, _physical_slot

    mismatches = []
    for num_slots, blocks_per_slot in [(4, 128), (1, 1), (8, 2560), (16, 64)]:
        for slot in range(num_slots):
            got = _initial_block_table(slot, blocks_per_slot)
            first_block = _physical_slot(slot) * blocks_per_slot
            want = list(range(first_block, first_block + blocks_per_slot))
            if got != want:
                mismatches.append(
                    {
                        "num_slots": num_slots,
                        "blocks_per_slot": blocks_per_slot,
                        "slot": slot,
                        "got_head": got[:5],
                        "want_head": want[:5],
                        "len_got": len(got),
                        "len_want": len(want),
                    }
                )
    return {"passed": not mismatches, "mismatches": mismatches}


def _run_numeric_equivalence() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import CapturedBatchDecodeGraph, DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    # Small blocks_per_slot -- this check only needs a handful of tokens
    # per slot, keeps allocation/capture cheap. num_slots=4 satisfies
    # CapturedBatchDecodeGraph(batch_size=2)'s num_slots >= 2*batch_size
    # requirement below.
    runner = DirectModelRunner(
        vllm_config, num_slots=4, block_size=16, blocks_per_slot=64, enable_block_table=False
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    # --- 1. Single-request path: build_attention_metadata + _slot_mapping ---
    runner.enable_block_table = False
    tok_off = runner.prefill(0, prompt_ids)
    logits_off = runner._forward(0, [tok_off], start_pos=runner.slot_kv_len[0], is_decode=True)
    runner.reset_slot(0)

    runner.enable_block_table = True
    tok_on = runner.prefill(0, prompt_ids)
    logits_on = runner._forward(0, [tok_on], start_pos=runner.slot_kv_len[0], is_decode=True)
    runner.reset_slot(0)

    checks["single_request_greedy_token_match"] = tok_off == tok_on
    checks["single_request_logits_bytewise_equal"] = bool(torch.equal(logits_off.cpu(), logits_on.cpu()))

    # --- 2. Batched path: build_attention_metadata_batch + _slot_mapping_batch ---
    def _run_batch_pass(enable: bool):
        runner.enable_block_table = enable
        slots = [0, 1]
        qo_len = len(prompt_ids)
        prefill_logits = runner._forward_batch(
            slots,
            [prompt_ids, prompt_ids],
            kv_lengths=[0, 0],
            qo_len=qo_len,
            is_decode=False,
            commit=True,
        )
        anchors = [
            int(prefill_logits[i * qo_len + qo_len - 1].argmax(dim=-1).item()) for i in range(len(slots))
        ]
        kv_lengths = [runner.slot_kv_len[s] for s in slots]
        decode_logits = runner._forward_batch(
            slots, anchors, kv_lengths, qo_len=1, is_decode=True, commit=True
        )
        for s in slots:
            runner.reset_slot(s)
        return prefill_logits, decode_logits, anchors

    prefill_off, decode_off, anchors_off = _run_batch_pass(False)
    prefill_on, decode_on, anchors_on = _run_batch_pass(True)

    checks["batched_anchor_tokens_match"] = anchors_off == anchors_on
    checks["batched_prefill_logits_bytewise_equal"] = bool(torch.equal(prefill_off.cpu(), prefill_on.cpu()))
    checks["batched_decode_logits_bytewise_equal"] = bool(torch.equal(decode_off.cpu(), decode_on.cpu()))

    # --- 3. CUDA-graph fill path: CapturedBatchDecodeGraph._fill_buffers ---
    # Exercised directly (no real graph capture needed -- _fill_buffers is
    # plain Python/tensor-copy code, not itself inside torch.cuda.graph()).
    runner.enable_block_table = False
    graph = CapturedBatchDecodeGraph(runner, batch_size=2, qo_len=1)
    fill_slots = [0, 1]
    fill_kv_lengths = [3, 5]
    fill_tokens = [7, 11]

    runner.enable_block_table = False
    graph._fill_buffers(fill_slots, fill_tokens, fill_kv_lengths)
    page_indices_off = graph.static_kv_page_indices.clone()
    slot_mapping_off = graph.static_slot_mapping.clone()
    kv_page_indptr_off = graph.static_kv_page_indptr.clone()

    runner.enable_block_table = True
    graph._fill_buffers(fill_slots, fill_tokens, fill_kv_lengths)
    page_indices_on = graph.static_kv_page_indices.clone()
    slot_mapping_on = graph.static_slot_mapping.clone()
    kv_page_indptr_on = graph.static_kv_page_indptr.clone()

    checks["graph_fill_buffers_page_indices_equal"] = bool(torch.equal(page_indices_off, page_indices_on))
    checks["graph_fill_buffers_slot_mapping_equal"] = bool(torch.equal(slot_mapping_off, slot_mapping_on))
    checks["graph_fill_buffers_kv_page_indptr_equal"] = bool(torch.equal(kv_page_indptr_off, kv_page_indptr_on))

    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    arange_result = _check_arange_equivalence()
    print(f"arange-equivalence check: {'PASS' if arange_result['passed'] else 'FAIL'}")
    if not arange_result["passed"]:
        print(f"  mismatches: {arange_result['mismatches']}")

    numeric_result = _run_numeric_equivalence()
    print(f"numeric-equivalence check: {'PASS' if numeric_result['passed'] else 'FAIL'}")
    for name, ok in numeric_result["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    overall = arange_result["passed"] and numeric_result["passed"]
    print(f"\n=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
