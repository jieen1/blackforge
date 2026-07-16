"""Real numerical eager-vs-graph parity check for
``CapturedBatchDecodeGraph`` (qo_len=1) -- NOT signal-probe. Built per an
explicit coordinator requirement, following an independent (Codex)
correctness review that found the earlier signal-probe-only verification
insufficient to catch GDN state pollution: this project's own identity-
recall task is plausibly dominated by full-attention layers' copy
mechanism, masking a GDN-specific perturbation a more GDN-sensitive task
might not tolerate.

This test drives the SAME real input (identical prompt, identical
kv_len, identical draft token) through TWO independent, identically
prefilled physical-slot groups:
  - ``eager_slots``: the already-verified single-call
    ``DirectModelRunner._forward_batch`` path (no CUDA graph at all).
  - ``graph_slots``: ``CapturedBatchDecodeGraph.replay()``.

and compares, directly and numerically:
  1. Full output logits -- max abs diff, cosine similarity, and
     ``torch.allclose`` at a generous-but-meaningful tolerance (eager
     uses a per-call, TIGHT kv_split_size; the graph uses a build-time
     FIXED, larger one -- see build_attention_metadata_batch's docstring
     -- so a different attention split/merge reduction order is
     EXPECTED to introduce some floating-point noise, the same
     "different shape/path -> different rounding" phenomenon already
     established for batch-composition effects. The check here is that
     the difference stays SMALL, not that it's exactly zero).
  2. Top-1/top-5 predicted token id agreement.
  3. The GDN recurrent state tensors themselves (``conv_state``/
     ``ssm_state``, read directly out of ``runner.kv_caches`` for the
     physical row each group's slot maps to) -- for a SINGLE token at
     IDENTICAL input, GDN's own math does not depend on attention's
     kv_split_size at all, so eager and graph should agree here far
     more tightly than the logits comparison. This is the single most
     direct test for the "capture's warmup polluted the real GDN state"
     bug class this round's fix targets: a real state-neutral capture
     should show these tensors bytewise-identical (or extremely close);
     a regression would show them meaningfully diverged even though the
     signal-probe-level text output might still look fine.

Usage:
    python -m benchmarks.cudagraph_eager_parity_check
    python -m benchmarks.cudagraph_eager_parity_check --repeat 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"

# Tolerances: generous relative to typical BF16/FP8 noise (this project's
# own established atol=1e-2/rtol=1e-2 convention for BF16 numerical
# checks), but still tight enough to catch a real several-extra-step
# state perturbation, which would show up as a much larger, obviously
# not-noise-shaped divergence.
LOGITS_ATOL = 0.5
LOGITS_RTOL = 0.05
GDN_STATE_ATOL = 1e-2
GDN_STATE_RTOL = 1e-2


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import (
        CapturedBatchDecodeGraph,
        DirectModelRunner,
        _physical_slot,
        build_vllm_config,
    )

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    block_size, blocks_per_slot = 16, 128
    batch = 2
    # eager_slots (batch) + graph_slots (batch) + CapturedBatchDecodeGraph's
    # own internally reserved warmup slots (batch) = 3*batch.
    runner = DirectModelRunner(
        vllm_config, num_slots=3 * batch, block_size=block_size, blocks_per_slot=blocks_per_slot
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    eager_slots = list(range(batch))
    graph_slots = list(range(batch, 2 * batch))

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    eager_next = [runner.prefill(s, prompt_ids) for s in eager_slots]
    graph_next = [runner.prefill(s, prompt_ids) for s in graph_slots]
    if eager_next != graph_next:
        return {"passed": False, "error": "prefill greedy tokens diverged between eager_slots and graph_slots"}

    eager_kv = [runner.slot_kv_len[s] for s in eager_slots]
    graph_kv = [runner.slot_kv_len[s] for s in graph_slots]
    if eager_kv != graph_kv:
        return {"passed": False, "error": "kv_len diverged between eager_slots and graph_slots after prefill"}

    # Identical draft token for both groups (the greedy next token from
    # the identical prefill).
    draft_tokens_eager = list(eager_next)
    draft_tokens_graph = list(graph_next)

    # --- Eager path (already-verified, no CUDA graph at all). ---
    eager_logits = runner._forward_batch(eager_slots, draft_tokens_eager, eager_kv).clone()

    # --- Graph path: capture (self-contained, uses its own reserved
    # warmup slots -- never touches eager_slots or graph_slots) then
    # replay at the IDENTICAL input. ---
    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    graph.capture()
    graph_logits = graph.replay(graph_slots, draft_tokens_graph, graph_kv).clone()

    # --- 1. Logits comparison. ---
    max_abs_diff = (eager_logits.float() - graph_logits.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        eager_logits.float().flatten(), graph_logits.float().flatten(), dim=0
    ).item()
    allclose = torch.allclose(eager_logits.float(), graph_logits.float(), atol=LOGITS_ATOL, rtol=LOGITS_RTOL)

    eager_top5 = eager_logits.float().topk(5, dim=-1)
    graph_top5 = graph_logits.float().topk(5, dim=-1)
    top1_match = [
        int(eager_top5.indices[i, 0].item()) == int(graph_top5.indices[i, 0].item()) for i in range(batch)
    ]
    top5_overlap = [
        len(set(eager_top5.indices[i].tolist()) & set(graph_top5.indices[i].tolist())) for i in range(batch)
    ]

    logits_result = {
        "max_abs_diff": max_abs_diff,
        "cosine_similarity": cos_sim,
        "allclose": allclose,
        "top1_match": top1_match,
        "top5_overlap_count": top5_overlap,
    }

    # --- 2. Direct GDN state tensor comparison -- the decisive check for
    # capture-warmup state pollution. Compare EVERY GDN layer's
    # conv_state/ssm_state row for the corresponding eager/graph physical
    # slots (both received the IDENTICAL single-token update above). ---
    gdn_state_results = []
    all_gdn_close = True
    for layer_name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[layer_name]
        for i in range(batch):
            eager_phys = _physical_slot(eager_slots[i])
            graph_phys = _physical_slot(graph_slots[i])
            conv_e = conv_state[eager_phys].float()
            conv_g = conv_state[graph_phys].float()
            ssm_e = ssm_state[eager_phys].float()
            ssm_g = ssm_state[graph_phys].float()
            conv_close = torch.allclose(conv_e, conv_g, atol=GDN_STATE_ATOL, rtol=GDN_STATE_RTOL)
            ssm_close = torch.allclose(ssm_e, ssm_g, atol=GDN_STATE_ATOL, rtol=GDN_STATE_RTOL)
            conv_max_diff = (conv_e - conv_g).abs().max().item()
            ssm_max_diff = (ssm_e - ssm_g).abs().max().item()
            gdn_state_results.append(
                {
                    "layer": layer_name,
                    "pair_idx": i,
                    "conv_close": conv_close,
                    "conv_max_diff": conv_max_diff,
                    "ssm_close": ssm_close,
                    "ssm_max_diff": ssm_max_diff,
                }
            )
            if not (conv_close and ssm_close):
                all_gdn_close = False

    passed = bool(allclose) and all(top1_match) and all_gdn_close
    return {
        "passed": passed,
        "logits": logits_result,
        "gdn_states_all_close": all_gdn_close,
        "gdn_state_sample": gdn_state_results[:4],  # first few layers, avoid huge output
        "num_gdn_layers_checked": len(runner.gdn_layer_names),
    }


def _run_subprocess() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.cudagraph_eager_parity_check", "--single-run-json"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once()
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once()
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess() for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}")
        if not r.get("passed"):
            print(f"  detail: {r}")

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
