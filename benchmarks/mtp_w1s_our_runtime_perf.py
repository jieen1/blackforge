"""This runtime's side of the REAL end-to-end performance comparison
(accepted tokens/s, ms/accepted token, ms/draft, TTFT, ITL, and --
the metric this whole project's premise rests on -- GPU-busy%/launch-gap)
against native vLLM, at the W1-S shape (frozen, versioned prompt token
ids, same fixture `w1s_native_bench.py` uses).

Unlike `mtp_w1s_our_runtime.py` (which only needed aggregate accept/reject
counts), every real GPU-issuing call (`mtp_prefill`'s target forward,
`mtp_verify_and_commit`'s verify/recompute) is bracketed with
`torch.cuda.Event` (the SAME technique `mtp_trace_driven_probe.py`'s
synthetic-trace probe used, now applied to the REAL, already-verified
MTP state machine on REAL data instead of a synthetic trace) so GPU-busy
time can be compared directly against wall-clock time.

This runtime processes slots ONE AT A TIME in round-robin (never batches
multiple slots into one kernel launch -- see the design doc's step-6
scope note) -- so "GPU busy% here" measures how much of THIS runtime's
own wall-clock time is real GPU work vs. Python-dispatch/launch overhead
for its ACTUAL current architecture, not a best-case synthetic scenario.
Eager mode only (no CUDA graph capture) -- explicitly the point of
comparison the coordinator asked for: does removing Python/vLLM
scheduling overhead show an advantage even without the graph-capture
optimization this project has not yet integrated into the real MTP
accept/reject flow.

Usage:
    python -m benchmarks.mtp_w1s_our_runtime_perf --max-tokens 256 --concurrency 4 --fixture n128 --num-requests 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _gpu_thermal() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    temp, clock, mem = [x.strip() for x in out.split(",")]
    return {"temperature_c": int(temp), "clock_sm_mhz": int(clock), "memory_used_mib": int(mem)}


def _run_batch(torch, runner, prompts_batch: list[list[int]], target_output_len: int) -> dict:
    num = len(prompts_batch)
    slots = list(range(num))
    for s in slots:
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    anchor = {}
    draft_tokens = {}
    committed_len = {s: 0 for s in slots}
    per_slot = {
        s: {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0, "gpu_busy_s": 0.0, "wall_s": 0.0}
        for s in slots
    }
    ttft_s = {}
    itl_samples: list[float] = []

    for i, s in enumerate(slots):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_evt.record()
        pr = runner.mtp_prefill(s, prompts_batch[i])
        end_evt.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        anchor[s] = pr["anchor"]
        draft_tokens[s] = pr["draft_tokens"]
        ttft_s[s] = t1 - t0
        per_slot[s]["wall_s"] += t1 - t0
        per_slot[s]["gpu_busy_s"] += start_evt.elapsed_time(end_evt) / 1000.0

    finished = set()
    while len(finished) < num:
        for s in slots:
            if s in finished:
                continue
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start_evt.record()
            decision = runner.mtp_verify_and_commit(s, anchor[s], draft_tokens[s])
            end_evt.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_acc = decision["num_accepted"]
            tokens_this_round = n_acc + 1
            per_slot[s]["num_drafts"] += 1
            per_slot[s]["num_draft_tokens"] += K
            per_slot[s]["num_accepted_tokens"] += n_acc
            per_slot[s]["wall_s"] += t1 - t0
            per_slot[s]["gpu_busy_s"] += start_evt.elapsed_time(end_evt) / 1000.0
            committed_len[s] += tokens_this_round
            itl_samples.append((t1 - t0) / tokens_this_round)

            anchor[s], draft_tokens[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                finished.add(s)

    return {"per_slot": per_slot, "ttft_s": ttft_s, "itl_samples": itl_samples}


def _run_measurement(torch, runner, prompts: list[list[int]], max_tokens: int, concurrency: int, rep: int) -> dict:
    """ONE repetition of the full W1-S request set against an
    ALREADY-LOADED runner (no reload between reps -- reps are for
    repeated-measurement variance, not independent process launches;
    see this file's module docstring / the design doc for why this is a
    deliberate, documented deviation from literal interleaved A/B this
    round, given the cost of reloading a 27B model per leg)."""
    thermal_before = _gpu_thermal()
    all_ttfts: list[float] = []
    all_itls: list[float] = []
    total_drafts = total_draft_tokens = total_accepted = 0
    total_gpu_busy_s = total_wall_s = 0.0

    t_start = time.perf_counter()
    num_batches = (len(prompts) + concurrency - 1) // concurrency
    for batch_idx, batch_start in enumerate(range(0, len(prompts), concurrency)):
        batch = prompts[batch_start : batch_start + concurrency]
        out = _run_batch(torch, runner, batch, max_tokens)
        for s, stats in out["per_slot"].items():
            total_drafts += stats["num_drafts"]
            total_draft_tokens += stats["num_draft_tokens"]
            total_accepted += stats["num_accepted_tokens"]
            total_gpu_busy_s += stats["gpu_busy_s"]
            total_wall_s += stats["wall_s"]
        all_ttfts.extend(out["ttft_s"].values())
        all_itls.extend(out["itl_samples"])
        elapsed = time.perf_counter() - t_start
        print(f"  ... rep {rep} batch {batch_idx + 1}/{num_batches} done ({elapsed:.0f}s elapsed)", flush=True)

    wall_s_e2e = time.perf_counter() - t_start
    thermal_after = _gpu_thermal()
    total_committed = total_accepted + total_drafts

    all_ttfts.sort()
    all_itls.sort()

    return {
        "rep": rep,
        "wall_s_e2e": wall_s_e2e,
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted,
        "draft_acceptance_rate_pct": total_accepted / total_draft_tokens * 100.0 if total_draft_tokens else float("nan"),
        "total_committed_tokens": total_committed,
        "accepted_tokens_per_sec": total_committed / wall_s_e2e if wall_s_e2e > 0 else float("nan"),
        "ms_per_accepted_token": wall_s_e2e * 1000.0 / total_committed if total_committed > 0 else float("nan"),
        "ms_per_draft": wall_s_e2e * 1000.0 / total_drafts if total_drafts > 0 else float("nan"),
        "ttft_mean_ms": sum(all_ttfts) / len(all_ttfts) * 1000.0 if all_ttfts else float("nan"),
        "ttft_p99_ms": all_ttfts[int(len(all_ttfts) * 0.99)] * 1000.0 if all_ttfts else float("nan"),
        "itl_mean_ms": sum(all_itls) / len(all_itls) * 1000.0 if all_itls else float("nan"),
        "itl_p99_ms": all_itls[int(len(all_itls) * 0.99)] * 1000.0 if all_itls else float("nan"),
        "num_itl_samples": len(all_itls),
        "gpu_busy_s_summed_across_slots": total_gpu_busy_s,
        "wall_s_summed_across_slots": total_wall_s,
        "gpu_busy_pct": total_gpu_busy_s / total_wall_s * 100.0 if total_wall_s > 0 else float("nan"),
        "launch_gap_pct": (1.0 - total_gpu_busy_s / total_wall_s) * 100.0 if total_wall_s > 0 else float("nan"),
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
    }


def _run_once(max_tokens: int, concurrency: int, fixture_key: str, num_requests: int | None, repeats: int) -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, W1_S_FIXTURE_N128, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {"n16": W1_S_FIXTURE, "n128": W1_S_FIXTURE_N128}[fixture_key]
    prompts = load_prompt_token_ids(fixture)
    if num_requests is not None:
        prompts = prompts[:num_requests]

    thermal_before_load = _gpu_thermal()

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, fixture.prompt_len + max_tokens + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=concurrency, block_size=16, blocks_per_slot=2560)
    thermal_after_load = _gpu_thermal()

    reps = [_run_measurement(torch, runner, prompts, max_tokens, concurrency, r + 1) for r in range(repeats)]

    return {
        "passed": True,
        "num_requests": len(prompts),
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "k": K,
        "repeats": repeats,
        "reps": reps,
        "thermal_before_load": thermal_before_load,
        "thermal_after_load": thermal_after_load,
        "fixture": fixture.path,
        "fixture_seed": fixture.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixture", choices=["n16", "n128"], default="n16")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    result = _run_once(args.max_tokens, args.concurrency, args.fixture, args.num_requests, args.repeats)

    import json

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
