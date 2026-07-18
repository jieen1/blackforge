"""Native vLLM side of the W1-S (controlled synthetic) acceptance-rate
comparison. Sends the FROZEN, VERSIONED prompt token ids from
`benchmarks/fixtures/w1s_prompts.json` directly to a running server's
`/v1/completions` endpoint as `prompt: list[int]` (the OpenAI completions
API's own token-array prompt form -- confirmed supported by reading
`vllm/entrypoints/openai/completion/protocol.py`'s `CompletionRequest
.prompt` type directly), NOT through `vllm bench serve`'s own
`--dataset-name random` generator -- this is what makes "both sides ran
the IDENTICAL input" a checked fact rather than "same formula, hopefully
reproduced the same way."

Spec-decode stats: scrapes `/metrics` before and after (the exact same
approach `vllm/benchmarks/serve.py`'s own `fetch_spec_decode_metrics()` +
delta computation uses -- reimplemented here directly, not imported, to
avoid coupling to vLLM's internal benchmarks module path) and reports the
delta, matching vLLM's own real acceptance-rate computation exactly.

Assumes the server is ALREADY running (this project's established
pattern: launch via `launch_test_server.py` in a separate step, this
script only drives the benchmark traffic against it).

Usage:
    python -m benchmarks.w1s_native_bench --port 8100 --max-tokens 256 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

from benchmarks.workloads import (
    D1_CTX16K_FIXTURE,
    D1_CTX32K_FIXTURE,
    D1_CTX64K_FIXTURE,
    W1_S_FIXTURE,
    W1_S_FIXTURE_N128,
    load_prompt_token_ids,
)

FIXTURES = {
    "n16": W1_S_FIXTURE,
    "n128": W1_S_FIXTURE_N128,
    # 2026-07-18, Phase D1 shape-generalization sweep: constructed,
    # same-formula/same-seed fixtures at longer context -- NOT the
    # official W2/W2-S line, see workloads.py's own docstring.
    "ctx16k": D1_CTX16K_FIXTURE,
    "ctx32k": D1_CTX32K_FIXTURE,
    # 2026-07-18, D1 sweep continuation: native has no equivalent fixed
    # per-slot capacity ceiling (paged KV cache sized from
    # gpu_memory_utilization at server startup), so this fixture works fine
    # here even though it is blocked for this runtime's own benchmark --
    # see workloads.py's D1_CTX64K_FIXTURE docstring.
    "ctx64k": D1_CTX64K_FIXTURE,
}


def _gpu_thermal() -> dict:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    temp, clock, mem = [x.strip() for x in out.split(",")]
    return {"temperature_c": int(temp), "clock_sm_mhz": int(clock), "memory_used_mib": int(mem)}


async def _fetch_spec_decode_counters(base_url: str, session: aiohttp.ClientSession) -> dict:
    """Faithful port of vllm/benchmarks/serve.py's fetch_spec_decode_metrics
    parsing logic (Prometheus text exposition format, `vllm:spec_decode*_total`
    counters, summed across any labels)."""
    counters = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}
    accepted_per_pos: dict[int, int] = {}
    async with session.get(f"{base_url}/metrics") as resp:
        if resp.status != 200:
            return {"counters": counters, "accepted_per_pos": accepted_per_pos, "found": False}
        text = await resp.text()

    found = False
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{")[0]
        if not metric_name.endswith("_total"):
            continue
        found = True
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if "num_accepted_tokens_per_pos" in metric_name:
            if 'position="' in line:
                pos = int(line.split('position="', 1)[1].split('"', 1)[0])
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + int(value)
        elif "num_drafts" in metric_name:
            counters["num_drafts"] += int(value)
        elif "num_draft_tokens" in metric_name:
            counters["num_draft_tokens"] += int(value)
        elif "num_accepted_tokens" in metric_name:
            counters["num_accepted_tokens"] += int(value)

    return {"counters": counters, "accepted_per_pos": accepted_per_pos, "found": found}


async def _send_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    sem: asyncio.Semaphore,
    stream: bool,
) -> dict:
    """`stream=False` (default): a single non-streaming completion, only
    total wall time is measured -- this is what the acceptance-rate
    comparison rounds used, since TTFT/ITL were not needed there.
    `stream=True` (2026-07-17 addition, for the real end-to-end
    performance comparison): uses SSE streaming so per-token arrival
    times are observable -- TTFT (time to the first streamed token) and
    ITL (inter-arrival time between subsequent tokens) are exactly the
    metrics `vllm bench serve` itself reports, computed the same way
    here (first chunk vs. `t0` for TTFT, consecutive chunk gaps for
    ITL) so the numbers are directly comparable."""
    # 2026-07-17 fix: `ignore_eos=True` on BOTH the streaming and
    # non-streaming paths. Without it, native vLLM legitimately stops
    # early for some frozen prompts (a REAL EOS prediction, not an
    # error -- confirmed by direct debugging: some prompts hit EOS
    # within 1-2 generated tokens). This runtime's own direct-runner
    # side has no EOS-checking logic at all (it always generates
    # exactly `target_output_len` tokens, unconditionally) -- for THIS
    # controlled-synthetic, FIXED-LENGTH performance comparison
    # (this project's `-S` line, not the real-traffic `-R` line where
    # early EOS is exactly what should be measured), both sides must
    # behave the same way: generate the full fixed length regardless of
    # any model-predicted stop signal. Omitting this was inflating
    # apparent "failures" (a 0-token completion isn't a failure, just an
    # early-EOS response) AND would have been a genuine, uncontrolled
    # confound for the performance numbers (fewer real generated tokens
    # = less real work = faster wall time, unrelated to the two
    # implementations' actual per-token cost).
    async with sem:
        t0 = time.perf_counter()
        if not stream:
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt_token_ids,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "ignore_eos": True,
                },
            ) as resp:
                body = await resp.json()
            t1 = time.perf_counter()
            ok = resp.status == 200 and "choices" in body
            return {"ok": ok, "wall_s": t1 - t0, "status": resp.status, "raw": body if not ok else None}

        token_times: list[float] = []
        status = None
        async with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "ignore_eos": True,
                "stream": True,
            },
        ) as resp:
            status = resp.status
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text", "") != "":
                    token_times.append(time.perf_counter())
        t_end = time.perf_counter()

        ok = status == 200 and len(token_times) > 0
        ttft = (token_times[0] - t0) if token_times else float("nan")
        itls = [b - a for a, b in zip(token_times, token_times[1:])]
        return {
            "ok": ok,
            "wall_s": t_end - t0,
            "status": status,
            "num_tokens": len(token_times),
            "ttft_s": ttft,
            "itls_s": itls,
        }


async def _run_once(
    port: int,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    fixture_key: str,
    stream: bool = False,
    num_requests: int | None = None,
) -> dict:
    base_url = f"http://127.0.0.1:{port}"
    fixture = FIXTURES[fixture_key]
    prompts = load_prompt_token_ids(fixture)
    if num_requests is not None:
        # 2026-07-18, Phase D1: bound cost at long-context spot-checks by
        # slicing the frozen fixture down (same convention as this
        # project's own `mtp_w1s_our_runtime_perf.py --num-requests`).
        prompts = prompts[:num_requests]

    async with aiohttp.ClientSession() as session:
        before = await _fetch_spec_decode_counters(base_url, session)
        if not before["found"]:
            return {"passed": False, "error": "no vllm:spec_decode* metrics found -- is --with-mtp enabled?"}

        sem = asyncio.Semaphore(concurrency)
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                _send_one(session, base_url, model, p, max_tokens, temperature, sem, stream)
                for p in prompts
            ]
        )
        t_end = time.perf_counter()

        after = await _fetch_spec_decode_counters(base_url, session)

    num_failed = sum(1 for r in results if not r["ok"])
    delta_drafts = after["counters"]["num_drafts"] - before["counters"]["num_drafts"]
    delta_draft_tokens = after["counters"]["num_draft_tokens"] - before["counters"]["num_draft_tokens"]
    delta_accepted = after["counters"]["num_accepted_tokens"] - before["counters"]["num_accepted_tokens"]

    all_pos = set(before["accepted_per_pos"]) | set(after["accepted_per_pos"])
    per_pos_delta = {
        pos: after["accepted_per_pos"].get(pos, 0) - before["accepted_per_pos"].get(pos, 0)
        for pos in all_pos
    }
    per_position_rate = (
        {pos: v / delta_drafts for pos, v in sorted(per_pos_delta.items())} if delta_drafts > 0 else {}
    )

    result = {
        "passed": num_failed == 0,
        "num_requests": len(prompts),
        "num_failed": num_failed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "concurrency": concurrency,
        "stream": stream,
        "wall_s": t_end - t_start,
        "num_drafts": delta_drafts,
        "num_draft_tokens": delta_draft_tokens,
        "num_accepted_tokens": delta_accepted,
        "draft_acceptance_rate_pct": (
            delta_accepted / delta_draft_tokens * 100.0 if delta_draft_tokens > 0 else float("nan")
        ),
        "mean_acceptance_length": 1 + (delta_accepted / delta_drafts) if delta_drafts > 0 else float("nan"),
        "per_position_acceptance_rate": per_position_rate,
        "accepted_tokens_per_sec": delta_accepted / (t_end - t_start) if (t_end - t_start) > 0 else float("nan"),
        "ms_per_accepted_token": (t_end - t_start) * 1000.0 / delta_accepted if delta_accepted > 0 else float("nan"),
        "ms_per_draft": (t_end - t_start) * 1000.0 / delta_drafts if delta_drafts > 0 else float("nan"),
        "fixture": fixture.path,
        "fixture_seed": fixture.seed,
    }

    if stream:
        all_ttfts = [r["ttft_s"] for r in results if r.get("ok") and "ttft_s" in r]
        all_itls = [itl for r in results if r.get("ok") for itl in r.get("itls_s", [])]
        result["ttft_mean_ms"] = sum(all_ttfts) / len(all_ttfts) * 1000.0 if all_ttfts else float("nan")
        result["ttft_p99_ms"] = (
            sorted(all_ttfts)[int(len(all_ttfts) * 0.99)] * 1000.0 if all_ttfts else float("nan")
        )
        result["itl_mean_ms"] = sum(all_itls) / len(all_itls) * 1000.0 if all_itls else float("nan")
        result["itl_p99_ms"] = sorted(all_itls)[int(len(all_itls) * 0.99)] * 1000.0 if all_itls else float("nan")
        result["num_itl_samples"] = len(all_itls)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="qwen3.6-sm120-test")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixture", choices=list(FIXTURES.keys()), default="n16")
    parser.add_argument("--stream", action="store_true", help="use SSE streaming to measure TTFT/ITL")
    parser.add_argument("--repeats", type=int, default=1, help="repeat against the SAME already-running server")
    parser.add_argument(
        "--num-requests", type=int, default=None, help="slice the frozen fixture down to this many requests"
    )
    args = parser.parse_args()

    reps = []
    for r in range(args.repeats):
        thermal_before = _gpu_thermal()
        rep_result = asyncio.run(
            _run_once(
                args.port,
                args.model,
                args.max_tokens,
                args.temperature,
                args.concurrency,
                args.fixture,
                args.stream,
                args.num_requests,
            )
        )
        thermal_after = _gpu_thermal()
        rep_result["rep"] = r + 1
        rep_result["thermal_before"] = thermal_before
        rep_result["thermal_after"] = thermal_after
        reps.append(rep_result)
        print(f"  ... native rep {r + 1}/{args.repeats} done", flush=True)

    result = {"passed": all(r.get("passed") for r in reps), "repeats": args.repeats, "reps": reps}
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
