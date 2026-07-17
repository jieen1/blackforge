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

from benchmarks.workloads import W1_S_FIXTURE, load_prompt_token_ids


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
) -> dict:
    async with sem:
        t0 = time.perf_counter()
        async with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        ) as resp:
            body = await resp.json()
        t1 = time.perf_counter()
        ok = resp.status == 200 and "choices" in body
        return {"ok": ok, "wall_s": t1 - t0, "status": resp.status, "raw": body if not ok else None}


async def _run_once(
    port: int, model: str, max_tokens: int, temperature: float, concurrency: int
) -> dict:
    base_url = f"http://127.0.0.1:{port}"
    prompts = load_prompt_token_ids(W1_S_FIXTURE)

    async with aiohttp.ClientSession() as session:
        before = await _fetch_spec_decode_counters(base_url, session)
        if not before["found"]:
            return {"passed": False, "error": "no vllm:spec_decode* metrics found -- is --with-mtp enabled?"}

        sem = asyncio.Semaphore(concurrency)
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                _send_one(session, base_url, model, p, max_tokens, temperature, sem)
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

    return {
        "passed": num_failed == 0,
        "num_requests": len(prompts),
        "num_failed": num_failed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "concurrency": concurrency,
        "wall_s": t_end - t_start,
        "num_drafts": delta_drafts,
        "num_draft_tokens": delta_draft_tokens,
        "num_accepted_tokens": delta_accepted,
        "draft_acceptance_rate_pct": (
            delta_accepted / delta_draft_tokens * 100.0 if delta_draft_tokens > 0 else float("nan")
        ),
        "mean_acceptance_length": 1 + (delta_accepted / delta_drafts) if delta_drafts > 0 else float("nan"),
        "per_position_acceptance_rate": per_position_rate,
        "fixture": W1_S_FIXTURE.path,
        "fixture_seed": W1_S_FIXTURE.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="qwen3.6-sm120-test")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    result = asyncio.run(
        _run_once(args.port, args.model, args.max_tokens, args.temperature, args.concurrency)
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
