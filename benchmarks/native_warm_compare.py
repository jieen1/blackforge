"""Native vLLM WARM prefix-cache-hit throughput at 64K/128K, c=4.

Apples-to-apples comparison against this runtime's own warm prefix-cache
measurement (benchmarks/prefix_cache_warm_throughput_check.py --fixture
ctx128k --concurrency 4 => 83.24 tok/s aggregate warm accepted throughput).

Pattern:
  1. Cold-populate: 4 concurrent requests, prompt=prefix_i (131072 tok),
     max_tokens=16.  Populates vLLM's APC with the 128K prefixes.
  2. Warm: 4 concurrent requests, prompt=prefix_i + fresh 10240-token
     suffix (141312 tok total), max_tokens=256, greedy, streaming.
     APC hits at the prefix boundary; only the ~10K suffix is re-prefilled.
  3. Repeat warm 3x (APC stays warm across repeats).

Uses the token-array API (prompt: list[int]) for exact prefix matching,
and scrapes vllm:spec_decode_* metrics deltas for accepted tok/s, exactly
like w1s_native_bench.py.

Usage:
    python -m benchmarks.native_warm_compare --port 8100 [--fixture ctx64k|ctx128k]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

from benchmarks.workloads import (
    CTX128K_FIXTURE,
    D1_CTX64K_FIXTURE,
    load_prompt_token_ids,
)

MODEL = "qwen3.6-sm120-test"
SUFFIX_LEN = 10240
COLD_MAX_TOKENS = 16
WARM_MAX_TOKENS = 256
CONCURRENCY = 4
WARM_REPEATS = 3

# Per-fixture config: frozen fixture, its prefix length, and this runtime's own
# warm reference numbers for the apples-to-apples comparison block. native_cold
# is None where we have not measured a native cold-populate throughput reference.
FIXTURES = {
    "ctx64k": {
        "fixture": D1_CTX64K_FIXTURE,
        "prefix_len": 65536,
        "ours_warm_tok_s": 114.28,
        "ours_warm_ttft_ms": 15681.0,
        "native_cold_ref_tok_s": None,
    },
    "ctx128k": {
        "fixture": CTX128K_FIXTURE,
        "prefix_len": 131072,
        "ours_warm_tok_s": 83.24,
        "ours_warm_ttft_ms": None,
        "native_cold_ref_tok_s": 3.27,
    },
}


def _build_suffix(length: int) -> list[int]:
    """Deterministic suffix that is NOT a substring of any fixture prefix.
    Uses a stride-7 pattern from a high base offset, distinct from the
    sequential-run formula used by the frozen fixtures."""
    base = 100000
    return [(base + i * 7) % 151665 for i in range(length)]


async def _fetch_spec_decode_counters(
    base_url: str, session: aiohttp.ClientSession
) -> dict:
    counters = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}
    async with session.get(f"{base_url}/metrics") as resp:
        if resp.status != 200:
            return {"counters": counters, "found": False}
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
        if "num_drafts" in metric_name and "tokens" not in metric_name:
            counters["num_drafts"] += int(value)
        elif "num_draft_tokens" in metric_name:
            counters["num_draft_tokens"] += int(value)
        elif "num_accepted_tokens" in metric_name and "per_pos" not in metric_name:
            counters["num_accepted_tokens"] += int(value)
    return {"counters": counters, "found": found}


async def _fetch_prefix_cache_metrics(
    base_url: str, session: aiohttp.ClientSession
) -> dict:
    """Scrape /metrics for gpu_prefix_cache_hit_rate if exposed."""
    result: dict = {}
    async with session.get(f"{base_url}/metrics") as resp:
        if resp.status != 200:
            return result
        text = await resp.text()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            continue
        if "prefix_cache" in line.lower() or "gpu_prefix_cache" in line.lower():
            result[line.split("{")[0].strip()] = line
    return result


async def _send_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    sem: asyncio.Semaphore,
    timeout_s: float = 600.0,
) -> dict:
    async with sem:
        token_times: list[float] = []
        t0 = time.perf_counter()
        status = None
        try:
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt_token_ids,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                status = resp.status
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("text", "") != "":
                        token_times.append(time.perf_counter())
        except Exception as exc:
            return {"ok": False, "error": str(exc), "ttft_s": float("nan")}
        t_end = time.perf_counter()
        ok = status == 200 and len(token_times) > 0
        ttft = (token_times[0] - t0) if token_times else float("nan")
        itls = [b - a for a, b in zip(token_times, token_times[1:])]
        return {
            "ok": ok,
            "status": status,
            "num_tokens": len(token_times),
            "ttft_s": ttft,
            "wall_s": t_end - t0,
            "itls_s": itls,
        }


async def _run_phase(
    base_url: str,
    prompts: list[list[int]],
    max_tokens: int,
    concurrency: int,
    timeout_s: float = 600.0,
) -> tuple[list[dict], dict, dict, float]:
    """Run one phase (cold or warm). Returns (results, before, after, wall)."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        before = await _fetch_spec_decode_counters(base_url, session)
        sem = asyncio.Semaphore(concurrency)
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                _send_streaming(session, base_url, p, max_tokens, sem, timeout_s)
                for p in prompts
            ]
        )
        t_end = time.perf_counter()
        after = await _fetch_spec_decode_counters(base_url, session)
    return list(results), before, after, t_end - t_start


def _summarize_phase(
    results: list[dict], before: dict, after: dict, wall: float, label: str
) -> dict:
    num_failed = sum(1 for r in results if not r["ok"])
    delta_drafts = after["counters"]["num_drafts"] - before["counters"]["num_drafts"]
    delta_draft_tokens = (
        after["counters"]["num_draft_tokens"] - before["counters"]["num_draft_tokens"]
    )
    delta_accepted = (
        after["counters"]["num_accepted_tokens"] - before["counters"]["num_accepted_tokens"]
    )
    ttfts = [r["ttft_s"] for r in results if r.get("ok") and "ttft_s" in r]
    itls = [itl for r in results if r.get("ok") for itl in r.get("itls_s", [])]
    total_streamed = sum(r.get("num_tokens", 0) for r in results if r.get("ok"))

    summary = {
        "label": label,
        "num_requests": len(results),
        "num_failed": num_failed,
        "wall_s": round(wall, 3),
        "delta_drafts": delta_drafts,
        "delta_draft_tokens": delta_draft_tokens,
        "delta_accepted": delta_accepted,
        "accepted_tok_per_s": round(delta_accepted / wall, 3) if wall > 0 else float("nan"),
        "streamed_tok_per_s": round(total_streamed / wall, 3) if wall > 0 else float("nan"),
        "total_streamed_tokens": total_streamed,
        "mean_acceptance_length": (
            round(1 + delta_accepted / delta_drafts, 3) if delta_drafts > 0 else float("nan")
        ),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts) * 1000, 1) if ttfts else float("nan"),
        "ttft_max_ms": round(max(ttfts) * 1000, 1) if ttfts else float("nan"),
        "ttft_per_req_ms": [round(t * 1000, 1) for t in ttfts],
        "itl_mean_ms": round(sum(itls) / len(itls) * 1000, 2) if itls else float("nan"),
    }
    return summary


def _gpu_mem_gib() -> float:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        return round(int(out.strip()) / 1024, 2)
    except Exception:
        return -1.0


async def _async_main(
    port: int, warm_repeats: int, timeout_s: float, fixture_key: str
) -> dict:
    base_url = f"http://127.0.0.1:{port}"

    fx = FIXTURES[fixture_key]
    fixture = fx["fixture"]
    prefix_len = fx["prefix_len"]
    ctx_label = f"{prefix_len // 1024}K"

    print(f"Loading {CONCURRENCY}x{ctx_label} prefixes from frozen fixture...", flush=True)
    prefixes = load_prompt_token_ids(fixture)[:CONCURRENCY]
    assert len(prefixes) == CONCURRENCY
    assert all(len(p) == prefix_len for p in prefixes)

    suffix = _build_suffix(SUFFIX_LEN)
    warm_prompts = [p + suffix for p in prefixes]
    assert all(len(p) == prefix_len + SUFFIX_LEN for p in warm_prompts)

    report: dict = {
        "config": {
            "model": MODEL,
            "fixture": fixture_key,
            "prefix_len": prefix_len,
            "suffix_len": SUFFIX_LEN,
            "warm_prompt_len": prefix_len + SUFFIX_LEN,
            "concurrency": CONCURRENCY,
            "cold_max_tokens": COLD_MAX_TOKENS,
            "warm_max_tokens": WARM_MAX_TOKENS,
            "warm_repeats": warm_repeats,
        },
        "gpu_mem_before_gib": _gpu_mem_gib(),
    }

    # Phase 1: cold populate
    print(f"\n--- Phase 1: COLD POPULATE (c={CONCURRENCY}, max_tokens={COLD_MAX_TOKENS}) ---", flush=True)
    results, before, after, wall = await _run_phase(
        base_url, prefixes, COLD_MAX_TOKENS, CONCURRENCY, timeout_s
    )
    cold_pop = _summarize_phase(results, before, after, wall, "cold_populate")
    report["cold_populate"] = cold_pop
    print(f"  wall={cold_pop['wall_s']}s, ttft_mean={cold_pop['ttft_mean_ms']}ms, "
          f"accepted_tok/s={cold_pop['accepted_tok_per_s']}", flush=True)
    print(f"  GPU mem: {_gpu_mem_gib()} GiB", flush=True)

    # Phase 2: warm repeats
    warm_summaries = []
    for rep in range(warm_repeats):
        print(f"\n--- Phase 2: WARM rep {rep+1}/{warm_repeats} "
              f"(c={CONCURRENCY}, max_tokens={WARM_MAX_TOKENS}, +{SUFFIX_LEN} suffix) ---", flush=True)
        results, before, after, wall = await _run_phase(
            base_url, warm_prompts, WARM_MAX_TOKENS, CONCURRENCY, timeout_s
        )
        warm = _summarize_phase(results, before, after, wall, f"warm_rep{rep+1}")
        warm_summaries.append(warm)
        print(f"  wall={warm['wall_s']}s, ttft_mean={warm['ttft_mean_ms']}ms, "
              f"ttft_max={warm['ttft_max_ms']}ms", flush=True)
        print(f"  accepted_tok/s={warm['accepted_tok_per_s']}, "
              f"streamed_tok/s={warm['streamed_tok_per_s']}, "
              f"mean_accept_len={warm['mean_acceptance_length']}", flush=True)
        print(f"  per-req TTFT(ms)={warm['ttft_per_req_ms']}", flush=True)
        print(f"  GPU mem: {_gpu_mem_gib()} GiB", flush=True)

    report["warm_reps"] = warm_summaries

    # Stable warm number: use the last repeat (most thermally stable)
    stable = warm_summaries[-1]
    report["warm_stable"] = {
        "accepted_tok_per_s": stable["accepted_tok_per_s"],
        "ttft_mean_ms": stable["ttft_mean_ms"],
        "ttft_max_ms": stable["ttft_max_ms"],
        "mean_acceptance_length": stable["mean_acceptance_length"],
    }

    # APC hit evidence
    report["apc_evidence"] = {
        "cold_populate_ttft_mean_ms": cold_pop["ttft_mean_ms"],
        "warm_ttft_mean_ms": stable["ttft_mean_ms"],
        "ttft_speedup_cold_over_warm": (
            round(cold_pop["ttft_mean_ms"] / stable["ttft_mean_ms"], 2)
            if stable["ttft_mean_ms"] > 0 else float("inf")
        ),
        "note": f"Warm TTFT should be dramatically lower than cold {ctx_label} prefill "
                f"(APC skips the {ctx_label} prefix, re-prefilling only the ~10K suffix).",
    }

    # Comparison
    ours_warm = fx["ours_warm_tok_s"]
    native_cold_ref = fx["native_cold_ref_tok_s"]
    native_warm = stable["accepted_tok_per_s"]
    report["comparison"] = {
        "ours_warm_tok_s": ours_warm,
        "native_warm_tok_s": native_warm,
        "native_cold_tok_s_ref": native_cold_ref,
        "ours_warm_vs_native_warm": (
            round(ours_warm / native_warm, 3) if native_warm > 0 else float("nan")
        ),
        "ours_warm_vs_native_cold": (
            round(ours_warm / native_cold_ref, 3)
            if native_cold_ref else float("nan")
        ),
        "native_warm_vs_native_cold": (
            round(native_warm / native_cold_ref, 3)
            if native_cold_ref and native_cold_ref > 0 else float("nan")
        ),
    }

    report["gpu_mem_after_gib"] = _gpu_mem_gib()

    # Scrape prefix cache metrics if available
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pcm = await _fetch_prefix_cache_metrics(base_url, session)
    if pcm:
        report["prefix_cache_metrics"] = pcm

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--warm-repeats", type=int, default=WARM_REPEATS)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default="ctx128k",
        help="Frozen prefix fixture: ctx64k (65536) or ctx128k (131072, default).",
    )
    args = parser.parse_args()

    report = asyncio.run(
        _async_main(args.port, args.warm_repeats, args.timeout_s, args.fixture)
    )

    ctx_label = f"{report['config']['prefix_len'] // 1024}K"
    print(f"\n{'='*78}")
    print(f"RESULTS — Native vLLM WARM prefix-cache-hit, {ctx_label}/c={CONCURRENCY}")
    print(f"{'='*78}")
    ws = report["warm_stable"]
    comp = report["comparison"]
    apc = report["apc_evidence"]
    print(f"  Native WARM accepted tok/s : {ws['accepted_tok_per_s']}")
    print(f"  Native WARM TTFT (mean)    : {ws['ttft_mean_ms']} ms")
    print(f"  Native WARM TTFT (max)     : {ws['ttft_max_ms']} ms")
    print(f"  Mean acceptance length     : {ws['mean_acceptance_length']}")
    print(f"  Cold populate TTFT (mean)  : {apc['cold_populate_ttft_mean_ms']} ms")
    print(f"  TTFT speedup (cold/warm)   : {apc['ttft_speedup_cold_over_warm']}x")
    print(f"  GPU mem peak               : {report['gpu_mem_after_gib']} GiB")
    print()
    print(f"  Our WARM tok/s             : {comp['ours_warm_tok_s']}")
    print(f"  Native WARM tok/s          : {comp['native_warm_tok_s']}")
    print(f"  Ours / Native WARM         : {comp['ours_warm_vs_native_warm']}x")
    print(f"  Native COLD ref            : {comp['native_cold_tok_s_ref']}")
    print(f"  Ours WARM / Native COLD    : {comp['ours_warm_vs_native_cold']}x")
    print(f"  Native WARM / Native COLD  : {comp['native_warm_vs_native_cold']}x")
    print(f"{'='*78}")

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
