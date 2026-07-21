"""Profile native vLLM (FlashInfer) decode steps using nsys.

Two modes:
  1. Server mode (default): Launch server, cold-populate, nsys-trace the GPU
     worker during warm decode, parse kernel breakdown.
  2. In-process mode (--in-process): Use vLLM LLM engine with torch profiler
     config, parse the resulting trace.

Reports kernel breakdown using the SAME categorization as decode_step_profile.py.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.native_nsys_profile --fixture ctx128k
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PORT = 8199
NSYS_OUTPUT = "/tmp/native_decode_profile"
SUFFIX_LEN = 10240
COLD_MAX_TOKENS = 16
WARM_MAX_TOKENS = 256
CONCURRENCY = 4


def _categorize_kernel(name: str) -> str:
    """Categorize a CUDA kernel name — SAME logic as decode_step_profile.py."""
    name_lower = name.lower()
    if any(k in name_lower for k in ["delta_rule", "gating", "ssm", "gdn",
                                      "fused_sigmoid", "recurrent"]):
        return "gdn"
    if any(k in name_lower for k in ["gqa", "flash", "fmha", "attention",
                                      "splitkv", "split_kv", "paged",
                                      "sm120", "decode_kernel",
                                      "batch_decode", "single_decode",
                                      "batch_prefill", "ragged",
                                      "flashinfer"]):
        return "attention"
    if any(k in name_lower for k in ["gemm", "cutlass", "cublas", "nvfp4",
                                      "matmul", "mma", "warp", "sm90_xmma",
                                      "ampere_", "ffma", "hmma"]):
        return "gemm"
    if any(k in name_lower for k in ["embedding", "layer_norm", "layernorm",
                                      "rmsnorm", "silu", "gelu", "softmax",
                                      "elementwise", "vectorized", "copy",
                                      "fill", "cat_", "index"]):
        return "other_compute"
    return "other"


def _build_suffix(length: int) -> list[int]:
    base = 100000
    return [(base + i * 7) % 151665 for i in range(length)]


def _find_vllm_processes(port: int) -> list[int]:
    """Find all vLLM processes related to our server."""
    result = subprocess.run(
        ["pgrep", "-f", f"vllm"],
        capture_output=True, text=True,
    )
    pids = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        pid = int(line.strip())
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
            if str(port) in cmdline or "EngineCore" in cmdline or "engine_core" in cmdline:
                pids.append(pid)
        except (FileNotFoundError, PermissionError):
            continue
    return pids


def _find_gpu_worker_pid(port: int) -> int:
    """Find the EngineCore GPU worker process."""
    result = subprocess.run(
        ["pgrep", "-f", "EngineCore"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            return int(line.strip())
    pids = _find_vllm_processes(port)
    return pids[-1] if pids else -1


def _launch_server(port: int) -> subprocess.Popen:
    """Launch native vLLM server with FlashInfer + MTP."""
    env = os.environ.copy()
    env["USE_LIBUV"] = "0"
    env["CUDA_HOME"] = "/usr/local/cuda-13.3"
    env["PATH"] = f"/home/bot/.venvs/vllm/bin:{env['CUDA_HOME']}/bin:{env['PATH']}"

    cmd = [
        "/home/bot/.venvs/vllm/bin/vllm", "serve", MODEL,
        "-O3",
        "--served-model-name", "qwen3.6-sm120-test",
        "--language-model-only",
        "--max-num-seqs", "4",
        "--max-model-len", "262144",
        "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.92",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--attention-backend", "FLASHINFER",
        "--kv-cache-dtype", "fp8_e4m3",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--speculative-config", json.dumps({
            "method": "mtp",
            "num_speculative_tokens": 3,
            "attention_backend": "FLASHINFER",
        }),
    ]
    print(f"Launching native FlashInfer server on port {port}...")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def _wait_for_server(port: int, timeout: int = 300) -> bool:
    import aiohttp
    async def _check():
        deadline = time.time() + timeout
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(3)
        return False
    return asyncio.run(_check())


async def _cold_populate(port: int, prefixes: list[list[int]]) -> None:
    import aiohttp
    async def _one(session, prefix, idx):
        payload = {
            "model": "qwen3.6-sm120-test",
            "prompt": prefix,
            "max_tokens": COLD_MAX_TOKENS,
            "temperature": 0,
        }
        async with session.post(
            f"http://127.0.0.1:{port}/v1/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            data = await resp.json()
            usage = data.get("usage", {})
            print(f"  Cold [{idx}]: {usage.get('prompt_tokens', '?')} prompt tokens")
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[_one(session, p, i) for i, p in enumerate(prefixes)])


async def _warm_decode(port: int, prefixes: list[list[int]], suffix: list[int]) -> dict:
    import aiohttp
    results = {"tokens": [], "times": []}
    async def _one(session, prefix, idx):
        prompt = prefix + suffix
        payload = {
            "model": "qwen3.6-sm120-test",
            "prompt": prompt,
            "max_tokens": WARM_MAX_TOKENS,
            "temperature": 0,
            "stream": True,
        }
        t0 = time.perf_counter()
        token_count = 0
        async with session.post(
            f"http://127.0.0.1:{port}/v1/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        d = json.loads(line[6:])
                        choices = d.get("choices", [])
                        if choices and choices[0].get("text"):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
        elapsed = time.perf_counter() - t0
        results["tokens"].append(token_count)
        results["times"].append(elapsed)
        print(f"  Warm [{idx}]: {token_count} tokens in {elapsed:.2f}s")
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[_one(session, p, i) for i, p in enumerate(prefixes)])
    total_tokens = sum(results["tokens"])
    max_time = max(results["times"]) if results["times"] else 1
    return {
        "total_tokens": total_tokens,
        "wall_time_s": round(max_time, 3),
        "aggregate_tok_s": round(total_tokens / max_time, 2) if max_time > 0 else 0,
    }


def _stop_server(port: int) -> None:
    subprocess.run(
        ["/home/bot/.venvs/vllm/bin/python",
         "/home/bot/project/sm120-flash-attention/vllm_integration/stop_test_server.py",
         "--port", str(port)],
        capture_output=True, timeout=30,
    )
    time.sleep(2)


def _parse_nsys_csv(csv_path: str) -> dict:
    """Parse nsys cuda_gpu_kern_sum CSV into categorized breakdown."""
    categories = {"attention": 0.0, "gemm": 0.0, "gdn": 0.0, "other_compute": 0.0, "other": 0.0}
    total_cuda_ms = 0.0
    top_kernels = []

    with open(csv_path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            time_ns = float(parts[1])
            count = int(parts[2])
            name = ",".join(parts[3:]).strip('"')
        except (ValueError, IndexError):
            continue

        time_ms = time_ns / 1_000_000
        cat = _categorize_kernel(name)
        categories[cat] += time_ms
        total_cuda_ms += time_ms
        top_kernels.append({"name": name, "total_ms": round(time_ms, 3), "count": count, "category": cat})

    top_kernels.sort(key=lambda x: x["total_ms"], reverse=True)

    return {
        "total_cuda_ms": round(total_cuda_ms, 3),
        "categories": {k: round(v, 3) for k, v in categories.items()},
        "pct": {k: round(v / total_cuda_ms * 100, 1) if total_cuda_ms > 0 else 0
                for k, v in categories.items()},
        "top_15_kernels": top_kernels[:15],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile native vLLM decode with nsys")
    parser.add_argument("--fixture", choices=["ctx128k", "ctx64k"], default="ctx128k")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--nsys-duration", type=int, default=60)
    args = parser.parse_args()

    port = args.port

    from benchmarks.workloads import CTX128K_FIXTURE, D1_CTX64K_FIXTURE, load_prompt_token_ids

    fixture_map = {
        "ctx128k": (CTX128K_FIXTURE, "128K"),
        "ctx64k": (D1_CTX64K_FIXTURE, "64K"),
    }
    fixture, label = fixture_map[args.fixture]
    P = fixture.prompt_len

    print(f"=== native_nsys_profile: {label}/c={CONCURRENCY} ===")

    prompts = load_prompt_token_ids(fixture)
    prefixes = [prompts[i] for i in range(CONCURRENCY)]
    suffix = _build_suffix(SUFFIX_LEN)

    server_proc = _launch_server(port)
    print("Waiting for server...")
    if not _wait_for_server(port):
        print("ERROR: Server did not start")
        server_proc.kill()
        return 1
    print("Server ready!")

    try:
        print(f"\n--- Cold populate {CONCURRENCY}x{P} prefixes ---")
        asyncio.run(_cold_populate(port, prefixes))

        print(f"\n--- Finding GPU worker process ---")
        worker_pid = _find_gpu_worker_pid(port)
        print(f"  Worker PID: {worker_pid}")

        if worker_pid <= 0:
            print("ERROR: Could not find GPU worker process")
            return 1

        nsys_report = f"{NSYS_OUTPUT}.nsys-rep"
        for f in glob.glob(f"{NSYS_OUTPUT}*"):
            os.remove(f)

        print(f"\n--- Starting nsys on PID {worker_pid}, duration={args.nsys_duration}s ---")
        nsys_proc = subprocess.Popen(
            ["nsys", "profile",
             "--pid", str(worker_pid),
             "--output", NSYS_OUTPUT,
             "--force-overwrite", "true",
             "--duration", str(args.nsys_duration),
             "--trace", "cuda",
             "--sample", "none",
             "--cpuctxsw", "none"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(3)

        print(f"\n--- Warm decode: {CONCURRENCY}x({P}+{SUFFIX_LEN}) ---")
        warm_result = asyncio.run(_warm_decode(port, prefixes, suffix))
        print(f"  Aggregate: {warm_result['aggregate_tok_s']:.1f} tok/s")

        print(f"\n  Waiting for nsys to finish...")
        try:
            nsys_proc.wait(timeout=args.nsys_duration + 60)
        except subprocess.TimeoutExpired:
            nsys_proc.send_signal(signal.SIGINT)
            try:
                nsys_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                nsys_proc.kill()

        if not os.path.exists(nsys_report):
            print(f"ERROR: nsys report not found at {nsys_report}")
            stderr = nsys_proc.stderr.read() if nsys_proc.stderr else ""
            print(f"  nsys stderr: {stderr[:500]}")
            return 1

        print(f"\n--- Parsing nsys kernel breakdown ---")
        csv_path = NSYS_OUTPUT + "_cuda_gpu_kern_sum.csv"
        stats_proc = subprocess.run(
            ["nsys", "stats",
             "--report", "cuda_gpu_kern_sum",
             "--format", "csv",
             "--output", NSYS_OUTPUT,
             nsys_report],
            capture_output=True, text=True, timeout=120,
        )

        if not os.path.exists(csv_path):
            for candidate in glob.glob(f"{NSYS_OUTPUT}*.csv"):
                csv_path = candidate
                break

        if not os.path.exists(csv_path):
            print(f"ERROR: nsys stats CSV not found")
            print(f"  stats stderr: {stats_proc.stderr[:500]}")
            return 1

        kernel_analysis = _parse_nsys_csv(csv_path)

        print(f"\n{'='*78}")
        print(f"NATIVE vLLM (FlashInfer) — {label}/c={CONCURRENCY}")
        print(f"{'='*78}")
        print(f"Aggregate throughput: {warm_result['aggregate_tok_s']:.1f} tok/s")
        print(f"Total CUDA time captured: {kernel_analysis.get('total_cuda_ms', 0):.1f} ms")
        print(f"\nKernel breakdown:")
        for cat, ms in kernel_analysis.get("categories", {}).items():
            pct = kernel_analysis.get("pct", {}).get(cat, 0)
            print(f"  {cat:>15s}: {ms:>10.1f} ms  ({pct:>5.1f}%)")
        print(f"\nTop 15 kernels:")
        for i, k in enumerate(kernel_analysis.get("top_15_kernels", [])):
            print(f"  {i+1:>2}. [{k['total_ms']:>8.1f}ms, {k['count']:>5}x] [{k['category']:>12s}] {k['name'][:80]}")

        result = {
            "label": f"native_flashinfer_{label}_c{CONCURRENCY}",
            "fixture": args.fixture,
            "P": P,
            "concurrency": CONCURRENCY,
            "suffix_len": SUFFIX_LEN,
            "warm_throughput": warm_result,
            "kernel_analysis": kernel_analysis,
        }
        print(f"\n{'='*78}")
        print(json.dumps(result, indent=2, default=str))

        out_path = f"/tmp/native_profile_{args.fixture}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")

    finally:
        print("\nStopping server...")
        _stop_server(port)
        try:
            server_proc.kill()
            server_proc.wait(timeout=10)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
