"""No-HTTP correct baseline: drives the REAL vLLM `LLM` class in-process.

Per the 2026-07-16 strategy reset (see notes/direct-model-runner-design.md):
after `runtime/direct_model_runner.py`'s hand-built attention/GDN metadata
approach produced a 100%-deterministic wrong answer (confirmed via 20/20
identical failures in benchmarks/single_prefill_regression.py), the
hypothesis shifted from "a low-level kernel bug" to "our own runner skips
real GPUModelRunner contract semantics" (padded/unpadded token counts,
persistent buffers, real metadata builders, warmup ordering, GDN state
batch bookkeeping).

Rather than re-deriving `GPUModelRunner`'s internals by hand, this module
uses vLLM's own `LLM` class directly. This is a genuinely different code
path from `direct_model_runner.py` -- real `Scheduler`, real
`KVCacheManager`-allocated cache, the real `SM120GQAMetadataBuilder` /
`GDNAttentionMetadataBuilder`, real warmup/profiling sequencing -- not a
re-implementation of any of it. `LLM` is not "HTTP with the server part
skipped": it never had an HTTP layer to begin with (that only exists for
`vllm serve`). Under the hood it uses `SyncMPClient` (a background
EngineCore process reached via local ZMQ, not a network call) -- still
"no HTTP" in every sense relevant to this project's redirect.

Confirmed empirically (2026-07-16): the *same* SM120GQABackend (decode v2
included) driven this way produces the correct " Paris" completion for
"The capital of France is" -- proving the kernels/backend themselves are
fine, and the bug in `direct_model_runner.py` is specifically in its own
hand-rolled orchestration, not in anything downstream of it.

Usage:
    ./.venv/bin/python -m runtime.vllm_inprocess_baseline
    ./.venv/bin/python -m runtime.vllm_inprocess_baseline --repeat 20
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")

SM120_VLLM_INTEGRATION = os.environ.get("SM120_VLLM_INTEGRATION", "")
EXPECTED_PREFIX = " Paris"
PROMPT = "The capital of France is"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"

# Side-effecting import, deliberately at true top level (unconditional, not
# inside `if __name__ == "__main__"`): vLLM's SyncMPClient spawns the
# EngineCore worker via multiprocessing's "spawn" start method, which
# re-executes this file from scratch in the child process. The
# registration must run again there too, or AttentionBackendEnum.CUSTOM
# resolves in the parent but not in the child that actually runs the
# model. See launch_test_server.py's own docstring for the identical
# reasoning (confirmed the hard way there first).
sys.path.insert(0, SM120_VLLM_INTEGRATION)
import register_sm120_backend  # noqa: E402,F401


def run_once(max_tokens: int = 8) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    llm = LLM(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        attention_backend=AttentionBackendEnum.CUSTOM,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        language_model_only=True,
    )
    outputs = llm.generate([PROMPT], SamplingParams(max_tokens=max_tokens, temperature=0.0))
    text = outputs[0].outputs[0].text
    return {"text": text, "passed": text.strip().startswith(EXPECTED_PREFIX.strip())}


def _run_subprocess() -> dict:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "runtime.vllm_inprocess_baseline", "--single-run-json"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = run_once()
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = run_once()
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess() for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}  text={r.get('text')!r}")
    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
