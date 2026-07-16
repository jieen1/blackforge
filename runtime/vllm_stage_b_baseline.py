"""Stage B of the 2026-07-16 three-stage ownership-transfer ladder (see
notes/direct-model-runner-design.md's "Step 2 result" section for Stage A,
the no-HTTP baseline this stage is compared against).

Stage A (``runtime/vllm_inprocess_baseline.py``): vLLM metadata + vLLM
cache -- the in-process oracle, verified 20/20 correct.

Stage B (this module): vLLM's real Scheduler/GPUModelRunner/metadata
builders/warmup sequence, UNCHANGED -- the only thing swapped out is the
actual KV (attention) and state (GDN) cache TENSOR allocation, replaced
with our own fixed-4-slot tensors from
``runtime.direct_model_runner.allocate_fixed_slot_kv_caches`` (the exact
same helper ``DirectModelRunner`` itself uses). This isolates one single
variable: does our own cache shape/binding/slot-addressing scheme work
correctly when real vLLM scheduling (not our hand-built metadata) drives
it?

Implementation: monkey-patches ``GPUModelRunner.initialize_kv_cache_tensors``
-- the one real vLLM method responsible for allocating+binding cache
tensors (see its own source: it ends with the same ``bind_kv_cache()`` call
our code already reuses) -- to call our allocator instead of vLLM's own
``_allocate_kv_cache_tensors``/``_reshape_kv_cache_tensors``. Every other
method on the real path (``initialize_attn_backend``,
``initialize_metadata_builders``, ``may_reinitialize_input_batch``, the
real ``Scheduler``, the real warmup) is untouched.

Usage:
    ./.venv/bin/python -m runtime.vllm_stage_b_baseline
    ./.venv/bin/python -m runtime.vllm_stage_b_baseline --repeat 20
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
EXPECTED_PREFIX = " Paris"
PROMPT = "The capital of France is"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
NUM_SLOTS = 4
BLOCK_SIZE = 16
BLOCKS_PER_SLOT = 128

# Both side-effecting, deliberately at true top level (see
# vllm_inprocess_baseline.py's identical note): vLLM's SyncMPClient spawns
# the EngineCore worker via multiprocessing's "spawn" start method, which
# re-executes this file from scratch in the child process, so both the
# backend registration and the monkeypatch below must apply there too, not
# just in the parent.
sys.path.insert(0, SM120_VLLM_INTEGRATION)
import register_sm120_backend  # noqa: E402,F401


def _install_stage_b_patch() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from runtime.direct_model_runner import allocate_fixed_slot_kv_caches

    def patched_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        # kernel_block_sizes unused: our allocator derives shapes from the
        # real layer objects (get_kv_cache_shape/get_state_shape), not this.
        del kernel_block_sizes
        static_forward_context = self.compilation_config.static_forward_context
        kv_caches = allocate_fixed_slot_kv_caches(
            static_forward_context,
            self.vllm_config,
            self.device,
            num_slots=NUM_SLOTS,
            block_size=BLOCK_SIZE,
            blocks_per_slot=BLOCKS_PER_SLOT,
        )
        return kv_caches

    GPUModelRunner.initialize_kv_cache_tensors = patched_initialize_kv_cache_tensors


_install_stage_b_patch()


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
        block_size=BLOCK_SIZE,
    )
    outputs = llm.generate([PROMPT], SamplingParams(max_tokens=max_tokens, temperature=0.0))
    text = outputs[0].outputs[0].text
    return {"text": text, "passed": text.strip().startswith(EXPECTED_PREFIX.strip())}


def _run_subprocess() -> dict:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "runtime.vllm_stage_b_baseline", "--single-run-json"],
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
        "stderr_tail": proc.stderr[-4000:],
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
        if not r.get("passed") and r.get("stderr_tail"):
            print(f"  stderr tail: {r['stderr_tail'][-500:]}")
    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
