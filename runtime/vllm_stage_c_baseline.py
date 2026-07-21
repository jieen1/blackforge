"""Stage C of the 2026-07-16 three-stage ownership-transfer ladder (see
notes/direct-model-runner-design.md for Stage A/B).

Stage A: vLLM metadata + vLLM cache -- in-process oracle, 20/20 correct.
Stage B: vLLM metadata + our own 4-slot cache -- 20/20 correct, cache
exonerated.
Stage C (this module): our own hand-built attention/GDN metadata
(``runtime.direct_model_runner.build_attention_metadata`` /
``build_gdn_metadata`` -- the SAME functions ``DirectModelRunner`` itself
calls, not a second copy) + Stage B's cache. Real vLLM Scheduler,
``GPUModelRunner``, warmup sequencing are still untouched; only the two
real metadata builders' ``.build()`` methods are monkey-patched to call our
own field-construction logic instead, deriving the few facts it needs
(new-token count, prior context length, prefill-vs-decode) from vLLM's own
real, scheduler-computed ``CommonAttentionMetadata`` rather than
re-deriving them independently -- this isolates ONE variable (our
metadata-construction logic itself), not "does our bookkeeping also track
requests correctly."

Single request, slot 0, matching every earlier stage's scope.
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
NUM_SLOTS = 4
BLOCK_SIZE = 16
BLOCKS_PER_SLOT = 128
SLOT = 0  # single request, always slot 0, matching stages A/B

sys.path.insert(0, SM120_VLLM_INTEGRATION)
import register_sm120_backend  # noqa: E402,F401


def _install_stage_c_patch() -> None:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadataBuilder
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from runtime.direct_model_runner import (
        allocate_fixed_slot_kv_caches,
        build_attention_metadata,
        build_gdn_metadata,
    )

    # Reuse Stage B's cache substitution -- Stage C only adds the metadata
    # substitution on top, per the ladder's "change one dimension at a time"
    # rule.
    def patched_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        del kernel_block_sizes
        static_forward_context = self.compilation_config.static_forward_context
        return allocate_fixed_slot_kv_caches(
            static_forward_context,
            self.vllm_config,
            self.device,
            num_slots=NUM_SLOTS,
            block_size=BLOCK_SIZE,
            blocks_per_slot=BLOCKS_PER_SLOT,
        )

    GPUModelRunner.initialize_kv_cache_tensors = patched_initialize_kv_cache_tensors

    def _derive_step_facts(common_attn_metadata):
        """Extract exactly the facts our hand-built metadata needs from
        vLLM's own real, scheduler-computed CommonAttentionMetadata --
        single request assumed (index 0), matching every stage's scope."""
        num_new_tokens = int(common_attn_metadata.num_actual_tokens)
        total_seq_len = int(common_attn_metadata.seq_lens[0].item())
        prior_kv_len = total_seq_len - num_new_tokens
        is_decode = num_new_tokens == 1
        return prior_kv_len, num_new_tokens, is_decode

    def patched_sm120_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        prior_kv_len, num_new_tokens, is_decode = _derive_step_facts(common_attn_metadata)
        return build_attention_metadata(
            prior_kv_len=prior_kv_len,
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=SLOT,
            block_size=BLOCK_SIZE,
            blocks_per_slot=BLOCKS_PER_SLOT,
            device=common_attn_metadata.query_start_loc.device,
        )

    SM120GQAMetadataBuilder.build = patched_sm120_build

    _gdn_slot_initialized = {"value": False}

    def patched_gdn_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        del common_prefix_len, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build
        prior_kv_len, num_new_tokens, is_decode = _derive_step_facts(common_attn_metadata)
        meta = build_gdn_metadata(
            slot_initialized=_gdn_slot_initialized["value"],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=SLOT,
            device=common_attn_metadata.query_start_loc.device,
        )
        if prior_kv_len == 0 and not is_decode:
            _gdn_slot_initialized["value"] = True
        return meta

    GDNAttentionMetadataBuilder.build = patched_gdn_build


_install_stage_c_patch()


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
        [sys.executable, "-m", "runtime.vllm_stage_c_baseline", "--single-run-json"],
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
            print(f"  stderr tail: {r['stderr_tail'][-800:]}")
    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
