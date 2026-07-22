"""Profile native vLLM (FlashInfer) decode steps using in-process torch.profiler.

Uses VLLM_ENABLE_V1_MULTIPROCESSING=0 to run EngineCore in-process,
allowing torch.profiler to capture GPU kernel breakdown directly.

SAME methodology as benchmarks/decode_step_profile.py for apples-to-apples comparison.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.native_decode_step_profile --fixture ctx128k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
SUFFIX_LEN = 10240
COLD_MAX_TOKENS = 16
WARM_MAX_TOKENS = 256
CONCURRENCY = 4
NUM_PROFILER_STEPS = 5


def _categorize_kernel(name: str) -> str:
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


def _analyze_profiler(prof, num_steps: int) -> dict:
    events = prof.key_averages()
    categories = {"attention": 0.0, "gemm": 0.0, "gdn": 0.0, "other_compute": 0.0, "other": 0.0}
    total_cuda_ms = 0.0
    top_kernels = []

    for evt in events:
        if evt.device_time_total > 0:
            cuda_ms = evt.device_time_total / 1000.0
            cat = _categorize_kernel(evt.key)
            categories[cat] += cuda_ms
            total_cuda_ms += cuda_ms
            top_kernels.append({"name": evt.key, "total_ms": round(cuda_ms, 3),
                                "count": evt.count, "category": cat})

    top_kernels.sort(key=lambda x: x["total_ms"], reverse=True)

    return {
        "total_cuda_ms": round(total_cuda_ms, 3),
        "per_step_cuda_ms": round(total_cuda_ms / num_steps, 3) if num_steps > 0 else 0,
        "categories": {k: round(v, 3) for k, v in categories.items()},
        "pct": {k: round(v / total_cuda_ms * 100, 1) if total_cuda_ms > 0 else 0
                for k, v in categories.items()},
        "top_15_kernels": top_kernels[:15],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=["ctx128k", "ctx64k"], default="ctx128k")
    parser.add_argument("--gpu-mem-util", type=float, default=0.92)
    args = parser.parse_args()

    from benchmarks.workloads import CTX128K_FIXTURE, D1_CTX64K_FIXTURE, load_prompt_token_ids

    fixture_map = {
        "ctx128k": (CTX128K_FIXTURE, "128K"),
        "ctx64k": (D1_CTX64K_FIXTURE, "64K"),
    }
    fixture, label = fixture_map[args.fixture]
    P = fixture.prompt_len

    print(f"=== native_decode_step_profile: {label}/c={CONCURRENCY} ===")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}")

    prompts = load_prompt_token_ids(fixture)
    prefixes = [prompts[i] for i in range(CONCURRENCY)]
    suffix = _build_suffix(SUFFIX_LEN)

    print("\n--- Creating in-process LLM engine ---")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        attention_backend="FLASHINFER",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        max_num_seqs=CONCURRENCY,
        max_model_len=262144,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=args.gpu_mem_util,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "attention_backend": "FLASHINFER",
        },
        optimization_level="3",
        language_model_only=True,
    )

    import torch
    print(f"GPU mem after load: {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated, "
          f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved")

    sampling_cold = SamplingParams(max_tokens=COLD_MAX_TOKENS, temperature=0)
    sampling_warm = SamplingParams(max_tokens=WARM_MAX_TOKENS, temperature=0)

    print(f"\n--- Cold populate {CONCURRENCY}x{P} prefixes ---")
    t0 = time.perf_counter()
    cold_outputs = llm.generate([p for p in prefixes], sampling_cold)
    cold_time = time.perf_counter() - t0
    for i, out in enumerate(cold_outputs):
        print(f"  Cold [{i}]: {len(out.prompt_token_ids)} prompt tokens, "
              f"{len(out.outputs[0].token_ids)} generated")
    print(f"  Cold populate time: {cold_time:.1f}s")

    print(f"\n--- Warm decode (no profiler): {CONCURRENCY}x({P}+{SUFFIX_LEN}) ---")
    warm_prompts = [p + suffix for p in prefixes]
    t0 = time.perf_counter()
    warm_outputs = llm.generate(warm_prompts, sampling_warm)
    warm_time = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in warm_outputs)
    print(f"  Warm decode: {total_tokens} tokens in {warm_time:.2f}s = {total_tokens/warm_time:.1f} tok/s")

    print(f"\n--- Profiled warm decode ({NUM_PROFILER_STEPS} steps under torch.profiler) ---")
    warm_prompts2 = [p + suffix for p in prefixes]

    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        profiled_outputs = llm.generate(warm_prompts2, sampling_warm)
        torch.cuda.synchronize()

    profiled_tokens = sum(len(o.outputs[0].token_ids) for o in profiled_outputs)
    print(f"  Profiled: {profiled_tokens} tokens")

    profiler_analysis = _analyze_profiler(prof, 1)

    print(f"\n{'='*78}")
    print(f"NATIVE vLLM (FlashInfer) — {label}/c={CONCURRENCY}")
    print(f"{'='*78}")
    print(f"Warm throughput: {total_tokens/warm_time:.1f} tok/s")
    print(f"Total CUDA time (profiled): {profiler_analysis['total_cuda_ms']:.1f} ms")
    print("\nKernel breakdown:")
    for cat, ms in profiler_analysis["categories"].items():
        pct = profiler_analysis["pct"][cat]
        print(f"  {cat:>15s}: {ms:>10.1f} ms  ({pct:>5.1f}%)")
    print("\nTop 15 kernels:")
    for i, k in enumerate(profiler_analysis["top_15_kernels"]):
        print(f"  {i+1:>2}. [{k['total_ms']:>8.1f}ms, {k['count']:>5}x] [{k['category']:>12s}] {k['name'][:80]}")

    result = {
        "label": f"native_flashinfer_{label}_c{CONCURRENCY}",
        "fixture": args.fixture,
        "P": P,
        "concurrency": CONCURRENCY,
        "suffix_len": SUFFIX_LEN,
        "warm_tok_s": round(total_tokens / warm_time, 2),
        "warm_time_s": round(warm_time, 3),
        "total_tokens": total_tokens,
        "kernel_analysis": profiler_analysis,
    }
    print(f"\n{'='*78}")
    print(json.dumps(result, indent=2, default=str))

    out_path = f"/tmp/native_profile_{args.fixture}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
