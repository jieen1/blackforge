#!/usr/bin/env python3
"""Laguna-S-2.1-NVFP4 · vLLM 长上下文基准（32K/64K/128K/200K × 3 runs + profiling）

官方推荐参数（model card）：
  - gpu_memory_utilization: 0.85
  - max_model_len: 262144
  - max_num_seqs: 32
  - sampling: temperature=0.7, top_p=0.95, top_k=20
  - DFlash: num_speculative_tokens=15
  - 官方性能（GB10）：prefill 600-800 tok/s, decode 13-14 tok/s (no spec)

本脚本用 greedy (temperature=0) 测速度基准（保证 token 数一致），
同时记录官方推荐采样参数供后续质量测试使用。

Usage:
    USE_LIBUV=0 /home/bot/.venvs/vllm/bin/python benchmarks/laguna_vllm_baseline.py
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("MAX_JOBS", "4")

MODEL = "poolside/Laguna-S-2.1-NVFP4"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks/fixtures/laguna_vllm_baseline.json",
)
CTX_LENGTHS = [32768, 65536, 131072, 200000]
NUM_RUNS = 3
DECODE_TOKENS = 128  # output tokens per run (enough to measure stable ITL)


def get_gpu_mem_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return -1.0


def make_prompt(tokenizer, target_tokens: int) -> str:
    """Generate a prompt that tokenizes to approximately target_tokens."""
    base_sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
    words_needed = int(target_tokens * 0.8)
    text = base_sentence * (words_needed // len(base_sentence.split()) + 2)
    words = text.split()
    # Binary search for right word count
    lo, hi = 1, len(words)
    best_n = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        n_tok = len(tokenizer.encode(candidate))
        if n_tok < target_tokens:
            lo = mid + 1
        else:
            best_n = mid
            hi = mid - 1
    # Fine-tune
    result_words = words[:best_n]
    result = " ".join(result_words)
    actual = len(tokenizer.encode(result))
    # Add/remove words to get closer
    idx = best_n
    while actual < target_tokens and idx < len(words):
        result_words.append(words[idx])
        idx += 1
        result = " ".join(result_words)
        actual = len(tokenizer.encode(result))
    return result


def measure_one_run(llm, prompt: str, max_tokens: int):
    """Measure prefill + decode timing for one run."""
    from vllm import SamplingParams
    import torch

    params_1 = SamplingParams(max_tokens=1, temperature=0)
    params_full = SamplingParams(max_tokens=max_tokens, temperature=0)

    # Prefill (TTFT)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([prompt], params_1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    # Full generation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], params_full)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    n_out = len(out[0].outputs[0].token_ids)
    decode_time = total - ttft
    itl = decode_time / max(n_out - 1, 1)

    return {
        "ttft_ms": ttft * 1000,
        "total_ms": total * 1000,
        "decode_ms": decode_time * 1000,
        "output_tokens": n_out,
        "itl_ms": itl * 1000,
        "decode_tps": 1.0 / itl if itl > 0 else 0,
    }


def profile_kernels(llm, prompt: str, max_tokens: int):
    """Get kernel-level breakdown using torch profiler."""
    from vllm import SamplingParams
    from torch.profiler import profile, ProfilerActivity

    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    # Warmup
    llm.generate([prompt], params)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        llm.generate([prompt], params)

    events = prof.key_averages()
    kernel_times = {}
    total_cuda_us = 0
    for evt in events:
        if evt.device_time_total > 0:
            name = evt.key.lower()
            if any(k in name for k in ["nvfp4", "cutlass", "gemm", "mma", "wgmma"]):
                cat = "gemm"
            elif any(k in name for k in ["flash", "fmha", "attention", "sdpa"]):
                cat = "attention"
            elif any(k in name for k in ["gdn", "mamba", "ssm", "conv1d", "selective"]):
                cat = "gdn_ssm"
            elif any(k in name for k in ["moe", "expert", "topk", "gate"]):
                cat = "moe_routing"
            elif any(k in name for k in ["rms_norm", "layernorm", "norm"]):
                cat = "norm"
            elif any(k in name for k in ["silu", "gelu", "act", "elementwise"]):
                cat = "activation"
            elif any(k in name for k in ["copy", "memcpy", "memset", "cat"]):
                cat = "memcpy"
            elif any(k in name for k in ["quant", "dequant", "fp8", "scale"]):
                cat = "quant"
            elif any(k in name for k in ["rope", "rotary", "pos"]):
                cat = "rope"
            elif any(k in name for k in ["sample", "top_k", "top_p", "sort"]):
                cat = "sampling"
            else:
                cat = "other"
            kernel_times[cat] = kernel_times.get(cat, 0) + evt.device_time_total
            total_cuda_us += evt.device_time_total

    kernel_pct = {}
    if total_cuda_us > 0:
        kernel_pct = {k: round(v / total_cuda_us * 100, 1)
                      for k, v in sorted(kernel_times.items(), key=lambda x: -x[1])}

    top_kernels = [
        {"name": evt.key[:100], "cuda_ms": round(evt.device_time_total / 1000, 2), "count": evt.count}
        for evt in sorted(events, key=lambda e: -e.cuda_time_total)[:15]
        if evt.device_time_total > 0
    ]

    return {
        "total_cuda_ms": round(total_cuda_us / 1000, 2),
        "kernel_pct": kernel_pct,
        "top_kernels": top_kernels,
    }


def main():
    results = {
        "_comment": "Laguna-S-2.1-NVFP4 vLLM baseline — official params, 3-run avg + profiling",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "vllm_version": "",
        "gpu": "",
        "official_config": {
            "gpu_memory_utilization": 0.85,
            "max_model_len": 262144,
            "max_num_seqs": 32,
            "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
            "dflash": {"model": "poolside/Laguna-S-2.1-DFlash-NVFP4", "num_speculative_tokens": 15},
            "official_perf_gb10": {
                "prefill_tok_s": "600-800",
                "decode_tok_s_no_spec": "13-14",
                "dflash_accept_per_step": "2.9-3.1",
            },
        },
        "test_config": {
            "max_model_len": 210000,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "dtype": "bfloat16",
            "num_runs": NUM_RUNS,
            "decode_tokens": DECODE_TOKENS,
            "ctx_lengths": CTX_LENGTHS,
            "temperature": 0,
        },
        "tests": {},
    }

    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        results["gpu"] = gpu_info
    except Exception:
        pass

    print(f"{'='*70}")
    print("  Laguna-S-2.1-NVFP4 · vLLM Baseline (Official Params)")
    print(f"  Context: {CTX_LENGTHS} | Runs: {NUM_RUNS} | Decode: {DECODE_TOKENS} tok")
    print("  GPU mem util: 0.85 | enforce_eager: True")
    print(f"{'='*70}\n")

    # ─── Load ───
    print("[1/3] Loading model...")
    t0 = time.perf_counter()
    from vllm import LLM
    import vllm
    results["vllm_version"] = vllm.__version__

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=210000,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
        max_num_seqs=32,
    )
    load_time = time.perf_counter() - t0
    mem_load = get_gpu_mem_mib()
    results["tests"]["load"] = {"load_time_s": round(load_time, 1), "gpu_mem_mib": mem_load}
    print(f"  ✓ {load_time:.1f}s, GPU mem: {mem_load:.0f} MiB\n")

    tokenizer = llm.get_tokenizer()

    # ─── Per-context benchmarks ───
    print("[2/3] Per-context speed (3 runs each)...")
    ctx_results = {}

    for ctx_len in CTX_LENGTHS:
        label = f"ctx{ctx_len//1024}k"
        print(f"\n  ═══ {label} ({ctx_len} tokens) ═══")
        prompt = make_prompt(tokenizer, ctx_len)
        actual_tokens = len(tokenizer.encode(prompt))
        print(f"  Actual input: {actual_tokens} tokens")

        runs = []
        for i in range(NUM_RUNS):
            r = measure_one_run(llm, prompt, DECODE_TOKENS)
            runs.append(r)
            print(f"    Run {i+1}: TTFT={r['ttft_ms']:.1f}ms  ITL={r['itl_ms']:.2f}ms  "
                  f"TPS={r['decode_tps']:.1f}  total={r['total_ms']:.1f}ms  out={r['output_tokens']}")

        avg_ttft = sum(r["ttft_ms"] for r in runs) / NUM_RUNS
        avg_itl = sum(r["itl_ms"] for r in runs) / NUM_RUNS
        avg_tps = sum(r["decode_tps"] for r in runs) / NUM_RUNS
        avg_total = sum(r["total_ms"] for r in runs) / NUM_RUNS
        avg_decode = sum(r["decode_ms"] for r in runs) / NUM_RUNS

        ctx_results[label] = {
            "actual_input_tokens": actual_tokens,
            "ttft_ms_avg": round(avg_ttft, 2),
            "itl_ms_avg": round(avg_itl, 3),
            "decode_tps_avg": round(avg_tps, 1),
            "total_ms_avg": round(avg_total, 2),
            "decode_ms_avg": round(avg_decode, 2),
            "prefill_tok_s": round(actual_tokens / (avg_ttft / 1000), 1),
            "output_tokens": runs[0]["output_tokens"],
            "gpu_mem_mib": get_gpu_mem_mib(),
            "runs": [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in runs],
        }
        print(f"  ► AVG: TTFT={avg_ttft:.1f}ms  ITL={avg_itl:.2f}ms  "
              f"TPS={avg_tps:.1f}  prefill={actual_tokens/(avg_ttft/1000):.0f} tok/s")

    results["tests"]["context_benchmarks"] = ctx_results

    # ─── Profiling ───
    print("\n[3/3] Kernel profiling...")
    profiling_results = {}

    for ctx_len in CTX_LENGTHS:
        label = f"ctx{ctx_len//1024}k"
        print(f"\n  Profiling {label}...")
        prompt = make_prompt(tokenizer, ctx_len)
        try:
            prof = profile_kernels(llm, prompt, DECODE_TOKENS)
            profiling_results[label] = prof
            print(f"    Total CUDA: {prof['total_cuda_ms']:.1f}ms")
            for cat, pct in list(prof["kernel_pct"].items())[:8]:
                print(f"      {cat:15s}: {pct:5.1f}%")
        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            profiling_results[label] = {"error": str(e)}

    results["tests"]["profiling"] = profiling_results
    results["gpu_mem_peak_mib"] = get_gpu_mem_mib()

    # ─── Save ───
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"  ✅ Saved: {RESULTS_PATH}")
    print(f"  GPU peak: {results['gpu_mem_peak_mib']:.0f} MiB")
    print(f"{'='*70}")

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
