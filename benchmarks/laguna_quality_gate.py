#!/usr/bin/env python3
"""Laguna L2 质量链：oracle A/B 比对 + prompt ids 断言护栏。

永久护栏：比对输出前，先断言两条路径的 prompt token ids 完全一致。
任何分词路径偏差（如 add_special_tokens=False 跳过 BOS）会在此处立即失败，
而不是在下游 logits 比对中变成难以定位的系统性偏移。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_quality_gate
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "poolside/Laguna-S-2.1-NVFP4"

GREEDY_PROMPTS = [
    "The capital of France is",
    "Write a Python function to check if a number is prime.",
    "What is 15 * 37? Show your work step by step.",
    "Explain the theory of relativity in simple terms.",
]

LAGUNA_EOS = (2, 24)
MAX_DECODE_TOKENS = 80


def assert_prompt_ids_equal(
    prompt: str,
    ids_a: list[int],
    ids_b: list[int],
    path_a: str,
    path_b: str,
) -> None:
    """永久护栏：断言两条路径的 prompt token ids 完全一致。"""
    if ids_a != ids_b:
        raise AssertionError(
            f"Prompt token ids 不一致！\n"
            f"  Prompt: {prompt!r}\n"
            f"  {path_a}: {ids_a[:20]}{'...' if len(ids_a) > 20 else ''} (len={len(ids_a)})\n"
            f"  {path_b}: {ids_b[:20]}{'...' if len(ids_b) > 20 else ''} (len={len(ids_b)})\n"
            f"  首个分歧位置: {next((i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b), min(len(ids_a), len(ids_b)))}\n"
            f"  诊断: 检查分词路径是否一致（BOS/add_special_tokens/post_processor）"
        )


def run_backend_greedy(backend, tokenizer, prompt: str) -> list[int]:
    """LagunaBackend 贪心生成，返回 output token ids。"""
    prompt_ids = tokenizer.encode(prompt)
    backend.reset_slot(0)
    first_token = backend.prefill(0, prompt_ids)
    tokens = [first_token]
    for _ in range(MAX_DECODE_TOKENS - 1):
        tok = backend.decode(0, tokens[-1])
        tokens.append(tok)
        if tok in LAGUNA_EOS:
            break
    backend.reset_slot(0)
    return tokens


def run_vllm_greedy(llm, prompt: str) -> list[int]:
    """Stock vLLM 贪心生成，返回 output token ids。"""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=MAX_DECODE_TOKENS)
    result = llm.generate([prompt], params)[0]
    return list(result.outputs[0].token_ids)


def main():
    import torch
    from transformers import AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print("Quality gate: prompt ids assertion + token-level A/B comparison")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # ── Phase 1: Load LagunaBackend ──
    print("=== Phase 1: Load LagunaBackend ===")
    from runtime.compat_vllm import EngineArgs

    args = EngineArgs(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
        async_scheduling=False,
    )
    config = args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend

    t0 = time.perf_counter()
    backend = LagunaBackend(config, num_slots=4, block_size=16, blocks_per_slot=512)
    backend_load_time = time.perf_counter() - t0
    print(f"  Loaded in {backend_load_time:.1f}s")

    # ── Phase 2: Load stock vLLM ──
    print("\n=== Phase 2: Load stock vLLM ===")
    del backend
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    from vllm import LLM

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    vllm_load_time = time.perf_counter() - t0
    vllm_tokenizer = llm.get_tokenizer()
    print(f"  Loaded in {vllm_load_time:.1f}s")

    # ── Phase 3: Tokenizer 一致性断言 ──
    print("\n=== Phase 3: Tokenizer 一致性断言 ===")
    tokenizer_match = True
    for prompt in GREEDY_PROMPTS:
        ids_hf = tokenizer.encode(prompt)
        ids_vllm = vllm_tokenizer.encode(prompt)
        try:
            assert_prompt_ids_equal(prompt, ids_hf, ids_vllm, "HF tokenizer", "vLLM tokenizer")
            print(f"  ✅ {prompt[:40]!r}... ids match (len={len(ids_hf)}, BOS={ids_hf[0]})")
        except AssertionError as e:
            print(f"  ❌ {e}")
            tokenizer_match = False

    if not tokenizer_match:
        print("\n❌ ABORT: Tokenizer 不一致，无法进行 A/B 比对")
        sys.exit(1)

    # ── Phase 4: A/B 贪心比对 ──
    print("\n=== Phase 4: A/B 贪心比对 ===")

    # 先跑 vLLM（已加载）
    vllm_outputs = {}
    for prompt in GREEDY_PROMPTS:
        vllm_outputs[prompt] = run_vllm_greedy(llm, prompt)
        text = vllm_tokenizer.decode(vllm_outputs[prompt], skip_special_tokens=True)
        print(f"  vLLM: {prompt[:40]!r} → {text[:60]!r}...")

    # 释放 vLLM，加载 backend
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    backend2 = LagunaBackend(config, num_slots=4, block_size=16, blocks_per_slot=512)

    results = []
    all_match = True
    for prompt in GREEDY_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        backend_tokens = run_backend_greedy(backend2, tokenizer, prompt)
        vllm_tokens = vllm_outputs[prompt]

        backend_text = tokenizer.decode(backend_tokens, skip_special_tokens=True)
        vllm_text = vllm_tokenizer.decode(vllm_tokens, skip_special_tokens=True)

        # Token-level comparison
        min_len = min(len(backend_tokens), len(vllm_tokens))
        first_diverge = None
        for i in range(min_len):
            if backend_tokens[i] != vllm_tokens[i]:
                first_diverge = i
                break
        if first_diverge is None and len(backend_tokens) != len(vllm_tokens):
            first_diverge = min_len

        match = first_diverge is None
        if not match:
            all_match = False

        result = {
            "prompt": prompt,
            "prompt_ids_len": len(prompt_ids),
            "prompt_bos": prompt_ids[0],
            "backend_tokens": len(backend_tokens),
            "vllm_tokens": len(vllm_tokens),
            "match": match,
            "first_diverge_at": first_diverge,
            "backend_text": backend_text[:200],
            "vllm_text": vllm_text[:200],
        }
        results.append(result)

        status = "✅ MATCH" if match else f"❌ DIVERGE@{first_diverge}"
        print(f"  {status}: {prompt[:40]!r}")
        if not match:
            print(f"    Backend: {backend_text[:80]!r}")
            print(f"    vLLM:    {vllm_text[:80]!r}")
            if first_diverge is not None and first_diverge < min_len:
                print(f"    Token@{first_diverge}: backend={backend_tokens[first_diverge]} vllm={vllm_tokens[first_diverge]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"QUALITY GATE: {'✅ PASS' if all_match and tokenizer_match else '❌ FAIL'}")
    print(f"  Tokenizer consistency: {'✅' if tokenizer_match else '❌'}")
    print(f"  Token-level A/B match: {sum(r['match'] for r in results)}/{len(results)}")
    print(f"{'='*60}")

    # Save
    fixture = {
        "model": MODEL,
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "guardrail": "prompt_ids_assertion",
        "tokenizer_consistency": tokenizer_match,
        "ab_match_all": all_match,
        "results": results,
    }
    out_path = Path("benchmarks/fixtures/laguna_quality_gate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()
