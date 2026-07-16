"""Signal-probe crosstalk test for DirectModelRunner.decode_batch, per the
2026-07-16 coordinator-approved acceptance criteria (replacing the
unachievable "bytewise-identical vs run-alone" bar -- see
notes/direct-model-runner-design.md's "batch>=2 numerical mismatch"
section for why that bar failed even under real vLLM's own production
path):

  1. Argmax/token-sequence plausibility: each slot's generated
     continuation must be fluent (decodable, non-garbage) text.
  2. Signal-probe / marker-token crosstalk detection (this project's
     established causal-mask-leakage methodology -- CLAUDE.md's "embed
     distinct marker tokens... verify masked positions never see future
     markers" -- adapted here to cross-REQUEST leakage in a batch): each
     active slot's prompt embeds a unique 5-digit number and repeats the
     cue phrase immediately before the completion point, a strong
     in-context copy pattern the model reliably reproduces. If
     decode_batch's physical-slot addressing is wrong, a slot's
     continuation echoes a DIFFERENT slot's number -- an unambiguous
     signal of cross-slot data leakage, sharply distinguished from
     ordinary batch-composition floating-point noise (which cannot
     coincidentally reproduce another slot's distinct 5-digit number).
  3. Same-batch internal self-consistency (replaces "run alone" as the
     correctness reference): two slots given the IDENTICAL prompt in the
     SAME batch call must produce bytewise-identical logits to each
     other, every step.

Modes exercised via CLI flags:
  --batch N            number of concurrently active slots
  --varlen             give each slot a different-length filler prefix
                        (heterogeneous prior_kv_len/page count across
                        the batch)
  --steps N            number of sequential decode_batch calls (default 8,
                        enough to fully spell out a 5-digit number; use a
                        large N for the continuous-generation check)
  --reuse              after `steps` rounds, release slot 0 and reuse it
                        with a NEW number, verifying no residue from the
                        prior occupant
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"

# Distinct 5-digit markers, no shared full-string overlap.
NUMBERS = [84317, 52968, 71053, 39642, 60284, 17395, 28471, 93856]
FILLER_SENTENCE = "The weather today is mild and pleasant. "


def _hash(tensor) -> str:
    return hashlib.sha256(tensor.float().cpu().numpy().tobytes()).hexdigest()[:16]


def _make_prompt(number: int, filler_repeats: int = 0) -> str:
    filler = FILLER_SENTENCE * filler_repeats
    return f"{filler}The value of X is {number}. The value of X is"


def _assign_numbers(batch: int) -> list[int]:
    numbers = list(NUMBERS[:batch])
    if batch >= 3:
        # Last slot duplicates the first -- a same-batch self-consistency
        # pair -- while every other slot keeps a distinct crosstalk target.
        numbers[-1] = numbers[0]
    return numbers


def _assign_filler_repeats(batch: int, varlen: bool) -> list[int]:
    if not varlen:
        return [0] * batch
    fillers = list(range(batch))
    if batch >= 3:
        # The self-consistency pair (slots 0 and batch-1, same number) must
        # also share the same filler length -- otherwise their prompts
        # aren't actually identical and "self-consistency" would be
        # comparing two genuinely different inputs, not testing anything.
        # Interior slots still get distinct lengths for varlen coverage.
        fillers[-1] = fillers[0]
    return fillers


def _run_once(batch: int, varlen: bool, steps: int, reuse: bool) -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    runner = DirectModelRunner(vllm_config, num_slots=batch, block_size=16, blocks_per_slot=128)
    tok = AutoTokenizer.from_pretrained(MODEL)

    numbers = _assign_numbers(batch)
    filler_repeats = _assign_filler_repeats(batch, varlen)
    prompts = [_make_prompt(n, f) for n, f in zip(numbers, filler_repeats)]
    prompt_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]

    slots = list(range(batch))
    next_tokens = [runner.prefill(slot, ids) for slot, ids in zip(slots, prompt_ids)]
    kv_lengths = [runner.slot_kv_len[s] for s in slots]

    generated = [[t] for t in next_tokens]
    cur_tokens = list(next_tokens)
    self_consistent = True
    for _ in range(steps):
        logits = runner._forward_batch(slots, cur_tokens, kv_lengths)
        if batch >= 3:
            # Same-batch self-consistency: slot 0 and slot batch-1 share
            # an identical prompt -- their logits must match bytewise.
            if _hash(logits[0]) != _hash(logits[-1]):
                self_consistent = False
        cur_tokens = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
        kv_lengths = [kv_len + 1 for kv_len in kv_lengths]
        for i, t in enumerate(cur_tokens):
            generated[i].append(t)

    texts = [tok.decode(g) for g in generated]

    crosstalk = []
    signal_ok = True
    for i in range(batch):
        text = texts[i]
        own = str(numbers[i])
        contains_own = own in text
        leaked_other = any(str(numbers[j]) in text for j in range(batch) if numbers[j] != numbers[i])
        crosstalk.append(
            {"slot": i, "number": numbers[i], "text": text, "contains_own": contains_own, "leaked_other": leaked_other}
        )
        if leaked_other or not contains_own:
            signal_ok = False

    result = {
        "passed": self_consistent and signal_ok,
        "batch": batch,
        "varlen": varlen,
        "steps": steps,
        "self_consistent": self_consistent,
        "signal_ok": signal_ok,
        "crosstalk": crosstalk,
    }

    if reuse:
        runner.reset_slot(0)
        new_number = 91827
        assert new_number not in numbers
        new_prompt = _make_prompt(new_number)
        new_ids = tok.encode(new_prompt, add_special_tokens=False)
        new_tok = runner.prefill(0, new_ids)
        gen_new = [new_tok]
        kv_len0 = runner.slot_kv_len[0]
        cur = new_tok
        for _ in range(8):
            logits = runner._forward_batch([0], [cur], [kv_len0])
            cur = int(logits[0].argmax(dim=-1).item())
            kv_len0 += 1
            gen_new.append(cur)
        text_new = tok.decode(gen_new)
        reuse_ok = str(new_number) in text_new and not any(str(n) in text_new for n in numbers if n != new_number)
        result["reuse_check"] = {"text": text_new, "reuse_ok": reuse_ok}
        result["passed"] = result["passed"] and reuse_ok

    return result


def _run_subprocess(batch: int, varlen: bool, steps: int, reuse: bool) -> dict:
    args = [
        sys.executable,
        "-m",
        "benchmarks.batch_decode_signal_probe",
        "--single-run-json",
        "--batch",
        str(batch),
        "--steps",
        str(steps),
    ]
    if varlen:
        args.append("--varlen")
    if reuse:
        args.append("--reuse")
    proc = subprocess.run(
        args,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=600,
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
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once(args.batch, args.varlen, args.steps, args.reuse)
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once(args.batch, args.varlen, args.steps, args.reuse)
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess(args.batch, args.varlen, args.steps, args.reuse) for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}")
        if not r.get("passed"):
            print(f"  detail: {r}")

    n_pass = sum(1 for r in results if r.get("passed"))
    print(
        f"\n=== {n_pass}/{args.repeat} passed "
        f"(batch={args.batch}, varlen={args.varlen}, steps={args.steps}, reuse={args.reuse}) ==="
    )
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
