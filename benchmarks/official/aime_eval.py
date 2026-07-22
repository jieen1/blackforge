#!/usr/bin/env python3
"""AIME math evaluation -- comparable to the official Qwen3.6-27B AIME score
(AIME26 = 94.1). Uses the AI-MO AIME validation set (90 AIME problems with
integer answers 000-999), the standard public AIME-style benchmark.

Methodology (standard for thinking models on competition math):
  * Thinking mode ON (the model reasons step by step then answers).
  * Greedy decoding, generous max_tokens (default 16384) so long reasoning finishes.
  * Final answer extracted from \\boxed{...} (preferred), else "answer is N",
    else the last standalone integer; matched exactly to the gold integer.
  * Score = accuracy (exact integer match).

Runs against any OpenAI-compatible chat endpoint (our runtime or stock vLLM).

Usage:
  python benchmarks/official/aime_eval.py --base-url http://localhost:8000/v1 \
      --model qwen3.6 --out evalplus_results/official/aime_our_runtime.json
  # quick subset:
  ... --limit 30 --concurrency 4
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import time
import urllib.request

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_IS_RE = re.compile(r"(?:final\s+)?answer\s*is[:\s]*\$?(-?\d+)", re.I)
INT_RE = re.compile(r"-?\d+")


def chat(base_url, model, prompt, max_tokens, temperature=0.0, timeout=1200):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "n": 1,
    }).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed: {last}")


def normalize_int(s):
    s = re.sub(r"[^\d-]", "", str(s))
    if s in ("", "-"):
        return None
    try:
        v = int(s)
        return v % 1000 if v < 0 or v > 999 else v  # AIME answers wrap to 000-999
    except ValueError:
        return None


def extract_answer(content, reasoning):
    # Prefer the final \boxed{...} in the visible content, then reasoning.
    for text in (content or "", reasoning or ""):
        m = list(BOXED_RE.finditer(text))
        if m:
            v = normalize_int(m[-1].group(1))
            if v is not None:
                return v
    for text in (content or "", reasoning or ""):
        m = list(ANSWER_IS_RE.finditer(text))
        if m:
            v = normalize_int(m[-1].group(1))
            if v is not None:
                return v
    for text in (content or "", reasoning or ""):
        m = list(INT_RE.finditer(text))
        if m:
            v = normalize_int(m[-1].group(0))
            if v is not None:
                return v
    return None


PROMPT_TMPL = ("{problem}\n\nPlease reason step by step, and put your final answer "
               "within \\boxed{{}}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    args = p.parse_args()

    from datasets import load_dataset
    print("Loading AI-MO/aimo-validation-aime ...", flush=True)
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    items = list(ds)
    if args.limit:
        items = items[:args.limit]
    print(f"Evaluating {len(items)} AIME problems, concurrency={args.concurrency}, "
          f"max_tokens={args.max_tokens}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = args.out.replace(".json", ".jsonl")
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
        print(f"Resuming: {len(done)} already done", flush=True)
    ckpt_f = open(ckpt, "a")

    def one(ex):
        if ex["id"] in done:
            return done[ex["id"]]
        prompt = PROMPT_TMPL.format(problem=ex["problem"])
        gold = normalize_int(ex["answer"])
        try:
            resp = chat(args.base_url, args.model, prompt, args.max_tokens)
            ch = resp["choices"][0]
            msg = ch["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            finish = ch.get("finish_reason")
            gen = resp.get("usage", {}).get("completion_tokens")
        except Exception as e:  # noqa: BLE001
            content = reasoning = ""
            finish = "error"
            gen = None
        pred = extract_answer(content, reasoning)
        rec = {"id": ex["id"], "gold": gold, "pred": pred,
               "correct": pred == gold, "finish_reason": finish, "gen_tokens": gen,
               "answer_tail": content[-160:]}
        ckpt_f.write(json.dumps(rec) + "\n")
        ckpt_f.flush()
        return rec

    t0 = time.time()
    results = []
    finished = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, e) for e in items]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
            finished += 1
            if finished % 5 == 0 or finished == len(items):
                acc = sum(r["correct"] for r in results) / len(results)
                rate = finished / max(1e-6, time.time() - t0)
                eta = (len(items) - finished) / rate if rate else 0
                print(f"  {finished}/{len(items)} acc={acc:.4f} "
                      f"({rate:.3f} q/s, ETA {eta/60:.1f}m)", flush=True)
    ckpt_f.close()

    n = len(results)
    overall = sum(r["correct"] for r in results) / n
    trunc = sum(1 for r in results if r["finish_reason"] == "length")
    report = {
        "benchmark": "AIME (AI-MO validation, 90 AIME problems)",
        "official_qwen36_27b_AIME26": 94.1,
        "model": args.model, "base_url": args.base_url,
        "date": datetime.datetime.now().isoformat(),
        "n": n, "thinking": "on", "max_tokens": args.max_tokens,
        "methodology": "thinking, greedy, exact integer match, \\boxed extraction",
        "accuracy": round(overall * 100, 2),
        "truncated": trunc,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== AIME accuracy = {report['accuracy']} "
          f"(official Qwen3.6-27B AIME26 = 94.1) ===", flush=True)
    print(f"truncated={trunc}/{n}  Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
