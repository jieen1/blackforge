"""Parallel HumanEval+ quality evaluation via OpenAI-compatible API.

Sends all 164 HumanEval+ prompts to a server with greedy decoding,
saves results in evalplus jsonl format for evaluation.
"""
import asyncio
import json
import os
import sys
import time

import aiohttp
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
MAX_TOKENS = 768
CONCURRENCY = 16


async def generate_one(session, base_url, model, task_id, prompt, semaphore):
    message = INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "n": 1,
    }
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"  [{task_id}] HTTP {resp.status}: {text[:200]}", flush=True)
                        await asyncio.sleep(2)
                        continue
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return task_id, content
            except Exception as e:
                print(f"  [{task_id}] attempt {attempt+1} error: {e}", flush=True)
                await asyncio.sleep(2)
    return task_id, None


async def run_all(base_url, model, dataset, output_path, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for task_id, task in dataset.items():
            prompt = task["prompt"].strip() + "\n"
            tasks.append(generate_one(session, base_url, model, task_id, prompt, semaphore))

        results = {}
        done_count = 0
        total = len(tasks)
        start = time.time()
        for coro in asyncio.as_completed(tasks):
            task_id, content = await coro
            done_count += 1
            if content is not None:
                results[task_id] = content
                if done_count % 10 == 0 or done_count == total:
                    elapsed = time.time() - start
                    rate = done_count / elapsed
                    eta = (total - done_count) / rate if rate > 0 else 0
                    print(f"  Progress: {done_count}/{total} ({rate:.1f} problems/s, ETA {eta:.0f}s)", flush=True)
            else:
                print(f"  FAILED: {task_id}", flush=True)

    raw_path = output_path.replace(".jsonl", ".raw.jsonl")
    with open(output_path, "w") as f_san, open(raw_path, "w") as f_raw:
        for task_id, task in dataset.items():
            if task_id not in results:
                continue
            prompt = task["prompt"].strip() + "\n"
            solution = results[task_id]
            sanitized = sanitize(solution, entrypoint=task["entry_point"])
            f_san.write(json.dumps({"task_id": task_id, "solution": sanitized}) + "\n")
            f_raw.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")

    print(f"\nSaved {len(results)}/{total} solutions to {output_path}")
    return len(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    print(f"Loading HumanEval+ dataset...")
    dataset = get_human_eval_plus()
    print(f"  {len(dataset)} problems loaded")
    print(f"Target: {args.base_url} model={args.model}")
    print(f"Concurrency: {args.concurrency}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    asyncio.run(run_all(args.base_url, args.model, dataset, args.output, args.concurrency))


if __name__ == "__main__":
    main()
