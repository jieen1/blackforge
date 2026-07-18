"""Generates the FROZEN, VERSIONED prompt token id fixtures the `-S`
(controlled synthetic) workload line depends on -- run ONCE, the output is
committed to the repo (`benchmarks/fixtures/*.json`) and loaded verbatim by
every measurement script from then on (`workloads.load_prompt_token_ids`).
Re-running this script would overwrite the frozen fixture with a
(deterministically identical, given the same library versions) but
DIFFERENTLY-timed regeneration -- the whole point of freezing is to remove
even that residual "did the formula really reproduce exactly" doubt, so
this script is intentionally not part of any measurement script's own
runtime path.

Uses vLLM's own `RandomDataset.generate_token_sequence()` formula exactly
(read directly from `vllm/benchmarks/datasets/datasets.py`, not
approximated): `allowed_tokens[(offset + request_index + arange(input_len))
% len(allowed_tokens)]`, `allowed_tokens = sorted(set(range(vocab_size)) -
set(tokenizer.all_special_ids))`, per-request `offset` drawn from a seeded
RNG. Only needs the tokenizer (CPU-only, no GPU/model load).

Usage:
    python -m benchmarks.generate_synthetic_fixtures
"""

from __future__ import annotations

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from benchmarks.workloads import (
    D1_CTX16K_FIXTURE,
    D1_CTX32K_FIXTURE,
    FIXTURES_DIR,
    W1_S_FIXTURE,
    W1_S_FIXTURE_N128,
)


def _generate(fixture) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(fixture.tokenizer)
    vocab_size = tok.vocab_size
    special_ids = set(tok.all_special_ids)
    allowed_tokens = sorted(set(range(vocab_size)) - special_ids)
    n = len(allowed_tokens)

    rng = random.Random(fixture.seed)
    prompts = []
    for req_index in range(fixture.num_requests):
        offset = rng.randrange(n)
        prompt = [allowed_tokens[(offset + req_index + i) % n] for i in range(fixture.prompt_len)]
        prompts.append(prompt)

    return {
        "tokenizer": fixture.tokenizer,
        "generation_formula": fixture.generation_formula,
        "seed": fixture.seed,
        "num_requests": fixture.num_requests,
        "prompt_len": fixture.prompt_len,
        "vocab_size": vocab_size,
        "num_allowed_tokens": n,
        "prompt_token_ids": prompts,
    }


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for fixture in (W1_S_FIXTURE, W1_S_FIXTURE_N128, D1_CTX16K_FIXTURE, D1_CTX32K_FIXTURE):
        out_path = FIXTURES_DIR / fixture.path
        if out_path.exists():
            print(f"skip (already exists, frozen): {out_path}")
            continue
        data = _generate(fixture)
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"wrote {out_path} ({len(data['prompt_token_ids'])} prompts x {fixture.prompt_len} tokens)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
