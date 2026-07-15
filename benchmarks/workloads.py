"""Frozen Phase 0 serving workloads.

The runner is intentionally not implemented until the target model and vLLM
launch command are pinned. These values are the contract for all comparisons.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: tuple[int, ...]


WORKLOADS = {
    "W1": Workload("W1", input_tokens=4096, output_tokens=1024, concurrency=(1, 4)),
    "W2": Workload("W2", input_tokens=32768, output_tokens=1024, concurrency=(1, 4)),
}


def main() -> None:
    print(json.dumps({name: asdict(workload) for name, workload in WORKLOADS.items()}, indent=2))


if __name__ == "__main__":
    main()
