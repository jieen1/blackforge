"""Inspect one checkpoint tensor through the indexed on-demand reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loader.checkpoint_index import load_checkpoint_index
from loader.tensor_reader import TensorReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("tensor_name")
    arguments = parser.parse_args()

    tensor = TensorReader(load_checkpoint_index(arguments.model_dir)).load(arguments.tensor_name)
    print(
        json.dumps(
            {
                "name": arguments.tensor_name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": tensor.numel(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
