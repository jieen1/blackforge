"""Confirm that the project CUDA extra can execute an SM120 tensor operation."""

from __future__ import annotations

import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    device = torch.device("cuda")
    matrix = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    product = matrix @ matrix.T
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "device": properties.name,
                "capability": [properties.major, properties.minor],
                "torch": torch.__version__,
                "result_checksum": float(product.sum().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
