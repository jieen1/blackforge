"""Print a validated checkpoint metadata summary without reading tensor payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loader.checkpoint_index import load_checkpoint_index
from loader.pack_manifest import create_manifest
from model.qwen36_config import load_qwen36_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    arguments = parser.parse_args()

    config = load_qwen36_config(arguments.model_dir)
    index = load_checkpoint_index(arguments.model_dir)
    index.validate_files()
    index.validate_qwen36(config)
    headers = index.validate_headers()
    index.validate_nvfp4_companions()
    print(
        json.dumps(
            {
                "checkpoint": index.summary() | {"validated_header_tensors": len(headers)},
                "manifest": create_manifest(index, config).__dict__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
