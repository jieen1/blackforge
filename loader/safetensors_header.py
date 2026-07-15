"""Minimal safetensors header reader for metadata-only checkpoint validation."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SafetensorHeaderError(ValueError):
    """Raised for malformed or unsupported safetensors metadata."""


@dataclass(frozen=True)
class TensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def read_safetensors_header(path: Path) -> dict[str, TensorMetadata]:
    """Return tensor metadata while reading only the safetensors JSON header."""
    with path.open("rb") as source:
        encoded_length = source.read(8)
        if len(encoded_length) != 8:
            raise SafetensorHeaderError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length == 0 or header_length > 64 * 1024 * 1024:
            raise SafetensorHeaderError(f"invalid safetensors header length: {header_length}")
        encoded_header = source.read(header_length)
    if len(encoded_header) != header_length:
        raise SafetensorHeaderError(f"truncated safetensors metadata: {path}")
    try:
        raw: dict[str, Any] = json.loads(encoded_header)
    except json.JSONDecodeError as error:
        raise SafetensorHeaderError(f"invalid safetensors JSON metadata: {path}") from error

    tensors: dict[str, TensorMetadata] = {}
    for name, metadata in raw.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise SafetensorHeaderError(f"invalid metadata for tensor {name}")
        try:
            tensors[name] = TensorMetadata(
                dtype=str(metadata["dtype"]),
                shape=tuple(int(size) for size in metadata["shape"]),
                data_offsets=tuple(int(offset) for offset in metadata["data_offsets"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SafetensorHeaderError(f"invalid layout metadata for tensor {name}") from error
    return tensors
