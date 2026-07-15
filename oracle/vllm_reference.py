"""Schema for a read-only vLLM capture hook.

Implement the hook in the local vLLM checkout only after selecting the exact
revision and model checkpoint. The runtime consumes artifacts, never imports a
modified vLLM tree in its serving path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CaptureManifest:
    vllm_revision: str
    model_id: str
    case_id: str
    tensor_names: tuple[str, ...]
    dtype: str


def write_manifest(path: Path, manifest: CaptureManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
