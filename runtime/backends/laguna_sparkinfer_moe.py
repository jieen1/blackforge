"""sparkinfer NVFP4 MoE kernel — standalone, zero vLLM dependency.

Loads Laguna NVFP4 checkpoint weights directly from safetensors,
prepares them for sparkinfer, and provides a clean forward() API.

Dependency: sparkinfer (editable install from jieen1/sparkinfer fork,
branch blackforge-main, or BF_SPARKINFER_PATH env fallback).

Scale convention (verified cosine=0.954 vs fp32 reference):
  - Fold weight_global_scale into block scales: bs_new = bs / wgs
  - Use unit w1_global_scale and unit a1_gscale
  - Kernel handles dynamic activation quantization internally

Performance (SM120, E=256, K=3072, I=1024, top_k=10):
  - CUDA graph: ~38μs/layer → 1.8ms for 47 layers
  - vs CUTLASS eager: ~186μs/layer → 8.73ms for 47 layers
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Sequence

import torch

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_moe")

# ---------------------------------------------------------------------------
# sparkinfer import: editable install preferred, env fallback
# ---------------------------------------------------------------------------
_BF_SPARKINFER_PATH = os.environ.get("BF_SPARKINFER_PATH", "")
if _BF_SPARKINFER_PATH and _BF_SPARKINFER_PATH not in sys.path:
    sys.path.insert(0, _BF_SPARKINFER_PATH)

try:
    from sparkinfer._lib.intrinsics import swizzle_block_scale
    from sparkinfer.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        build_tp_moe_fp4_binding,
        plan_sparkinfer_fp4_moe_weights,
        prepare_sparkinfer_fp4_moe_weights,
        sparkinfer_moe_fp4,
    )
    from sparkinfer.moe.fused_moe import is_supported as sparkinfer_is_supported
except ImportError as exc:
    raise ImportError(
        "sparkinfer not found. Install via: pip install -e /path/to/sparkinfer "
        "or set BF_SPARKINFER_PATH=/path/to/sparkinfer"
    ) from exc

# Remove cutlass-dsl base_dsl from sys.path — it contains a torch.py
# that shadows the real torch module in spawned subprocesses.
sys.path[:] = [
    p for p in sys.path
    if "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl" not in p
]


def sparkinfer_version() -> str:
    """Return sparkinfer git commit sha for version stamping."""
    try:
        import sparkinfer
        pkg_dir = pathlib.Path(sparkinfer.__file__).parent
        git_dir = pkg_dir.parent / ".git"
        if git_dir.exists():
            result = subprocess.run(
                ["git", "-C", str(git_dir.parent), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EXPERTS = 256
TOP_K = 10
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 1024
MOE_LAYER_IDS = list(range(1, 48))  # layers 1-47


def _find_checkpoint(model_id: str = "poolside/Laguna-S-2.1-NVFP4") -> pathlib.Path:
    """Resolve HF cache snapshot path for the model."""
    cache_root = pathlib.Path.home() / ".cache/huggingface/hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        snaps = sorted(snapshots.iterdir())
        if snaps:
            return snaps[0]
    raise FileNotFoundError(f"Cannot find checkpoint for {model_id} in {cache_root}")


def load_moe_layer_weights(
    ckpt: pathlib.Path,
    layer_idx: int,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Load one MoE layer's per-expert weights directly from safetensors.

    Returns dict: gate_w, up_w, down_w, gate_sf, up_sf, down_sf,
    gate_gs, up_gs, down_gs — each [E, ...] on device.
    """
    from safetensors import safe_open

    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_idx}.mlp.experts"
    needed_shards: set[str] = set()
    for eid in range(NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for sfx in ("weight_packed", "weight_scale", "weight_global_scale"):
                needed_shards.add(weight_map[f"{prefix}.{eid}.{proj}.{sfx}"])

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(needed_shards):
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    tensors[k] = f.get_tensor(k)

    result = {}
    for name, proj in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
        result[f"{name}_w"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_packed"] for e in range(NUM_EXPERTS)]
        ).to(device)
        result[f"{name}_sf"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_scale"] for e in range(NUM_EXPERTS)]
        ).to(device)
        result[f"{name}_gs"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_global_scale"] for e in range(NUM_EXPERTS)]
        ).to(device).float()
    return result


def prepare_sparkinfer_layer(
    raw: dict[str, torch.Tensor],
    device: str | torch.device = "cuda",
):
    """Prepare sparkinfer expert weights from raw checkpoint tensors.

    Scale convention: fold weight_global_scale into block scales,
    use unit w1_global_scale and unit a1_gscale.
    """
    gate_sf_f = (raw["gate_sf"].float() / raw["gate_gs"].view(-1, 1, 1)).to(torch.float8_e4m3fn)
    up_sf_f = (raw["up_sf"].float() / raw["up_gs"].view(-1, 1, 1)).to(torch.float8_e4m3fn)
    down_sf_f = (raw["down_sf"].float() / raw["down_gs"].view(-1, 1, 1)).to(torch.float8_e4m3fn)

    w13_fp4 = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous()
    w13_sf = swizzle_block_scale(torch.cat([up_sf_f, gate_sf_f], dim=1).contiguous())
    w2_sf = swizzle_block_scale(down_sf_f.contiguous())

    ones_E = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device)
    ones_0 = torch.ones((), dtype=torch.float32, device=device)

    wplan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="nvfp4", source_format="modelopt_nvfp4",
        activation="silu", params_dtype=torch.bfloat16,
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE, w13_layout="w31",
    )
    return prepare_sparkinfer_fp4_moe_weights(
        plan=wplan, w1_global_scale=ones_E, w2_global_scale=ones_E,
        w1_fp4=w13_fp4, w1_blockscale=w13_sf,
        w2_fp4=raw["down_w"].contiguous(), w2_blockscale=w2_sf,
        a1_gscale=ones_0, a2_gscale=ones_0, params_dtype=torch.bfloat16,
    )


class SparkinferMoELayer:
    """One MoE layer backed by sparkinfer kernel."""

    def __init__(self, experts, workspace, device="cuda"):
        self.experts = experts
        self.workspace = workspace
        self.device = torch.device(device)
        self._output_buf: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        M = hidden.shape[0]
        if self._output_buf is None or self._output_buf.shape[0] < M:
            self._output_buf = torch.empty(
                M, HIDDEN_SIZE, dtype=hidden.dtype, device=self.device,
            )
        out = self._output_buf[:M]
        binding = build_tp_moe_fp4_binding(
            scratch=self.workspace, a=hidden, experts=self.experts,
            topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32),
            quant_mode="nvfp4", input_scales_static=True, output=out,
        )
        return sparkinfer_moe_fp4(binding=binding)


class SparkinferMoEModel:
    """All 47 MoE layers loaded from checkpoint, ready for inference."""

    def __init__(
        self,
        ckpt: pathlib.Path | str | None = None,
        layer_ids: Sequence[int] = MOE_LAYER_IDS,
        device: str = "cuda",
    ):
        if ckpt is None:
            ckpt = _find_checkpoint()
        self.ckpt = pathlib.Path(ckpt)
        self.layer_ids = list(layer_ids)
        self.device = device
        self.layers: dict[int, SparkinferMoELayer] = {}
        self._workspace = allocate_tp_moe_workspace_pool()
        self.version = sparkinfer_version()
        logger.info(
            "SparkinferMoEModel init: %d layers, sparkinfer@%s",
            len(self.layer_ids), self.version,
        )

    def load_layer(self, layer_idx: int) -> SparkinferMoELayer:
        t0 = time.time()
        raw = load_moe_layer_weights(self.ckpt, layer_idx, self.device)
        experts = prepare_sparkinfer_layer(raw, self.device)
        layer = SparkinferMoELayer(experts, self._workspace, self.device)
        self.layers[layer_idx] = layer
        del raw
        torch.cuda.empty_cache()
        logger.info("Layer %d prepared in %.1fs", layer_idx, time.time() - t0)
        return layer

    def load_all(self) -> None:
        t0 = time.time()
        for lid in self.layer_ids:
            self.load_layer(lid)
        total = time.time() - t0
        logger.info(
            "All %d MoE layers loaded in %.1fs (%.1fs/layer)",
            len(self.layer_ids), total, total / len(self.layer_ids),
        )

    def forward_layer(
        self, layer_idx: int,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if layer_idx not in self.layers:
            self.load_layer(layer_idx)
        return self.layers[layer_idx].forward(hidden, topk_ids, topk_weights)
