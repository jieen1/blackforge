"""B7-V1: Single-point consolidation for ALL vLLM dependencies.

Every ``from vllm.*`` import in the production path goes through this
module.  Dependencies are classified into three tiers:

**Self-written (thin)** — pure dataclasses / constants / trivial utilities
re-implemented here with zero vLLM import.  These survive even if vLLM
is uninstalled.

**Re-exported (medium)** — stable public API symbols that vLLM exposes
and that we consume without modification.  Imported lazily so the module
can be loaded (for its self-written symbols) even without vLLM installed.

**Re-exported (thick)** — model graph construction and MTP loading.
These are the last to be replaced (pulled by A1/A2/A3/E1 evidence).

Migration invariant (architecture.md §3.6): replacing any symbol here
must preserve bit-level parity on the greedy fixed-prompt suite.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "FLA_CHUNK_SIZE",
    "AttentionBackendEnum",
    "EngineArgs",
    "GDNAttentionMetadata",
    "SM120GQAMetadata",
    "VllmConfig",
    "bind_kv_cache",
    "compute_causal_conv1d_metadata",
    "get_distributed_init_method",
    "get_gemma_rms_norm",
    "get_model",
    "get_open_port",
    "get_vllm_ir",
    "init_worker_distributed_environment",
    "load_eagle_model",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "register_backend",
    "set_current_vllm_config",
    "set_forward_context",
]

# ---------------------------------------------------------------------------
# Self-written: SM120GQAMetadata (thin — pure dataclass)
# ---------------------------------------------------------------------------


@dataclass
class SM120GQAMetadata:
    """Attention metadata for the SM120 GQA decode/prefill kernel.

    Field-for-field compatible with vLLM's
    ``vllm.v1.attention.backends.sm120_gqa.SM120GQAMetadata``.
    """

    num_actual_tokens: int
    num_reqs: int
    qo_indptr: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    page_size: int
    is_pure_decode: bool
    kv_split_size: int
    max_num_splits: int
    decode_qo_len: int


# ---------------------------------------------------------------------------
# Self-written: GDNAttentionMetadata (thin — pure dataclass)
# ---------------------------------------------------------------------------


@dataclass
class GDNAttentionMetadata:
    """GDN (Gated DeltaNet) attention metadata.

    Field-for-field compatible with vLLM's
    ``vllm.v1.attention.backends.gdn_attn.GDNAttentionMetadata``.
    """

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int
    has_initial_state: torch.Tensor | None = None
    spec_query_start_loc: torch.Tensor | None = None
    non_spec_query_start_loc: torch.Tensor | None = None
    spec_state_indices_tensor: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = None
    spec_sequence_masks: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Self-written: constants (thin)
# ---------------------------------------------------------------------------

FLA_CHUNK_SIZE: int = 64


# ---------------------------------------------------------------------------
# Self-written: network utilities (thin)
# ---------------------------------------------------------------------------


def get_open_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_distributed_init_method(ip: str, port: int) -> str:
    """Build a ``tcp://`` URI for torch.distributed init."""
    if ":" in ip:
        return f"tcp://[{ip}]:{port}"
    return f"tcp://{ip}:{port}"


# ---------------------------------------------------------------------------
# Self-written: compute_causal_conv1d_metadata (thin — pure computation)
# ---------------------------------------------------------------------------

_PAD_SLOT_ID = -1


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor, *, device: torch.device
) -> tuple[dict[int, dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Pre-compute causal_conv1d chunk metadata from CPU query_start_loc.

    Re-implementation of vLLM's
    ``vllm.v1.attention.backends.utils.compute_causal_conv1d_metadata``.
    """
    import numpy as np

    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict[str, Any]] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(
            np.repeat(np.arange(len(nums)), nums.numpy())
        ).pin_memory()
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(mlist)
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist: list[int] = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(int(num)))
        offsetlist_t = torch.tensor(
            offsetlist, dtype=torch.int32, pin_memory=True
        )
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist_t

        if batch_ptr is None:
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,),
                _PAD_SLOT_ID,
                dtype=torch.int32,
                device=device,
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,),
                _PAD_SLOT_ID,
                dtype=torch.int32,
                device=device,
            )
        batch_ptr[:mlist_len] = mlist.to(device)
        token_chunk_offset_ptr[:mlist_len] = offsetlist_t.to(device)

    assert batch_ptr is not None and token_chunk_offset_ptr is not None
    return nums_dict, batch_ptr, token_chunk_offset_ptr


# ---------------------------------------------------------------------------
# Self-written: FLA chunk index helpers (thin)
# ---------------------------------------------------------------------------


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Build per-chunk (batch_idx, chunk_idx) pairs for FLA kernels."""
    import triton

    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = triton.cdiv(lens, chunk_size)
    indices = torch.cat(
        [torch.arange(n) for n in num_chunks.tolist()]
    )
    return torch.stack(
        [indices.eq(0).cumsum(0) - 1, indices], 1
    ).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Build cumulative chunk-count offsets for FLA kernels."""
    import triton

    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = triton.cdiv(lens, chunk_size)
    return torch.cat(
        [cu_seqlens.new_tensor([0]), num_chunks]
    ).cumsum(-1)


# ---------------------------------------------------------------------------
# Re-exported: medium/thick dependencies (vLLM public API)
#
# These are imported at module level so that ``direct_model_runner.py``
# can do ``from runtime.compat_vllm import X`` at its own module level.
# All vLLM imports in the production path are consolidated here — this
# is the B7-V1 "single point" contract.
# ---------------------------------------------------------------------------

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,  # noqa: E402
    register_backend,  # noqa: E402
)
from vllm.v1.worker.gpu_worker import (  # noqa: E402
    init_worker_distributed_environment,  # noqa: E402
)
from vllm.v1.worker.utils import bind_kv_cache  # noqa: E402


def load_eagle_model(*args, **kwargs):
    """Thick dependency: MTP model loading (replaced by A3 evidence)."""
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        load_eagle_model as _load_eagle_model,
    )

    return _load_eagle_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Re-exported: vLLM IR ops and model layers (used by norm patches)
# ---------------------------------------------------------------------------


def get_vllm_ir():
    """Lazy import of vLLM's IR op system (used by gemma_norm_patch / triton_norm_ops)."""
    from vllm import ir

    return ir


def get_gemma_rms_norm():
    """Lazy import of GemmaRMSNorm (used by gemma_norm_patch)."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    return GemmaRMSNorm
