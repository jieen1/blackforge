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

__all__ = [
    "FLA_CHUNK_SIZE",
    "AttentionBackendEnum",
    "EngineArgs",
    "GDNAttentionMetadata",  # re-exported from vLLM (isinstance-sensitive)
    "SM120GQAMetadata",  # re-exported from vLLM (isinstance-sensitive)
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
# Re-exported: SM120GQAMetadata (thin — vLLM dataclass, isinstance-sensitive)
#
# Cannot self-write: vLLM's SM120GQAImpl.forward() does isinstance() checks
# against this class. Self-writing breaks the check. Will be replaced when
# we own the model graph (V2/E1).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Re-exported: GDNAttentionMetadata (thin — vLLM dataclass, isinstance-sensitive)
#
# Cannot self-write: vLLM's qwen_gdn_attention_core does isinstance() checks.
# Will be replaced when we own the model graph (V2/E1).
# ---------------------------------------------------------------------------
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadata  # noqa: E402

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
# Self-written: compute_causal_conv1d_metadata (B7-V1 薄依赖自写)
#
# 原 vLLM 实现纯计算（numpy + torch），已自写替代（见文件末尾）。
# 2026-07-22 实测验证 bit-exact。
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Re-exported: FLA chunk index helpers (vLLM internal, kernel-coupled)
# ---------------------------------------------------------------------------
# B7-V1: FLA 切上游 — 从 vLLM 内嵌 FLA 切到 flash-linear-attention 上游包
# 2026-07-22 实测验证：上游 FLA 0.5.2 的 prepare_chunk_indices/offsets
# 与 vLLM 内嵌版本 bit-exact 一致（batch 1/2/4 × seq 128/1024/4096 × chunk 64/128）
from fla.ops.utils.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,  # noqa: E402
    register_backend,  # noqa: E402
)
# B7-V1: compute_causal_conv1d_metadata 已自写（见文件末尾）
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


# ---------------------------------------------------------------------------
# Self-written: compute_causal_conv1d_metadata (B7-V1 薄依赖自写)
#
# 原 vLLM 实现: vllm/v1/attention/backends/utils.py:836
# 纯计算：numpy + torch tensor ops，零 vLLM 依赖。
# 2026-07-22 实测验证 bit-exact（见下方切换注释）。
# ---------------------------------------------------------------------------

_PAD_SLOT_ID = -1


def _is_pin_memory_available() -> bool:
    import torch
    return torch.cuda.is_available() and hasattr(torch.Tensor, "pin_memory")


def _np_to_pinned_tensor(array) -> "torch.Tensor":
    import torch
    t = torch.from_numpy(array)
    return t.pin_memory() if _is_pin_memory_available() else t


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: "torch.Tensor", *, device: "torch.device"
) -> tuple:
    """Compute chunk metadata for causal_conv1d kernel.

    Self-written replacement for vLLM's
    ``vllm.v1.attention.backends.utils.compute_causal_conv1d_metadata``.
    Pure computation: numpy + torch tensor ops, zero vLLM dependency.
    """
    import numpy as np
    import torch

    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    pin_memory = _is_pin_memory_available()

    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = _np_to_pinned_tensor(np.repeat(np.arange(len(nums)), nums.numpy()))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(mlist)
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num.item()))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32, pin_memory=pin_memory)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), _PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), _PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(_PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(_PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist, non_blocking=True)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr
