"""A2: Triton NVFP4 GEMM kernel for SM120 decode shapes.

Optimized for M=1..16 (decode/verify), purely memory-bandwidth-bound.
Key insight: CUTLASS 128×128 tiles underutilize 188 SMs on small-N shapes.
This kernel uses tunable BLOCK_N to maximize SM occupancy.

NVFP4 format:
- Weight: packed uint8, 2 FP4 (e2m1) values per byte, K-contiguous
- Block scale: FP8 e4m3, one per 128 elements along K
- Global alpha: FP32 scalar = 1/(a_global_scale * b_global_scale)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# FP4 e2m1 lookup table: 4-bit index -> bf16 value
# e2m1: {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {+, -}
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)


@triton.jit
def _nvfp4_gemm_kernel(
    # Pointers
    a_ptr,          # [M, K/2] packed FP4 activation
    b_ptr,          # [N, K/2] packed FP4 weight
    scale_a_ptr,    # [M, K//128] FP8 block scales for activation
    scale_b_ptr,    # [N, K//128] FP8 block scales for weight
    out_ptr,        # [M, N] bf16 output
    # Scalars
    alpha,          # global scale (fp32)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # must be 128 (NVFP4 block scale granularity)
    # Strides
    stride_a_m, stride_a_k,
    stride_b_n, stride_b_k,
    stride_sa_m, stride_sa_k,
    stride_sb_n, stride_sb_k,
    stride_out_m, stride_out_n,
):
    pid = tl.program_id(0)
    n_offset = pid * BLOCK_N
    n_range = n_offset + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # Accumulator: [M, BLOCK_N] in fp32
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # K-dimension loop (BLOCK_K=128 per iteration, matches block scale)
    num_k_blocks = K // BLOCK_K
    for kb in range(num_k_blocks):
        k_start = kb * BLOCK_K
        k_half_start = k_start // 2  # packed index

        # Load activation block: [M, BLOCK_K/2] packed uint8
        m_range = tl.arange(0, M)
        a_half_range = k_half_start + tl.arange(0, BLOCK_K // 2)
        a_packed = tl.load(
            a_ptr + m_range[:, None] * stride_a_m + a_half_range[None, :] * stride_a_k,
            mask=(m_range[:, None] < M),
            other=0,
        )  # [M, BLOCK_K/2] uint8

        # Unpack FP4: each uint8 -> 2 values (low nibble first)
        a_lo = (a_packed & 0x0F).to(tl.uint8)  # [M, BLOCK_K/2]
        a_hi = ((a_packed >> 4) & 0x0F).to(tl.uint8)
        # Interleave to get [M, BLOCK_K] — approximate by processing lo and hi separately

        # Load weight block: [BLOCK_N, BLOCK_K/2] packed uint8
        b_half_range = k_half_start + tl.arange(0, BLOCK_K // 2)
        b_packed = tl.load(
            b_ptr + n_range[:, None] * stride_b_n + b_half_range[None, :] * stride_b_k,
            mask=n_mask[:, None],
            other=0,
        )  # [BLOCK_N, BLOCK_K/2] uint8

        b_lo = (b_packed & 0x0F).to(tl.uint8)
        b_hi = ((b_packed >> 4) & 0x0F).to(tl.uint8)

        # Load block scales: [M, 1] and [BLOCK_N, 1]
        sa = tl.load(
            scale_a_ptr + m_range * stride_sa_m + kb * stride_sa_k,
            mask=(m_range < M),
            other=1.0,
        )  # [M] fp8 -> treated as float
        sb = tl.load(
            scale_b_ptr + n_range * stride_sb_n + kb * stride_sb_k,
            mask=n_mask,
            other=1.0,
        )  # [BLOCK_N]

        # Convert scales from fp8 representation to float
        # The scales are stored as fp8_e4m3 but we load them as raw bytes
        # For now, treat as float32 (the actual conversion depends on storage format)
        sa_f = sa.to(tl.float32)
        sb_f = sb.to(tl.float32)

        # Compute partial products for low and high nibbles
        # For each pair of FP4 values packed in one byte:
        # result[m,n] += lut[a_lo[m,k]] * lut[b_lo[n,k]] * sa[m] * sb[n]
        #              + lut[a_hi[m,k]] * lut[b_hi[n,k]] * sa[m] * sb[n]

        # Since we can't do LUT lookup in Triton efficiently,
        # we'll use the mathematical decomposition of e2m1:
        # value = (-1)^sign * 2^(exp-1) * (1 + mantissa) for exp>0
        # value = (-1)^sign * 0.5 * mantissa for exp=0 (subnormal)
        # This is complex. For the prototype, let's use a simpler approach:
        # convert FP4 to float using bit manipulation

        # FP4 e2m1: bit3=sign, bit2-1=exp, bit0=mantissa
        # We'll compute: value = sign * (2^exp) * (1 + mantissa/2) for exp>0
        #                         sign * 0.5 * mantissa for exp=0

        # For now, accumulate using the packed dot product approach
        # This is a placeholder — the actual implementation needs proper FP4 decode
        pass

    # Apply global alpha and store
    out = acc * alpha
    tl.store(
        out_ptr + m_range[:, None] * stride_out_m + n_range[None, :] * stride_out_n,
        out.to(tl.bfloat16),
        mask=n_mask[None, :],
    )
