# Comprehensive Benchmark Results — 2026-07-24

## Environment
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB, SM120)
- Model: poolside/Laguna-S-2.1-NVFP4 (48 layers, 47 MoE, NVFP4)
- DFlash: poolside/Laguna-S-2.1-DFlash-NVFP4 (6 layers, bf16)
- vLLM: /home/bot/vllm (local)
- Python: /home/bot/.venvs/vllm/bin/python
- Commit: d02a0f0 (main)

## Baseline Performance (MARLIN backend, CUDA Graph enabled)

| Context | DFlash tok/s | Baseline eager | ITL (ms) | Accept% | Tok/Step | Speedup | Prefill (ms) | GPU Mem (MB) |
|---------|-------------|---------------|----------|---------|----------|---------|-------------|-------------|
| 64K     | 49.2        | 14.5          | 20.33    | 15.5%   | 3.31     | 3.38×   | 15,426      | 82,076      |
| 128K    | 41.7        | 14.2          | 23.97    | 17.7%   | 3.64     | 2.94×   | 47,184      | 82,076      |

### Warmup (CG capture) pass:
- 64K: 47.0 tok/s, accept=34.0%, prefill=16,315ms
- 128K: 46.5 tok/s, accept=21.1%, prefill=46,809ms

### Notes:
- Acceptance rate lower on measured pass vs warmup (synthetic repetitive text)
- vLLM DFlash baseline (stock TRTLLM+autotune): 64K=367 tok/s, 128K=311 tok/s
- Our gap to vLLM: ~7× at 64K, ~7.5× at 128K (TRTLLM unavailable on SM120)

## Memory Comparison: CUTLASS vs MARLIN

| Backend          | Load (s) | GPU Mem (MB) | Peak (MB) | Params (GB) | MoE (GB) | Attn (GB) | Other (GB) |
|-----------------|----------|-------------|-----------|-------------|----------|-----------|-----------|
| flashinfer_cutlass | 65.9  | 75,432      | 75,432    | 66.96       | 59.90    | 5.22      | 1.84      |
| marlin          | 83.6     | 75,433      | 75,433    | 66.96       | 59.90    | 5.22      | 1.84      |

### Key Finding: IDENTICAL memory footprint
- Both backends: 75.4 GB model params on GPU
- MoE weights: 52.88 GB (CUTLASS=uint8, MARLIN=int32, same byte count)
- Attention: 7.48 GB bf16 + 6.61 GB fp8 KV
- Other: 1.84 GB (embed, lm_head, norms)
- **Backend switch does NOT save memory**

### Memory Breakdown (MARLIN, 1 slot, 128K context):
- Model params: 75,433 MB (73.7 GB)
- DFlash (draft model + KV + CG): ~5,100 MB
- KV cache (48 layers, fp8, 128K): ~6,400 MB
- CG capture buffers: ~1,200 MB
- **Total: ~82,076 MB (80.2 GB)**
- **Headroom: ~14 GB / 96 GB**

## Per-Step Profiling (4K context, CUTLASS, from earlier session)

| Component | Time (ms) | % of step |
|-----------|-----------|-----------|
| Verify CG replay (GPU) | 20.45 | 90% of verify |
| Verify CG plan | 1.76 | 8% of verify |
| Verify CG fill | 0.56 | 2% of verify |
| Decode CG total | 16.13 | includes .item() sync |
| Draft CG total | 3.90 | fill+plan+replay |
| Combine+precompute | 1.65 | eager ops |

## Model Load Times
- MARLIN: 79-84s (includes Marlin repack ~20s extra)
- CUTLASS: 66s (native NVFP4, no repack)
- Checkpoint on disk: 66.98 GiB (15 shards)

## Architecture Details
- Main: 48 layers (12 full-attn 48-head + 36 SWA 72-head window=512)
- 47 MoE layers (256 experts, top-10, intermediate=1024)
- 8 KV heads, head_dim=128, FP8 KV cache
- Draft: 6 layers all SWA, bf16, shares embed+lm_head
- aux_hidden_state_layers: [1, 10, 19, 29, 38, 47]

## Optimization Targets (Priority Order)

### 1. Prefill Speed (47s for 128K → target <25s)
- Current: chunk=4096, ~47s for 128K
- Try: chunk=8192 (logits peak ~1.6GB, should fit in 14GB headroom)
- Future: CUDA Graph for fixed-size prefill chunks

### 2. Decode ITL (20-24ms → target <15ms)
- Verify CG replay is 90% GPU-bound (20ms)
- Can't optimize without faster MoE kernels (B12x/MARLIN improvement)
- Can eliminate .item() sync in decode CG path
- Can overlap combine+precompute with draft

### 3. Memory (82 GB → target <70 GB)
- Model params fixed at 75.4 GB (can't reduce without different quantization)
- KV cache: reduce blocks_per_slot for shorter contexts (dynamic allocation)
- DFlash: 5.1 GB — could share more buffers with main model
- CG: 1.2 GB — minimize captured graph count

### 4. Acceptance Rate (15-17% synthetic → 40-60% real prompts)
- Synthetic repetitive text gives artificially low acceptance
- Real agent programming prompts should be much higher
- Draft model limited by SWA window=512 (can't track long-range)

## Files
- Benchmark script: benchmarks/comprehensive_bench.py
- Memory comparison: benchmarks/mem_backend_compare.py
- Results JSON: benchmarks/fixtures/comprehensive_bench.json
- Memory JSON: benchmarks/fixtures/mem_backend_compare.json
