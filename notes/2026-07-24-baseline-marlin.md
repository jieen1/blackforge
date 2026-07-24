# Baseline: MARLIN MoE + Verify-Only + CUDA Graph (2026-07-24)

## Configuration
- MoE backend: MARLIN (confirmed fastest for SM120)
- Speculative decoding: verify-only (no redundant M=1 decode)
- CUDA Graph: verify (M=16) + draft (M=16)
- KV cache: FP8 (float8_e4m3fn)
- Prefill chunk: 8192 tokens
- gpu_memory_utilization: 0.88

## Results (CG captured at 64K kv_len)

| Context | tok/s | Accept | Tok/Step | Step(ms) | Prefill(ms) |
|---------|-------|--------|----------|----------|-------------|
| Agent (233 tok) | 266.7 | 65.7% | 9.44 | 35.4 | 150 |
| 64K (eager-first) | 77.6 | 51.2% | 7.50 | 96.7 | 12,122 |
| 128K (CG) | 210.9 | 55.1% | 8.23 | 39.0 | 28,098 |

## Step Latency Breakdown (from earlier profiling, 64K CG)
| Stage | ms | % |
|-------|-----|---|
| verify_cg (MoE-limited) | 28.8 | 81.5% |
| draft_cg | 4.4 | 12.6% |
| precompute | 1.6 | 4.4% |
| argmax+accept | 0.5 | 1.3% |
| **TOTAL** | **35.3** | |

## Memory
- Total allocated: 86.9 GB / 96 GB
- Model params: 66.96 GiB (checkpoint index)
- KV + scratch + CG + draft: ~12 GiB
- Load residue: ~8 GiB (auditable)

## Key Findings
1. CG capture MUST happen at large kv_len (≥64K) — capturing at small kv
   causes illegal memory access at large kv (FlashInfer plan grid baked)
2. MARLIN > CUTLASS > B12x for MoE on SM120 (B12x has weight format issues)
3. Prefill is 12s/28s for 64K/128K — major optimization target
4. verify_cg at 28.8ms is 81.5% of step — MoE kernel is the bottleneck

## Optimization Roadmap
1. Prefill: larger chunks (16384), prefill CUDA graph
2. MoE: optimize MARLIN path (routing, shared expert fusion)
3. Memory: reduce load residue, optimize KV allocation
4. Production: multi-slot, streaming, EOS handling

## Update: MoE Backend Comparison (2026-07-24 13:40)

| Backend | 64K tok/s | 64K step | 128K tok/s | 128K step | Memory |
|---------|-----------|----------|------------|-----------|--------|
| **MARLIN** | **192** | **36.7ms** | **191** | **39.2ms** | 86.9GB |
| CUTLASS | 136 | 40.0ms | 171 | 42.5ms | 86.5GB |
| TRTLLM | ❌ SM100 only | | | | |

- TRTLLM: `is_device_capability_family(100)` — SM120 not supported
- CUTEDSL: same SM100 restriction
- vLLM referent (377 tok/s) used CUTLASS (auto), not TRTLLM
- Referent advantage is from vLLM native compile + CG, not MoE backend

## Update: Prefill Chunk Size (2026-07-24 13:42)

| Chunk | 64K prefill | 128K prefill | Memory |
|-------|-------------|--------------|--------|
| 8192 | 11.2s | 26.3s | 86.9GB |
| 16384 | 13.3s | 29.0s | 88.5GB |

chunk=16384 is SLOWER (+15-20%) and uses +1.6GB. Keeping 8192.

## Update: Memory Audit (2026-07-24 13:44)

No load residue. Backend init = 79.4 GB:
- Model params (MARLIN): 71.90 GB = 66.96 GiB checkpoint
- KV cache (48 layers): 6.72 GB
- SWA scratch: 0.64 GB
- Buffers: 0.10 GB
- Unaccounted: 0.04 GB

Generate peak (~87 GB) = backend 79.4 + draft ~2 + CG ~3 + activations ~2.
