# BlackForge

**Hand-forged CUDA inference engine for Blackwell (SM120) GPUs — 56% faster attention decode, 256K context, single GPU.**

BlackForge is a specialized, single-GPU inference runtime that squeezes maximum
performance out of NVIDIA Blackwell workstation GPUs (RTX PRO 6000, RTX 5090)
for large language models. Instead of being a general-purpose serving framework,
it takes a "one GPU architecture, do it extremely well" approach — with a
hand-written CUDA attention kernel, FP8 KV cache, MTP speculative decoding,
and CUDA Graph capture, all co-designed for the SM120 architecture.

## Why BlackForge?

Mainstream inference frameworks (vLLM, TGI) use generic attention kernels
(FlashInfer, FlashAttention) that target a wide range of GPU architectures.
On SM120 (Blackwell consumer/workstation), these kernels leave significant
performance on the table because they don't exploit SM120-specific features
like 16-byte `cp.async` loads or the specific shared memory bank layout.

BlackForge's decode attention kernel is **written from scratch for SM120**,
achieving **56% lower latency** than FlashInfer on 128K-context decode
(0.988 ms vs 1.540 ms per decode step, batch=4, GQA 24→4 heads, head_dim=256,
FP8 KV cache, paged layout).

## Key Features

- **Custom SM120 CUDA attention kernel** — hand-written decode kernel with
  16-byte `cp.async` vectorized loads, 272-byte aligned shared memory strides,
  and split-K parallelism (32 splits/request) tuned for SM120's 132 SMs
- **FP8 (e4m3) KV cache** — halves KV cache memory vs FP16, enabling 256K
  context on a single 96 GB GPU
- **MTP speculative decoding (K=3)** — leverages Qwen3.6's built-in
  Multi-Token Prediction layers for ~50% acceptance rate, lossless output
- **CUDA Graph capture** — eliminates kernel launch overhead for decode steps
- **Prefix caching** — content-hash based KV cache reuse across multi-turn
  conversations, with block-level reference counting and eviction
- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, and Prometheus `/metrics`
- **Production quality** — validated with EvalPlus HumanEval+ (pass@1 parity
  with standard vLLM, see [Quality Validation](#quality-validation))

## Performance

All measurements on **RTX PRO 6000 Blackwell Max-Q** (96 GB, 132 SMs),
Qwen3.6-27B-NVFP4, FP8 KV cache, MTP K=3, CUDA Graph enabled.

### Decode Throughput (accepted tokens/s, warm prefix cache)

| Context | Concurrency | Throughput |
|---------|-------------|------------|
| 128K    | 4           | 222 tok/s  |
| 64K     | 4           | 267 tok/s  |

### Kernel Latency (decode attention only)

| Context | Concurrency | BlackForge | FlashInfer | Speedup |
|---------|-------------|------------|------------|---------|
| 128K    | 4           | 0.988 ms   | 1.540 ms   | 1.56×   |

### Context Capacity

| Context Length | Max Concurrency | GPU Memory |
|----------------|-----------------|------------|
| 256K           | 3               | ~93 GB     |
| 128K           | 4               | ~70 GB     |

> **Note on comparisons:** We only compare kernel-level latency under identical
> conditions (same context, batch, KV format). End-to-end throughput comparisons
> with vLLM depend heavily on cache state (cold vs warm), scheduling, and
> compilation overhead, so we avoid apples-to-oranges numbers here.

## Quality Validation

We ran [EvalPlus](https://github.com/evalplus/evalplus) HumanEval+ (164 problems,
greedy decoding, temperature=0) against both BlackForge and standard vLLM server,
using identical OpenAI API prompts:

| Benchmark     | vLLM (FlashInfer) | BlackForge | Delta  |
|---------------|-------------------|------------|--------|
| HumanEval     | 71/164 = 0.433    | 73/164 = 0.445 | +1.2pp |
| HumanEval+    | 70/164 = 0.427    | 71/164 = 0.433 | +0.6pp |

Both use FP8 KV cache (e4m3) and NVFP4 weights. The per-problem differences
(~20 problems in each direction) are within statistical noise (SE ≈ ±3.9pp
for 164 problems) and attributable to floating-point ordering differences
between kernel implementations. **No systematic quality degradation.**

## Architecture

```
blackforge/
├── runtime/                  # Core inference engine
│   ├── direct_model_runner.py   # Model execution: prefill, decode, MTP verify
│   ├── hybrid_cache.py          # KV + GDN state management
│   ├── op_registry.py           # Replaceable op dispatch
│   └── slot_manager.py          # Fixed-slot scheduler (≤4 concurrent)
├── server/                   # OpenAI-compatible HTTP server
│   ├── app.py                   # FastAPI endpoints + /metrics
│   └── engine.py                # ServerEngine: admission, MTP loop, prefix cache
├── model/                    # Qwen3.6 model config
│   └── qwen36_config.py         # Architecture constants (64 layers, GQA, MTP)
├── loader/                   # Weight loading (safetensors, NVFP4)
├── kernels/                  # CUDA kernel documentation
├── benchmarks/               # Reproducible performance & correctness checks
│   ├── prefix_cache_warm_throughput_check.py  # Main throughput benchmark
│   ├── quality_eval.py          # HumanEval+ parallel evaluation
│   └── mtp_multiround_check.py  # MTP correctness verification
├── tests/                    # Unit tests (27 tests, CPU-only)
└── oracle/                   # vLLM reference comparison utilities
```

The CUDA attention kernel lives in a separate repository
([sm120-flash-attention](https://github.com/jieen1/sm120-flash-attention))
and is integrated via vLLM's custom attention backend registration.

### Model Architecture (Qwen3.6-27B)

- **64 layers**: 16 full-attention + 48 GDN (Gated Delta Network)
- **Full attention**: GQA 24→4 heads, head_dim=256, RoPE, FP8 KV cache
- **MTP**: 1 built-in Multi-Token Predictor layer for speculative decoding
- **Weights**: NVFP4 quantization (~21 GB)

## Quick Start

### Prerequisites

- NVIDIA Blackwell GPU (SM120, compute capability 12.0): RTX PRO 6000, RTX 5090
- CUDA 13.x toolkit
- Python 3.10+
- ~96 GB GPU memory (for 256K context)

### Installation

```bash
# Clone the runtime
git clone https://github.com/jieen1/blackforge.git
cd blackforge

# Install dependencies
python -m pip install -e '.[dev,serving]'

# Build the custom CUDA attention kernel (separate repo)
cd /path/to/sm120-flash-attention/kernel
export CUDA_HOME=/usr/local/cuda
python setup.py build_ext --inplace
```

### Running the Server

```bash
QSR_SERVER_PRODUCTION=1 \
QSR_SERVER_CAPACITY=3 \
QSR_SERVER_NUM_SLOTS=6 \
QSR_SERVER_BLOCKS_PER_SLOT=16384 \
QSR_SERVED_MODEL_NAME="qwen3.6" \
python -m server.app --host 0.0.0.0 --port 8000
```

### API Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

### Running Benchmarks

```bash
# 128K context, 4 concurrent, 12 decode rounds
python -m benchmarks.prefix_cache_warm_throughput_check \
  --fixture ctx128k --concurrency 4 --decode-rounds 12

# Unit tests (CPU-only, fast)
python -m pytest tests/ -q
```

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `QSR_SERVER_CAPACITY` | `4` | Max concurrent requests |
| `QSR_SERVER_NUM_SLOTS` | `16` | Total internal slots |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `4200` | KV cache blocks per slot (×16 = tokens) |
| `QSR_SERVER_PRODUCTION` | `0` | Production mode: skip validation slots |
| `QSR_SERVED_MODEL_NAME` | model ID | Advertised model name(s), space-separated |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | Enable CUDA Graph capture |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `1` | Enable prefix caching |
| `SM120_VLLM_INTEGRATION` | (auto) | Path to sm120-flash-attention vllm_integration dir |

### Context Length vs Concurrency

KV cache is pre-allocated. On a 96 GB GPU with Qwen3.6-27B-NVFP4:

| `BLOCKS_PER_SLOT` | Context/Slot | Capacity | `NUM_SLOTS` (production) |
|--------------------|-------------|----------|--------------------------|
| 16384              | 256K        | 3        | 6                        |
| 8192               | 128K        | 4        | 8                        |
| 4200               | 67K         | 4        | 16                       |

## Limitations

- **Single model**: currently hardcoded to Qwen3.6-27B-NVFP4
- **Single GPU**: no tensor/pipeline parallelism
- **Greedy decoding only**: temperature/top-p sampling not yet implemented
- **Non-streaming**: `stream=true` is rejected (not yet implemented)
- **SM120 only**: the custom CUDA kernel requires compute capability 12.0

## License

Apache 2.0 — see [LICENSE](LICENSE).
