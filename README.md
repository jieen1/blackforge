# BlackForge

**Blackwell inference, forged for speed.**

BlackForge is a full-stack inference engine built from the ground up for
NVIDIA Blackwell (SM120) GPUs. It features hand-written CUDA attention
kernels, FP8 KV cache, MTP speculative decoding, and CUDA Graph capture —
all co-designed to extract maximum performance from SM120's unique hardware.

Currently optimized for **Qwen3.6-27B** (NVFP4). More model support coming.

[中文文档](#中文文档)

---

## Why BlackForge?

Mainstream inference frameworks (vLLM, TGI) use generic attention kernels
(FlashInfer, FlashAttention) targeting a wide range of GPU architectures.
On SM120 (Blackwell consumer/workstation), these kernels leave significant
performance on the table — they don't exploit SM120-specific features like
16-byte `cp.async` loads or the specific shared memory bank layout.

BlackForge's decode attention kernel is **written from scratch for SM120**,
achieving **56% lower latency** than FlashInfer on 128K-context decode
(0.988 ms vs 1.540 ms per decode step, batch=4, GQA 24→4 heads,
head_dim=256, FP8 KV cache, paged layout).

## Key Features

- **Custom SM120 CUDA attention kernel** — 16-byte `cp.async` vectorized
  loads, 272-byte aligned shared memory strides, split-K parallelism
  (32 splits/request) tuned for SM120's 132 SMs
- **FP8 (e4m3) KV cache** — halves KV memory vs FP16, enabling 256K
  context on a single 96 GB GPU
- **MTP speculative decoding (K=3)** — leverages Qwen3.6's built-in
  Multi-Token Prediction layers, ~50% acceptance rate, lossless output
- **CUDA Graph capture** — eliminates kernel launch overhead for decode
- **Prefix caching** — content-hash KV cache reuse across multi-turn
  conversations, block-level reference counting and eviction
- **OpenAI + Anthropic compatible API** — `/v1/chat/completions`,
  `/v1/completions`, `/v1/messages`, `/v1/models`, Prometheus `/metrics`
- **Quality validated** — EvalPlus HumanEval+ pass@1 parity with
  standard vLLM (see [Quality Validation](#quality-validation))

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

> **Note:** We only compare kernel-level latency under identical conditions.
> End-to-end throughput comparisons with vLLM depend on cache state,
> scheduling, and compilation overhead, so we avoid apples-to-oranges numbers.

## Quality Validation

[EvalPlus](https://github.com/evalplus/evalplus) HumanEval+ (164 problems,
greedy decoding, temperature=0), identical OpenAI API prompts on both servers:

| Benchmark  | vLLM (FlashInfer)    | BlackForge           | Delta  |
|------------|----------------------|----------------------|--------|
| HumanEval  | 71/164 = 0.433       | 73/164 = 0.445       | +1.2pp |
| HumanEval+ | 70/164 = 0.427       | 71/164 = 0.433       | +0.6pp |

Both use FP8 KV cache (e4m3) and NVFP4 weights. Per-problem differences
(~20 in each direction) are within statistical noise (SE ≈ ±3.9pp).
**No systematic quality degradation.**

## Architecture

```
blackforge/
├── runtime/                  # Core inference engine
│   ├── direct_model_runner.py   # Prefill, decode, MTP verify
│   ├── hybrid_cache.py          # KV + GDN state management
│   ├── op_registry.py           # Replaceable op dispatch
│   └── slot_manager.py          # Fixed-slot scheduler (≤4 concurrent)
├── server/                   # HTTP server (OpenAI + Anthropic API)
│   ├── app.py                   # FastAPI endpoints + /metrics
│   └── engine.py                # Admission, MTP loop, prefix cache
├── model/                    # Model architecture config
├── loader/                   # Weight loading (safetensors, NVFP4)
├── kernels/                  # CUDA kernel documentation
├── benchmarks/               # Reproducible perf & correctness checks
├── tests/                    # Unit tests (27 tests, CPU-only)
└── oracle/                   # vLLM reference comparison utilities
```

The CUDA attention kernel lives in
[sm120-flash-attention](https://github.com/jieen1/sm120-flash-attention),
integrated via vLLM's custom attention backend registration.

## Quick Start

### Prerequisites

- NVIDIA Blackwell GPU (SM120, CC 12.0): RTX PRO 6000, RTX 5090
- CUDA 13.x, Python 3.10+, ~96 GB GPU memory (for 256K context)

### Installation

```bash
git clone https://github.com/jieen1/blackforge.git
cd blackforge
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
SM120_VLLM_INTEGRATION=/path/to/sm120-flash-attention/vllm_integration \
python -m server.app --host 0.0.0.0 --port 8000
```

### API Usage

```bash
# OpenAI format
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'

# Anthropic format
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'
```

### Benchmarks

```bash
python -m benchmarks.prefix_cache_warm_throughput_check \
  --fixture ctx128k --concurrency 4 --decode-rounds 12

python -m pytest tests/ -q
```

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `QSR_SERVER_CAPACITY` | `4` | Max concurrent requests |
| `QSR_SERVER_NUM_SLOTS` | `16` | Total internal slots |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `4200` | KV blocks per slot (×16 = tokens) |
| `QSR_SERVER_PRODUCTION` | `0` | Production mode: skip validation slots |
| `QSR_SERVED_MODEL_NAME` | model ID | Advertised model name(s) |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | Enable CUDA Graph capture |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `1` | Enable prefix caching |
| `SM120_VLLM_INTEGRATION` | (auto) | Path to sm120-flash-attention integration |

### Context Length vs Concurrency (96 GB GPU)

| `BLOCKS_PER_SLOT` | Context/Slot | Capacity | `NUM_SLOTS` (production) |
|--------------------|-------------|----------|--------------------------|
| 16384              | 256K        | 3        | 6                        |
| 8192               | 128K        | 4        | 8                        |
| 4200               | 67K         | 4        | 16                       |

## Roadmap

- [ ] More model support (Qwen3 series, other hybrid architectures)
- [ ] Streaming response support
- [ ] Temperature / top-p sampling
- [ ] Dynamic KV cache allocation (flexible context vs concurrency)
- [ ] Multi-GPU support

## Limitations

- **Single model**: currently Qwen3.6-27B-NVFP4 only
- **Single GPU**: no tensor/pipeline parallelism
- **Greedy decoding only**: sampling not yet implemented
- **Non-streaming**: `stream=true` rejected
- **SM120 only**: requires compute capability 12.0

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 中文文档

### 项目简介

BlackForge 是一个专为 NVIDIA Blackwell（SM120）GPU 打造的全栈推理引擎。
通过手写 CUDA attention kernel、FP8 KV cache、MTP 投机解码和 CUDA Graph
捕获等深度优化，在 SM120 架构上实现极致推理性能。

当前已适配 **Qwen3.6-27B**（NVFP4 量化），后续将支持更多模型。

### 核心优势

- **自研 SM120 CUDA kernel** — 针对 Blackwell 硬件特性手写，decode attention
  比 FlashInfer 快 56%（128K 上下文，0.988ms vs 1.540ms）
- **FP8 KV cache** — 显存占用减半，单卡 96GB 支持 256K 上下文
- **MTP 投机解码** — 利用 Qwen3.6 内置 MTP 层，~50% 接受率，输出无损
- **质量验证** — HumanEval+ pass@1 与标准 vLLM 持平，无质量下降

### 性能数据

测试环境：RTX PRO 6000 Blackwell Max-Q（96GB，132 SMs）

| 上下文 | 并发 | 吞吐量（warm） |
|--------|------|----------------|
| 128K   | 4    | 222 tok/s      |
| 64K    | 4    | 267 tok/s      |

### 快速开始

```bash
git clone https://github.com/jieen1/blackforge.git
cd blackforge
python -m pip install -e '.[dev,serving]'

# 启动服务（256K 上下文，3 并发）
QSR_SERVER_PRODUCTION=1 \
QSR_SERVER_CAPACITY=3 \
QSR_SERVER_NUM_SLOTS=6 \
QSR_SERVER_BLOCKS_PER_SLOT=16384 \
QSR_SERVED_MODEL_NAME="qwen3.6" \
python -m server.app --host 0.0.0.0 --port 8000
```

### 许可证

Apache 2.0
