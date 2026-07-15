# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains the implementation plan in
`项目实施规划.md`. Build the runtime according to its proposed, intentionally
narrow structure:

- `model/`: Qwen3.6 configuration, layers, and MTP implementation.
- `loader/`: Hugging Face/NVFP4 loading and offline weight packing.
- `runtime/`: engine, fixed-slot scheduler, cache, CUDA Graphs, and sampling.
- `kernels/`: SM120 CUDA kernels and their Python/C++ bindings.
- `server/`: OpenAI-compatible streaming API.
- `oracle/`, `tests/`, and `benchmarks/`: correctness references, regression
  tests, and reproducible performance measurements.

Keep components small and layer boundaries explicit. Register replaceable
operations through a shared `OpRegistry`; do not embed backend-specific calls
throughout model code.

## Build, Test, and Development Commands

Install the lightweight development tools, then run these repository-root
commands:

```bash
python -m pip install -e '.[dev]'        # development dependencies
python -m pytest -q                      # correctness suite
python -m pytest tests/test_gdn.py -q   # focused regression
python -m benchmarks.workloads            # print frozen W1/W2 contracts
```

Keep environment setup, CUDA/toolchain versions, and model-location variables
in `README.md` or dedicated developer documentation. Do not make unit tests require
downloading model weights unless explicitly marked as integration tests.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` for modules/functions,
`PascalCase` for classes, and type annotations on public interfaces. Name CUDA
sources descriptively (for example, `nvfp4_gemm_sm120.cu`). Favor clear,
fixed-scope interfaces over generic abstractions: this runtime supports one
model family, one GPU architecture, one GPU, and at most four concurrent slots.

Add a formatter and linter with the first Python code (for example, Ruff), and
run them before submitting changes. Keep generated packed weights, profiles,
and large checkpoints out of Git.

## Testing Guidelines

Add tests beside each capability using `test_<unit>.py` and
`test_<behavior>` names. Compare model layers, logits, MTP acceptance, and GDN
state against the vLLM oracle. Cover prefill, multi-step decode, slot reset and
reuse, batches 1--4, and CUDA Graph replay. For quantized kernels, record
error metrics and top-k-logit agreement; greedy fixtures must not show
systematic token drift.

## Commits & Pull Requests

There is no Git history yet, so use concise imperative commit subjects, e.g.
`Add GDN state reset test`. Keep commits narrowly scoped. Pull requests should
state the affected workload, correctness evidence, benchmark comparison (using
accepted tokens/s and ITL), hardware/CUDA environment, and any changed runtime
assumptions. Include profiler screenshots only when they substantiate a
performance claim.
