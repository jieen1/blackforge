# 2026-07-22 Comprehensive Quality Review

Scope: system architecture, code standards, test quality, documentation,
and startup/ops scripts. Review method: static analysis (`ruff`), full unit
test run (`pytest`), `py_compile` on GPU/vLLM-dependent modules, and manual
cross-file reading. Environment: RTX PRO 6000 Blackwell (SM120, CC 12.0),
torch 2.11.0+cu130; the external vLLM fork is **not** installed in the dev
venv, so GPU/vLLM runtime paths were validated statically (`py_compile` +
ruff) rather than executed.

## 1. System architecture

**Strengths**
- Clear, narrow layering that matches the frozen scope (one model, one GPU,
  ≤4 concurrent): `runtime/` (engine, fixed-slot scheduler, hybrid KV/GDN
  cache, OpRegistry), `server/` (OpenAI + Anthropic API, admission, MTP loop,
  prefix cache), `loader/`, `model/`, `oracle/`.
- `OpRegistry` keeps backend-specific calls out of model code (replaceable ops).
- Oracle-comparison methodology (`oracle/`) + frozen W1/W2 workload contracts
  (`benchmarks/workloads.py`, guarded by `tests/test_workloads.py`).
- Prefix cache with block-level refcounting/eviction; session affinity (P4b).

**Findings / risks**
- The runtime drives an **external vLLM fork as a library** plus the
  `sm120-flash-attention` integration (`SM120_VLLM_INTEGRATION` path). Neither
  is vendored or version-pinned here (only `torch==2.11.0` is pinned). This is
  a reproducibility/portability risk. → Documented in README; pinning the
  integration commit is recommended.
- `runtime/direct_model_runner.py` is a single ~6,000-line file (prefill,
  decode, MTP verify, CUDA-graph reconcile). High maintainability cost; not
  unit-testable without GPU. → Flagged; refactor deferred (high risk).
- `runtime/vllm_*_baseline.py` (bridge/stage-b/stage-c/inprocess) are historical
  baselines; `vllm_bridge_backend` is still imported by
  `benchmarks/real_forward_smoke.py`, so they are **not** dead code. → Documented
  as legacy reference baselines in README rather than deleted.
- `model/` (config only) and `kernels/` (docs only) are intentionally thin
  because the model graph + kernels come from the vLLM integration. → Clarified
  in README (was previously easy to misread as "missing implementation").

## 2. Code standards

**Before:** `ruff check .` reported **666 errors** (508 line-too-long, 36
unsorted-imports, 28 f-string-no-placeholders, 25 unused-import, 18
multiple-statements-semicolon, 15 unused-variable, 2 bare-except, 2
lambda-assignment, …). The linter was configured but never enforced; no
formatter was adopted.

**After:** `ruff check .` → **0 errors**; `ruff format --check` clean on all
production packages.
- Substantive fixes done by hand in production code: 2 bare-`except` →
  `except Exception`, 2 lambda assignments → `def`, 2 multiple-statements →
  split, 8 unused local variables removed/ inlined (3 in
  `direct_model_runner.py` verified side-effect-free + no new `F821`).
- Safe auto-fixes (import sorting, unused imports, f-string placeholders,
  quoted annotations, `format`→f-string) applied repo-wide.
- Adopted `ruff format` as the formatter (root-cause fix for line-length +
  consistent style); 10 genuinely un-splittable long strings wrapped manually
  with byte-identical results.
- `benchmarks/` (diagnostic microbench/probe/repro scripts) is style-relaxed via
  `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` (`E501,E701,E702,I001,
  F841`) while bug-catching rules (`F821,F811,F401,E9xx`) stay active.
- Enforcement added: `.pre-commit-config.yaml` (ruff lint + format) and
  `.github/workflows/ci.yml` (ruff lint + format check + unit tests).

## 3. Test quality

**State:** 171 unit tests, all passing in ~1s, CPU-only. Good structure
(`test_<unit>.py`, classes, parametrization). Integration/E2E scripts
(`test_real_world.py`, `test_api_compat.py`, `test_e2e_256k_longctx.py`) are
cleanly excluded from collection via `conftest.py` with documented manual run
commands. torch-dependent tests use `pytest.importorskip`, so the suite runs
without torch/GPU (enables fast CI).

**Gaps / risks**
- The heavy GPU/vLLM-dependent logic (`direct_model_runner.py`, `runtime/engine.py`,
  `server/engine.py`) has **no unit tests**; correctness there relies on oracle
  comparison + manual integration scripts. Inherent to GPU code, but the largest
  correctness risk surface.
- No coverage tooling configured. Recommended: add `pytest-cov` for the
  CPU-testable packages once the engine is exercised in a GPU CI lane.

## 4. Documentation

**Fixed**
- Stale `Limitations`: removed the false "Non-streaming: `stream=true` rejected"
  (streaming is implemented); refined the greedy line to note sampling fields
  are accepted for compatibility but decoding is greedy (verified: the runtime
  is greedy-only, no temperature/top-p implementation).
- Stale `Roadmap`: checked "Streaming response support" (implemented); left
  "Temperature / top-p sampling" unchecked (accurately not implemented).
- Stale architecture-tree test count "27 tests" → "170+ tests".
- Added an architecture clarification (runtime = serving/runtime layer; model
  graph + kernels come from the vLLM integration; legacy baselines explained).
- Added a **Naming** note reconciling BlackForge (product/repo) vs
  `qwen-sm120-runtime` (dir) vs `QSR_` (env prefix).
- Added a **Development** section wiring up the new `make` targets, pre-commit,
  and CI.

**Remaining**
- `PROGRESS.md` is a 251 KB tracked changelog; consider archiving/splitting if
  it keeps growing (no action taken — historical record).

## 5. Startup / ops scripts

**Before:** none (no Makefile, Dockerfile, systemd unit, or startup script).
The server was started only via `python -m server.app` documented in README.

**After:**
- `Makefile` with `help/install/install-cuda/lint/format/format-check/test/
  verify-cuda/workloads/serve/clean`. Verified: `make help`, `make lint`,
  `make format-check`, `make workloads`, `make verify-cuda` (SM120 CC 12.0),
  `make test` all pass.
- `.pre-commit-config.yaml` and `.github/workflows/ci.yml` (see §2).

## Verification evidence

- `ruff check .` → All checks passed!
- `ruff format --check runtime server loader model oracle tests tools` → 58 files already formatted
- `python -m pytest -q` → 171 passed
- `python -m py_compile` on `direct_model_runner.py`, `triton_norm_ops.py`,
  `vllm_bridge_backend.py`, `server/app.py`, `server/engine.py`,
  `runtime/engine.py`, `runtime/hybrid_cache.py` → OK
- `make verify-cuda` → device RTX PRO 6000 Blackwell, capability [12, 0]

## Recommended follow-ups (not done — out of scope / higher risk)

1. Pin/vendor the vLLM fork + sm120-flash-attention integration commit.
2. Split `runtime/direct_model_runner.py` into focused modules.
3. Add a GPU CI lane (or documented nightly) that runs the oracle comparison
   and integration scripts; add `pytest-cov`.
4. Decide retention policy for `benchmarks/` diagnostic scripts and `PROGRESS.md`.
