# BlackForge (qwen-sm120-runtime) — developer & ops tasks.
#
# Run `make help` for an overview. Server tuning is done through QSR_*
# environment variables (see README "Configuration"); `make serve` only
# sets the listen address.

PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
# Packages that hold production code (formatted + lint-strict). benchmarks/
# is diagnostic scratch and is lint-relaxed via pyproject per-file-ignores.
PKGS = runtime server loader model oracle tests tools

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev + serving extras (editable)
	$(PYTHON) -m pip install -e '.[dev,serving]'

install-cuda: ## Install the pinned PyTorch CUDA runtime extra
	$(PYTHON) -m pip install -e '.[cuda]'

lint: ## Ruff lint gate for the whole repo (must stay green)
	$(PYTHON) -m ruff check .

format: ## Auto-fix lint issues and format the production packages
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format $(PKGS)

format-check: ## Verify production packages are formatted (no writes)
	$(PYTHON) -m ruff format --check $(PKGS)

test: ## Run the CPU-only unit test suite
	$(PYTHON) -m pytest -q

verify-cuda: ## Smoke-test that an SM120 CUDA op executes
	$(PYTHON) -m tools.verify_cuda

workloads: ## Print the frozen Phase-0 W1/W2 workload contracts
	$(PYTHON) -m benchmarks.workloads

serve: ## Start the OpenAI/Anthropic-compatible server (tune via QSR_* env)
	$(PYTHON) -m server.app --host $(HOST) --port $(PORT)

clean: ## Remove build and test caches
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

.PHONY: help install install-cuda lint format format-check test verify-cuda workloads serve clean
