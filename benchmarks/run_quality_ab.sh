#!/usr/bin/env bash
# Quality A/B orchestration: our custom runtime vs the original model on stock vLLM.
#
# IMPORTANT (fair comparison): both backends must serve the SAME weights
# (unsloth/Qwen3.6-27B-NVFP4) so the test isolates the backend, not the quant.
# Our runtime serves it as model "qwen3.6-rt"; for the vLLM side, start vLLM on
# the same repo (the 2026-07-21 baseline did exactly this).
#
# Flow (restart required between backends; single GPU holds one model):
#   1) Ensure our runtime is up (~/vllm_server/vllm_ctl.sh start qwen3.6-rt), then:
#        benchmarks/run_quality_ab.sh ours
#   2) Switch to stock vLLM on the same NVFP4 weights, then:
#        benchmarks/run_quality_ab.sh vllm
#   3) Regression gate (exit 1 on regression):
#        benchmarks/run_quality_ab.sh compare
#
# Each `ours`/`vllm` run executes all four dimensions (tool, agent, longctx, code).
# code (HumanEval+, 164 problems @ 4096 tok) is the slow part (~10-20 min).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/bot/.venvs/vllm/bin/python
URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen3.6}"
OUT=evalplus_results/quality
mkdir -p "$OUT"

run_suite() {
  local label="$1"
  echo ">>> Running full quality suite as '${label}' against ${URL} model=${MODEL}"
  "$PY" -u benchmarks/quality_regression.py --base-url "$URL" --model "$MODEL" \
      --label "$label" --dims tool,agent,longctx,code --concurrency 4 \
      --max-tokens-code 4096 --out "$OUT/${label}.json"
}

case "${1:-}" in
  ours)   run_suite our_runtime ;;
  vllm)   run_suite vllm_baseline ;;
  compare)
    "$PY" benchmarks/quality_compare.py \
        --candidate "$OUT/our_runtime.json" \
        --baseline  "$OUT/vllm_baseline.json"
    ;;
  *) echo "usage: $0 {ours|vllm|compare}"; exit 2 ;;
esac
