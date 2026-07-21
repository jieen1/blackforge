# Server

The server exposes the fixed-slot engine (`server/engine.py`'s
`ServerEngine`, a thin continuous-batching wrapper around the ONE
production-validated `DirectModelRunner` runtime in
`runtime/direct_model_runner.py`) through an OpenAI-compatible interface.
Do not add multi-model or multi-GPU routing here.

## Endpoints

- `POST /v1/chat/completions`, `POST /v1/completions` — non-streaming,
  greedy-only (temperature 0 / top_p 1.0 / n 1; anything else is a clean
  400, not a crash). A request whose `prompt + max_tokens` would exceed the
  per-slot capacity is rejected with a clean 400 BEFORE it reaches the
  runtime. Each response carries non-standard
  `debug_committed_token_ids` / `debug_prompt_token_ids`, used solely by
  `benchmarks/server_e2e_check.py` (any real OpenAI client ignores them).
- `GET /health` — liveness + slot occupancy.
- `GET /debug/stats` — the engine's admission/round counters plus the P0
  prompt-prefix-overlap, P4a prefix-cache hit-rate, and P4b session-affinity
  instrumentation.

## Configuration

Set via env (read at import) or `python -m server.app` flags:

| Env | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `QSR_SERVER_CAPACITY` | `--capacity` | 4 | concurrent production slots |
| `QSR_SERVER_NUM_SLOTS` | `--num-slots` | 16 | total slots (prod + ref + diag + captured-graph warmup) |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `--blocks-per-slot` | 4200 | per-slot KV ceiling (`× block_size` tokens) |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `--no-cudagraph` | 1 | captured decode graph |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `--no-prefix-cache` | 1 | persistent prefix cache (P4a) |
| `QSR_SERVER_ENABLE_SESSION_AFFINITY` | `--session-affinity` | 0 | opt-in session-affinity warm-slot retention (P4b) |
| `QSR_SERVER_SESSION_TTL_S` | `--session-ttl-s` | 30.0 | warm-slot retention TTL in seconds (P4b) |
| `QSR_SERVER_KV_CACHE_DTYPE` | — | fp8_e4m3 | KV cache dtype |
| `QSR_SERVER_GPU_MEM_UTIL` | — | 0.85 | vLLM `gpu_memory_utilization` |

### Prefix cache (P4a)

Default **ON** — this is the product value: the server serves warm prefix
hits across requests via the P0→P3 persistent content-addressed cache
(`enable_block_table` + `enable_prefix_cache` +
`enable_persistent_prefix_cache` on the runner). Turn 2+ of a growing
conversation, or unrelated requests sharing a system prompt / codebase
bundle, skip re-prefilling the cached prefix. The hit depth `L` and the
prefill tokens saved are reported by `/debug/stats` under
`prefix_cache_hits` / `prefix_cache_misses` / `prefix_cache_hit_rate` /
`prefix_cache_hit_L_samples` / `prefix_cache_hit_tokens_saved`.

`--no-prefix-cache` (or `QSR_SERVER_ENABLE_PREFIX_CACHE=0`) rolls back to
the pre-P4a server **byte-for-byte**: the runner is constructed with no
cache flags, `mtp_prefill_with_cache` delegates straight to
`mtp_prefill_batch`, and `prefix_cache_hits` stays 0. `_finish_request`
still does an unconditional `reset_slot` in P4a — the content-hash cache
survives reset by design (R10). Session affinity (`session_id` + warm-slot
TTL retention) is the separate P4b optimization (below).

### Session affinity (P4b)

Default **OFF** — opt-in via `--session-affinity` (or
`QSR_SERVER_ENABLE_SESSION_AFFINITY=1`). When a request carries a `session_id`,
the server retains the finished slot **warm** for `QSR_SERVER_SESSION_TTL_S`
(default 30.0s) so the next turn of the same session continues IN PLACE with
**zero restore** — it skips the GDN-checkpoint copy + block `touch` that a
content-hash hit (P4a) otherwise performs, via the runtime's gated
`mtp_prefill_warm_continue`. The content-hash cache (P4a) remains the
correctness-bearing fallback: if the next turn's prompt does not reproduce the
retained slot's committed prefix exactly (the tokenizer re-segments at the turn
boundary, and the runtime's committed boundary can run a few tokens past the
`max_tokens`-truncated response), the request falls back to the normal
cold/content-hash path. A retained slot's blocks stay pinned (`ref_cnt >= 1`)
against `capacity=4`, bounded by the TTL; expiry (or shutdown/crash) resets +
releases them exactly once (published blocks keep their hash, so they stay
hit-able at `ref_cnt == 0`, R10). Requires the prefix cache:
`--session-affinity` + `--no-prefix-cache` is refused at startup. Without a
`session_id` (or with the flag off) behavior is byte-for-byte P4a
(`_finish_request` does the unconditional `reset_slot`). `/debug/stats` reports
`session_warm_continuations` (+ a bounded `session_warm_continuation_samples`
list), `session_retentions`, `session_expirations`, and `session_warm_fallbacks`;
a warm turn advances `session_warm_continuations` but NOT `prefix_cache_hits`
(the definitive zero-restore signal).

### Long context (`blocks_per_slot`)

`blocks_per_slot` is the fixed per-slot KV capacity ceiling
(`blocks_per_slot * block_size` tokens); `capacity_ok` rejects any request
whose `prompt + max_tokens + K` would exceed it BEFORE it reaches the
runtime. The default 4200 (⇒ 67200-token ceiling) admits a real ≥64K
request (`ceil((65536 + max_tokens)/16) ≈ 4113`). The KV cache is allocated
up front for the whole shared `BlockPool` (`(num_slots + 1) *
blocks_per_slot` blocks), so this is a fixed startup cost paid regardless
of how many slots are active: at `num_slots=16` / `blocks_per_slot=4200`
the server starts at ~79.5 GiB of the 95.6 GiB card (verified — see
`notes/prefix-cache-implementation-log.md`, P4a). A single 64K request is
admissible; 4×64K-simultaneous is gated by prefill activation memory (a
pre-existing runtime characteristic, session-notes §16.4), not block
availability — reduced-concurrency 64K is the achievable shape.

## Validation

`python -m benchmarks.server_e2e_check` starts the real server (uvicorn +
genuine HTTP over a real socket) and verifies: a basic round-trip,
independent-reference-replay correctness, genuine concurrent batching,
defensive rejections, the P4a turn-1/turn-2 prefix-cache hit (the hit
engages, INV1 end-to-end token match vs an independent cold reference, and
a TTFT drop), and the P4b session-affinity warm-continue (turn 2 continues
the retained warm slot in place: `session_warm_continuations` +1 while
`prefix_cache_hits` does NOT advance — zero restore — plus the exact-prefix
precondition, INV1 end-to-end match, and a TTFT drop).
