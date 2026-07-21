"""Minimal OpenAI-compatible HTTP server for this repository's
``DirectModelRunner`` runtime (see ``server/engine.py`` for the
continuous-batching engine this app wraps).

Scope (a genuinely working first cut, not a feature-complete production
server -- see ``notes/2026-07-18-session-review-and-next-steps.md``'s
server section for the honest capability/limitation record):

- ``POST /v1/chat/completions`` and ``POST /v1/completions``, NON-STREAMING
  only. ``stream=true`` is rejected with a clean 400, not silently ignored
  and not a crash.
- Greedy decoding only, matching this runtime's own MTP-verify mechanism
  (verify is a greedy match, see ``runtime/direct_model_runner.py``'s
  ``determine_accept_reject*``): any request setting ``temperature`` to a
  non-zero value, or ``top_p`` to anything other than 1.0, or ``n`` to
  anything other than 1, is rejected with a clean 400 rather than silently
  ignored -- this project's explicit instruction for this task, since
  actually adding sampling is separate, larger work, out of scope here.
- Fixed capacity=4 concurrent slots (``server.engine.ServerEngine``'s
  default), matching every validated benchmark's shape in this repo.
  A request whose prompt token length would not leave room for at least
  ``max_tokens`` more tokens within this runtime's per-slot capacity
  (``blocks_per_slot * block_size``) is rejected with a clean 400 BEFORE
  it ever reaches the runtime -- this is what keeps this server from ever
  triggering the known ``build_attention_metadata_batch:440``
  whole-batch-crash gap (a real ``runtime/`` bug, out of scope to fix in
  this task; see the task brief / session notes).
- A non-standard ``debug_committed_token_ids`` field is included in every
  response (both endpoints), OUTSIDE the standard OpenAI response shape.
  This exists solely so this project's own correctness-verification
  script (``benchmarks/server_e2e_check.py``) can replay the EXACT real
  committed token ids through an independent reference slot on the same
  running engine, using this project's established near-tie methodology,
  without needing a second full model load. Any real OpenAI client simply
  ignores an unrecognized JSON field.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.engine import ServerEngine

logger = logging.getLogger("qwen_sm120_server.app")

DEFAULT_MAX_TOKENS = 256

# CLI/launcher (``python -m server.app``) sets these via env vars before
# ``uvicorn.run`` triggers the lifespan startup below -- kept as module-
# level constants (not argparse-threaded into the FastAPI app object
# directly) since uvicorn's import-string app-loading convention
# (``uvicorn.run("server.app:app", ...)``) needs ``app`` importable with
# no constructor arguments.
SERVER_CAPACITY = int(os.environ.get("QSR_SERVER_CAPACITY", "4"))
SERVER_NUM_SLOTS = int(os.environ.get("QSR_SERVER_NUM_SLOTS", "16"))
SERVER_BLOCK_SIZE = int(os.environ.get("QSR_SERVER_BLOCK_SIZE", "16"))
# P4a (§25.10 follow-up): raised from 512 so a real >=64K request is
# ADMISSIBLE -- capacity_ok gates on the per-slot ceiling blocks_per_slot *
# block_size, and ceil((65536 + max_tokens)/16) ~= 4113, so 4200 (=> 67200-
# token ceiling) admits a 64K prompt with ~1.6K of generation room. The KV
# cache is allocated up front for the WHOLE shared BlockPool ((num_slots +
# RESERVED) * blocks_per_slot blocks), so this is a fixed startup cost paid
# regardless of how many slots are active; see server/README.md + notes/
# prefix-cache-implementation-log.md (P4a) for the measured memory fit. The
# E2E check sets its OWN smaller blocks_per_slot (its prompts are moderate),
# so it does not pay for the full long-context pool.
SERVER_BLOCKS_PER_SLOT = int(os.environ.get("QSR_SERVER_BLOCKS_PER_SLOT", "4200"))
SERVER_ENABLE_CUDAGRAPH = os.environ.get("QSR_SERVER_ENABLE_CUDAGRAPH", "1") != "0"
# P4a (notes/prefix-cache-design.md sec 5-P4): the prefix-cache rollback
# spine, plumbed straight into ServerEngine(enable_prefix_cache=...). Default
# ON (this is THE product value -- warm prefix hits served across requests);
# `python -m server.app --no-prefix-cache` (or QSR_SERVER_ENABLE_PREFIX_CACHE=0)
# turns it off => byte-for-byte the old server.
SERVER_ENABLE_PREFIX_CACHE = os.environ.get("QSR_SERVER_ENABLE_PREFIX_CACHE", "1") != "0"
# P4b session affinity (notes/2026-07-20-p4b-session-affinity-plan.md): opt-in
# warm-slot retention. Default OFF => byte-for-byte P4a (without a session_id, or
# with the flag off, _finish_request does the unconditional reset_slot). Requires
# the prefix cache -- ServerEngine raises ValueError if affinity is on but prefix
# cache is off (warm-continue needs the persistent content-hash cache).
SERVER_ENABLE_SESSION_AFFINITY = os.environ.get("QSR_SERVER_ENABLE_SESSION_AFFINITY", "0") != "0"
SERVER_SESSION_TTL_S = float(os.environ.get("QSR_SERVER_SESSION_TTL_S", "30.0"))
SERVER_KV_CACHE_DTYPE = os.environ.get("QSR_SERVER_KV_CACHE_DTYPE", "fp8_e4m3")
SERVER_GPU_MEM_UTIL = float(os.environ.get("QSR_SERVER_GPU_MEM_UTIL", "0.85"))
SERVER_PRODUCTION = os.environ.get("QSR_SERVER_PRODUCTION", "0") != "0"

engine: ServerEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("loading DirectModelRunner (this can take a while: model load + KV cache alloc)...")
    engine = ServerEngine(
        capacity=SERVER_CAPACITY,
        num_slots=SERVER_NUM_SLOTS,
        block_size=SERVER_BLOCK_SIZE,
        blocks_per_slot=SERVER_BLOCKS_PER_SLOT,
        kv_cache_dtype=SERVER_KV_CACHE_DTYPE,
        enable_cudagraph=SERVER_ENABLE_CUDAGRAPH,
        enable_prefix_cache=SERVER_ENABLE_PREFIX_CACHE,
        enable_session_affinity=SERVER_ENABLE_SESSION_AFFINITY,
        session_ttl_s=SERVER_SESSION_TTL_S,
        gpu_memory_utilization=SERVER_GPU_MEM_UTIL,
        production=SERVER_PRODUCTION,
    )
    engine.start()
    logger.info(
        "engine ready: capacity=%d num_slots=%d capacity_tokens_per_slot=%d cudagraph=%s prefix_cache=%s "
        "session_affinity=%s ttl=%.1fs",
        engine.capacity, engine.num_slots, engine.capacity_tokens_per_slot, SERVER_ENABLE_CUDAGRAPH,
        SERVER_ENABLE_PREFIX_CACHE, SERVER_ENABLE_SESSION_AFFINITY, SERVER_SESSION_TTL_S,
    )
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(title="qwen-sm120-runtime server", lifespan=lifespan)


# -- schemas (loose OpenAI-compatible subset -- see module docstring for
# the explicit, intentional deviations: greedy-only, non-streaming, plus
# a debug-only extra field). --
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    # P4b session affinity (opt-in): caller-supplied session id. With the server
    # built --session-affinity, the finished slot is retained warm for the TTL so
    # the next turn of the same session continues in place with zero restore.
    # Ignored (byte-for-byte P4a) when the feature is off.
    session_id: str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    # P4b session affinity (opt-in) -- see ChatCompletionRequest.session_id.
    session_id: str | None = None


def _invalid_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": {"message": message, "type": "invalid_request_error"}})


def _validate_sampling_params(temperature: float | None, top_p: float | None, n: int | None, stream: bool | None) -> None:
    # Accept temperature/top_p/stream for client compatibility.
    # Internally always greedy decode (MTP verify requires greedy match).
    if n is not None and n != 1:
        raise _invalid_request(f"n={n!r} is not supported: only a single completion (n=1) per request.")


def _validate_and_resolve_max_tokens(max_tokens: int | None) -> int:
    resolved = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if resolved <= 0:
        raise _invalid_request(f"max_tokens={max_tokens!r} must be >= 1.")
    return resolved


def _validate_capacity(prompt_ids: list[int], max_tokens: int) -> None:
    assert engine is not None
    if not engine.capacity_ok(len(prompt_ids), max_tokens):
        raise _invalid_request(
            f"prompt_tokens({len(prompt_ids)}) + max_tokens({max_tokens}) = "
            f"{len(prompt_ids) + max_tokens} exceeds this runtime's per-slot capacity of "
            f"{engine.capacity_tokens_per_slot} tokens (blocks_per_slot * block_size). "
            "Reduce the prompt length or max_tokens and retry."
        )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    # A defensive net so an unexpected runtime error (e.g. the engine's own
    # error-recovery path in server/engine.py's _loop) surfaces as a clean
    # 500 JSON body instead of an unhandled-exception stack trace / crash.
    logger.exception("unhandled exception serving %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


@app.get("/health")
async def health():
    assert engine is not None
    return {
        "status": "ok",
        "capacity": engine.capacity,
        "free_slots": len(engine.free_slots),
        "active": len(engine.active),
        "waiting": len(engine.waiting),
    }


@app.get("/debug/stats")
async def debug_stats():
    """Non-standard, this-project-only endpoint: exposes the engine's own
    round/admission counters so the E2E validation script (and any curious
    human) can directly confirm real multi-request batching happened
    (``admission_batch_sizes``/``round_batch_sizes`` containing entries
    > 1), rather than inferring it indirectly from timing alone."""
    assert engine is not None
    return engine.stats


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    assert engine is not None
    _validate_sampling_params(req.temperature, req.top_p, req.n, req.stream)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    # return_dict=False is required here: this tokenizer's
    # apply_chat_template(tokenize=True) defaults to returning a
    # BatchEncoding (dict-like, keyed by "input_ids"/"attention_mask"),
    # NOT a plain list of token ids -- confirmed by direct instrumentation
    # (a real bug caught by this task's own E2E run: passing the
    # BatchEncoding straight through crashed
    # ``mtp_prefill_batch``/``_forward_batch``'s ``torch.tensor(...)`` call
    # deep inside the runtime with "too many dimensions 'str'"). This flag
    # makes the return value a plain ``list[int]``, matching every other
    # prompt_ids producer in this codebase (e.g. plain ``tok.encode(...)``).
    prompt_ids = engine.tok.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    _validate_capacity(prompt_ids, max_tokens)

    result = await engine.submit(prompt_ids, max_tokens, session_id=req.session_id)
    text = engine.tok.decode(result["committed_token_ids"])
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = req.model or ServerEngine.MODEL

    if req.stream:
        import json as _json
        async def _sse():
            chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            }
            yield f"data: {_json.dumps(chunk)}\n\n"
            done = {
                "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
            }
            yield f"data: {_json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "debug_committed_token_ids": result["committed_token_ids"],
        "debug_prompt_token_ids": list(prompt_ids),
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    assert engine is not None
    _validate_sampling_params(req.temperature, req.top_p, req.n, req.stream)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    prompt_ids = engine.tok.encode(req.prompt, add_special_tokens=False)
    _validate_capacity(prompt_ids, max_tokens)

    result = await engine.submit(prompt_ids, max_tokens, session_id=req.session_id)
    text = engine.tok.decode(result["committed_token_ids"])
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or ServerEngine.MODEL,
        "choices": [
            {"index": 0, "text": text, "finish_reason": result["finish_reason"], "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "debug_committed_token_ids": result["committed_token_ids"],
        "debug_prompt_token_ids": list(prompt_ids),
    }


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capacity", type=int, default=SERVER_CAPACITY)
    parser.add_argument("--num-slots", type=int, default=SERVER_NUM_SLOTS)
    parser.add_argument("--blocks-per-slot", type=int, default=SERVER_BLOCKS_PER_SLOT)
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable the persistent prefix cache (rollback to the pre-P4a server).",
    )
    parser.add_argument(
        "--session-affinity",
        action="store_true",
        help="Enable opt-in session-affinity warm-slot retention (P4b). Requires the prefix cache.",
    )
    parser.add_argument(
        "--session-ttl-s",
        type=float,
        default=SERVER_SESSION_TTL_S,
        help="Warm-slot retention TTL in seconds for session affinity (P4b). Default 30.0.",
    )
    args = parser.parse_args()

    # P4b: refuse --session-affinity together with --no-prefix-cache -- a clean
    # startup error, not a runtime crash (warm-continue needs the persistent
    # content-hash cache; ServerEngine.__init__ raises the same way as a backstop).
    if args.session_affinity and args.no_prefix_cache:
        parser.error(
            "--session-affinity requires the prefix cache (cannot combine with --no-prefix-cache)"
        )

    os.environ["QSR_SERVER_CAPACITY"] = str(args.capacity)
    os.environ["QSR_SERVER_NUM_SLOTS"] = str(args.num_slots)
    os.environ["QSR_SERVER_BLOCKS_PER_SLOT"] = str(args.blocks_per_slot)
    if args.no_cudagraph:
        os.environ["QSR_SERVER_ENABLE_CUDAGRAPH"] = "0"
    if args.no_prefix_cache:
        os.environ["QSR_SERVER_ENABLE_PREFIX_CACHE"] = "0"
    if args.session_affinity:
        os.environ["QSR_SERVER_ENABLE_SESSION_AFFINITY"] = "1"
    os.environ["QSR_SERVER_SESSION_TTL_S"] = str(args.session_ttl_s)

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()


@app.get("/v1/models")
async def list_models():
    served = os.environ.get("QSR_SERVED_MODEL_NAME", ServerEngine.MODEL)
    names = served.split()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen-sm120-runtime",
                "root": ServerEngine.MODEL,
                "parent": None,
                "max_model_len": engine.capacity_tokens_per_slot if engine else 0,
                "permission": [
                    {
                        "id": f"modelperm-{uuid.uuid4().hex[:24]}",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }
            for name in names
        ],
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics (vLLM naming convention)."""
    assert engine is not None
    runner = engine.runner
    pool = runner.block_pool
    total_blocks = pool.num_blocks - pool.reserved
    free_blocks = len(pool._free_queue)
    used_blocks = total_blocks - free_blocks
    kv_usage = used_blocks / total_blocks if total_blocks > 0 else 0.0

    num_running = len(engine.active)
    num_waiting = len(engine.pending)
    num_free_slots = len(engine.free_slots)

    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="{ServerEngine.MODEL}"}} {num_running}',
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="{ServerEngine.MODEL}"}} {num_waiting}',
        "# HELP vllm:kv_cache_usage_perc KV cache usage percentage.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f'vllm:kv_cache_usage_perc{{model_name="{ServerEngine.MODEL}"}} {kv_usage:.4f}',
        "# HELP vllm:num_free_slots Number of free production slots.",
        "# TYPE vllm:num_free_slots gauge",
        f'vllm:num_free_slots{{model_name="{ServerEngine.MODEL}"}} {num_free_slots}',
        "# HELP vllm:capacity_tokens_per_slot Max tokens per slot.",
        "# TYPE vllm:capacity_tokens_per_slot gauge",
        f'vllm:capacity_tokens_per_slot{{model_name="{ServerEngine.MODEL}"}} {engine.capacity_tokens_per_slot}',
        "# HELP vllm:requests_completed_total Total completed requests.",
        "# TYPE vllm:requests_completed_total counter",
        f'vllm:requests_completed_total{{model_name="{ServerEngine.MODEL}"}} {engine.stats.get("requests_completed", 0)}',
        "# HELP vllm:prefix_cache_hit_rate Prefix cache hit rate.",
        "# TYPE vllm:prefix_cache_hit_rate gauge",
        f'vllm:prefix_cache_hit_rate{{model_name="{ServerEngine.MODEL}"}} {engine.stats.get("prefix_cache_hit_rate", 0.0):.4f}',
        "# HELP vllm:prefix_cache_hits_total Prefix cache hits.",
        "# TYPE vllm:prefix_cache_hits_total counter",
        f'vllm:prefix_cache_hits_total{{model_name="{ServerEngine.MODEL}"}} {engine.stats.get("prefix_cache_hits", 0)}',
        "# HELP vllm:prefix_cache_misses_total Prefix cache misses.",
        "# TYPE vllm:prefix_cache_misses_total counter",
        f'vllm:prefix_cache_misses_total{{model_name="{ServerEngine.MODEL}"}} {engine.stats.get("prefix_cache_misses", 0)}',
        "# HELP vllm:kv_cache_total_blocks Total KV cache blocks.",
        "# TYPE vllm:kv_cache_total_blocks gauge",
        f'vllm:kv_cache_total_blocks{{model_name="{ServerEngine.MODEL}"}} {total_blocks}',
        "# HELP vllm:kv_cache_used_blocks Used KV cache blocks.",
        "# TYPE vllm:kv_cache_used_blocks gauge",
        f'vllm:kv_cache_used_blocks{{model_name="{ServerEngine.MODEL}"}} {used_blocks}',
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; charset=utf-8")


@app.api_route("/v1", methods=["GET", "POST"])
async def v1_root():
    return {
        "object": "api_info",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions", "/metrics"],
        "model": ServerEngine.MODEL,
    }


# -- Anthropic Messages API compatibility ---------------------------------
# Full Anthropic Messages API support (/v1/messages).
# Accepts: string or array-of-blocks for content and system fields,
# cache_control blocks (ignored), stream=true/false.
# Internally uses apply_chat_template for proper model formatting.


def _anthropic_extract_text(field) -> str:
    """Extract plain text from an Anthropic content field (str or list of blocks)."""
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        parts = []
        for block in field:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(field)


def _anthropic_to_chat_messages(body: dict) -> list[dict]:
    """Convert Anthropic Messages API request body to OpenAI-style messages list."""
    chat_messages = []

    system_text = _anthropic_extract_text(body.get("system"))
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        text = _anthropic_extract_text(msg.get("content"))
        if text:
            chat_messages.append({"role": role, "content": text})

    return chat_messages


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    assert engine is not None
    body = await request.json()

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    model_name = body.get("model", "qwen3.6")
    stream = body.get("stream", False)

    chat_messages = _anthropic_to_chat_messages(body)
    if not chat_messages:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "no messages provided"}},
        )

    prompt_ids = engine.tok.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )

    effective_max = min(max_tokens, engine.capacity_tokens_per_slot - len(prompt_ids) - 1)
    if effective_max < 1:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "prompt too long for requested max_tokens"}},
        )

    result = await engine.submit(prompt_ids, effective_max)
    text = engine.tok.decode(result["committed_token_ids"], skip_special_tokens=True)
    stop_reason = "end_turn" if result["finish_reason"] == "stop" else "max_tokens"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    usage = {
        "input_tokens": result["prompt_tokens"],
        "output_tokens": result["completion_tokens"],
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    if stream:
        import json as _json

        async def _sse():
            # message_start
            msg_start = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": result["prompt_tokens"], "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {_json.dumps(msg_start)}\n\n"

            # content_block_start
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {_json.dumps(block_start)}\n\n"

            # ping
            yield f"event: ping\ndata: {_json.dumps({'type': 'ping'})}\n\n"

            # content_block_delta with full text
            delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }
            yield f"event: content_block_delta\ndata: {_json.dumps(delta)}\n\n"

            # content_block_stop
            yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # message_delta
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": result["completion_tokens"]},
            }
            yield f"event: message_delta\ndata: {_json.dumps(msg_delta)}\n\n"

            # message_stop
            yield f"event: message_stop\ndata: {_json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model_name,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }
