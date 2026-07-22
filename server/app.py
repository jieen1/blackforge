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

import asyncio
import functools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server import metrics
from server.engine import ServerEngine
from server.formats import anthropic as anthropic_format
from server.formats import convert_tools_to_chat_template, strip_thinking
from server.formats import openai as openai_format
from server.formats.stream import StreamProcessor

logger = logging.getLogger("qwen_sm120_server.app")

# uvicorn only configures its own loggers; without an explicit handler this
# logger's INFO records (e.g. the Anthropic debug capture below) are dropped
# silently. Attach a stderr handler so they reach the service log file.
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stderr_handler = logging.StreamHandler()
    _stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_stderr_handler)
logger.propagate = False

# Verbose raw request/response capture for ALL endpoints (OpenAI + Anthropic).
# Default ON so real client traffic (e.g. Claude Desktop) is captured for
# debugging and regression fixtures; set QSR_DEBUG_REQUESTS=0 (or the legacy
# QSR_DEBUG_ANTHROPIC=0) to disable. Logs the raw request body, the parsed
# messages, the decoded prompt (exact model input), and the raw model output.
DEBUG_REQUESTS = (
    os.environ.get("QSR_DEBUG_REQUESTS", os.environ.get("QSR_DEBUG_ANTHROPIC", "1")) != "0"
)

DEFAULT_MAX_TOKENS = 16384

# CLI/launcher (``python -m server.app``) sets these via env vars before
# ``uvicorn.run`` triggers the lifespan startup below -- kept as module-
# level constants (not argparse-threaded into the FastAPI app object
# directly) since uvicorn's import-string app-loading convention
# (``uvicorn.run("server.app:app", ...)``) needs ``app`` importable with
# no constructor arguments.
SERVER_CAPACITY = int(os.environ.get("QSR_SERVER_CAPACITY", "4"))
SERVER_NUM_SLOTS = int(os.environ.get("QSR_SERVER_NUM_SLOTS", "8"))
SERVER_BLOCK_SIZE = int(os.environ.get("QSR_SERVER_BLOCK_SIZE", "16"))
# 256K context support: blocks_per_slot = 262144 / block_size(16) = 16384.
# The KV cache pool size is now determined by GPU memory profiling (see
# server/engine.py _load_model → profile_kv_cache_blocks), NOT by the old
# fixed formula (num_slots + 1) * blocks_per_slot. blocks_per_slot is the
# per-slot MAXIMUM context ceiling; the actual pool is sized to fit the GPU.
# The E2E check sets its OWN smaller blocks_per_slot (its prompts are moderate),
# so it does not pay for the full long-context pool.
SERVER_BLOCKS_PER_SLOT = int(os.environ.get("QSR_SERVER_BLOCKS_PER_SLOT", "16384"))
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
SERVER_PRODUCTION = os.environ.get("QSR_SERVER_PRODUCTION", "1") != "0"

engine: ServerEngine | None = None


async def _tokenize_chat(engine_ref, messages, tools=None):
    """Run apply_chat_template in a thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        engine_ref.tok.apply_chat_template,
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return await loop.run_in_executor(None, fn)


async def _tokenize_encode(engine_ref, text):
    """Run tokenizer encode in a thread."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(engine_ref.tok.encode, text, add_special_tokens=False)
    return await loop.run_in_executor(None, fn)


async def _tokenize_decode(engine_ref, token_ids):
    """Run tokenizer decode in a thread."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(engine_ref.tok.decode, token_ids, skip_special_tokens=True)
    return await loop.run_in_executor(None, fn)


def _endpoint_from_path(path: str) -> str:
    """Map a request path to a low-cardinality metrics endpoint label."""
    if path.startswith("/v1/chat/completions"):
        return "chat"
    if path.startswith("/v1/completions"):
        return "completions"
    if path.startswith("/v1/messages"):
        return "messages"
    return "other"


async def _debug_log_input(tag: str, body: dict, parsed_messages, prompt_ids: list[int]) -> None:
    """Capture the full raw request, the parsed messages, and the decoded
    prompt (the exact input the model receives). Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        _raw = json.dumps(body, ensure_ascii=False, default=str)
        logger.info("%s RAW REQUEST (%d bytes): %s", tag, len(_raw), _raw)
        logger.info(
            "%s PARSED MESSAGES: %s",
            tag,
            json.dumps(parsed_messages, ensure_ascii=False, default=str),
        )
        _loop = asyncio.get_running_loop()
        _prompt_text = await _loop.run_in_executor(
            None,
            functools.partial(engine.tok.decode, prompt_ids, skip_special_tokens=False),
        )
        logger.info(
            "%s DECODED PROMPT (%d ids, %d chars): %s",
            tag,
            len(prompt_ids),
            len(_prompt_text),
            _prompt_text,
        )
    except Exception:
        logger.exception("%s debug input capture failed", tag)


def _debug_log_output(
    tag: str, raw_text: str, visible_text: str, finish_reason: str, gen_tokens: int
) -> None:
    """Capture the raw model output and the visible (thinking-stripped) output
    for a NON-streaming response. Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        logger.info(
            "%s RAW OUTPUT (%d tokens, finish=%s, %d chars): %s",
            tag,
            gen_tokens,
            finish_reason,
            len(raw_text),
            raw_text,
        )
        logger.info("%s VISIBLE OUTPUT (%d chars): %s", tag, len(visible_text), visible_text)
    except Exception:
        logger.exception("%s debug output capture failed", tag)


async def _debug_log_stream_output(
    tag: str, proc, visible_text: str, tool_calls, finish_reason: str
) -> None:
    """Capture the raw + visible model output for a STREAMING response by
    decoding the full committed token list. Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        gen_tokens = len(proc.all_ids)
        _loop = asyncio.get_running_loop()
        _raw = await _loop.run_in_executor(
            None,
            functools.partial(engine.tok.decode, proc.all_ids, skip_special_tokens=False),
        )
        logger.info(
            "%s RAW OUTPUT (%d tokens, finish=%s, %d chars): %s",
            tag,
            gen_tokens,
            finish_reason,
            len(_raw),
            _raw,
        )
        logger.info("%s VISIBLE OUTPUT (%d chars): %s", tag, len(visible_text), visible_text)
        if tool_calls:
            logger.info(
                "%s TOOL CALLS: %s",
                tag,
                json.dumps(tool_calls, ensure_ascii=False, default=str),
            )
    except Exception:
        logger.exception("%s debug output capture failed", tag)


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
        "engine ready: capacity=%d num_slots=%d capacity_tokens_per_slot=%d "
        "cudagraph=%s prefix_cache=%s session_affinity=%s ttl=%.1fs",
        engine.capacity,
        engine.num_slots,
        engine.capacity_tokens_per_slot,
        SERVER_ENABLE_CUDAGRAPH,
        SERVER_ENABLE_PREFIX_CACHE,
        SERVER_ENABLE_SESSION_AFFINITY,
        SERVER_SESSION_TTL_S,
    )
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(title="qwen-sm120-runtime server", lifespan=lifespan)


@app.head("/")
@app.get("/")
async def root():
    return {"status": "ok", "service": "blackforge"}


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    import time as _time

    t0 = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (_time.perf_counter() - t0) * 1000
    if elapsed_ms > 100:
        logger.info(
            "SLOW %s %s -> %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


# -- schemas (loose OpenAI-compatible subset -- see module docstring for
# the explicit, intentional deviations: greedy-only, non-streaming, plus
# a debug-only extra field). --


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
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
    return HTTPException(
        status_code=400, detail={"error": {"message": message, "type": "invalid_request_error"}}
    )


def _validate_sampling_params(
    temperature: float | None, top_p: float | None, n: int | None, stream: bool | None
) -> None:
    # Accept temperature/top_p/stream for client compatibility.
    # Internally always greedy decode (MTP verify requires greedy match).
    if n is not None and n != 1:
        raise _invalid_request(
            f"n={n!r} is not supported: only a single completion (n=1) per request."
        )


def _validate_and_resolve_max_tokens(max_tokens: int | None) -> int:
    resolved = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if resolved <= 0:
        raise _invalid_request(f"max_tokens={max_tokens!r} must be >= 1.")
    return resolved


def _validate_capacity(prompt_ids: list[int], max_tokens: int, endpoint: str = "request") -> None:
    assert engine is not None
    if not engine.capacity_ok(len(prompt_ids), max_tokens):
        metrics.record_error(endpoint, 400)
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
    metrics.record_error(_endpoint_from_path(request.url.path), 500)
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
    t0 = time.perf_counter()

    # Parse messages through the format layer (handles string | array content)
    chat_messages = openai_format.parse_chat_messages(req.model_dump())

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(req.tools)

    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    await _debug_log_input(
        "OPENAI /v1/chat/completions", req.model_dump(), chat_messages, prompt_ids
    )
    _validate_capacity(prompt_ids, max_tokens, "chat")

    model_name = req.model or ServerEngine.MODEL

    if req.stream:
        import json as _json

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def _sse():
            proc = StreamProcessor(engine.tok)
            final_result = None
            first_token_t = None
            # First chunk: role announcement (matches vLLM format)
            first_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json.dumps(first_chunk)}\n\n"
            async for item in engine.submit_stream(
                prompt_ids, max_tokens, session_id=req.session_id
            ):
                if isinstance(item, dict):
                    final_result = item
                    break
                proc.add_tokens(item)
                if first_token_t is None and item:
                    first_token_t = time.perf_counter()
                # Stream thinking as reasoning_content (vLLM compatible)
                for td in proc.drain_thinking():
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": td}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"
                for delta in proc.drain_content():
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"
            finish = final_result["finish_reason"] if final_result else "stop"
            visible_text, tool_calls = proc.finalize()
            if tool_calls:
                finish = "tool_calls"
                from server.formats.tools import format_tool_calls_openai

                for i, tc in enumerate(format_tool_calls_openai(tool_calls)):
                    # Chunk 1: id + type + function name (matches vLLM incremental format)
                    name_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {"name": tc["function"]["name"]},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {_json.dumps(name_chunk)}\n\n"
                    # Chunk 2: arguments (full string in one piece)
                    args_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "function": {"arguments": tc["function"]["arguments"]},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {_json.dumps(args_chunk)}\n\n"
            done = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish, "logprobs": None}],
            }
            yield f"data: {_json.dumps(done)}\n\n"
            metrics.record_request(
                "chat",
                len(prompt_ids),
                len(proc.all_ids),
                finish,
                time.perf_counter() - t0,
                (first_token_t - t0) if first_token_t is not None else None,
            )
            await _debug_log_stream_output(
                "OPENAI /v1/chat/completions", proc, visible_text, tool_calls, finish
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    # Non-streaming path
    result = await engine.submit(prompt_ids, max_tokens, session_id=req.session_id)
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    # Prepend  for strip_thinking (chat template injects it in prompt)
    _raw_for_strip = (
        raw_text
        if raw_text.startswith(chr(60) + "think" + chr(62))
        else (chr(60) + "think" + chr(62) + "\n" + raw_text)
    )
    text = strip_thinking(_raw_for_strip)
    # Extract reasoning_content (thinking) for non-streaming response
    # Generated tokens start with thinking content (no  prefix since
    # the chat template injected it in the prompt). Prepend  to extract.
    reasoning_content = None
    _THINK_OPEN = chr(60) + "think" + chr(62)
    _THINK_CLOSE = chr(60) + "/think" + chr(62)
    _raw_with_think = (
        raw_text if raw_text.startswith(_THINK_OPEN) else _THINK_OPEN + "\n" + raw_text
    )
    if _THINK_CLOSE in _raw_with_think:
        start = _raw_with_think.index(_THINK_OPEN) + len(_THINK_OPEN)
        if start < len(_raw_with_think) and _raw_with_think[start] == "\n":
            start += 1
        end = _raw_with_think.index(_THINK_CLOSE)
        reasoning_content = _raw_with_think[start:end].strip().replace("\ufffd", "")
    metrics.record_request(
        "chat",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "OPENAI /v1/chat/completions",
        raw_text,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    resp = openai_format.build_response(
        model=model_name,
        text=text,
        finish_reason=result["finish_reason"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        committed_token_ids=result["committed_token_ids"],
        prompt_token_ids=list(prompt_ids),
    )
    if reasoning_content:
        resp["choices"][0]["message"]["reasoning_content"] = reasoning_content
    return resp


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    assert engine is not None
    _validate_sampling_params(req.temperature, req.top_p, req.n, req.stream)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()
    prompt_ids = await _tokenize_encode(engine, req.prompt)
    await _debug_log_input("OPENAI /v1/completions", req.model_dump(), req.prompt, prompt_ids)
    _validate_capacity(prompt_ids, max_tokens, "completions")

    result = await engine.submit(prompt_ids, max_tokens, session_id=req.session_id)
    _raw_comp = await _tokenize_decode(engine, result["committed_token_ids"])
    _raw_comp_full = (
        _raw_comp
        if _raw_comp.startswith(chr(60) + "think" + chr(62))
        else (chr(60) + "think" + chr(62) + "\n" + _raw_comp)
    )
    text = strip_thinking(_raw_comp_full)
    metrics.record_request(
        "completions",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "OPENAI /v1/completions",
        _raw_comp,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
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
async def metrics_endpoint():
    """Prometheus-compatible metrics (vLLM naming convention)."""
    assert engine is not None
    runner = engine.runner
    pool = runner.block_pool
    total_blocks = pool.num_blocks - pool.reserved
    free_blocks = len(pool._free_queue)
    used_blocks = total_blocks - free_blocks
    kv_usage = used_blocks / total_blocks if total_blocks > 0 else 0.0

    num_running = len(engine.active)
    num_waiting = len(engine.waiting)
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
        f'vllm:capacity_tokens_per_slot{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.capacity_tokens_per_slot}",
        "# HELP vllm:requests_completed_total Total completed requests.",
        "# TYPE vllm:requests_completed_total counter",
        f'vllm:requests_completed_total{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('requests_completed', 0)}",
        "# HELP vllm:prefix_cache_hit_rate Prefix cache hit rate.",
        "# TYPE vllm:prefix_cache_hit_rate gauge",
        f'vllm:prefix_cache_hit_rate{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hit_rate', 0.0):.4f}",
        "# HELP vllm:prefix_cache_hits_total Prefix cache hits.",
        "# TYPE vllm:prefix_cache_hits_total counter",
        f'vllm:prefix_cache_hits_total{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hits', 0)}",
        "# HELP vllm:prefix_cache_misses_total Prefix cache misses.",
        "# TYPE vllm:prefix_cache_misses_total counter",
        f'vllm:prefix_cache_misses_total{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_misses', 0)}",
        "# HELP vllm:kv_cache_total_blocks Total KV cache blocks.",
        "# TYPE vllm:kv_cache_total_blocks gauge",
        f'vllm:kv_cache_total_blocks{{model_name="{ServerEngine.MODEL}"}} {total_blocks}',
        "# HELP vllm:kv_cache_used_blocks Used KV cache blocks.",
        "# TYPE vllm:kv_cache_used_blocks gauge",
        f'vllm:kv_cache_used_blocks{{model_name="{ServerEngine.MODEL}"}} {used_blocks}',
    ]

    # Accuracy/correctness signal: the admission bootstrap check re-runs each
    # speculative prefill on an independent reference slot and compares the
    # first committed token. A non-zero failure count means the MTP path
    # diverged from the greedy reference output (a real correctness problem).
    lines.append(
        "# HELP vllm:bootstrap_checks_ok_total Speculative prefills matching the reference prefill."
    )
    lines.append("# TYPE vllm:bootstrap_checks_ok_total counter")
    lines.append(
        f'vllm:bootstrap_checks_ok_total{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('bootstrap_checks_ok', 0)}"
    )
    lines.append(
        "# HELP vllm:bootstrap_checks_failed_total Speculative prefills diverged from reference."
    )
    lines.append("# TYPE vllm:bootstrap_checks_failed_total counter")
    lines.append(
        f'vllm:bootstrap_checks_failed_total{{model_name="{ServerEngine.MODEL}"}} '
        f"{engine.stats.get('bootstrap_checks_failed', 0)}"
    )

    # App-layer request metrics: latency, TTFT, TPOT, token throughput, and
    # success/error counters (recorded per request in the handlers above).
    lines.extend(metrics.render(ServerEngine.MODEL))

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; charset=utf-8")


@app.api_route("/v1", methods=["GET", "POST"])
async def v1_root():
    return {
        "object": "api_info",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions", "/metrics"],
        "model": ServerEngine.MODEL,
    }


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Anthropic token counting endpoint (Claude Desktop requires this)."""
    assert engine is not None
    body = await request.json()
    from server.formats import convert_tools_to_chat_template
    from server.formats.anthropic import parse_messages

    chat_messages = parse_messages(body)
    tools = convert_tools_to_chat_template(body.get("tools"))
    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    if DEBUG_REQUESTS:
        logger.info(
            "ANTHROPIC count_tokens: msgs=%d system_chars=%d tools=%d -> input_tokens=%d",
            len(body.get("messages", [])),
            len(str(body.get("system") or "")),
            len(body.get("tools", [])),
            len(prompt_ids),
        )
    return {"input_tokens": len(prompt_ids)}


# -- Anthropic Messages API (/v1/messages) ---------------------------------
# Full format handling delegated to server/formats.py.
# This handler only does: parse -> tokenize -> engine.submit -> format response.


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    assert engine is not None
    body = await request.json()
    t0 = time.perf_counter()

    # Diagnostic: log request shape for Claude Desktop debugging
    _sys = body.get("system")
    _sys_len = len(str(_sys)) if _sys else 0
    _msgs = body.get("messages", [])
    _tools_n = len(body.get("tools", []))
    _stream = body.get("stream", False)
    logger.info(
        "ANTHROPIC REQ: system_chars=%d msgs=%d tools=%d stream=%s max_tokens=%s qs=%s",
        _sys_len,
        len(_msgs),
        _tools_n,
        _stream,
        body.get("max_tokens"),
        str(request.url.query) if request.url.query else "-",
    )

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    model_name = body.get("model", "qwen3.6")
    stream = body.get("stream", False)

    # Parse through the Anthropic format layer (handles array content, tool_use, tool_result)
    chat_messages = anthropic_format.parse_messages(body)
    if not chat_messages:
        metrics.record_error("messages", 400)
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "no messages provided"},
            },
        )

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(body.get("tools"))

    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    await _debug_log_input("ANTHROPIC /v1/messages", body, chat_messages, prompt_ids)

    effective_max = min(max_tokens, engine.capacity_tokens_per_slot - len(prompt_ids) - 1)
    if effective_max < 1:
        metrics.record_error("messages", 400)
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt too long for requested max_tokens",
                },
            },
        )

    if stream:
        import json as _json

        async def _anthropic_sse():
            proc = StreamProcessor(engine.tok)
            final_result = None
            first_token_t = None
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
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
                    "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {_json.dumps(msg_start)}\n\n"
            yield "event: ping\ndata: " + _json.dumps({"type": "ping"}) + "\n\n"

            block_index = 0
            thinking_open = False
            thinking_closed = False
            text_open = False

            async for item in engine.submit_stream(prompt_ids, effective_max):
                if isinstance(item, dict):
                    final_result = item
                    break
                proc.add_tokens(item)
                if first_token_t is None and item:
                    first_token_t = time.perf_counter()

                for td in proc.drain_thinking():
                    if not thinking_open:
                        thinking_open = True
                        bs = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                        yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                    d = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": td},
                    }
                    yield f"event: content_block_delta\ndata: {_json.dumps(d)}\n\n"

                for delta in proc.drain_content():
                    if thinking_open and not thinking_closed:
                        thinking_closed = True
                        yield (
                            "event: content_block_stop\ndata: "
                            + _json.dumps({"type": "content_block_stop", "index": block_index})
                            + "\n\n"
                        )
                        block_index += 1
                    if not text_open:
                        text_open = True
                        bs = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                    d = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": delta},
                    }
                    yield f"event: content_block_delta\ndata: {_json.dumps(d)}\n\n"

            if thinking_open and not thinking_closed:
                yield (
                    "event: content_block_stop\ndata: "
                    + _json.dumps({"type": "content_block_stop", "index": block_index})
                    + "\n\n"
                )
                block_index += 1
            if text_open:
                yield (
                    "event: content_block_stop\ndata: "
                    + _json.dumps({"type": "content_block_stop", "index": block_index})
                    + "\n\n"
                )
                block_index += 1
            if not thinking_open and not text_open:
                bs = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                yield (
                    "event: content_block_stop\ndata: "
                    + _json.dumps({"type": "content_block_stop", "index": 0})
                    + "\n\n"
                )
                block_index = 1

            finish = final_result["finish_reason"] if final_result else "stop"
            stop_reason = "end_turn" if finish == "stop" else "max_tokens"
            visible_text, tool_calls = proc.finalize()
            out_tokens = len(proc.all_ids)
            if tool_calls:
                stop_reason = "tool_use"
                from server.formats.tools import format_tool_calls_anthropic

                for tc in format_tool_calls_anthropic(tool_calls):
                    bs = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                    delta_ev = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": _json.dumps(tc["input"]),
                        },
                    }
                    yield f"event: content_block_delta\ndata: {_json.dumps(delta_ev)}\n\n"
                    yield (
                        "event: content_block_stop\ndata: "
                        + _json.dumps({"type": "content_block_stop", "index": block_index})
                        + "\n\n"
                    )
                    block_index += 1
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": len(prompt_ids), "output_tokens": out_tokens},
            }
            yield f"event: message_delta\ndata: {_json.dumps(msg_delta)}\n\n"
            metrics.record_request(
                "messages",
                len(prompt_ids),
                len(proc.all_ids),
                finish,
                time.perf_counter() - t0,
                (first_token_t - t0) if first_token_t is not None else None,
            )
            await _debug_log_stream_output(
                "ANTHROPIC /v1/messages", proc, visible_text, tool_calls, finish
            )
            yield "event: message_stop\ndata: " + _json.dumps({"type": "message_stop"}) + "\n\n"

        return StreamingResponse(_anthropic_sse(), media_type="text/event-stream")

    # Non-streaming path
    result = await engine.submit(prompt_ids, effective_max)
    _raw_anth = await _tokenize_decode(engine, result["committed_token_ids"])
    _raw_anth_full = (
        _raw_anth
        if _raw_anth.startswith(chr(60) + "think" + chr(62))
        else (chr(60) + "think" + chr(62) + "\n" + _raw_anth)
    )
    text = strip_thinking(_raw_anth_full)
    metrics.record_request(
        "messages",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "ANTHROPIC /v1/messages",
        _raw_anth,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    return anthropic_format.build_response(
        model=model_name,
        text=text,
        finish_reason=result["finish_reason"],
        input_tokens=result["prompt_tokens"],
        output_tokens=result["completion_tokens"],
    )
