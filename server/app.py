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

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
SERVER_BLOCKS_PER_SLOT = int(os.environ.get("QSR_SERVER_BLOCKS_PER_SLOT", "512"))
SERVER_ENABLE_CUDAGRAPH = os.environ.get("QSR_SERVER_ENABLE_CUDAGRAPH", "1") != "0"
SERVER_KV_CACHE_DTYPE = os.environ.get("QSR_SERVER_KV_CACHE_DTYPE", "fp8_e4m3")
SERVER_GPU_MEM_UTIL = float(os.environ.get("QSR_SERVER_GPU_MEM_UTIL", "0.85"))

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
        gpu_memory_utilization=SERVER_GPU_MEM_UTIL,
    )
    engine.start()
    logger.info(
        "engine ready: capacity=%d num_slots=%d capacity_tokens_per_slot=%d cudagraph=%s",
        engine.capacity, engine.num_slots, engine.capacity_tokens_per_slot, SERVER_ENABLE_CUDAGRAPH,
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


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False


def _invalid_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": {"message": message, "type": "invalid_request_error"}})


def _validate_sampling_params(temperature: float | None, top_p: float | None, n: int | None, stream: bool | None) -> None:
    if stream:
        raise _invalid_request(
            "stream=true is not supported by this server (non-streaming only, first cut)."
        )
    if temperature is not None and temperature != 0:
        raise _invalid_request(
            f"temperature={temperature!r} is not supported: this runtime is greedy-decode only "
            "(MTP verify is a greedy match, see runtime/direct_model_runner.py). "
            "Omit temperature or set it to 0."
        )
    if top_p is not None and top_p != 1.0:
        raise _invalid_request(
            f"top_p={top_p!r} is not supported: this runtime does not implement nucleus sampling. "
            "Omit top_p or set it to 1.0."
        )
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

    result = await engine.submit(prompt_ids, max_tokens)
    text = engine.tok.decode(result["committed_token_ids"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or ServerEngine.MODEL,
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

    result = await engine.submit(prompt_ids, max_tokens)
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
    args = parser.parse_args()

    os.environ["QSR_SERVER_CAPACITY"] = str(args.capacity)
    os.environ["QSR_SERVER_NUM_SLOTS"] = str(args.num_slots)
    os.environ["QSR_SERVER_BLOCKS_PER_SLOT"] = str(args.blocks_per_slot)
    if args.no_cudagraph:
        os.environ["QSR_SERVER_ENABLE_CUDAGRAPH"] = "0"

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
