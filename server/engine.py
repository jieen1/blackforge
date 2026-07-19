"""``ServerEngine``: a thin, always-on continuous-batching wrapper around
``DirectModelRunner`` (see ``runtime/direct_model_runner.py``), the ONLY
production-validated driving path in this repository (per the 2026-07-19
audit -- ``runtime/engine.py``/``hybrid_cache.py``/``slot_manager.py``/
``op_registry.py``/``vllm_bridge_backend.py`` are an older, now-bypassed
architecture nothing validated actually uses; this module does not import
or build on any of them).

Design: this is ``benchmarks/mtp_sustained_realistic_workload_check.py``'s
``_run_sustained`` admission/scheduling loop (arrival queue, free-slot
admission, mid-flight admission, MTP verify/commit loop -- capacity=4,
K=3, ``enable_cudagraph=True``, ref/diag slot reservation), adapted from a
synthetic wall-clock-gated request pool to a live ``asyncio`` loop driven
by real incoming HTTP requests. It intentionally reuses the SAME slot
layout that script (and ``mtp_async_arrival_check.py`` before it)
established: for ``capacity`` production slots ``0..capacity-1``, a
dedicated reference slot per production slot at ``capacity+p`` (used for
an always-on, cheap admission-time bootstrap correctness check -- the
same technique ``_run_sustained`` uses every admission) and a dedicated
margin-diagnostic slot at ``2*capacity+p`` (only touched on an actual
first-token divergence). ``num_slots`` must be >= ``3*capacity`` for this
reservation, and if ``enable_cudagraph`` is on, >= ``2*capacity`` again
for the captured verify graph's own warmup reservation (see
``CapturedBatchDecodeGraph``'s docstring) -- the project's own established
``num_slots=16``/``capacity=4`` default satisfies both simultaneously and
is reused here unchanged.

Single-event-loop design, no background thread: the engine's round loop
is an ``asyncio`` task running in the SAME event loop FastAPI serves HTTP
requests on. Each loop iteration does one (possibly-empty) admission step
plus, if any slot is active, exactly one real (blocking, synchronous)
``mtp_verify_and_commit_batch`` round, then yields via
``await asyncio.sleep(...)``. A blocking GPU round briefly pauses request
intake, but round durations are small (single-digit-to-tens of ms at this
project's validated shapes), so multiple HTTP requests arriving within one
inter-round gap reliably land in ``waiting`` together and get admitted in
ONE ragged ``mtp_prefill_batch`` call -- genuine continuous batching, not
one-request-at-a-time serialization. This also avoids running CUDA calls
from more than one Python thread, sidestepping any cross-thread CUDA
context question entirely.

Explicitly out of scope for this pass (see module docstring of
``server/app.py`` and the session notes for the full list): streaming,
temperature/top-p sampling (greedy/MTP-verify only), request cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"

logger = logging.getLogger("qwen_sm120_server.engine")


@dataclass
class GenerationRequest:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    future: "asyncio.Future[dict]"
    created_t: float = field(default_factory=time.perf_counter)


class ServerEngine:
    """Owns the one ``DirectModelRunner`` instance this process ever
    constructs, plus the admission/round-robin bookkeeping needed to drive
    it as a live, continuously-batched service."""

    MODEL = "unsloth/Qwen3.6-27B-NVFP4"
    K = 3

    def __init__(
        self,
        *,
        capacity: int = 4,
        num_slots: int = 16,
        block_size: int = 16,
        blocks_per_slot: int = 512,
        kv_cache_dtype: str = "fp8_e4m3",
        enable_cudagraph: bool = True,
        gpu_memory_utilization: float = 0.85,
        idle_sleep_s: float = 0.005,
    ) -> None:
        # Slot layout (matches mtp_sustained_realistic_workload_check.py's
        # _run_sustained exactly): capacity production slots [0, capacity),
        # capacity dedicated reference slots [capacity, 2*capacity) (the
        # always-on admission bootstrap check), capacity dedicated
        # margin-diagnostic slots [2*capacity, 3*capacity) (only touched on
        # an actual divergence). If enable_cudagraph, CapturedBatchDecodeGraph
        # /CapturedMTPDraftStepGraph additionally reserve the LAST capacity
        # slots of num_slots permanently for their own disposable warmup
        # (see those classes' docstrings) -- confirmed to always be a
        # suffix of [num_slots - capacity, num_slots), so as long as that
        # whole top range is otherwise unused, any real batch_size <=
        # capacity graph stays confined to it.
        min_slots = 3 * capacity + (capacity if enable_cudagraph else 0)
        if num_slots < min_slots:
            raise ValueError(
                f"num_slots={num_slots} must be >= {min_slots} for capacity={capacity}, "
                f"enable_cudagraph={enable_cudagraph} (production + reference + "
                "margin-diagnostic slots, plus captured-graph warmup reservation if cudagraph is on)"
            )

        sys.path.insert(0, SM120_VLLM_INTEGRATION)
        import register_sm120_backend  # noqa: F401
        from transformers import AutoTokenizer

        from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

        # Reused verbatim (not re-implemented) -- see this project's own
        # "reuse this project's own established validation method" convention,
        # already applied the same way by mtp_sustained_realistic_workload_check.py.
        from benchmarks.mtp_async_arrival_check import NEAR_TIE_LOGIT_MARGIN, _near_tie_margin_diag

        self._near_tie_margin_diag = _near_tie_margin_diag
        self.near_tie_logit_margin = NEAR_TIE_LOGIT_MARGIN

        self.capacity = capacity
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.capacity_tokens_per_slot = block_size * blocks_per_slot
        self.idle_sleep_s = idle_sleep_s

        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.eos_token_id = self.tok.eos_token_id

        max_model_len = max(8192, self.capacity_tokens_per_slot + 256)
        vllm_config = build_vllm_config(
            model=self.MODEL,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": self.K,
                "attention_backend": "CUSTOM",
            },
        )
        self.runner = DirectModelRunner(
            vllm_config,
            num_slots=num_slots,
            block_size=block_size,
            blocks_per_slot=blocks_per_slot,
            enable_cudagraph=enable_cudagraph,
        )

        self.free_slots: list[int] = list(range(capacity))
        self.active: dict[int, dict[str, Any]] = {}
        self.waiting: list[GenerationRequest] = []
        self.pending: list[GenerationRequest] = []
        self.ref_slot_for = {p: capacity + p for p in range(capacity)}
        self.diag_slot_for = {p: 2 * capacity + p for p in range(capacity)}

        self._stop = False
        self._task: asyncio.Task | None = None

        # Observability only -- proves real batching happened (item 4 of
        # this task), not a correctness signal by itself.
        self.stats: dict[str, Any] = {
            "rounds": 0,
            "admissions": 0,
            "admission_batch_sizes": [],
            "round_batch_sizes": [],
            "bootstrap_checks_ok": 0,
            "bootstrap_checks_failed": 0,
            "bootstrap_failures": [],
            "requests_completed": 0,
        }

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="server-engine-loop")

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            await self._task

    # -- request-facing API ---------------------------------------------
    def capacity_ok(self, prompt_len: int, max_tokens: int) -> bool:
        return prompt_len + max_tokens <= self.capacity_tokens_per_slot

    async def submit(self, prompt_ids: list[int], max_tokens: int) -> dict:
        """Enqueue a generation request; resolves when the request finishes
        (EOS or max_tokens), or raises if the engine loop hit an
        unexpected error while this request was active (see ``_loop``'s
        try/except -- a defensive net so one bad round cannot silently
        wedge the whole server)."""
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[dict]" = loop.create_future()
        req = GenerationRequest(
            request_id=str(uuid.uuid4()), prompt_ids=list(prompt_ids), max_tokens=max_tokens, future=fut
        )
        self.pending.append(req)
        return await fut

    # -- internal admission-time correctness check -----------------------
    def _admission_bootstrap_check(self, slot: int, req: GenerationRequest, anchor: int) -> None:
        """Same technique ``_run_sustained``'s admission block uses for
        every real admission: an independent single-slot ``prefill()`` of
        the SAME prompt on a dedicated, reused reference slot, compared
        against the batched path's own anchor token, near-tie tolerant.
        Always-on (cheap: one extra prefill per admission), logged into
        ``self.stats`` rather than blocking the request -- a production
        analogue of the same check this project's benchmarks already run,
        not a new correctness methodology."""
        ref_slot = self.ref_slot_for[slot]
        if self.runner.slot_kv_len[ref_slot] != 0:
            self.runner.reset_slot(ref_slot)
        ref_first = self.runner.prefill(ref_slot, req.prompt_ids)
        if ref_first == anchor:
            self.stats["bootstrap_checks_ok"] += 1
            return
        diag_slot = self.diag_slot_for[slot]
        if self.runner.slot_kv_len[diag_slot] != 0:
            self.runner.reset_slot(diag_slot)
        diag = self._near_tie_margin_diag(self.runner, diag_slot, req.prompt_ids, anchor)
        if diag["within_tolerance"]:
            self.stats["bootstrap_checks_ok"] += 1
        else:
            self.stats["bootstrap_checks_failed"] += 1
            self.stats["bootstrap_failures"].append(
                {"request_id": req.request_id, "slot": slot, "diag": diag}
            )
            logger.warning("admission bootstrap check FAILED for request %s: %s", req.request_id, diag)

    def _finish_request(self, slot: int, req: GenerationRequest, committed_tokens: list[int], finish_reason: str) -> None:
        """Resolve ``req``'s future and release ``slot`` back to the free
        pool. Shared by both the admission-time immediate-finish edge cases
        (anchor is EOS; max_tokens==1) and the normal round-loop finish
        path, so both go through exactly one code path."""
        result = {
            "committed_token_ids": committed_tokens,
            "finish_reason": finish_reason,
            "prompt_tokens": len(req.prompt_ids),
            "completion_tokens": len(committed_tokens),
        }
        if not req.future.done():
            req.future.set_result(result)
        self.stats["requests_completed"] += 1
        self.runner.reset_slot(slot)
        self.free_slots.append(slot)

    # -- the engine loop --------------------------------------------------
    async def _loop(self) -> None:
        while not self._stop:
            try:
                await self._step()
            except Exception as exc:  # pragma: no cover -- defensive net, see submit()'s docstring
                logger.exception("engine round failed, failing active requests and resetting slots")
                for slot, st in list(self.active.items()):
                    fut = st["req"].future
                    if not fut.done():
                        fut.set_exception(exc)
                    try:
                        self.runner.reset_slot(slot)
                    except Exception:
                        logger.exception("reset_slot(%d) itself failed during error recovery", slot)
                    self.free_slots.append(slot)
                self.active.clear()
                await asyncio.sleep(0.05)

    async def _step(self) -> None:
        while self.pending:
            self.waiting.append(self.pending.pop(0))

        if self.free_slots and self.waiting:
            n = min(len(self.free_slots), len(self.waiting))
            admit_now = [(self.free_slots.pop(0), self.waiting.pop(0)) for _ in range(n)]
            new_slots = [s for s, _ in admit_now]
            new_prompts = [r.prompt_ids for _, r in admit_now]
            try:
                for slot, _ in admit_now:
                    if self.runner.slot_kv_len[slot] != 0:
                        self.runner.reset_slot(slot)
                prefill_result = self.runner.mtp_prefill_batch(new_slots, new_prompts)
            except Exception as exc:
                # A real gap this task's own E2E run caught empirically:
                # ``admit_now``'s slots/requests were already popped out of
                # ``free_slots``/``self.waiting`` above, so if
                # ``mtp_prefill_batch`` itself raises (e.g. a malformed
                # prompt), the outer ``_loop``'s broad except (which only
                # fails futures for requests already recorded in
                # ``self.active``) would never see these requests at all --
                # their futures would hang forever AND their slots would
                # leak (never returned to ``free_slots``). Recover exactly
                # the requests actually affected, here, precisely.
                logger.exception("admission failed for %d request(s); failing their futures", len(admit_now))
                for slot, req in admit_now:
                    if not req.future.done():
                        req.future.set_exception(exc)
                    try:
                        self.runner.reset_slot(slot)
                    except Exception:
                        logger.exception("reset_slot(%d) during admission-error recovery also failed", slot)
                    self.free_slots.append(slot)
            else:
                self.stats["admissions"] += 1
                self.stats["admission_batch_sizes"].append(len(admit_now))

                for slot, req in admit_now:
                    anchor = prefill_result[slot]["anchor"]
                    drafts = prefill_result[slot]["draft_tokens"]
                    self._admission_bootstrap_check(slot, req, anchor)

                    # 2026-07-19, real bug found by this task's own E2E
                    # validation run (a genuine first-token content
                    # mismatch between the server's HTTP response and an
                    # independent single-slot reference replay of the SAME
                    # prompt): ``anchor`` -- the FIRST real generated token,
                    # produced by ``mtp_prefill_batch``'s prefill forward --
                    # is never part of any later round's own
                    # ``decision["committed"]`` list (that list only ever
                    # contains draft-continuation/bonus tokens for
                    # POSITIONS AFTER the anchor; every subsequent round's
                    # own anchor is already folded into the PRIOR round's
                    # last committed entry, so no further gap ever
                    # accumulates -- confirmed by tracing
                    # mtp_verify_and_commit_batch/determine_accept_reject_
                    # batch's own construction). The established benchmark
                    # scripts this engine's loop is otherwise a direct port
                    # of (``mtp_sustained_realistic_workload_check.py``'s
                    # ``_run_sustained``, ``mtp_async_arrival_check.py``'s
                    # ``_run_async_arrival``) have this SAME gap in their own
                    # ``committed_tokens`` bookkeeping -- it never surfaced
                    # there because nothing in this project before this task
                    # ever treated ``committed_tokens``'s literal CONTENT as
                    # load-bearing (only informational substring checks /
                    # length counts use it); this server's HTTP response body
                    # is the first place in this repository where that
                    # content is actually served to a caller, so the gap is a
                    # real, user-visible bug here that must be fixed, not a
                    # "known, accepted" quirk to reproduce. Seeding
                    # ``committed_tokens`` with ``[anchor]`` here (once, at
                    # admission) is the complete fix -- it also needs the two
                    # edge cases below (anchor itself is EOS; max_tokens==1)
                    # that a bare append doesn't handle.
                    if anchor == self.eos_token_id:
                        self._finish_request(slot, req, committed_tokens=[], finish_reason="stop")
                        continue
                    committed_tokens = [anchor]
                    if len(committed_tokens) >= req.max_tokens:
                        self._finish_request(slot, req, committed_tokens=committed_tokens, finish_reason="length")
                        continue

                    self.active[slot] = {
                        "req": req,
                        "anchor": anchor,
                        "drafts": drafts,
                        "committed_tokens": committed_tokens,
                    }

        if not self.active:
            await asyncio.sleep(self.idle_sleep_s)
            return

        active_slots = list(self.active.keys())
        decisions = self.runner.mtp_verify_and_commit_batch(
            active_slots,
            {s: self.active[s]["anchor"] for s in active_slots},
            {s: self.active[s]["drafts"] for s in active_slots},
        )
        self.stats["rounds"] += 1
        self.stats["round_batch_sizes"].append(len(active_slots))

        newly_finished: list[int] = []
        for s in active_slots:
            st = self.active[s]
            req: GenerationRequest = st["req"]
            decision = decisions[s]
            new_tokens = decision["committed"]

            finish_reason: str | None = None
            kept: list[int] = []
            for t in new_tokens:
                if len(st["committed_tokens"]) + len(kept) >= req.max_tokens:
                    finish_reason = "length"
                    break
                if t == self.eos_token_id:
                    finish_reason = "stop"
                    break
                kept.append(t)
            st["committed_tokens"].extend(kept)
            if finish_reason is None and len(st["committed_tokens"]) >= req.max_tokens:
                finish_reason = "length"

            if finish_reason is None:
                st["anchor"], st["drafts"] = decision["next_anchor"], decision["next_draft_tokens"]
                continue

            self._finish_request(s, req, st["committed_tokens"], finish_reason)
            newly_finished.append(s)

        for s in newly_finished:
            del self.active[s]

        await asyncio.sleep(0)
