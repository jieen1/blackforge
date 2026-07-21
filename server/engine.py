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
from collections import deque
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"

logger = logging.getLogger("qwen_sm120_server.engine")

# P0 (2026-07-19, notes/prefix-cache-design.md sec 5 -- instrumentation
# only, no caching logic): bounded window sizes for the prompt-prefix-
# overlap logging §25.9 recommended building BEFORE committing to P2/P3
# engineering, so real traffic can tell us which sharing pattern (§1.1's
# fan-out vs. sequential-growth vs. incidental cross-request) actually
# dominates on this machine.
_PREFIX_OVERLAP_HISTORY = 64
_PREFIX_OVERLAP_SAMPLES_KEPT = 200
# P4a (notes/prefix-cache-design.md sec 5-P4 / §25.9): bounded window for
# the per-hit L samples kept in self.stats (mirrors _PREFIX_OVERLAP_SAMPLES_
# KEPT above) so a long-running server's stats dict stays bounded no matter
# how many warm prefix hits it serves.
_PREFIX_CACHE_HIT_SAMPLES_KEPT = 200
# P4b (notes/2026-07-20-p4b-session-affinity-plan.md §1D): bounded window for the
# per-warm-continuation samples kept in self.stats (mirrors _PREFIX_CACHE_HIT_
# SAMPLES_KEPT) so a long-running server's stats dict stays bounded no matter how
# many warm-slot continuations it serves.
_SESSION_WARM_CONTINUATION_SAMPLES_KEPT = 200


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Exact token-id longest-common-prefix length -- the same notion of
    "shared prefix" the eventual chained-block-hash cache (P3) will use,
    just without the hashing/blocking machinery this early. O(min(len(a),
    len(b))) worst case, O(1) in the common no-overlap case (stops at the
    first mismatch)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class GenerationRequest:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    future: "asyncio.Future[dict]"
    created_t: float = field(default_factory=time.perf_counter)
    # P4b session affinity: optional caller-supplied session id. When the engine
    # is built with enable_session_affinity (off by default), a finished request
    # with a session_id retains its slot WARM for session_ttl_s so the next turn
    # of the same session continues in place with zero restore. None (or affinity
    # off) => byte-for-byte the P4a path.
    session_id: str | None = None


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
        enable_prefix_cache: bool = True,
        enable_session_affinity: bool = False,
        session_ttl_s: float = 30.0,
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

        # P4b session affinity (notes/2026-07-20-p4b-session-affinity-plan.md §3.2):
        # warm-slot continuation needs the persistent cache (slot_committed_tokens,
        # block table, content-hash fallback). Refuse affinity-on / prefix-cache-off
        # at construction -- a clean startup error, not a runtime crash.
        if enable_session_affinity and not enable_prefix_cache:
            raise ValueError(
                "enable_session_affinity requires enable_prefix_cache (warm-slot "
                "continuation needs the persistent content-hash cache)"
            )

        sys.path.insert(0, SM120_VLLM_INTEGRATION)
        import register_sm120_backend  # noqa: F401
        from transformers import AutoTokenizer

        from runtime.direct_model_runner import DirectModelRunner, build_vllm_config, _DEFAULT_PREFILL_CHUNK_SIZE
        self._prefill_chunk_size = _DEFAULT_PREFILL_CHUNK_SIZE

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
        # P4a (notes/prefix-cache-design.md sec 5-P4 -- server integration,
        # the product value): this ONE flag is the rollback spine. When ON
        # (the default), construct the runner with the full P0->P3 persistent
        # prefix-cache stack enabled (P0/P1 block-table indirection + ref-
        # counting allocator, P2 same-round fan-out, P3 persistent content-
        # addressed cache) so the server SERVES warm prefix hits across
        # requests. When OFF, construct EXACTLY as today (no cache flags) =>
        # byte-for-byte the old server: every DirectModelRunner cache flag
        # defaults False, and mtp_prefill_with_cache delegates straight to
        # mtp_prefill_batch when enable_persistent_prefix_cache is off, so the
        # _step call-site swap below is safe either way.
        self.enable_prefix_cache = enable_prefix_cache
        # P4b session-affinity knobs (off by default => byte-for-byte P4a).
        self.enable_session_affinity = enable_session_affinity
        self.session_ttl_s = session_ttl_s
        cache_kwargs: dict[str, bool] = {}
        if enable_prefix_cache:
            cache_kwargs = {
                "enable_block_table": True,
                "enable_prefix_cache": True,
                "enable_persistent_prefix_cache": True,
            }
        self.runner = DirectModelRunner(
            vllm_config,
            num_slots=num_slots,
            block_size=block_size,
            blocks_per_slot=blocks_per_slot,
            enable_cudagraph=enable_cudagraph,
            **cache_kwargs,
        )

        self.free_slots: list[int] = list(range(capacity))
        self.active: dict[int, dict[str, Any]] = {}
        self.waiting: list[GenerationRequest] = []
        self.pending: list[GenerationRequest] = []
        # P4b session affinity: retained WARM slots keyed by session_id. Each value
        # is {slot, expire_t, prior_len, committed_full}. A retained slot is NOT
        # reset_slot-ed (its blocks stay pinned at ref_cnt>=1, so BlockPool can
        # never evict/reuse them -- INV9/INV2 hold by construction); it is
        # reset+released exactly once on TTL expiry, warm-continue admission,
        # crash recovery, or shutdown (BlockPool.free raises on double-free).
        self.retained: dict[str, dict[str, Any]] = {}
        self.ref_slot_for = {p: capacity + p for p in range(capacity)}
        self.diag_slot_for = {p: 2 * capacity + p for p in range(capacity)}

        self._stop = False
        self._task: asyncio.Task | None = None

        # P0 (2026-07-19, notes/prefix-cache-design.md sec 5 --
        # instrumentation only, no caching logic): bounded rolling window of
        # recently-admitted requests' own prompt token ids, consulted by
        # _log_prefix_overlap to measure real prompt-prefix overlap against
        # requests admitted in EARLIER rounds (pattern B / incidental
        # cross-request sharing). Same-round fan-out (pattern A) is
        # measured directly from admit_now, no history needed for that case.
        self._recent_prompts: deque[tuple[str, list[int]]] = deque(maxlen=_PREFIX_OVERLAP_HISTORY)

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
            # P0 prompt-prefix-overlap logging (§25.9's cheap recommendation
            # -- instrumentation only, never consulted by admission/
            # scheduling/generation logic in this phase): a bounded log of
            # per-admitted-request overlap samples, plus running counts of
            # how many admissions had at least one full block_size worth of
            # overlap against another same-round request (pattern A) or a
            # recently-admitted earlier request (pattern B / incidental).
            "prefix_overlap_samples": [],
            "prefix_overlap_same_round_events": 0,
            "prefix_overlap_history_events": 0,
            # P4a hit-rate instrumentation (§25.9, the production analogue of
            # P0's overlap logging): per-admitted-prompt persistent-cache hit
            # depth, captured by reconcile_prefix_hit JUST BEFORE each prefill
            # (read-only O(blocks) probe). hits = L>0 count, misses = L==0
            # count, hit_rate = hits/(hits+misses), hit_L_samples = bounded
            # rolling list of per-hit L, hit_tokens_saved = sum of L across
            # hits (the prefill tokens a warm hit skipped). With the cache flag
            # off, reconcile_prefix_hit returns 0 for every prompt => hits
            # stays 0 and these degrade gracefully to the no-cache truth.
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_hit_L_samples": [],
            "prefix_cache_hit_tokens_saved": 0,
            # P4b session-affinity instrumentation (plan §1D): warm-continue
            # admissions -- provably NO restore, because the warm path bypasses
            # restore_cached_prefix AND reconcile_prefix_hit, so these are distinct
            # from prefix_cache_hits. Plus retention/expiration/fallback counters
            # and a bounded rolling list of per-warm-continuation samples.
            "session_warm_continuations": 0,
            "session_warm_continuation_samples": [],
            "session_retentions": 0,
            "session_expirations": 0,
            "session_warm_fallbacks": 0,
        }

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="server-engine-loop")

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            await self._task
        # P4b: release any still-retained warm slots on shutdown so pinned blocks
        # cannot leak (each reset+released exactly once).
        self._release_all_retained()

    # -- request-facing API ---------------------------------------------
    def capacity_ok(self, prompt_len: int, max_tokens: int) -> bool:
        # 2026-07-19, 256K-feasibility task: a real admission-time gap found
        # by that task's own zero-margin ctx256k probe (`prompt_len +
        # max_tokens == capacity_tokens_per_slot` EXACTLY) -- it crashed
        # with `slot N kv_len {capacity+1} exceeds this slot's {capacity}
        # -token capacity` deep inside `_mtp_forward_batch`/
        # `build_attention_metadata_batch`, NOT here, because this exact
        # formula (no margin) says "OK" for that shape. Root cause: MTP's
        # own draft-ahead mechanism (`_mtp_sync_and_propose_batch` /
        # `_mtp_run_continuation_steps`) computes each round's K
        # (`self.K`, always 3 in this repo) speculative candidate
        # continuations AHEAD of the target model's own confirmed
        # `committed_tokens` count, so the draft model's internal kv_len
        # can transiently run up to K tokens past `prompt_len + max_tokens`
        # even though no MORE than `max_tokens` tokens are ever actually
        # committed to the response. This is the exact same "raw
        # prompt_len+max_tokens arithmetic under-counts real capacity need"
        # class of gap this project's own `mtp_w1s_our_runtime_perf.py`
        # ctx64k/ctx256k guards already established for `blocks_per_slot`
        # sizing -- generalized here to this admission check, which had the
        # SAME zero-margin formula. Reserving `self.K` tokens of margin is
        # the minimal, correct fix (not a blanket safety-factor guess): it
        # exactly covers the one concrete mechanism that overshoots, without
        # rejecting any request that would otherwise fit.
        return prompt_len + max_tokens + self.K <= self.capacity_tokens_per_slot

    async def submit(
        self, prompt_ids: list[int], max_tokens: int, session_id: str | None = None
    ) -> dict:
        """Enqueue a generation request; resolves when the request finishes
        (EOS or max_tokens), or raises if the engine loop hit an
        unexpected error while this request was active (see ``_loop``'s
        try/except -- a defensive net so one bad round cannot silently
        wedge the whole server)."""
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[dict]" = loop.create_future()
        req = GenerationRequest(
            request_id=str(uuid.uuid4()), prompt_ids=list(prompt_ids), max_tokens=max_tokens,
            future=fut, session_id=session_id,
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

    def _log_prefix_overlap(self, admit_now: list[tuple[int, GenerationRequest]]) -> None:
        """P0 instrumentation (2026-07-19, notes/prefix-cache-design.md sec
        5 -- §25.9's cheap recommendation): for each request in THIS
        round's admission batch, measure its longest-common-token-prefix
        overlap against (a) every OTHER request admitted in the same round
        (pattern A -- simultaneous fan-out, §1.1) and (b) every recently-
        admitted request from an EARLIER round still held in
        ``self._recent_prompts`` (pattern B -- sequential per-conversation
        growth, and incidental cross-request sharing, §1.1). Purely
        additive to ``self.stats``; reads no cache, writes no cache, and
        never affects admission/scheduling/generation behavior -- the
        whole point is to gather real hit-rate/pattern data on THIS
        machine's actual traffic before committing engineering to P2/P3.
        """
        new_prompts = [(req.request_id, req.prompt_ids) for _, req in admit_now]
        for i, (rid, prompt) in enumerate(new_prompts):
            same_round_best = 0
            for j, (_other_rid, other_prompt) in enumerate(new_prompts):
                if j == i:
                    continue
                same_round_best = max(same_round_best, _longest_common_prefix_len(prompt, other_prompt))

            history_best = 0
            history_best_rid: str | None = None
            for other_rid, other_prompt in self._recent_prompts:
                overlap = _longest_common_prefix_len(prompt, other_prompt)
                if overlap > history_best:
                    history_best = overlap
                    history_best_rid = other_rid

            self.stats["prefix_overlap_samples"].append(
                {
                    "request_id": rid,
                    "prompt_tokens": len(prompt),
                    "same_round_overlap_tokens": same_round_best,
                    "history_overlap_tokens": history_best,
                    "history_overlap_source": history_best_rid,
                }
            )
            if len(self.stats["prefix_overlap_samples"]) > _PREFIX_OVERLAP_SAMPLES_KEPT:
                self.stats["prefix_overlap_samples"].pop(0)
            if same_round_best >= self.block_size:
                self.stats["prefix_overlap_same_round_events"] += 1
            if history_best >= self.block_size:
                self.stats["prefix_overlap_history_events"] += 1

        for rid, prompt in new_prompts:
            self._recent_prompts.append((rid, prompt))

    def _record_prefix_cache_hits(
        self, admit_now: list[tuple[int, GenerationRequest]], hit_depths: list[int]
    ) -> None:
        """P4a hit-rate instrumentation (§25.9): fold each admitted prompt's
        pre-prefill reconcile_prefix_hit depth into self.stats. Purely
        additive observability -- never consulted by admission/scheduling/
        generation logic. A hit (L>0) means the persistent content-addressed
        cache served [0, L) of this prompt from a previously-populated prefix
        (skipping L prefill tokens); a miss (L==0) means a cold prefill. With
        the cache flag off every L is 0 => hits stays 0, hit_rate 0.0."""
        for (_slot, req), L in zip(admit_now, hit_depths):
            if L > 0:
                self.stats["prefix_cache_hits"] += 1
                self.stats["prefix_cache_hit_tokens_saved"] += L
                self.stats["prefix_cache_hit_L_samples"].append(
                    {"request_id": req.request_id, "prompt_tokens": len(req.prompt_ids), "hit_L": L}
                )
                if len(self.stats["prefix_cache_hit_L_samples"]) > _PREFIX_CACHE_HIT_SAMPLES_KEPT:
                    self.stats["prefix_cache_hit_L_samples"].pop(0)
            else:
                self.stats["prefix_cache_misses"] += 1
        total = self.stats["prefix_cache_hits"] + self.stats["prefix_cache_misses"]
        self.stats["prefix_cache_hit_rate"] = (self.stats["prefix_cache_hits"] / total) if total else 0.0

    def _expire_retained_slots(self) -> None:
        """P4b: reset+release every retained warm slot whose TTL has passed,
        returning its capacity to the free pool BEFORE admission. reset_slot
        frees the slot's blocks but KEEPS their published hashes (R10), so an
        expired warm slot's prefix stays hit-able via the content-hash path at
        ref_cnt==0. Called at the top of every _step."""
        now = time.perf_counter()
        for sid in [s for s, r in self.retained.items() if r["expire_t"] <= now]:
            ret = self.retained.pop(sid)
            try:
                self.runner.reset_slot(ret["slot"])
            except Exception:
                logger.exception(
                    "reset_slot(%d) failed expiring retained session %s", ret["slot"], sid
                )
            self.free_slots.append(ret["slot"])
            self.stats["session_expirations"] += 1

    def _release_all_retained(self) -> None:
        """P4b crash/shutdown cleanup: reset+release every retained warm slot
        exactly once and clear the retention map, so a crash or shutdown cannot
        leak pinned blocks (INV9 still holds -- they are never reused -- but the
        pool would otherwise shrink). Popping as we go makes this idempotent and
        guards against double-release (BlockPool.free raises on double-free)."""
        for sid in list(self.retained.keys()):
            ret = self.retained.pop(sid)
            try:
                self.runner.reset_slot(ret["slot"])
            except Exception:
                logger.exception(
                    "reset_slot(%d) failed releasing retained session %s", ret["slot"], sid
                )
            self.free_slots.append(ret["slot"])

    def _activate_slot(
        self, slot: int, req: GenerationRequest, anchor: int, drafts: list[int]
    ) -> None:
        """Shared post-prefill bookkeeping for BOTH the normal free-slot admission
        path and the P4b warm-continue path: run the always-on admission bootstrap
        correctness check, handle the two immediate-finish edge cases (anchor is
        EOS; max_tokens==1), seed committed_tokens with [anchor], and record the
        slot as active. Factored out of _step so the committed-token seeding logic
        is NOT duplicated (behavior-preserving refactor)."""
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
            return
        committed_tokens = [anchor]
        if len(committed_tokens) >= req.max_tokens:
            self._finish_request(slot, req, committed_tokens=committed_tokens, finish_reason="length")
            return

        self.active[slot] = {
            "req": req,
            "anchor": anchor,
            "drafts": drafts,
            "committed_tokens": committed_tokens,
        }

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
        # P4b session affinity: optionally retain the finished slot WARM (do NOT
        # reset_slot, do NOT release -- its blocks stay pinned at ref_cnt>=1 so the
        # next turn of the same session continues in place with zero restore). Off
        # by default; without a session_id (or flag off) this is the unchanged P4a
        # path (unconditional reset_slot + release). Retain on BOTH stop (EOS) and
        # length finishes in v1 (the TTL bounds the wasted capacity).
        if self.enable_session_affinity and req.session_id and self.enable_prefix_cache:
            # Newest finish wins: if this session_id already maps to a stale
            # retained slot from an earlier turn (a different slot), reset+release
            # the old one first so it cannot leak -- exactly once (BlockPool.free
            # raises on double-free).
            old = self.retained.get(req.session_id)
            if old is not None and old["slot"] != slot:
                self.runner.reset_slot(old["slot"])
                self.free_slots.append(old["slot"])
            # The warm-continue boundary is the runtime's AUTHORITATIVE committed
            # state (slot_kv_len / slot_committed_tokens), NOT the server's
            # max_tokens-truncated view: MTP's final verify round can commit a few
            # tokens past max_tokens (the server truncates the response, but the
            # runtime's live KV/GDN state and committed-token record include them --
            # empirically slot_kv_len ran 2 past len(prompt)+len(committed) at
            # max_tokens=32). Warm-continue must continue from that true boundary
            # (slot_kv_len), so store the runtime's full committed sequence; the
            # next turn reproduces it exactly to fire the zero-restore path, else it
            # falls back to the content-hash hit (opportunistic -- plan §5 risk #1).
            prior_len = self.runner.slot_kv_len[slot]
            committed_full = list(self.runner.slot_committed_tokens[slot])
            self.retained[req.session_id] = {
                "slot": slot,
                "expire_t": time.perf_counter() + self.session_ttl_s,
                "prior_len": prior_len,
                "committed_full": committed_full,
            }
            self.stats["session_retentions"] += 1
            return
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
                # P4b: a crash must not leak retained warm slots' pinned blocks --
                # reset+release each exactly once and clear the retention map.
                self._release_all_retained()
                await asyncio.sleep(0.05)

    async def _step(self) -> None:
        while self.pending:
            self.waiting.append(self.pending.pop(0))

        # P4b: expire retained warm slots past their TTL BEFORE admission, so the
        # freed capacity is available to this step's requests.
        self._expire_retained_slots()

        # P4b warm-continue admissions first (returning sessions), only when
        # enabled. A matching session_id continues its retained WARM slot in place
        # with ZERO restore (mtp_prefill_warm_continue); a prefix mismatch or a
        # runtime error falls back to the normal cold/content-hash path (the
        # correctness-bearing fallback). This path never calls reconcile_prefix_hit
        # / _record_prefix_cache_hits, so prefix_cache_hits is untouched by warm
        # turns -- the definitive zero-restore signal the e2e asserts on.
        if self.enable_session_affinity and self.retained:
            for req in list(self.waiting):
                if not req.session_id or req.session_id not in self.retained:
                    continue
                ret = self.retained.pop(req.session_id)
                self.waiting.remove(req)
                slot, prior_len = ret["slot"], ret["prior_len"]
                committed_full = ret["committed_full"]
                # Two-layer prefix guard (server side): the new prompt must EXTEND
                # the retained P1+C1 exactly through prior_len. The runtime method
                # re-checks the same condition against slot_committed_tokens.
                match = (
                    len(req.prompt_ids) > prior_len
                    and req.prompt_ids[:prior_len] == committed_full[:prior_len]
                )
                if not match:
                    self.runner.reset_slot(slot)
                    self.free_slots.append(slot)
                    self.stats["session_warm_fallbacks"] += 1
                    # Re-admit normally (content-hash hit / cold). waiting is a list,
                    # so insert at the front to be picked up by the admission below.
                    self.waiting.insert(0, req)
                    continue
                try:
                    res = self.runner.mtp_prefill_warm_continue(slot, req.prompt_ids, prior_len)
                except Exception:
                    logger.exception(
                        "warm-continue failed for session %s; falling back", req.session_id
                    )
                    self.runner.reset_slot(slot)
                    self.free_slots.append(slot)
                    self.stats["session_warm_fallbacks"] += 1
                    self.waiting.insert(0, req)
                    continue
                self.stats["session_warm_continuations"] += 1
                self.stats["session_warm_continuation_samples"].append(
                    {
                        "request_id": req.request_id,
                        "session_id": req.session_id,
                        "slot": slot,
                        "prior_len": prior_len,
                        "prompt_tokens": len(req.prompt_ids),
                        "suffix_len": len(req.prompt_ids) - prior_len,
                    }
                )
                if (
                    len(self.stats["session_warm_continuation_samples"])
                    > _SESSION_WARM_CONTINUATION_SAMPLES_KEPT
                ):
                    self.stats["session_warm_continuation_samples"].pop(0)
                self._activate_slot(slot, req, res["anchor"], res["draft_tokens"])

        if self.free_slots and self.waiting:
            n = min(len(self.free_slots), len(self.waiting))
            admit_now = [(self.free_slots.pop(0), self.waiting.pop(0)) for _ in range(n)]
            new_slots = [s for s, _ in admit_now]
            new_prompts = [r.prompt_ids for _, r in admit_now]
            try:
                for slot, _ in admit_now:
                    if self.runner.slot_kv_len[slot] != 0:
                        self.runner.reset_slot(slot)
                # P4a hit-rate instrumentation (§25.9): probe each admitted
                # prompt's persistent-cache hit depth JUST BEFORE the prefill.
                # reconcile_prefix_hit is a read-only O(blocks) dict probe; the
                # single-event-loop design guarantees no await runs between this
                # probe and the mtp_prefill_with_cache call below, so it sees the
                # EXACT cache state the prefill will (the same L the entrypoint's
                # own internal reconciliation recomputes). With the cache flag
                # off it returns 0 for every prompt => prefix_cache_hits stays 0.
                hit_depths = [self.runner.reconcile_prefix_hit(p) for p in new_prompts]
                # P4a call-site swap: the unified production prefill entrypoint
                # (persistent-hit + P2 same-round fan-out + cold). With the cache
                # flag off it delegates to mtp_prefill_batch => byte-for-byte the
                # old path; a slot with L==0 takes the cold/fanout path => the
                # old output for it. The surrounding admission error-recovery
                # block below is unchanged (still resets slots + fails futures +
                # returns slots to free_slots on raise).
                prefill_result = self.runner.mtp_prefill_with_cache(new_slots, new_prompts, chunk_size=self._prefill_chunk_size)
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
                self._log_prefix_overlap(admit_now)
                self._record_prefix_cache_hits(admit_now, hit_depths)

                for slot, req in admit_now:
                    anchor = prefill_result[slot]["anchor"]
                    drafts = prefill_result[slot]["draft_tokens"]
                    # P4b: shared post-prefill bookkeeping (bootstrap check + the
                    # anchor-EOS / max_tokens==1 immediate-finish edge cases +
                    # committed_tokens=[anchor] seeding + active[slot] record),
                    # factored into _activate_slot so the warm-continue path reuses
                    # the exact same logic (behavior-preserving refactor).
                    self._activate_slot(slot, req, anchor, drafts)

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
