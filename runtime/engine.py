"""Pure-Python control plane for the fixed-slot eager executor.

The engine deliberately does not know about tensors or a model backend. A
backend registers ``prefill`` and ``decode`` operations in :class:`OpRegistry`;
this module supplies each call with stable cache metadata and owns the request
lifecycle around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.hybrid_cache import CacheView, HybridCache
from runtime.op_registry import OpRegistry
from runtime.slot_manager import SlotAssignment, SlotError


class EngineError(RuntimeError):
    """Raised when a request cannot follow the eager engine contract."""


class RequestState(str, Enum):
    """The only lifecycle states a fixed physical slot may occupy."""

    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    FAILED = "failed"
    RELEASED = "released"


@dataclass(frozen=True)
class ExecutionRequest:
    """Backend-neutral input for one eager model operation."""

    request_id: str
    phase: RequestState
    cache: CacheView
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class RequestSnapshot:
    """Immutable externally visible request state."""

    request_id: str
    state: RequestState
    cache: CacheView
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class StepResult:
    """The opaque backend output paired with the resulting request state."""

    request: RequestSnapshot
    output: Any


@dataclass
class _Request:
    assignment: SlotAssignment
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    state: RequestState = RequestState.PREFILL
    generated_token_ids: list[int] = field(default_factory=list)


class EagerEngine:
    """Drive prefill and one-token decode without moving physical slots.

    ``prefill`` and ``decode`` registry operations receive an
    :class:`ExecutionRequest`. They may return any backend-specific result,
    such as logits. The engine appends tokens to ``HybridCache`` before the
    operation so the backend always observes the cache extent it must write.
    """

    def __init__(self, cache: HybridCache, registry: OpRegistry) -> None:
        self._cache = cache
        self._registry = registry
        self._requests: dict[str, _Request] = {}

    def submit(
        self, request_id: str, prompt_token_ids: Sequence[int], *, max_new_tokens: int
    ) -> RequestSnapshot:
        """Acquire a fixed slot and enqueue its prompt for prefill."""
        if (
            request_id in self._requests
            and self._requests[request_id].state is not RequestState.RELEASED
        ):
            raise EngineError(f"request is already known: {request_id}")
        tokens = _token_tuple(prompt_token_ids, name="prompt_token_ids", allow_empty=False)
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        try:
            cache_view = self._cache.acquire(request_id)
        except (SlotError, ValueError) as error:
            raise EngineError(f"unable to schedule request {request_id!r}: {error}") from error
        self._requests[request_id] = _Request(
            assignment=SlotAssignment(cache_view.slot_id, request_id, cache_view.generation),
            prompt_token_ids=tokens,
            max_new_tokens=max_new_tokens,
        )
        return self.request(request_id)

    def prefill(self, request_id: str) -> StepResult:
        """Run a prompt through the registered prefill operation."""
        request = self._require_state(request_id, RequestState.PREFILL)
        return self._execute(request_id, request, "prefill", request.prompt_token_ids)

    def prefill_ready(self) -> tuple[RequestSnapshot, ...]:
        """Return pending prefills in stable physical-slot order."""
        return self._snapshots_in_slot_order(RequestState.PREFILL)

    def prefill_all(self) -> tuple[StepResult, ...]:
        """Execute every pending prefill in stable physical-slot order."""
        return tuple(self.prefill(snapshot.request_id) for snapshot in self.prefill_ready())

    def decode(self, request_id: str, token_id: int) -> StepResult:
        """Consume one sampled token and run the registered decode operation."""
        request = self._require_state(request_id, RequestState.DECODE)
        return self._execute(
            request_id, request, "decode", _token_tuple([token_id], name="token_id")
        )

    def decode_ready(self) -> tuple[RequestSnapshot, ...]:
        """Return requests ready for one-token decode in physical-slot order."""
        return self._snapshots_in_slot_order(RequestState.DECODE)

    def decode_batch(self, token_ids: Mapping[str, int]) -> tuple[StepResult, ...]:
        """Decode a selected ready set, always in physical-slot order.

        The mapping may omit ready requests, which makes it possible for an
        outer sampler to advance only requests with a chosen token. Unknown or
        non-decode request IDs are rejected before any model operation runs.
        """
        for request_id in token_ids:
            self._require_state(request_id, RequestState.DECODE)
        return tuple(
            self.decode(snapshot.request_id, token_ids[snapshot.request_id])
            for snapshot in self.decode_ready()
            if snapshot.request_id in token_ids
        )

    def complete(self, request_id: str) -> RequestSnapshot:
        """Stop an active request; callers must still release its slot."""
        request = self._lookup(request_id)
        if request.state not in (RequestState.PREFILL, RequestState.DECODE):
            raise EngineError(f"request {request_id!r} cannot complete from {request.state.value}")
        request.state = RequestState.COMPLETED
        return self.request(request_id)

    def release(self, request_id: str) -> None:
        """Reset cache ownership for a completed or failed request."""
        request = self._lookup(request_id)
        if request.state not in (RequestState.COMPLETED, RequestState.FAILED):
            raise EngineError(f"request {request_id!r} must complete before release")
        self._cache.release(request.assignment)
        request.state = RequestState.RELEASED

    def request(self, request_id: str) -> RequestSnapshot:
        """Return the current immutable view of one submitted request."""
        request = self._lookup(request_id)
        if request.state is RequestState.RELEASED:
            raise EngineError(f"request {request_id!r} has been released")
        return self._snapshot(request_id, request)

    def active(self) -> tuple[RequestSnapshot, ...]:
        """Return all unreleased requests in stable physical-slot order."""
        return tuple(
            self._snapshot(view.request_id, self._requests[view.request_id])
            for view in self._cache.active()
        )

    def _execute(
        self, request_id: str, request: _Request, operation_name: str, token_ids: tuple[int, ...]
    ) -> StepResult:
        operation = self._registry.resolve(operation_name)
        cache_view = self._cache.append(request.assignment, len(token_ids))
        context = ExecutionRequest(
            request_id=request_id,
            phase=request.state,
            cache=cache_view,
            token_ids=token_ids,
            generated_token_ids=tuple(request.generated_token_ids),
            max_new_tokens=request.max_new_tokens,
        )
        try:
            output = operation(context)
        except Exception as error:
            request.state = RequestState.FAILED
            raise EngineError(
                f"{operation_name} operation failed for request {request_id!r}"
            ) from error

        if request.state is RequestState.PREFILL:
            request.state = RequestState.DECODE
        else:
            request.generated_token_ids.extend(token_ids)
            if len(request.generated_token_ids) >= request.max_new_tokens:
                request.state = RequestState.COMPLETED
        return StepResult(request=self._snapshot(request_id, request), output=output)

    def _snapshots_in_slot_order(self, state: RequestState) -> tuple[RequestSnapshot, ...]:
        return tuple(snapshot for snapshot in self.active() if snapshot.state is state)

    def _snapshot(self, request_id: str, request: _Request) -> RequestSnapshot:
        return RequestSnapshot(
            request_id=request_id,
            state=request.state,
            cache=next(view for view in self._cache.active() if view.request_id == request_id),
            prompt_token_ids=request.prompt_token_ids,
            generated_token_ids=tuple(request.generated_token_ids),
            max_new_tokens=request.max_new_tokens,
        )

    def _lookup(self, request_id: str) -> _Request:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise EngineError(f"unknown request: {request_id!r}") from error

    def _require_state(self, request_id: str, expected: RequestState) -> _Request:
        request = self._lookup(request_id)
        if request.state is not expected:
            raise EngineError(
                f"request {request_id!r} is {request.state.value}; expected {expected.value}"
            )
        return request


def _token_tuple(
    token_ids: Sequence[int], *, name: str, allow_empty: bool = False
) -> tuple[int, ...]:
    tokens = tuple(token_ids)
    if not tokens and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in tokens):
        raise TypeError(f"{name} must contain only integers")
    return tokens
