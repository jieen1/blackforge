"""Stateful stream processor for model output.

Handles the two-phase nature of Qwen3.6 streaming output:
1. Thinking phase: think-block...think-close -- must be completely suppressed
2. Content phase: visible text, possibly followed by tool-call XML

The processor buffers tokens and emits safe content deltas while
suppressing thinking blocks and tool-call XML from the content stream.
Tool calls are extracted at the end and emitted as structured chunks.
"""

from __future__ import annotations

from server.formats.thinking import strip_thinking
from server.formats.tools import parse_tool_calls, find_tool_call_start

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


class StreamProcessor:
    """Accumulates token IDs and produces safe content deltas.

    Usage::

        proc = StreamProcessor(tokenizer)
        for token_batch in stream:
            proc.add_tokens(token_batch)
            for delta in proc.drain_content():
                yield delta  # safe visible text, no thinking, no tool XML
        # After stream ends:
        visible_text, tool_calls = proc.finalize()
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._all_ids: list[int] = []
        self._thinking_done = False
        self._tool_call_started = False
        self._emitted_len = 0
        self._last_decode_len = 0
        self._cached_raw = ""

    def add_tokens(self, token_ids: list[int]) -> None:
        self._all_ids.extend(token_ids)

    @property
    def all_ids(self) -> list[int]:
        return self._all_ids

    def _get_raw(self) -> str:
        """Decode all accumulated tokens (cached when unchanged)."""
        n = len(self._all_ids)
        if n == self._last_decode_len:
            return self._cached_raw
        self._cached_raw = self._tok.decode(self._all_ids, skip_special_tokens=True)
        self._last_decode_len = n
        return self._cached_raw

    def drain_content(self) -> list[str]:
        """Return list of safe content deltas since last call.

        Returns empty list if still in thinking phase or if tool call
        XML has started (content is frozen at that point).
        """
        if self._tool_call_started:
            return []

        raw = self._get_raw()

        # Phase 1: detect thinking completion
        if not self._thinking_done:
            if _THINK_CLOSE in raw:
                # Normal case: think block closed
                self._thinking_done = True
            elif _THINK_OPEN in raw:
                # Think block opened but not yet closed -- still thinking
                return []
            elif len(raw) > 50:
                # No think tags at all after 50 chars -- no thinking phase
                self._thinking_done = True
            else:
                return []

        visible = strip_thinking(raw)

        # Check for tool call XML start
        tc_start = find_tool_call_start(visible)
        if tc_start >= 0:
            self._tool_call_started = True
            safe = visible[:tc_start]
            if len(safe) > self._emitted_len:
                delta = safe[self._emitted_len:]
                self._emitted_len = len(safe)
                return [delta]
            return []

        # Normal content delta
        if len(visible) > self._emitted_len:
            delta = visible[self._emitted_len:]
            self._emitted_len = len(visible)
            return [delta]
        return []

    def finalize(self) -> tuple[str, list[dict]]:
        """Called after stream ends. Returns (visible_text, tool_calls)."""
        raw = self._tok.decode(self._all_ids, skip_special_tokens=True)
        visible = strip_thinking(raw)
        visible_text, tool_calls = parse_tool_calls(visible)
        return visible_text, tool_calls
