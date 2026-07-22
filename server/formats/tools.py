"""Tool-call parsing and formatting.

The Qwen3.6 model emits tool calls as XML-ish text inside its generation.
This module parses that text into structured tool-call objects and
formats them for each API style (OpenAI / Anthropic).
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _repair_json(value: str) -> str:
    """Attempt to repair common JSON formatting errors from model output.

    Models occasionally produce near-valid JSON with predictable mutations:
    - ``{("key": ...)}`` instead of ``[{"key": ...}]`` (set-literal confusion)
    - Trailing commas before ``]`` or ``}``
    """
    repaired = value.strip()
    # Pattern: {("key": val, ...)}] -> [{"key": val, ...}]
    # The model sometimes wraps a dict in set-literal syntax {( ... )}
    # instead of putting it in an array [{ ... }].
    if repaired.startswith("{("):
        inner = repaired[2:]  # strip leading {(
        if inner.endswith("})]"):
            inner = inner[:-3] + "}]"
        elif inner.endswith(")}"):
            inner = inner[:-2] + "}]"
        elif inner.endswith(")"):
            inner = inner[:-1] + "}]"
        repaired = "[{" + inner
    # Trailing commas: ,] or ,}
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse tool calls from model output.

    Returns (visible_text, tool_calls) where visible_text is the output
    with tool_call blocks removed, and tool_calls is a list of dicts
    with keys: name, arguments (dict).
    """
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        func_name = match.group(1).strip()
        params_block = match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _PARAM_RE.finditer(params_block):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                # Attempt repair of common model JSON errors
                try:
                    arguments[param_name] = json.loads(_repair_json(param_value))
                except (json.JSONDecodeError, ValueError):
                    arguments[param_name] = param_value
        tool_calls.append({"name": func_name, "arguments": arguments})
    visible = _TOOL_CALL_RE.sub("", text).strip()
    return visible, tool_calls


def format_tool_calls_openai(tool_calls: list[dict], start_id: int = 0) -> list[dict]:
    """Format parsed tool calls for OpenAI chat completion response."""
    result = []
    for i, tc in enumerate(tool_calls):
        result.append(
            {
                "id": f"call_{start_id + i:04d}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
        )
    return result


def format_tool_calls_anthropic(tool_calls: list[dict], start_id: int = 0) -> list[dict]:
    """Format parsed tool calls as Anthropic tool_use content blocks.

    Each tool_use block gets a globally unique ID (uuid4-based) so that
    IDs never collide across turns in a multi-turn conversation.  The
    previous sequential scheme (toolu_0000, toolu_0001, ...) reused the
    same IDs in every assistant turn, which confused Claude Desktop's
    tool_result matching.
    """
    import uuid as _uuid

    result = []
    for tc in tool_calls:
        result.append(
            {
                "type": "tool_use",
                "id": f"toolu_{_uuid.uuid4().hex[:24]}",
                "name": tc["name"],
                "input": tc["arguments"],
            }
        )
    return result


# Anthropic server-side tool types that cannot be executed by a local model.
# These are skipped during tool conversion (the model cannot call them).
_SERVER_TOOL_TYPE_PREFIXES = (
    "web_search_",
    "code_execution_",
    "computer_",
    "text_editor_",
    "bash_",
)


def convert_tools_to_chat_template(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI/Anthropic tool definitions to the format expected
    by the Qwen3.6 chat template (list of function dicts).

    The chat template expects tools as a list of dicts, each with
    type=function and a function sub-dict with name/description/parameters.

    Anthropic server-side tools (web_search_20250305, code_execution_*, etc.)
    are skipped because they cannot be executed by a local model.
    """
    if not tools:
        return None
    converted = []
    for tool in tools:
        # Skip Anthropic server-side tools (web_search, code_execution, etc.)
        tool_type = tool.get("type", "")
        if any(tool_type.startswith(p) for p in _SERVER_TOOL_TYPE_PREFIXES):
            continue
        if "function" in tool:
            converted.append(tool)
        elif "name" in tool:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {})),
                    },
                }
            )
        else:
            converted.append(tool)
    return converted or None


# -- Streaming tool-call detection ------------------------------------------

_TOOL_CALL_OPEN = chr(60) + "tool_call" + chr(62)


def find_tool_call_start(text: str) -> int:
    """Find the earliest position where a tool call block might be starting.

    Returns the index of the first character of the potential tool call,
    or -1 if no tool call start is detected.

    We look for progressively shorter prefixes of the opening tag to catch
    partial matches at the end of a streaming buffer (e.g. the model has
    emitted '<tool' but not yet '_call>').
    """
    # Full tag present
    idx = text.find(_TOOL_CALL_OPEN)
    if idx >= 0:
        return idx
    # Partial prefixes at the very end of the text (streaming edge case)
    for length in range(len(_TOOL_CALL_OPEN) - 1, 0, -1):
        prefix = _TOOL_CALL_OPEN[:length]
        if text.endswith(prefix):
            return len(text) - length
    return -1
