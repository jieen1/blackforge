"""Content block parsing.

Both OpenAI and Anthropic allow content to be either a plain string
or a list of typed content blocks. This module normalises both into
plain text, and also extracts structured blocks when needed.
"""

from __future__ import annotations

from typing import Any


def extract_text(field: Any) -> str:
    """Extract plain text from a flexible content field.

    Accepts:
    - None -> empty string
    - str -> returned as-is
    - list of blocks -> concatenated text from type=text entries
    - list of str -> joined with newlines
    """
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        parts: list[str] = []
        for block in field:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(field)


def extract_blocks(field: Any) -> list[dict]:
    """Return the raw content blocks from a flexible content field.

    If field is a plain string it is wrapped as a text block.
    If field is already a list of dicts it is returned as-is.
    """
    if field is None:
        return []
    if isinstance(field, str):
        return [{"type": "text", "text": field}]
    if isinstance(field, list):
        out: list[dict] = []
        for block in field:
            if isinstance(block, dict):
                out.append(block)
            elif isinstance(block, str):
                out.append({"type": "text", "text": block})
        return out
    return [{"type": "text", "text": str(field)}]
