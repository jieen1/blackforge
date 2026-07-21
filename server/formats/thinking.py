"""Thinking / reasoning block removal.

The Qwen3.6 chat template injects a <think> tag at the start of
assistant generation. The model produces reasoning content followed by
</think> and the actual answer. We strip all of this.

Observed patterns:
1. <think>...</think> (normal paired tags)
2. Thinking Process:...\n</think>\n\nAnswer (orphan close tag)
3. <think>... (unclosed, hit max_tokens)
4. Thinking Process:... (no tags at all, rare)
"""

from __future__ import annotations

import re

# Pattern 1: Properly closed <think>...</think> blocks
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Pattern 2: Orphan </think> without opening tag
# Matches everything from start of string up to and including first </think>
_ORPHAN_CLOSE_RE = re.compile(r"\A.*?</think>\s*", re.DOTALL)

# Pattern 3: Unclosed <think> block (hit max_tokens mid-thinking)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)

# Pattern 4: Plain-text thinking prefix (no XML tags at all)
_THINKING_PREFIX_RE = re.compile(
    r"\A(?:Here.s a thinking process|Thinking Process):.*", re.DOTALL
)


def strip_thinking(text: str) -> str:
    """Remove all thinking/reasoning content from model output."""
    # Pattern 1: paired <think>...</think>
    text = _THINK_BLOCK_RE.sub("", text)
    # Pattern 2: orphan </think> (e.g. "Thinking Process:...\n</think>")
    text = _ORPHAN_CLOSE_RE.sub("", text)
    # Pattern 3: unclosed <think> (hit max_tokens)
    text = _UNCLOSED_THINK_RE.sub("", text)
    # Pattern 4: plain-text prefix fallback (no tags at all)
    text = _THINKING_PREFIX_RE.sub("", text)
    return text.strip()
