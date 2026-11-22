"""Evolvable Context Compactor (Baseline v0).

This file is the primary mutable target of OpenRSI's Harness-RSI evolution loop.
Baseline v0 uses a naive tail sliding window that simply keeps the most recent
messages up to `max_tokens`, dropping earlier conversation turns and constraints.
"""

from __future__ import annotations

from typing import Any, Callable

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base."""
    return len(_ENC.encode(text))


def compact(
    messages: list[dict[str, Any]],
    max_tokens: int,
    llm_call: Callable[[str], str] | None = None,
) -> str:
    """Compact a long conversation + tool-log trajectory into <= max_tokens.

    Args:
        messages: Ordered list of message dicts with keys `role` ('user',
            'assistant', 'tool') and `content` (str), plus optional `turn` (int).
        max_tokens: Hard token budget for the returned compacted context string.
        llm_call: Optional callable(prompt: str) -> str backed by `codex exec`
            for semantic summarization if desired.

    Returns:
        A compacted context string to pass to the downstream executor.
    """
    _ = llm_call  # Baseline v0 does not use semantic summarization.

    kept: list[str] = []
    used = 0
    for msg in reversed(messages):
        role = msg.get("role", "unknown").upper()
        turn = msg.get("turn", "?")
        content = msg.get("content", "")
        block = f"[Turn {turn} | {role}]\n{content}\n"
        block_tokens = count_tokens(block)

        if used + block_tokens <= max_tokens:
            kept.append(block)
            used += block_tokens
        else:
            remaining = max_tokens - used
            if remaining > 40:
                encoded = _ENC.encode(block)
                kept.append(_ENC.decode(encoded[-remaining:]))
            break

    kept.reverse()
    return "\n".join(kept)
