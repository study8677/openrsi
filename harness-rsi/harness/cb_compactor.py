"""Evolvable CompactBench Compactor (Baseline v0).

Primary mutable target for OpenRSI Harness-RSI on CompactBench's multi-cycle
drift benchmark. Baseline v0 is a naive tail-truncation compactor that keeps
only the most recent turns within `request.token_budget`.
"""

from __future__ import annotations

from typing import ClassVar

import tiktoken
from compactbench.compactors import Compactor
from compactbench.contracts import CompactionArtifact, CompactionRequest

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode_ordinary(text))


class OpenRSICompactor(Compactor):
    """Evolvable context compactor for CompactBench multi-cycle drift."""

    name: ClassVar[str] = "openrsi-compactor"
    version: ClassVar[str] = "0.1.0"

    async def compact(self, request: CompactionRequest) -> CompactionArtifact:
        budget = max(128, int(request.token_budget or 2048))
        blocks: list[str] = []

        if request.previous_artifact is not None and request.previous_artifact.summary_text:
            blocks.append(f"[PREVIOUS SUMMARY]\n{request.previous_artifact.summary_text}")

        for turn in request.turns:
            speaker = getattr(turn, "speaker", None) or getattr(turn, "role", "unknown")
            tid = getattr(turn, "turn_id", getattr(turn, "id", "?"))
            text = getattr(turn, "text", None) or getattr(turn, "content", "")
            blocks.append(f"[Turn {tid} | {speaker}]\n{text}")

        # Baseline v0: naive tail truncation up to budget
        kept: list[str] = []
        used = 0
        for blk in reversed(blocks):
            t = _count_tokens(blk)
            if used + t <= budget:
                kept.append(blk)
                used += t
            else:
                rem = budget - used
                if rem > 32:
                    ids = _ENC.encode_ordinary(blk)
                    kept.append(_ENC.decode(ids[-rem:]))
                break

        kept.reverse()
        summary = "\n\n".join(kept).strip()
        return CompactionArtifact(summary_text=summary)
