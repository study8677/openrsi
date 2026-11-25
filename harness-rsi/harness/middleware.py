"""Plug-and-play long-horizon compaction middleware powered by OpenRSICompactor.

Wraps any multi-turn agent or `codex exec` / `codex app-server` workflow so that
when accumulated conversation + tool logs exceed `trigger_tokens`, older turns
are incrementally compacted into the RSI-evolved Delta Evidence Ledger while
keeping the most recent `keep_recent_turns` raw turns intact.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from compactbench.contracts import CompactionArtifact, CompactionRequest, Turn
from compactbench.providers import Provider

from evolver.cb_adapter import CodexExecProvider
from evolver.codex_client import count_tokens
from harness.cb_compactor import OpenRSICompactor


@dataclass
class CompactionEvent:
    cycle: int
    turns_compacted: int
    tokens_before: int
    tokens_after: int
    compression_ratio: float


@dataclass
class SessionCompactionMiddleware:
    """Stateful multi-cycle compaction middleware for long-running coding agents."""

    budget_tokens: int = 2048
    trigger_tokens: int = 8192
    keep_recent_turns: int = 4
    model: str = "codex-default"
    provider: Provider | None = None
    _compactor: OpenRSICompactor = field(init=False)
    _artifact: CompactionArtifact | None = field(default=None, init=False)
    _pending_turns: list[dict[str, Any]] = field(default_factory=list, init=False)
    _cycle: int = field(default=0, init=False)
    _next_turn_id: int = field(default=1, init=False)
    history: list[CompactionEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = CodexExecProvider(model=None if self.model == "codex-default" else self.model)
        self._compactor = OpenRSICompactor(provider=self.provider, model=self.model)

    def add_message(self, role: str, content: str) -> None:
        """Append a new user/assistant/tool turn to the active session buffer."""
        self._pending_turns.append(
            {
                "turn_id": self._next_turn_id,
                "role": role,
                "content": content,
            }
        )
        self._next_turn_id += 1

    def current_context_tokens(self) -> int:
        return count_tokens(self.render_context())

    def render_context(self) -> str:
        """Render the compact prompt context: [Evidenced Ledger] + [Recent Raw Turns]."""
        parts: list[str] = []
        if self._artifact is not None and self._artifact.summary_text:
            parts.append(f"=== COMPACTED EVIDENCE LEDGER (cycle {self._cycle - 1}) ===\n{self._artifact.summary_text}")
        if self._pending_turns:
            recent_lines = [
                f"[Turn {t['turn_id']} | {t['role'].upper()}]\n{t['content']}"
                for t in self._pending_turns
            ]
            parts.append("=== RECENT TURNS ===\n" + "\n\n".join(recent_lines))
        return "\n\n".join(parts)

    async def maybe_compact(self, force: bool = False) -> CompactionEvent | None:
        """Compact older pending turns into the Delta Evidence Ledger if over trigger_tokens."""
        tokens_before = self.current_context_tokens()
        if not force and tokens_before < self.trigger_tokens:
            return None

        if len(self._pending_turns) <= self.keep_recent_turns and not force:
            return None

        split_idx = (
            len(self._pending_turns)
            if force and len(self._pending_turns) <= self.keep_recent_turns
            else max(1, len(self._pending_turns) - self.keep_recent_turns)
        )
        to_compact = self._pending_turns[:split_idx]
        retained = self._pending_turns[split_idx:]

        cb_turns = [
            Turn(
                id=str(t["turn_id"]),
                role=t["role"] if t["role"] in ("user", "assistant", "tool", "system") else "user",
                content=t["content"],
            )
            for t in to_compact
        ]
        req = CompactionRequest(
            cycle=self._cycle,
            turns=cb_turns,
            previous_artifact=self._artifact,
            token_budget=self.budget_tokens,
        )
        self._artifact = await self._compactor.compact(req)
        self._pending_turns = retained
        self._cycle += 1

        tokens_after = self.current_context_tokens()
        event = CompactionEvent(
            cycle=self._cycle - 1,
            turns_compacted=len(to_compact),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=round(max(0.0, 1.0 - tokens_after / max(1, tokens_before)), 4),
        )
        self.history.append(event)
        return event

    def maybe_compact_sync(self, force: bool = False) -> CompactionEvent | None:
        """Synchronous convenience wrapper around `maybe_compact`."""
        return asyncio.run(self.maybe_compact(force=force))
