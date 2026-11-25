"""Unit tests for SessionCompactionMiddleware and trace_analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from compactbench.providers import CompletionRequest, CompletionResponse, Provider

from evolver.trace_analyzer import analyze_run
from harness.middleware import SessionCompactionMiddleware


class _MockLedgerProvider(Provider):
    key: ClassVar[str] = "mock-ledger"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Return a valid delta-ledger addition respecting OpenRSICompactor's format
        return CompletionResponse(
            text="A | H | RULE: Must preserve exact UTC ISO8601 timestamps and never use third-party deps.",
            prompt_tokens=100,
            completion_tokens=20,
            model="mock",
            raw={},
        )


def test_middleware_triggers_and_preserves_recent_turns() -> None:
    mw = SessionCompactionMiddleware(
        budget_tokens=512,
        trigger_tokens=120,
        keep_recent_turns=2,
        provider=_MockLedgerProvider(),
    )
    for i in range(6):
        mw.add_message("user", f"Turn {i}: " + ("diagnostic log payload " * 12))

    event = mw.maybe_compact_sync()
    assert event is not None
    assert event.cycle == 0
    assert event.turns_compacted == 4
    assert event.tokens_after < event.tokens_before
    rendered = mw.render_context()
    assert "COMPACTED EVIDENCE LEDGER" in rendered
    assert "Turn 4" in rendered and "Turn 5" in rendered
    assert "Turn 0" not in rendered


def test_middleware_no_trigger_below_threshold() -> None:
    mw = SessionCompactionMiddleware(
        budget_tokens=512,
        trigger_tokens=5000,
        keep_recent_turns=2,
        provider=_MockLedgerProvider(),
    )
    mw.add_message("user", "short hello")
    assert mw.maybe_compact_sync() is None


def test_trace_analyzer_on_v1_if_present() -> None:
    res = Path(__file__).resolve().parent.parent / "reports" / "cb_practice48_openrsi_v1.jsonl"
    if not res.exists():
        return
    rep = analyze_run(res)
    assert rep.num_cases == 48
    assert rep.total_probes == 4546
    assert len(rep.family_modes) == 8
    assert len(rep.top_actionable_findings) >= 1
