"""Deep structural trace analyzer for CompactBench runs.

Separates failures into actionable root causes (strictly aggregate stats, zero
question/answer leakage) so Harness-RSI knows *mechanistically* why information
is lost:
  - Birth loss (missed on origin cycle: extraction/prompt gap)
  - Drift / eviction loss (survived origin cycle, lost later: packing/update gap)
  - False abstention (reader saw no answer and returned NOT_STATED)
  - Confusion / wrong answer (reader answered wrongly: entity/scope/reference mixup)
  - Stale answer (superseded decision/value survived)
  - Ledger fallback / saturation rate (budget ceiling causing row eviction)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table


class FamilyFailureModes(BaseModel):
    family: str
    probes: int
    mean_score: float
    failed_probes: int
    birth_loss: int
    drift_loss: int
    false_abstention: int
    wrong_answer: int
    stale_count: int
    trapped_count: int


class TraceDiagnosticReport(BaseModel):
    results_path: str
    traces_path: str | None
    num_cases: int
    num_cycles: int
    total_probes: int
    overall_score: float
    saturated_cycles: int
    saturated_cycle_rate: float
    fallback_cycles: int
    fallback_cycle_rate: float
    mean_rows_per_ledger: float
    family_modes: list[FamilyFailureModes]
    top_actionable_findings: list[str]


def analyze_run(
    results_path: Path,
    traces_path: Path | None = None,
    saturation_threshold_ratio: float = 0.88,
) -> TraceDiagnosticReport:
    """Compute structural root-cause diagnostics from CompactBench results + traces."""
    results_path = Path(results_path)
    lines = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    num_cases = 0
    num_cycles = 0
    total_probes = 0
    overall_score = 0.0
    saturated_cycles = 0

    fam_scores: dict[str, list[float]] = defaultdict(list)
    fam_failed: Counter[str] = Counter()
    fam_birth: Counter[str] = Counter()
    fam_drift: Counter[str] = Counter()
    fam_abstain: Counter[str] = Counter()
    fam_wrong: Counter[str] = Counter()
    fam_stale: Counter[str] = Counter()
    fam_trap: Counter[str] = Counter()

    for ev in lines:
        etype = ev.get("event")
        if etype == "run_end":
            metrics = ev.get("metrics") or {}
            overall_score = float(metrics.get("score", 0.0) or 0.0)
            continue
        if etype != "case_complete":
            continue

        num_cases += 1
        cr = ev.get("case_result") or {}
        case_fam = cr.get("family", "unknown")

        # Track whether a probe key passed at its origin cycle within this case
        origin_passed: dict[tuple[str, int], bool] = {}

        for cyc in cr.get("cycles") or []:
            num_cycles += 1
            cidx = int(cyc.get("cycle", 0))
            art_tok = int(cyc.get("artifact_tokens", 0) or 0)
            budget = max(1, int(cyc.get("budget_tokens", 2048) or 2048))
            if art_tok >= int(budget * saturation_threshold_ratio):
                saturated_cycles += 1

            for pr in cyc.get("probe_results") or []:
                total_probes += 1
                fam = pr.get("family") or case_fam or "unknown"
                key = str(pr.get("key", "?"))
                orig = int(pr.get("origin_cycle", 0))
                sc = float(pr.get("score", 0.0) or 0.0)
                abstained = bool(pr.get("abstained", False))
                expects_abs = bool(pr.get("expects_abstention", False))
                stale = bool(pr.get("stale", False))
                trapped = bool(pr.get("trapped", False))

                fam_scores[fam].append(sc)
                if cidx == orig:
                    origin_passed[(key, orig)] = sc >= 0.999

                if sc < 0.999:
                    fam_failed[fam] += 1
                    if cidx == orig or not origin_passed.get((key, orig), False):
                        fam_birth[fam] += 1
                    else:
                        fam_drift[fam] += 1

                    if abstained and not expects_abs:
                        fam_abstain[fam] += 1
                    else:
                        fam_wrong[fam] += 1
                    if stale:
                        fam_stale[fam] += 1
                    if trapped:
                        fam_trap[fam] += 1

    # Inspect traces for ledger row count & fallback detection
    fallback_cycles = 0
    trace_cycles = 0
    row_counts: list[int] = []
    if traces_path is not None and Path(traces_path).exists():
        with Path(traces_path).open(encoding="utf-8") as fp:
            for raw in fp:
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                trace_cycles += 1
                art = rec.get("artifact") or {}
                summary = str(art.get("summary_text") or art.get("summaryText") or "")
                summary_lines = [ln for ln in summary.splitlines() if ln.strip()]
                row_counts.append(max(0, len(summary_lines) - 1))
                # Heuristic: fallback extractive lines contain raw `[turn | speaker]` tags
                raw_excerpt_lines = sum(1 for ln in summary_lines if "| [" in ln and " | " in ln)
                if not summary_lines or raw_excerpt_lines >= max(3, len(summary_lines) // 2):
                    fallback_cycles += 1

    family_modes: list[FamilyFailureModes] = []
    for fam, scores in sorted(fam_scores.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1]))):
        family_modes.append(
            FamilyFailureModes(
                family=fam,
                probes=len(scores),
                mean_score=round(sum(scores) / max(1, len(scores)), 4),
                failed_probes=fam_failed[fam],
                birth_loss=fam_birth[fam],
                drift_loss=fam_drift[fam],
                false_abstention=fam_abstain[fam],
                wrong_answer=fam_wrong[fam],
                stale_count=fam_stale[fam],
                trapped_count=fam_trap[fam],
            )
        )

    sat_rate = saturated_cycles / max(1, num_cycles)
    fb_rate = fallback_cycles / max(1, trace_cycles) if trace_cycles > 0 else 0.0
    mean_rows = sum(row_counts) / max(1, len(row_counts))

    findings: list[str] = []
    if sat_rate >= 0.60:
        findings.append(
            f"Budget saturation is high ({saturated_cycles}/{num_cycles} cycles = {sat_rate:.1%} >= 88% of budget): "
            "rows are actively competing for budget; tighter record density or smarter eviction directly recovers recall."
        )
    if fb_rate >= 0.05:
        findings.append(
            f"Patch parser fell back to raw excerpts in {fallback_cycles}/{trace_cycles} cycles ({fb_rate:.1%}): "
            "make delta-patch parsing tolerant of minor formatting noise instead of rejecting the entire patch."
        )
    for fm in family_modes[:4]:
        dominant = "false-abstention (fact dropped/missing)" if fm.false_abstention >= fm.wrong_answer else "wrong-answer (entity/scope/value confusion)"
        phase = "birth-loss (missed on origin cycle)" if fm.birth_loss >= fm.drift_loss else "drift-loss (evicted/corrupted in later cycles)"
        findings.append(
            f"Weak family `{fm.family}` ({fm.mean_score:.2f}): {fm.failed_probes} failures — "
            f"primarily {phase} ({fm.birth_loss} birth vs {fm.drift_loss} drift) & {dominant} "
            f"({fm.false_abstention} abstain vs {fm.wrong_answer} wrong, stale={fm.stale_count})."
        )

    return TraceDiagnosticReport(
        results_path=str(results_path),
        traces_path=str(traces_path) if traces_path else None,
        num_cases=num_cases,
        num_cycles=num_cycles,
        total_probes=total_probes,
        overall_score=round(overall_score, 4),
        saturated_cycles=saturated_cycles,
        saturated_cycle_rate=round(sat_rate, 4),
        fallback_cycles=fallback_cycles,
        fallback_cycle_rate=round(fb_rate, 4),
        mean_rows_per_ledger=round(mean_rows, 1),
        family_modes=family_modes,
        top_actionable_findings=findings,
    )


def print_trace_diagnostics(console: Console, report: TraceDiagnosticReport) -> None:
    console.rule("[bold cyan]OpenRSI CompactBench Structural Trace Diagnostics")
    console.print(
        f"Cases: [bold]{report.num_cases}[/bold] | Cycles: [bold]{report.num_cycles}[/bold] | "
        f"Probes: [bold]{report.total_probes}[/bold] | Overall Score: [bold]{report.overall_score:.3f}[/bold]"
    )
    console.print(
        f"Budget Saturation (>=88%): [bold]{report.saturated_cycles}/{report.num_cycles} ({report.saturated_cycle_rate:.1%})[/bold] | "
        f"Fallback Cycles: [bold]{report.fallback_cycles} ({report.fallback_cycle_rate:.1%})[/bold] | "
        f"Mean Ledger Rows: [bold]{report.mean_rows_per_ledger:.1f}[/bold]\n"
    )

    table = Table(title="Failure Root-Cause Breakdown by Family (Weak → Strong)")
    table.add_column("Family", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Failed", justify="right")
    table.add_column("Birth Loss", justify="right")
    table.add_column("Drift Loss", justify="right")
    table.add_column("False Abstain", justify="right")
    table.add_column("Wrong Ans", justify="right")
    table.add_column("Stale", justify="right")

    for fm in report.family_modes:
        table.add_row(
            fm.family,
            f"{fm.mean_score:.3f}",
            str(fm.failed_probes),
            str(fm.birth_loss),
            str(fm.drift_loss),
            str(fm.false_abstention),
            str(fm.wrong_answer),
            str(fm.stale_count),
        )
    console.print(table)

    console.print("\n[bold yellow]Actionable Root-Cause Findings:[/bold yellow]")
    for idx, f in enumerate(report.top_actionable_findings, start=1):
        console.print(f"  {idx}. {f}")
