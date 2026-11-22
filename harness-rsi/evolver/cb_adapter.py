"""CompactBench adapter & Harness-RSI evolution loop backed by local `codex exec`."""

from __future__ import annotations

import asyncio
import json
import py_compile
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compactbench.contracts import CompactionRequest
from compactbench.providers import CompletionRequest, CompletionResponse, Provider
from compactbench.runner import (
    CaseCompleteEvent,
    RunArgs,
    RunEndEvent,
    iter_events,
    resolve_compactor_class,
    run_experiment,
)
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from evolver.codex_client import call_codex_text, count_tokens
from evolver.evaluator import HARNESS_RSI_ROOT

DEFAULT_CB_COMPACTOR_PATH = HARNESS_RSI_ROOT / "harness" / "cb_compactor.py"
REPORTS_DIR = HARNESS_RSI_ROOT / "reports"
CB_ARCHIVE_DIR = REPORTS_DIR / "cb_archive"


class CodexExecProvider(Provider):
    """CompactBench Provider backed by local `codex exec` (ChatGPT OAuth, $0 extra API cost)."""

    key: ClassVar[str] = "codex-exec"

    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "low",
        timeout_sec: int = 180,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._timeout_sec = timeout_sec

    @property
    def endpoint_kind(self) -> str:
        return "custom"

    def _complete_sync(self, request: CompletionRequest) -> CompletionResponse:
        parts: list[str] = []
        if request.system:
            parts.append(f"=== SYSTEM INSTRUCTIONS ===\n{request.system.strip()}")
        if request.cached_prefix:
            parts.append(request.cached_prefix.strip())
        parts.append(request.prompt.strip())
        full_prompt = "\n\n".join(parts)

        with tempfile.TemporaryDirectory(prefix="openrsi_cb_codex_") as tmpdir:
            tmp = Path(tmpdir)
            out_path = tmp / "last_message.txt"
            schema_path: Path | None = None

            cmd = [
                "codex",
                "exec",
                "-C",
                str(tmp),
                "-s",
                "read-only",
                "-c",
                f'model_reasoning_effort="{self._reasoning_effort}"',
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "-o",
                str(out_path),
            ]
            chosen_model = self._model or (
                request.model if request.model and request.model != "codex-default" else None
            )
            if chosen_model:
                cmd.extend(["-m", chosen_model])

            if request.json_schema:
                schema_path = tmp / "schema.json"
                schema_path.write_text(json.dumps(request.json_schema), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])

            cmd.append("-")

            proc = subprocess.run(
                cmd,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self._timeout_sec,
                check=False,
            )
            text = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""
            if proc.returncode != 0 and not text:
                raise RuntimeError(
                    f"codex exec failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}"
                )

        return CompletionResponse(
            text=text,
            prompt_tokens=count_tokens(full_prompt),
            completion_tokens=count_tokens(text),
            model=chosen_model or "codex-default",
            raw={"provider": "codex-exec"},
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await asyncio.to_thread(self._complete_sync, request)


class CBRunSummary(BaseModel):
    method_spec: str
    suite: str
    seed_group: str
    overall_accuracy: float
    drift_resistance: float = 0.0
    decay: float = 0.0
    stale_rate: float = 0.0
    trap_rate: float = 0.0
    mean_artifact_tokens: float = 0.0
    budget_violations: int = 0
    by_family: dict[str, float] = {}
    cycle_accuracies: dict[str, float] = {}
    num_cases: int = 0
    total_probes: int = 0
    correct_probes: int = 0
    elapsed_sec: float = 0.0
    failure_samples: list[str] = []
    results_path: str = ""


def run_compactbench_eval(
    method_spec: str = f"{DEFAULT_CB_COMPACTOR_PATH}:OpenRSICompactor",
    suite: str = "starter",
    seed_group: str = "default",
    cases_per_family: int | None = 1,
    drift_cycles: int | None = None,
    budget_tokens: int = 2048,
    model: str | None = None,
    concurrency: int = 2,
    reasoning_effort: str = "low",
) -> CBRunSummary:
    """Run a CompactBench suite via local `codex exec` and return structured metrics."""
    t0 = time.monotonic()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"cb_results_{suite}_{seed_group}_{ts}.jsonl"

    provider = CodexExecProvider(model=model, reasoning_effort=reasoning_effort)
    args = RunArgs(
        method_spec=method_spec,
        suite=suite,
        provider_key="codex-exec",
        model=model or "codex-default",
        output_path=out_path,
        reader_provider_key="codex-exec",
        reader_model=model or "codex-default",
        drift_cycles=drift_cycles,
        cases_per_family=cases_per_family,
        seed_group=seed_group,
        budget_tokens=budget_tokens,
        concurrency=concurrency,
    )

    result_file = asyncio.run(run_experiment(args, provider=provider, reader_provider=provider))

    cases_count = 0
    total_probes = 0
    correct_probes = 0
    cycle_correct: dict[int, float] = {}
    cycle_total: dict[int, int] = {}
    failure_samples: list[str] = []
    overall_acc = 0.0
    drift_resistance = 0.0
    decay = 0.0
    stale_rate = 0.0
    trap_rate = 0.0
    mean_artifact_tokens = 0.0
    budget_violations = 0
    by_family: dict[str, float] = {}

    for ev in iter_events(result_file):
        if isinstance(ev, CaseCompleteEvent):
            cases_count += 1
            cr = ev.case_result
            fam = getattr(cr, "family", "unknown")
            for cyc in getattr(cr, "cycles", []):
                cidx = getattr(cyc, "cycle", 0)
                probes = getattr(cyc, "probe_results", []) or []
                for pr in probes:
                    total_probes += 1
                    pr_score = float(getattr(pr, "score", 0.0) or 0.0)
                    cycle_total[cidx] = cycle_total.get(cidx, 0) + 1
                    cycle_correct[cidx] = cycle_correct.get(cidx, 0.0) + pr_score
                    if pr_score >= 0.999:
                        correct_probes += 1
                    elif len(failure_samples) < 18:
                        key = getattr(pr, "key", "?")
                        cat = getattr(pr, "category", "?")
                        orig = getattr(pr, "origin_cycle", 0)
                        stale = getattr(pr, "stale", False)
                        trapped = getattr(pr, "trapped", False)
                        failure_samples.append(
                            f"[family={fam} cycle={cidx} origin={orig} cat={cat}] key={key} "
                            f"score={pr_score:.2f} stale={stale} trapped={trapped}"
                        )
        elif isinstance(ev, RunEndEvent):
            metrics = ev.metrics
            if metrics is not None:
                val = getattr(metrics, "score", None)
                if isinstance(val, (int, float)):
                    overall_acc = float(val)
                drift_resistance = float(getattr(metrics, "drift_resistance", 0.0) or 0.0)
                decay = float(getattr(metrics, "decay", 0.0) or 0.0)
                stale_rate = float(getattr(metrics, "stale_rate", 0.0) or 0.0)
                trap_rate = float(getattr(metrics, "trap_rate", 0.0) or 0.0)
                mean_artifact_tokens = float(getattr(metrics, "mean_artifact_tokens", 0.0) or 0.0)
                budget_violations = int(getattr(metrics, "budget_violations", 0) or 0)
                raw_fam = getattr(metrics, "by_family", None) or {}
                for fk, fv in raw_fam.items():
                    fscore = getattr(fv, "score", None)
                    if fscore is None and isinstance(fv, dict):
                        fscore = fv.get("score")
                    if isinstance(fscore, (int, float)):
                        by_family[str(fk)] = round(float(fscore), 4)

    if overall_acc == 0.0 and total_probes > 0:
        overall_acc = sum(cycle_correct.values()) / total_probes

    cycle_accs = {
        f"cycle_{k}": round(cycle_correct.get(k, 0.0) / max(1, cycle_total[k]), 4)
        for k in sorted(cycle_total.keys())
    }

    return CBRunSummary(
        method_spec=method_spec,
        suite=suite,
        seed_group=seed_group,
        overall_accuracy=round(overall_acc, 4),
        drift_resistance=round(drift_resistance, 4),
        decay=round(decay, 4),
        stale_rate=round(stale_rate, 4),
        trap_rate=round(trap_rate, 4),
        mean_artifact_tokens=round(mean_artifact_tokens, 1),
        budget_violations=budget_violations,
        by_family=by_family,
        cycle_accuracies=cycle_accs,
        num_cases=cases_count,
        total_probes=total_probes,
        correct_probes=correct_probes,
        elapsed_sec=round(time.monotonic() - t0, 2),
        failure_samples=failure_samples,
        results_path=str(result_file),
    )


def print_cb_summary(console: Console, title: str, s: CBRunSummary) -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Method", s.method_spec)
    table.add_row("Suite / Seed", f"{s.suite} / {s.seed_group}")
    table.add_row("Cases / Probes", f"{s.num_cases} cases / {s.correct_probes}/{s.total_probes} full-credit")
    table.add_row("Official Score", f"{s.overall_accuracy:.3f} ({s.overall_accuracy:.1%})")
    table.add_row("Drift Resistance / Decay", f"{s.drift_resistance:.3f} / decay={s.decay:+.3f}")
    table.add_row("Mean Artifact Tokens / Violations", f"{s.mean_artifact_tokens:.0f} tok / {s.budget_violations} violations")
    if s.cycle_accuracies:
        cyc_str = " | ".join(f"{k}: {v:.2f}" for k, v in s.cycle_accuracies.items())
        table.add_row("Per-Cycle Score", cyc_str)
    if s.by_family:
        fam_str = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(s.by_family.items()))
        table.add_row("By Family", fam_str)
    table.add_row("Elapsed", f"{s.elapsed_sec:.1f}s")
    console.print(table)


def _extract_python(text: str) -> str | None:
    blocks = re.findall(r"^```python[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.DOTALL | re.MULTILINE)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    blocks = re.findall(r"^```[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.DOTALL | re.MULTILINE)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return None


def run_compactbench_evolution(
    generations: int = 3,
    suite: str = "starter",
    cases_per_family: int | None = 1,
    drift_cycles: int | None = None,
    budget_tokens: int = 2048,
    model: str | None = None,
    concurrency: int = 2,
    compactor_path: Path = DEFAULT_CB_COMPACTOR_PATH,
) -> list[dict[str, Any]]:
    """Run Harness-RSI evolution on `harness/cb_compactor.py` using CompactBench."""
    console = Console()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CB_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    method_spec = f"{compactor_path}:OpenRSICompactor"
    best_code = compactor_path.read_text(encoding="utf-8")
    history: list[dict[str, Any]] = []

    console.rule(f"[bold blue]CompactBench Gen 0 — Baseline ({suite})")
    best = run_compactbench_eval(
        method_spec=method_spec,
        suite=suite,
        seed_group="openrsi_train",
        cases_per_family=cases_per_family,
        drift_cycles=drift_cycles,
        budget_tokens=budget_tokens,
        model=model,
        concurrency=concurrency,
    )
    print_cb_summary(console, "Gen 0 (Baseline) — CompactBench", best)
    (CB_ARCHIVE_DIR / "gen_00_cb_compactor.py").write_text(best_code, encoding="utf-8")
    history.append(
        {
            "generation": 0,
            "status": "baseline",
            "accuracy": best.overall_accuracy,
            "cycle_accuracies": best.cycle_accuracies,
            "probes": f"{best.correct_probes}/{best.total_probes}",
        }
    )

    for gen in range(1, generations + 1):
        console.rule(f"[bold magenta]CompactBench Gen {gen} / {generations}")
        if best.overall_accuracy >= 0.99:
            console.print("[green]Reached >=99% accuracy on CompactBench! Stopping early.[/green]")
            break

        prompt = f"""You are the Meta-Agent Proposer in OpenRSI evolving `harness/cb_compactor.py` for CompactBench.
CompactBench tests MULTI-CYCLE context compaction drift across 8 adversarial families:
buried_constraint, conditional_scope, decision_override, entity_confusion, numeric_precision,
provenance_authority, reference_resolution, and task_continuity.

Critical rules for `OpenRSICompactor(Compactor)`:
1. Keep class name `OpenRSICompactor(Compactor)` with `name = "openrsi-compactor"` and `async def compact(self, request: CompactionRequest) -> CompactionArtifact`.
2. In cycle 0, `request.turns` is the opening conversation; in later cycles, `request.previous_artifact` is YOUR previous output and `request.turns` contains ONLY the new turns!
3. You may use deterministic Python extraction/structuring AND/OR `await self.provider.complete(CompletionRequest(model=self.model, prompt=..., max_tokens=...))` (hybrid structured ledger + LLM update works especially well for multi-cycle drift and decision overrides!).
4. Keep the returned `CompactionArtifact(summary_text=...)` strictly within `request.token_budget` tokens (verify with `tiktoken.get_encoding("cl100k_base")`).
5. Return the COMPLETE updated `harness/cb_compactor.py` inside a single ```python ... ``` code block.

## Current Best Performance
- Official Score: {best.overall_accuracy:.3f} ({best.overall_accuracy:.1%})
- Drift Resistance: {best.drift_resistance:.3f} | Decay (first - last): {best.decay:+.3f}
- Mean Artifact Tokens: {best.mean_artifact_tokens:.0f} | Budget Violations: {best.budget_violations} (aim for 0 violations by staying <= 90% of `request.token_budget`!)
- Per-Cycle Score: {best.cycle_accuracies}
- Per-Family Score: {best.by_family}
- Sample Failures:
{chr(10).join("  - " + f for f in best.failure_samples) or "  (none)"}

## Current Best `harness/cb_compactor.py`
```python
{best_code}
```

Analyze why information drifted or got lost across cycles, and output the improved `harness/cb_compactor.py` inside a single ```python ... ``` block.
"""
        console.print("[cyan]Proposing candidate patch for harness/cb_compactor.py...[/cyan]")
        reply = call_codex_text(prompt, model=model, timeout_sec=240)
        cand_code = _extract_python(reply)
        if not cand_code:
            console.print("[red]No Python code block returned. Skipping.[/red]")
            history.append({"generation": gen, "status": "invalid", "accuracy": best.overall_accuracy})
            continue

        compactor_path.write_text(cand_code, encoding="utf-8")
        try:
            py_compile.compile(str(compactor_path), doraise=True)
            resolve_compactor_class(method_spec)
        except Exception as exc:
            console.print(f"[red]Candidate failed compile/load check: {exc}. Rolling back.[/red]")
            compactor_path.write_text(best_code, encoding="utf-8")
            history.append({"generation": gen, "status": "invalid", "accuracy": best.overall_accuracy})
            continue

        cand = run_compactbench_eval(
            method_spec=method_spec,
            suite=suite,
            seed_group="openrsi_train",
            cases_per_family=cases_per_family,
            drift_cycles=drift_cycles,
            budget_tokens=budget_tokens,
            model=model,
            concurrency=concurrency,
        )
        print_cb_summary(console, f"Gen {gen} Candidate — CompactBench", cand)

        if cand.overall_accuracy > best.overall_accuracy + 1e-4:
            console.print(
                f"[bold green]KEPT Gen {gen}![/bold green] "
                f"{best.overall_accuracy:.2%} -> {cand.overall_accuracy:.2%}"
            )
            best_code = cand_code
            best = cand
            (CB_ARCHIVE_DIR / f"gen_{gen:02d}_cb_compactor.py").write_text(best_code, encoding="utf-8")
            history.append(
                {
                    "generation": gen,
                    "status": "kept",
                    "accuracy": cand.overall_accuracy,
                    "cycle_accuracies": cand.cycle_accuracies,
                    "probes": f"{cand.correct_probes}/{cand.total_probes}",
                }
            )
        else:
            console.print(
                f"[yellow]REVERTED Gen {gen}[/yellow] "
                f"({cand.overall_accuracy:.2%} <= {best.overall_accuracy:.2%})"
            )
            compactor_path.write_text(best_code, encoding="utf-8")
            history.append(
                {
                    "generation": gen,
                    "status": "reverted",
                    "accuracy": cand.overall_accuracy,
                    "cycle_accuracies": cand.cycle_accuracies,
                    "probes": f"{cand.correct_probes}/{cand.total_probes}",
                }
            )

    (REPORTS_DIR / "cb_evolution_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    gens = [h["generation"] for h in history]
    accs = [h["accuracy"] for h in history]
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(gens, accs, marker="o", linewidth=2, label=f"CompactBench ({suite}) Accuracy")
    plt.ylim(-0.05, 1.08)
    plt.xticks(gens)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("OpenRSI Harness-RSI on CompactBench (Multi-Cycle Drift)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "cb_evolution_curve.png", dpi=150)
    plt.close()
    return history
