"""OpenRSI Harness-RSI CLI (`openrsi`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from evolver.cb_adapter import (
    DEFAULT_CB_COMPACTOR_PATH,
    print_cb_summary,
    run_compactbench_eval,
    run_compactbench_evolution,
)
from evolver.engine import _print_eval_table, run_evolution
from evolver.evaluator import HARNESS_RSI_ROOT, evaluate_suite

app = typer.Typer(help="OpenRSI — Harness-RSI Context Compaction Evolution Engine")
console = Console()


@app.command("generate-arena")
def generate_arena() -> None:
    """Generate/refresh the `arena/context-compaction` benchmark cases."""
    gen_path = HARNESS_RSI_ROOT / "arena" / "context-compaction" / "generator.py"
    spec = importlib.util.spec_from_file_location("context_compaction_gen", gen_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summaries = mod.generate_all()
    for s in summaries:
        console.print(
            f"[green]✓[/green] [bold]{s['id']}[/bold] ({s['split']}) — "
            f"{s['num_turns']} turns, raw={s['raw_tokens']} tokens, budget={s['max_tokens']} tokens"
        )


@app.command("eval")
def eval_cmd(
    split: str = typer.Option("train", help="Split to evaluate: train | held_out | all"),
    model: Optional[str] = typer.Option(None, help="Optional Codex model override (e.g. gpt-5.1-codex-mini)"),
    no_llm_compactor: bool = typer.Option(False, help="Disable llm_call inside compactor.py"),
) -> None:
    """Evaluate the current `harness/compactor.py` on the context-compaction arena."""
    report = evaluate_suite(
        split=split,
        model=model,
        allow_llm_in_compactor=not no_llm_compactor,
    )
    _print_eval_table(console, f"OpenRSI Eval — {split}", report)


@app.command("evolve")
def evolve_cmd(
    generations: int = typer.Option(3, min=1, max=20, help="Number of RSI evolution generations"),
    model: Optional[str] = typer.Option(None, help="Optional Codex model override"),
    eval_held_out: bool = typer.Option(True, help="Audit baseline and promoted candidates on held_out"),
) -> None:
    """Run the Harness-RSI self-improvement loop on `harness/compactor.py`."""
    run_evolution(
        generations=generations,
        model=model,
        eval_held_out=eval_held_out,
    )


@app.command("cb-eval")
def cb_eval_cmd(
    method: str = typer.Option(
        f"{DEFAULT_CB_COMPACTOR_PATH}:OpenRSICompactor",
        help="CompactBench method spec (e.g. built-in:codex-cli or path/to/file.py:Class)",
    ),
    suite: str = typer.Option("starter", help="CompactBench suite: starter | practice | drills"),
    seed_group: str = typer.Option("default", help="Seed group name for reproducible splits"),
    cases_per_family: Optional[int] = typer.Option(1, help="Cases per family (keep small for fast checks)"),
    drift_cycles: Optional[int] = typer.Option(None, help="Override number of drift cycles"),
    budget_tokens: int = typer.Option(2048, help="Token budget per compaction cycle"),
    model: Optional[str] = typer.Option(None, help="Optional Codex model override"),
    concurrency: int = typer.Option(2, help="Parallel codex exec workers"),
) -> None:
    """Evaluate a compactor on CompactBench via local `codex exec` (ChatGPT OAuth)."""
    summary = run_compactbench_eval(
        method_spec=method,
        suite=suite,
        seed_group=seed_group,
        cases_per_family=cases_per_family,
        drift_cycles=drift_cycles,
        budget_tokens=budget_tokens,
        model=model,
        concurrency=concurrency,
    )
    print_cb_summary(console, f"CompactBench Eval — {suite}", summary)


@app.command("cb-evolve")
def cb_evolve_cmd(
    generations: int = typer.Option(3, min=1, max=20, help="Number of RSI evolution generations"),
    suite: str = typer.Option("starter", help="CompactBench suite: starter | practice | drills"),
    cases_per_family: Optional[int] = typer.Option(1, help="Cases per family"),
    drift_cycles: Optional[int] = typer.Option(None, help="Override number of drift cycles"),
    budget_tokens: int = typer.Option(2048, help="Token budget per compaction cycle"),
    model: Optional[str] = typer.Option(None, help="Optional Codex model override"),
    concurrency: int = typer.Option(2, help="Parallel codex exec workers"),
) -> None:
    """Run Harness-RSI evolution on `harness/cb_compactor.py` using CompactBench."""
    run_compactbench_evolution(
        generations=generations,
        suite=suite,
        cases_per_family=cases_per_family,
        drift_cycles=drift_cycles,
        budget_tokens=budget_tokens,
        model=model,
        concurrency=concurrency,
    )


if __name__ == "__main__":
    app()
