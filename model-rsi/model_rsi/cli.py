"""Typer CLI for OpenRSI Model-RSI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from model_rsi.flywheel import import_harness_evolution_traces
from model_rsi.protocol import TrajectoryStore, save_tasks_jsonl
from model_rsi.reporting import load_evolution_history, plot_evolution_curve
from model_rsi.tasks.countdown import generate_countdown_suite
from model_rsi.tasks.regression import get_regression_suite

app = typer.Typer(
    name="model-rsi",
    help="OpenRSI Model-RSI CLI: verifiable tasks, trajectory protocol, flywheel & evolution reports.",
    no_args_is_help=True,
)
console = Console()


@app.command("generate-tasks")
def generate_tasks(
    out_dir: Path = typer.Option(Path("data"), help="Output directory for JSONL task splits"),
    seed: int = typer.Option(42, help="Deterministic RNG seed"),
) -> None:
    """Generate strictly disjoint Countdown (train/val/held_out) + Regression task suites."""
    tasks = generate_countdown_suite(seed=seed) + get_regression_suite()
    for split in ("train", "val", "held_out", "regression"):
        split_tasks = [t for t in tasks if t.split == split]
        save_tasks_jsonl(split_tasks, out_dir / f"{split}.jsonl")
        console.print(f"[green]Saved[/green] {len(split_tasks):3d} tasks -> {out_dir / f'{split}.jsonl'}")


@app.command("import-harness")
def import_harness(
    harness_reports: Path = typer.Option(
        Path("../harness-rsi/reports"),
        help="Path to harness-rsi/reports directory (read-only)",
    ),
    store_path: Path = typer.Option(
        Path("data/trajectories.jsonl"),
        help="Destination Model-RSI TrajectoryStore JSONL",
    ),
    min_score: float = typer.Option(0.80, help="Minimum score to import"),
) -> None:
    """Import verified breakthrough traces from Harness-RSI into the Model-RSI TrajectoryStore."""
    store = TrajectoryStore(store_path)
    n = import_harness_evolution_traces(harness_reports, store, min_score=min_score)
    console.print(f"[cyan]Imported {n} verified Harness-RSI traces -> {store_path}[/cyan]")


@app.command("status")
def status(
    history_path: Path = typer.Option(
        Path("reports/evolution_history.json"),
        help="Path to evolution_history.json",
    ),
) -> None:
    """Display the multi-generation Model-RSI evolution ledger."""
    history = load_evolution_history(history_path)
    if not history:
        console.print(f"[yellow]No evolution history found at {history_path}[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="OpenRSI Model-RSI Evolution History")
    table.add_column("Gen", justify="right")
    table.add_column("Paradigm")
    table.add_column("Status")
    table.add_column("Held-out Pass@1", justify="right")
    table.add_column("Val Pass@1", justify="right")
    table.add_column("Regression", justify="right")
    table.add_column("Notes")

    for h in history:
        style = "green" if h.status == "promoted" else ("red" if h.status == "rejected" else "cyan")
        table.add_row(
            str(h.generation),
            h.paradigm,
            f"[{style}]{h.status}[/{style}]",
            f"{h.held_out_pass_at_1:.2%}",
            f"{h.val_pass_at_1:.2%}",
            f"{h.regression_score:.2%}",
            h.notes[:60],
        )
    console.print(table)


@app.command("plot")
def plot(
    history_path: Path = typer.Option(Path("reports/evolution_history.json")),
    out_path: Path = typer.Option(Path("reports/evolution_curve.png")),
) -> None:
    """Render the Model-RSI evolution curve PNG from evolution_history.json."""
    history = load_evolution_history(history_path)
    if not history:
        console.print("[red]No history to plot.[/red]")
        raise typer.Exit(code=1)
    plot_evolution_curve(history, out_path)
    console.print(f"[green]Saved evolution curve -> {out_path}[/green]")


@app.command("export-patch-dataset")
def export_patch_dataset(
    traces_path: Path = typer.Option(
        Path("../harness-rsi/reports/cb_practice48_openrsi_v1.traces.jsonl"),
        help="Read-only Harness-RSI CompactBench traces JSONL",
    ),
    out_path: Path = typer.Option(
        Path("data/compactbench_patch_distill_v1.jsonl.gz"),
        help="Output gzipped Compactor patch distillation dataset",
    ),
    train_cases: int = typer.Option(36, help="Strictly isolated train case count"),
    seed: int = typer.Option(42, help="Deterministic case_id split seed"),
) -> None:
    """Build & verify strictly train-isolated OpenRSICompactor patch distillation examples."""
    from model_rsi.tasks.compactbench_patch import (
        build_compactor_patch_distill_examples,
        save_patch_distill_gz,
    )

    examples = build_compactor_patch_distill_examples(
        traces_path, train_cases=train_cases, seed=seed
    )
    saved = save_patch_distill_gz(examples, out_path)
    n_audit = sum(1 for e in examples if e.get("audit"))
    console.print(
        f"[green]Exported {len(examples)} atomically verified patch examples "
        f"(main={len(examples) - n_audit}, audit={n_audit}) -> {saved}[/green]"
    )


@app.command("audit-splits")
def audit_splits(
    flywheel_gz: Path = typer.Option(
        Path("data/compactbench_flywheel_v1.jsonl.gz"),
        help="Gzipped Phase 4 Flywheel tasks",
    ),
    patch_gz: Path = typer.Option(
        Path("data/compactbench_patch_distill_v1.jsonl.gz"),
        help="Gzipped Phase 4.2 Compactor patch dataset",
    ),
) -> None:
    """Verify zero case_id leakage across train / val / held_out / official starter."""
    import gzip
    import json

    from compactbench.scenarios.generate import expand_suite, get_suite
    from model_rsi.protocol import VerifiableTask
    from model_rsi.tasks.compactbench_patch import load_patch_distill_gz

    tasks = [
        VerifiableTask.model_validate(json.loads(line))
        for line in gzip.open(flywheel_gz, "rt", encoding="utf-8")
        if line.strip()
    ]
    by_split: dict[str, set[str]] = {}
    for t in tasks:
        by_split.setdefault(t.split, set()).add(str(t.metadata.get("case_id", "")))

    starter_ids = {
        c.case_id
        for c in expand_suite(get_suite("starter"), seed_group="default", cases_per_family=1)
    }
    patch_ids = {str(e.get("case_id", "")) for e in load_patch_distill_gz(patch_gz)}

    assert by_split["train"].isdisjoint(by_split["val"]), "train/val overlap!"
    assert by_split["train"].isdisjoint(by_split["held_out"]), "train/held_out overlap!"
    assert by_split["val"].isdisjoint(by_split["held_out"]), "val/held_out overlap!"
    assert patch_ids.issubset(by_split["train"]), "patch dataset leaked outside train!"
    assert patch_ids.isdisjoint(starter_ids), "patch dataset overlaps starter!"

    console.print(
        f"[bold green]SPLIT AUDIT PASSED (0 leaks):[/bold green] "
        f"train={len(by_split['train'])}, val={len(by_split['val'])}, "
        f"held_out={len(by_split['held_out'])}, patch_train={len(patch_ids)}, "
        f"starter={len(starter_ids)}"
    )


if __name__ == "__main__":
    app()
