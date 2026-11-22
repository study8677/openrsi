"""Harness-RSI Evolution Engine for `harness/compactor.py`."""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from evolver.codex_client import call_codex_text
from evolver.evaluator import (
    DEFAULT_CASES_DIR,
    DEFAULT_COMPACTOR_PATH,
    HARNESS_RSI_ROOT,
    EvalReport,
    evaluate_suite,
    load_compactor_module,
)

REPORTS_DIR = HARNESS_RSI_ROOT / "reports"
ARCHIVE_DIR = REPORTS_DIR / "archive"


class GenerationRecord(BaseModel):
    generation: int
    status: str  # "baseline" | "kept" | "reverted" | "invalid"
    train_score: float
    train_pass_rate: float
    train_compression_ratio: float
    held_out_score: float | None = None
    held_out_pass_rate: float | None = None
    notes: str = ""


def _extract_python_code(text: str) -> str | None:
    blocks = re.findall(r"^```python[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.DOTALL | re.MULTILINE)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    blocks = re.findall(r"^```[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.DOTALL | re.MULTILINE)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return None


def _build_proposer_prompt(
    current_code: str,
    train_report: EvalReport,
    history: list[GenerationRecord],
) -> str:
    diag_blocks = []
    for c in train_report.cases:
        diag_blocks.append(
            f"- Case `{c.case_id}`: score={c.score:.4f}, pass_rate={c.passed}/{c.total} ({c.pass_rate:.2%}), "
            f"tokens={c.compressed_tokens}/{c.max_tokens} (raw={c.raw_tokens}, compression={c.compression_ratio:.1%}, budget_ok={c.budget_ok})\n"
            f"  Failed tests: {c.failed_tests}\n"
            f"  Failure snippet:\n{c.failure_summary[:700]}"
        )
    history_lines = [
        f"  Gen {h.generation} [{h.status}]: train_score={h.train_score:.4f}, pass_rate={h.train_pass_rate:.2%}, "
        f"compression={h.train_compression_ratio:.1%} | {h.notes}"
        for h in history
    ]

    return f"""You are the Meta-Agent Proposer in OpenRSI (Harness-RSI).
Your goal is to improve `harness/compactor.py` so that long multi-turn coding agent trajectories
(conversations + noisy tool logs) are compacted to <= `max_tokens` WITHOUT losing critical constraints,
requirement overrides, negative-memory pitfalls, buried log warnings, or exact API/error-code contracts.

## Rules & Constraints
1. Keep the exact public function signature:
   `compact(messages: list[dict[str, Any]], max_tokens: int, llm_call: Callable[[str], str] | None = None) -> str`
2. Use ONLY Python standard library + `tiktoken` (and optionally `llm_call(prompt)` if helpful, though fast deterministic + structured extraction or hybrid approaches are both welcome).
3. Your compaction logic MUST be general-purpose — NEVER hardcode task names, specific variable names, or test answers from the train cases (it will be audited on unseen held-out cases!).
4. Strictly respect `max_tokens` (verify token count with `tiktoken.get_encoding("cl100k_base")` and trim cleanly if needed).
5. Return the COMPLETE updated `harness/compactor.py` inside a single ```python ... ``` block.

## Evolution History So Far
{chr(10).join(history_lines) if history_lines else "  (Baseline only)"}

## Latest Train Diagnostics (mean_score={train_report.mean_score:.4f}, mean_pass_rate={train_report.mean_pass_rate:.2%})
{chr(10).join(diag_blocks)}

## Current Best `harness/compactor.py`
```python
{current_code}
```

Think carefully about WHY constraints and buried log details were lost, then output the improved `harness/compactor.py` inside a single ```python ... ``` code block.
"""


def _plot_evolution_curve(history: list[GenerationRecord], out_path: Path) -> None:
    gens = [h.generation for h in history]
    train_scores = [h.train_score for h in history]
    comp_ratios = [h.train_compression_ratio for h in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(gens, train_scores, marker="o", linewidth=2, label="Train Score (PassRate × Budget)")
    plt.plot(gens, comp_ratios, marker="s", linestyle="--", linewidth=1.5, label="Compression Ratio")

    ho_gens = [h.generation for h in history if h.held_out_score is not None]
    ho_scores = [h.held_out_score for h in history if h.held_out_score is not None]
    if ho_gens:
        plt.plot(ho_gens, ho_scores, marker="^", linewidth=2, label="Held-Out Score")

    plt.ylim(-0.05, 1.08)
    plt.xticks(gens)
    plt.xlabel("Generation")
    plt.ylabel("Score / Ratio")
    plt.title("OpenRSI Harness-RSI: Context Compaction Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _print_eval_table(console: Console, title: str, report: EvalReport) -> None:
    table = Table(title=title)
    table.add_column("Case", style="cyan")
    table.add_column("Split")
    table.add_column("Tokens (Comp/Max/Raw)", justify="right")
    table.add_column("Compress", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Time", justify="right")

    for c in report.cases:
        tok_str = f"{c.compressed_tokens}/{c.max_tokens}/{c.raw_tokens}"
        if not c.budget_ok:
            tok_str = f"[red]{tok_str}[/red]"
        table.add_row(
            c.case_id,
            c.split,
            tok_str,
            f"{c.compression_ratio:.1%}",
            f"{c.passed}/{c.total} ({c.pass_rate:.0%})",
            f"{c.score:.4f}",
            f"{c.elapsed_sec:.1f}s",
        )
    table.add_section()
    table.add_row(
        "MEAN",
        report.split,
        f"{report.mean_compressed_tokens:.0f} avg",
        f"{report.mean_compression_ratio:.1%}",
        f"{report.mean_pass_rate:.1%}",
        f"{report.mean_score:.4f}",
        "",
    )
    console.print(table)


def run_evolution(
    generations: int = 3,
    model: str | None = None,
    eval_held_out: bool = True,
    compactor_path: Path = DEFAULT_COMPACTOR_PATH,
    cases_dir: Path = DEFAULT_CASES_DIR,
) -> list[GenerationRecord]:
    """Run the Harness-RSI evolution loop on `harness/compactor.py`."""
    console = Console()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    history: list[GenerationRecord] = []
    best_code = compactor_path.read_text(encoding="utf-8")

    console.rule("[bold blue]Generation 0 — Baseline Evaluation")
    best_train = evaluate_suite("train", compactor_path=compactor_path, cases_dir=cases_dir, model=model)
    _print_eval_table(console, "Gen 0 (Baseline) — Train", best_train)

    ho_score = None
    ho_pass = None
    if eval_held_out:
        ho_report = evaluate_suite("held_out", compactor_path=compactor_path, cases_dir=cases_dir, model=model)
        _print_eval_table(console, "Gen 0 (Baseline) — Held-Out", ho_report)
        ho_score = ho_report.mean_score
        ho_pass = ho_report.mean_pass_rate

    (ARCHIVE_DIR / "gen_00_compactor.py").write_text(best_code, encoding="utf-8")
    history.append(
        GenerationRecord(
            generation=0,
            status="baseline",
            train_score=best_train.mean_score,
            train_pass_rate=best_train.mean_pass_rate,
            train_compression_ratio=best_train.mean_compression_ratio,
            held_out_score=ho_score,
            held_out_pass_rate=ho_pass,
            notes="Initial sliding-window baseline",
        )
    )

    for gen in range(1, generations + 1):
        console.rule(f"[bold magenta]Generation {gen} / {generations}")
        if best_train.mean_score >= 0.9999 and best_train.mean_compression_ratio >= 0.85:
            console.print("[green]Already reached near-perfect score with >=85% compression. Stopping early.[/green]")
            break

        console.print("[cyan]Proposing candidate patch for harness/compactor.py via codex exec...[/cyan]")
        proposer_prompt = _build_proposer_prompt(best_code, best_train, history)
        raw_reply = call_codex_text(proposer_prompt, model=model, timeout_sec=240)
        candidate_code = _extract_python_code(raw_reply)

        if not candidate_code:
            console.print("[red]Proposer did not return a valid Python code block. Skipping.[/red]")
            history.append(
                GenerationRecord(
                    generation=gen,
                    status="invalid",
                    train_score=best_train.mean_score,
                    train_pass_rate=best_train.mean_pass_rate,
                    train_compression_ratio=best_train.mean_compression_ratio,
                    notes="No python code block in proposer output",
                )
            )
            continue

        # Write candidate & syntax/smoke check before full evaluation
        compactor_path.write_text(candidate_code, encoding="utf-8")
        try:
            py_compile.compile(str(compactor_path), doraise=True)
            mod = load_compactor_module(compactor_path)
            smoke = mod.compact([{"turn": 1, "role": "user", "content": "hello world"}], max_tokens=200, llm_call=None)
            assert isinstance(smoke, str) and len(smoke) > 0
        except Exception as exc:
            console.print(f"[red]Candidate failed syntax/smoke check: {exc}. Rolling back.[/red]")
            compactor_path.write_text(best_code, encoding="utf-8")
            history.append(
                GenerationRecord(
                    generation=gen,
                    status="invalid",
                    train_score=best_train.mean_score,
                    train_pass_rate=best_train.mean_pass_rate,
                    train_compression_ratio=best_train.mean_compression_ratio,
                    notes=f"Syntax/smoke error: {exc}",
                )
            )
            continue

        cand_train = evaluate_suite("train", compactor_path=compactor_path, cases_dir=cases_dir, model=model)
        _print_eval_table(console, f"Gen {gen} Candidate — Train", cand_train)

        improved = (
            cand_train.mean_score > best_train.mean_score + 1e-4
            or (
                abs(cand_train.mean_score - best_train.mean_score) <= 1e-4
                and cand_train.mean_compression_ratio > best_train.mean_compression_ratio + 0.01
            )
        )

        if improved:
            console.print(
                f"[bold green]KEPT Gen {gen}![/bold green] "
                f"train_score: {best_train.mean_score:.4f} -> {cand_train.mean_score:.4f}"
            )
            best_code = candidate_code
            best_train = cand_train
            (ARCHIVE_DIR / f"gen_{gen:02d}_compactor.py").write_text(best_code, encoding="utf-8")

            gen_ho_score = None
            gen_ho_pass = None
            if eval_held_out:
                ho_rep = evaluate_suite("held_out", compactor_path=compactor_path, cases_dir=cases_dir, model=model)
                _print_eval_table(console, f"Gen {gen} (Promoted) — Held-Out", ho_rep)
                gen_ho_score = ho_rep.mean_score
                gen_ho_pass = ho_rep.mean_pass_rate

            history.append(
                GenerationRecord(
                    generation=gen,
                    status="kept",
                    train_score=cand_train.mean_score,
                    train_pass_rate=cand_train.mean_pass_rate,
                    train_compression_ratio=cand_train.mean_compression_ratio,
                    held_out_score=gen_ho_score,
                    held_out_pass_rate=gen_ho_pass,
                    notes="Improved train score; promoted",
                )
            )
        else:
            console.print(
                f"[yellow]REVERTED Gen {gen}[/yellow] "
                f"(candidate {cand_train.mean_score:.4f} <= best {best_train.mean_score:.4f})"
            )
            compactor_path.write_text(best_code, encoding="utf-8")
            history.append(
                GenerationRecord(
                    generation=gen,
                    status="reverted",
                    train_score=cand_train.mean_score,
                    train_pass_rate=cand_train.mean_pass_rate,
                    train_compression_ratio=cand_train.mean_compression_ratio,
                    notes="No improvement; rolled back",
                )
            )

    (REPORTS_DIR / "evolution_history.json").write_text(
        json.dumps([h.model_dump() for h in history], indent=2),
        encoding="utf-8",
    )
    curve_path = REPORTS_DIR / "evolution_curve.png"
    _plot_evolution_curve(history, curve_path)
    console.print(f"\n[bold green]Saved evolution history & curve to {curve_path}[/bold green]")
    return history
