"""Deterministic evaluator for `arena/context-compaction`."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evolver.codex_client import call_codex_text, count_tokens, run_codex_in_workspace

HARNESS_RSI_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES_DIR = HARNESS_RSI_ROOT / "arena" / "context-compaction" / "cases"
DEFAULT_COMPACTOR_PATH = HARNESS_RSI_ROOT / "harness" / "compactor.py"


class CaseResult(BaseModel):
    case_id: str
    split: str
    raw_tokens: int
    max_tokens: int
    compressed_tokens: int
    compression_ratio: float
    budget_ok: bool
    passed: int
    total: int
    pass_rate: float
    score: float
    elapsed_sec: float
    failed_tests: list[str]
    failure_summary: str
    compacted_preview: str


class EvalReport(BaseModel):
    split: str
    cases: list[CaseResult]
    mean_pass_rate: float
    mean_score: float
    mean_compression_ratio: float
    mean_compressed_tokens: float


def load_compactor_module(compactor_path: Path = DEFAULT_COMPACTOR_PATH) -> Any:
    """Dynamically load `harness/compactor.py` so mutations take effect immediately."""
    spec = importlib.util.spec_from_file_location("openrsi_compactor_dynamic", compactor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load compactor module from {compactor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "compact"):
        raise RuntimeError(f"{compactor_path} does not define `compact(messages, max_tokens, llm_call=None)`")
    return module


def _parse_pytest_output(stdout: str, stderr: str) -> tuple[int, int, list[str], str]:
    combined = f"{stdout}\n{stderr}"
    passed = 0
    failed = 0
    errors = 0
    m_pass = re.search(r"(\d+)\s+passed", combined)
    m_fail = re.search(r"(\d+)\s+failed", combined)
    m_err = re.search(r"(\d+)\s+error", combined)
    if m_pass:
        passed = int(m_pass.group(1))
    if m_fail:
        failed = int(m_fail.group(1))
    if m_err:
        errors = int(m_err.group(1))

    failed_tests = re.findall(r"FAILED\s+[^\s:]+::([^\s]+)", combined)
    total = passed + failed + errors
    if total == 0:
        # e.g. SyntaxError / ImportError during collection
        total = 8
        failed_tests = ["COLLECTION_OR_IMPORT_ERROR"]

    # Keep a concise failure summary (last 1800 chars)
    summary_lines = [
        line for line in combined.splitlines()
        if any(k in line for k in ("FAILED", "AssertionError", "Error:", "E       ", "assert "))
    ]
    failure_summary = "\n".join(summary_lines[:40]) if summary_lines else combined[-1200:]
    return passed, total, failed_tests, failure_summary


def evaluate_case(
    case_dir: Path,
    compactor_module: Any,
    model: str | None = None,
    allow_llm_in_compactor: bool = True,
) -> CaseResult:
    """Run one context-compaction benchmark case end-to-end."""
    t0 = time.monotonic()
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    messages = json.loads((case_dir / "trajectory.json").read_text(encoding="utf-8"))
    final_task = (case_dir / "final_task.md").read_text(encoding="utf-8").strip()
    hidden_test_path = case_dir / "test_hidden.py"

    llm_fn = (lambda p: call_codex_text(p, model=model)) if allow_llm_in_compactor else None
    compacted_context = compactor_module.compact(
        messages,
        max_tokens=meta["max_tokens"],
        llm_call=llm_fn,
    )
    if not isinstance(compacted_context, str):
        raise TypeError("compact() must return a str")

    compressed_tokens = count_tokens(compacted_context)
    raw_tokens = int(meta["raw_tokens"])
    max_tokens = int(meta["max_tokens"])
    compression_ratio = max(0.0, 1.0 - (compressed_tokens / max(1, raw_tokens)))
    budget_ok = compressed_tokens <= max_tokens
    budget_factor = 1.0 if budget_ok else (max_tokens / max(1, compressed_tokens))

    # Materialize clean temp workspace with starter files and run downstream codex exec
    with tempfile.TemporaryDirectory(prefix=f"openrsi_eval_{meta['id']}_") as tmpdir:
        workdir = Path(tmpdir)
        starter_dir = case_dir / "starter_files"
        for item in starter_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, workdir / item.name)

        executor_prompt = (
            "You are continuing a software engineering task after context compaction.\n"
            "Below is the ONLY conversation history available to you. Follow every active constraint, "
            "override, and edge case mentioned in it carefully.\n\n"
            "=== COMPACTED CONVERSATION CONTEXT ===\n"
            f"{compacted_context}\n"
            "=== END COMPACTED CONTEXT ===\n\n"
            "=== FINAL TASK ===\n"
            f"{final_task}\n\n"
            "Write clean, complete code directly into the target file in the current directory. "
            "Do not ask follow-up questions."
        )

        run_codex_in_workspace(
            prompt=executor_prompt,
            workdir=workdir,
            model=model,
            sandbox="workspace-write",
            timeout_sec=240,
        )

        # Run immutable hidden pytest suite against workdir (never copied into agent's view beforehand)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workdir)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(hidden_test_path), "-q", "--tb=short", "--no-header"],
            cwd=str(workdir),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        passed, total, failed_tests, failure_summary = _parse_pytest_output(proc.stdout, proc.stderr)

    pass_rate = passed / max(1, total)
    score = round(pass_rate * budget_factor, 4)
    elapsed = round(time.monotonic() - t0, 2)

    return CaseResult(
        case_id=meta["id"],
        split=meta["split"],
        raw_tokens=raw_tokens,
        max_tokens=max_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(compression_ratio, 4),
        budget_ok=budget_ok,
        passed=passed,
        total=total,
        pass_rate=round(pass_rate, 4),
        score=score,
        elapsed_sec=elapsed,
        failed_tests=failed_tests,
        failure_summary=failure_summary if passed < total else "",
        compacted_preview=compacted_context[:400],
    )


def evaluate_suite(
    split: str = "train",
    compactor_path: Path = DEFAULT_COMPACTOR_PATH,
    cases_dir: Path = DEFAULT_CASES_DIR,
    model: str | None = None,
    allow_llm_in_compactor: bool = True,
) -> EvalReport:
    """Evaluate compactor across all cases matching `split` ('train', 'held_out', or 'all')."""
    compactor_module = load_compactor_module(compactor_path)
    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir() and (d / "meta.json").exists())

    results: list[CaseResult] = []
    for cdir in case_dirs:
        meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        if split != "all" and meta["split"] != split:
            continue
        results.append(
            evaluate_case(
                cdir,
                compactor_module=compactor_module,
                model=model,
                allow_llm_in_compactor=allow_llm_in_compactor,
            )
        )

    if not results:
        raise RuntimeError(f"No benchmark cases found for split='{split}' in {cases_dir}")

    n = len(results)
    return EvalReport(
        split=split,
        cases=results,
        mean_pass_rate=round(sum(r.pass_rate for r in results) / n, 4),
        mean_score=round(sum(r.score for r in results) / n, 4),
        mean_compression_ratio=round(sum(r.compression_ratio for r in results) / n, 4),
        mean_compressed_tokens=round(sum(r.compressed_tokens for r in results) / n, 1),
    )
