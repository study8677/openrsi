"""Verifiable Python Coding Task & Isolated Subprocess Sandbox Verifier.

Supports both a built-in deterministic algorithmic coding suite (with strict
train/val/held_out isolation) and loading open-source MBPP sanitized problems.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from model_rsi.protocol import SplitName, VerifiableTask, VerificationResult

CODE_SYSTEM_PROMPT = (
    "You are an expert Python engineer. "
    "Reason carefully inside <think>...</think> tags, then provide a complete, "
    "self-contained Python solution inside a single ```python ... ``` code block."
)

_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_python_code(response: str) -> str | None:
    blocks = _PY_BLOCK_RE.findall(response)
    if blocks:
        return blocks[-1].strip()
    blocks = _ANY_BLOCK_RE.findall(response)
    if blocks:
        return blocks[-1].strip()
    return None


def verify_code_exec(
    task: VerifiableTask,
    response: str,
    *,
    timeout_sec: float = 4.0,
) -> VerificationResult:
    """Execute candidate Python code against immutable unit tests in an isolated subprocess."""
    t0 = time.perf_counter()
    code = extract_python_code(response)
    has_think = bool(_THINK_RE.search(response))

    if not code:
        return VerificationResult(
            passed=False,
            reward=0.0,
            format_reward=0.0,
            correctness_reward=0.0,
            extracted_answer=None,
            reason="No ```python ... ``` code block found",
            execution_ms=(time.perf_counter() - t0) * 1000.0,
        )

    test_code: str = task.verifier_spec["test_code"]
    format_reward = 0.15 if has_think else 0.05

    with tempfile.TemporaryDirectory(prefix="model_rsi_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "solution_and_tests.py"
        full_script = (
            "import sys\n"
            "sys.setrecursionlimit(1000)\n\n"
            f"{code}\n\n"
            "# --- Immutable Verifier Tests ---\n"
            f"{test_code}\n"
            "print('__OPENRSI_ALL_TESTS_PASSED__')\n"
        )
        script_path.write_text(full_script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                reward=format_reward * 0.2,
                format_reward=format_reward,
                correctness_reward=0.0,
                extracted_answer=code,
                reason=f"Sandbox execution timed out after {timeout_sec}s",
                execution_ms=(time.perf_counter() - t0) * 1000.0,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        if proc.returncode == 0 and "__OPENRSI_ALL_TESTS_PASSED__" in proc.stdout:
            return VerificationResult(
                passed=True,
                reward=1.0 if has_think else 0.85,
                format_reward=1.0 if has_think else 0.5,
                correctness_reward=1.0,
                extracted_answer=code,
                reason="All sandbox unit tests passed",
                execution_ms=elapsed,
            )

        err_tail = (proc.stderr or proc.stdout or "Unknown error").strip()[-400:]
        return VerificationResult(
            passed=False,
            reward=format_reward,
            format_reward=format_reward,
            correctness_reward=0.0,
            extracted_answer=code,
            reason=f"Unit test failure (rc={proc.returncode}): {err_tail}",
            execution_ms=elapsed,
        )


def load_mbpp_tasks(
    *,
    train_limit: int = 80,
    val_limit: int = 20,
    held_out_limit: int = 40,
) -> list[VerifiableTask]:
    """Load open-source MBPP sanitized tasks via HuggingFace `datasets` with strict split isolation."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    tasks: list[VerifiableTask] = []

    def _convert(split_rows: Any, split_name: SplitName, limit: int) -> None:
        for i, row in enumerate(split_rows):
            if i >= limit:
                break
            prompt_text = row["prompt"].strip()
            tests = row.get("test_list", [])
            test_imports = row.get("test_imports", [])
            test_code = "\n".join(list(test_imports) + list(tests))
            # Include one public signature hint from first assert so function name is unambiguous
            sig_hint = f"\nExample test: `{tests[0]}`" if tests else ""
            tasks.append(
                VerifiableTask(
                    task_id=f"mbpp_{split_name}_{row['task_id']}",
                    domain="code_exec",
                    split=split_name,
                    difficulty=2,
                    system_prompt=CODE_SYSTEM_PROMPT,
                    prompt=f"Write a Python function to solve the following task:\n{prompt_text}{sig_hint}",
                    verifier_spec={"test_code": test_code},
                    metadata={"source": "mbpp_sanitized", "mbpp_task_id": row["task_id"]},
                )
            )

    _convert(ds["train"], "train", train_limit)
    _convert(ds["validation"], "val", val_limit)
    _convert(ds["test"], "held_out", held_out_limit)
    return tasks


def builtin_code_tasks() -> list[VerifiableTask]:
    """Lightweight offline verifiable Python coding tasks for unit tests and smoke runs."""
    specs = [
        (
            "code_train_001",
            "train",
            1,
            "Write a function `rle_encode(s: str) -> str` that performs run-length encoding (e.g. 'aaabbc' -> 'a3b2c1', '' -> '').",
            "assert rle_encode('') == ''\nassert rle_encode('a') == 'a1'\nassert rle_encode('aaabbc') == 'a3b2c1'\nassert rle_encode('abba') == 'a1b2a1'",
        ),
        (
            "code_train_002",
            "train",
            2,
            "Write a function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping closed intervals and returns them sorted.",
            "assert merge_intervals([]) == []\nassert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([[5,7],[1,2],[2,6]]) == [[1,7]]",
        ),
        (
            "code_heldout_001",
            "held_out",
            2,
            "Write a function `is_balanced(s: str) -> bool` that checks whether brackets `()[]{}` in `s` are properly balanced (ignoring non-bracket characters).",
            "assert is_balanced('([{}])') is True\nassert is_balanced('a(b[c]d)e') is True\nassert is_balanced('([)]') is False\nassert is_balanced('(((') is False\nassert is_balanced('') is True",
        ),
    ]
    return [
        VerifiableTask(
            task_id=tid,
            domain="code_exec",
            split=split,  # type: ignore[arg-type]
            difficulty=diff,
            system_prompt=CODE_SYSTEM_PROMPT,
            prompt=prompt,
            verifier_spec={"test_code": tests},
        )
        for tid, split, diff, prompt, tests in specs
    ]
