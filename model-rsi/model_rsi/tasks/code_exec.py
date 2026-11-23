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
    test_mode: str = str(task.verifier_spec.get("test_mode", "inline"))
    entry_point: str = str(task.verifier_spec.get("entry_point", ""))
    format_reward = 0.15 if has_think else 0.05

    with tempfile.TemporaryDirectory(prefix="model_rsi_sandbox_") as tmpdir:
        tmp = Path(tmpdir)
        script_path = tmp / "run_verifier.py"

        if test_mode == "kodcode":
            # KodCode tests import from `solution` and define `def test_*():` functions
            (tmp / "solution.py").write_text(code + "\n", encoding="utf-8")
            (tmp / "test_suite.py").write_text(test_code + "\n", encoding="utf-8")
            full_script = (
                "import sys\n"
                "sys.path.insert(0, '.')\n"
                "sys.setrecursionlimit(1000)\n"
                "import test_suite\n"
                "ran = 0\n"
                "for name, obj in list(vars(test_suite).items()):\n"
                "    if name.startswith('test_') and callable(obj):\n"
                "        obj()\n"
                "        ran += 1\n"
                "assert ran > 0, 'No test_* functions found'\n"
                "print('__OPENRSI_ALL_TESTS_PASSED__')\n"
            )
        elif test_mode == "humaneval":
            # HumanEval+ defines `check(candidate)` and expects `check(<entry_point>)`
            prefix = str(task.verifier_spec.get("prompt_prefix", ""))
            # If candidate already defines the function, don't duplicate the header
            body = code if (f"def {entry_point}" in code) else (prefix + "\n" + code)
            full_script = (
                "import sys\n"
                "sys.setrecursionlimit(1000)\n\n"
                f"{body}\n\n"
                "# --- Immutable HumanEval+ Verifier ---\n"
                f"{test_code}\n"
                f"check({entry_point})\n"
                "print('__OPENRSI_ALL_TESTS_PASSED__')\n"
            )
        else:
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


def load_kodcode_tasks(
    *,
    train_limit: int = 80,
    val_limit: int = 20,
    min_pass_pct: float = 0.20,
    max_pass_pct: float = 0.85,
    seed: int = 42,
) -> list[VerifiableTask]:
    """Load frontier-difficulty verified Python tasks from `KodCode/KodCode-V1` (streaming)."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    tasks: list[VerifiableTask] = []
    total_needed = train_limit + val_limit

    for idx, row in enumerate(ds):
        if len(tasks) >= total_needed:
            break
        pct = float(row.get("gpt_pass_percentage") or 0.0)
        if not (min_pass_pct <= pct <= max_pass_pct):
            continue
        question = str(row.get("question", "")).strip()
        test_code = str(row.get("test", "")).strip()
        if not question or not test_code or len(question) > 1500:
            continue
        # Skip tests requiring external third-party packages or pytest fixtures
        if "import pytest" in test_code or "@pytest" in test_code or "numpy" in test_code or "pandas" in test_code:
            continue
        if "def test_" not in test_code or "from solution import" not in test_code:
            continue

        test_info = row.get("test_info") or []
        sig_hint = ""
        if isinstance(test_info, list) and test_info and isinstance(test_info[0], dict):
            decl = test_info[0].get("function_declaration", "")
            if decl:
                sig_hint = f"\n\nUse this exact function signature:\n```python\n{decl}\n```"

        split: SplitName = "train" if len(tasks) < train_limit else "val"
        diff = 1 if pct >= 0.65 else (2 if pct >= 0.40 else 3)
        tasks.append(
            VerifiableTask(
                task_id=f"kodcode_{split}_{row.get('question_id', idx)}",
                domain="code_exec",
                split=split,
                difficulty=diff,
                system_prompt=CODE_SYSTEM_PROMPT,
                prompt=f"Solve the following Python programming problem:{sig_hint}\n\n{question}",
                verifier_spec={"test_code": test_code, "test_mode": "kodcode"},
                metadata={
                    "source": "KodCode/KodCode-V1",
                    "subset": row.get("subset", ""),
                    "gpt_pass_percentage": pct,
                },
            )
        )
    return tasks


def load_humanevalplus_tasks(*, limit: int = 40, seed: int = 42) -> list[VerifiableTask]:
    """Load official `evalplus/humanevalplus` tasks as an external Held-out coding benchmark."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("evalplus/humanevalplus", split="test")
    ds = ds.shuffle(seed=seed)

    tasks: list[VerifiableTask] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        prompt_code = str(row["prompt"]).strip()
        entry_point = str(row["entry_point"]).strip()
        test_code = str(row["test"]).strip()
        tid = str(row["task_id"]).replace("/", "_")
        tasks.append(
            VerifiableTask(
                task_id=f"humanevalplus_ho_{tid}",
                domain="code_exec",
                split="held_out",
                difficulty=3,
                system_prompt=CODE_SYSTEM_PROMPT,
                prompt=(
                    "Complete the following Python function. Return the COMPLETE function "
                    "implementation (including imports and signature) inside a ```python ... ``` block:\n\n"
                    f"```python\n{prompt_code}\n```"
                ),
                verifier_spec={
                    "test_code": test_code,
                    "test_mode": "humaneval",
                    "entry_point": entry_point,
                    "prompt_prefix": prompt_code,
                },
                metadata={"source": "evalplus/humanevalplus", "entry_point": entry_point},
            )
        )
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
