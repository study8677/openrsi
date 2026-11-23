"""Verifiable Countdown Arithmetic Reasoning Task & Deterministic AST Verifier.

Inspired by TinyZero / DeepSeek-R1-Zero verifiable reasoning benchmarks.
Supports parametric difficulty (3, 4, 5 numbers), strict disjoint train/val/held-out
splits, and an unhackable AST + exact-Fraction verifier.
"""

from __future__ import annotations

import ast
import operator
import random
import re
import time
from collections import Counter
from fractions import Fraction
from typing import Any

from model_rsi.protocol import SplitName, VerifiableTask, VerificationResult

COUNTDOWN_SYSTEM_PROMPT = (
    "You are a precise mathematical reasoning assistant. "
    "Solve the Countdown problem step by step inside <think>...</think> tags, "
    "then output ONLY the final arithmetic expression inside <answer>...</answer> tags. "
    "You must use EVERY given number EXACTLY once, using only +, -, *, / and parentheses."
)

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _eval_ast_exact(node: ast.AST, used_numbers: list[int]) -> Fraction:
    """Safely evaluate an arithmetic AST using exact Fractions, collecting literals."""
    if isinstance(node, ast.Expression):
        return _eval_ast_exact(node.body, used_numbers)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Only positive integer literals are allowed")
        if node.value < 0:
            raise ValueError("Negative literals are not allowed")
        used_numbers.append(int(node.value))
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        left = _eval_ast_exact(node.left, used_numbers)
        right = _eval_ast_exact(node.right, used_numbers)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op_fn(left, right)
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def _clean_expression(raw: str) -> str:
    expr = raw.strip()
    # Strip trailing "= target" if model wrote "3 + 4 = 7"
    if "=" in expr:
        expr = expr.split("=")[0].strip()
    # Normalize unicode math operators
    expr = (
        expr.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .strip("` \n\t.")
    )
    return expr


def verify_countdown(task: VerifiableTask, response: str) -> VerificationResult:
    """Deterministic AST verifier for Countdown tasks."""
    t0 = time.perf_counter()
    numbers: list[int] = list(task.verifier_spec["numbers"])
    target: int = int(task.verifier_spec["target"])

    think_match = _THINK_RE.search(response)
    answer_matches = _ANSWER_RE.findall(response)

    has_think = think_match is not None and len(think_match.group(1).strip()) >= 8
    has_answer = len(answer_matches) > 0

    if not has_answer:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed=False,
            reward=0.0,
            format_reward=0.0,
            correctness_reward=0.0,
            extracted_answer=None,
            reason="Missing <answer>...</answer> tag",
            execution_ms=elapsed,
        )

    raw_answer = answer_matches[-1].strip()
    expr = _clean_expression(raw_answer)
    format_reward = 0.15 if (has_think and has_answer and bool(expr)) else 0.05

    if not expr or len(expr) > 200:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed=False,
            reward=0.0,
            format_reward=0.0,
            correctness_reward=0.0,
            extracted_answer=expr,
            reason="Empty or overly long expression",
            execution_ms=elapsed,
        )

    try:
        tree = ast.parse(expr, mode="eval")
        used_numbers: list[int] = []
        val = _eval_ast_exact(tree, used_numbers)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed=False,
            reward=format_reward * 0.3,
            format_reward=format_reward,
            correctness_reward=0.0,
            extracted_answer=expr,
            reason=f"Invalid expression: {exc}",
            execution_ms=elapsed,
        )

    if Counter(used_numbers) != Counter(numbers):
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed=False,
            reward=format_reward * 0.5,
            format_reward=format_reward,
            correctness_reward=0.0,
            extracted_answer=expr,
            reason=f"Number multiset mismatch: used={sorted(used_numbers)}, required={sorted(numbers)}",
            execution_ms=elapsed,
            details={"used_numbers": used_numbers, "value": float(val)},
        )

    if val == Fraction(target, 1):
        elapsed = (time.perf_counter() - t0) * 1000.0
        # Strict pass requires exact match; full 1.0 reward also rewards clean <think> reasoning
        total_reward = 1.0 if has_think else 0.85
        return VerificationResult(
            passed=True,
            reward=total_reward,
            format_reward=1.0 if has_think else 0.5,
            correctness_reward=1.0,
            extracted_answer=expr,
            reason="Exact match verified",
            execution_ms=elapsed,
            details={"used_numbers": used_numbers, "value": target},
        )

    # Valid numbers used, wrong target: tiny closeness bonus for RLVR gradient signal
    diff = abs(float(val) - target)
    closeness = 0.05 if diff <= 3 else 0.0
    elapsed = (time.perf_counter() - t0) * 1000.0
    return VerificationResult(
        passed=False,
        reward=round(format_reward + closeness, 4),
        format_reward=format_reward,
        correctness_reward=0.0,
        extracted_answer=expr,
        reason=f"Evaluated to {val}, expected {target}",
        execution_ms=elapsed,
        details={"used_numbers": used_numbers, "value": float(val)},
    )


def _sample_solvable_instance(
    rng: random.Random,
    num_count: int,
    max_num: int,
    target_min: int,
    target_max: int,
) -> tuple[list[int], int, str]:
    """Construct a guaranteed-solvable Countdown puzzle by random expression tree."""
    ops = ["+", "-", "*", "/"]
    for _ in range(500):
        nums = [rng.randint(1, max_num) for _ in range(num_count)]
        items: list[tuple[Fraction, str]] = [(Fraction(n, 1), str(n)) for n in nums]
        rng.shuffle(items)
        ok = True
        while len(items) > 1:
            a_val, a_str = items.pop()
            b_val, b_str = items.pop()
            op = rng.choice(ops)
            if op == "+":
                val = a_val + b_val
                expr = f"({a_str} + {b_str})"
            elif op == "-":
                if a_val >= b_val:
                    val = a_val - b_val
                    expr = f"({a_str} - {b_str})"
                else:
                    val = b_val - a_val
                    expr = f"({b_str} - {a_str})"
            elif op == "*":
                val = a_val * b_val
                expr = f"({a_str} * {b_str})"
            else:
                if b_val != 0 and (a_val / b_val).denominator == 1:
                    val = a_val / b_val
                    expr = f"({a_str} / {b_str})"
                elif a_val != 0 and (b_val / a_val).denominator == 1:
                    val = b_val / a_val
                    expr = f"({b_str} / {a_str})"
                else:
                    ok = False
                    break
            items.append((val, expr))
            rng.shuffle(items)

        if not ok:
            continue
        final_val, final_expr = items[0]
        if final_val.denominator == 1 and target_min <= int(final_val) <= target_max:
            # Strip outer parens for neatness
            if final_expr.startswith("(") and final_expr.endswith(")"):
                final_expr = final_expr[1:-1]
            rng.shuffle(nums)
            return nums, int(final_val), final_expr

    raise RuntimeError("Failed to sample solvable Countdown instance")


def build_countdown_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an arithmetic expression that equals {target}. "
        f"You must use each number exactly once with only +, -, *, / and parentheses. "
        f"Show your reasoning in <think>...</think> and put ONLY the final expression in <answer>...</answer>."
    )


def generate_countdown_suite(
    *,
    seed: int = 42,
    train_per_level: dict[int, int] | None = None,
    val_per_level: dict[int, int] | None = None,
    held_out_per_level: dict[int, int] | None = None,
) -> list[VerifiableTask]:
    """Generate strictly disjoint train / val / held_out Countdown suites.

    Difficulty mapping:
      - Level 1: 3 numbers in [1, 15], target in [10, 60]
      - Level 2: 4 numbers in [1, 25], target in [15, 100]
      - Level 3: 5 numbers in [1, 25], target in [20, 150] (hard frontier / OOD)
    """
    level_cfg = {
        1: (3, 15, 10, 60),
        2: (4, 25, 15, 100),
        3: (5, 25, 20, 150),
    }
    train_counts = train_per_level or {1: 60, 2: 80, 3: 40}
    val_counts = val_per_level or {1: 15, 2: 20, 3: 10}
    held_out_counts = held_out_per_level or {1: 20, 2: 30, 3: 20}

    rng = random.Random(seed)
    seen_signatures: set[tuple[tuple[int, ...], int]] = set()
    tasks: list[VerifiableTask] = []

    split_specs: list[tuple[SplitName, dict[int, int]]] = [
        ("train", train_counts),
        ("val", val_counts),
        ("held_out", held_out_counts),
    ]

    for split, counts in split_specs:
        for level, count in sorted(counts.items()):
            num_count, max_num, t_min, t_max = level_cfg[level]
            created = 0
            while created < count:
                nums, target, ref_expr = _sample_solvable_instance(
                    rng, num_count, max_num, t_min, t_max
                )
                sig = (tuple(sorted(nums)), target)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                created += 1
                tasks.append(
                    VerifiableTask(
                        task_id=f"countdown_{split}_l{level}_{created:03d}",
                        domain="countdown",
                        split=split,
                        difficulty=level,
                        system_prompt=COUNTDOWN_SYSTEM_PROMPT,
                        prompt=build_countdown_prompt(nums, target),
                        verifier_spec={
                            "numbers": nums,
                            "target": target,
                            "num_count": num_count,
                        },
                        metadata={"reference_solution": ref_expr},
                    )
                )
    return tasks
