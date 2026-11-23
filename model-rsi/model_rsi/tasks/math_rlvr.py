"""Verifiable Math RLVR Tasks & Deterministic Symbolic/Numeric Verifier.

Supports mature open-source RLVR math datasets:
  - Train / Val: `open-r1/DAPO-Math-17k-Processed` (and `openai/gsm8k` optional)
  - Held-out:    `HuggingFaceH4/MATH-500` (official standard math benchmark)
"""

from __future__ import annotations

import re
import time
from fractions import Fraction
from typing import Any

from model_rsi.protocol import SplitName, VerifiableTask, VerificationResult

MATH_RLVR_SYSTEM_PROMPT = (
    "You are a rigorous mathematical problem solver. "
    "Solve the problem step by step concisely, and put ONLY your final answer "
    "inside \\boxed{...} on the last line."
)

_HASH_ANS_RE = re.compile(r"####\s*([^\n]+)")
_ANSWER_LINE_RE = re.compile(r"(?:Final\s+Answer|Answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} with proper nested brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\fbox{")
        if idx == -1:
            return None
        start = idx + len("\\fbox{")
    else:
        start = idx + len("\\boxed{")

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def normalize_math_answer(raw: str) -> str:
    """Canonicalize LaTeX/numeric math answers for deterministic equivalence checking."""
    s = raw.strip()
    # Strip outer $ signs and trailing punctuation
    s = s.strip("$ \t\n.")
    # Unwrap \text{...}, \mathrm{...}, \mathbf{...}
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^}]*)\}", r"\1", s)
    # Normalize fraction commands and sizing macros
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("−", "-").replace("–", "-")
    # Remove thousands commas inside numbers (e.g. 1,200 -> 1200)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    # Strip trailing % or degree symbols if comparing pure numbers
    return s


def _to_exact_fraction(norm: str) -> Fraction | None:
    """Parse plain integers, decimals, or \\frac{a}{b} into an exact Fraction."""
    m_frac = re.fullmatch(r"(-?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", norm)
    if m_frac:
        sign = -1 if m_frac.group(1) == "-" else 1
        num, den = int(m_frac.group(2)), int(m_frac.group(3))
        if den != 0:
            return Fraction(sign * num, den)
    m_slash = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", norm)
    if m_slash:
        num, den = int(m_slash.group(1)), int(m_slash.group(2))
        if den != 0:
            return Fraction(num, den)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", norm):
        try:
            return Fraction(norm)
        except Exception:
            return None
    return None


def answers_equivalent(pred_raw: str, gold_raw: str) -> bool:
    """Deterministic equivalence check (exact Fraction -> normalized string -> optional SymPy)."""
    p = normalize_math_answer(pred_raw)
    g = normalize_math_answer(gold_raw)
    if not p or not g:
        return False
    if p.lower() == g.lower():
        return True

    # Exact rational / numeric match
    pf = _to_exact_fraction(p)
    gf = _to_exact_fraction(g)
    if pf is not None and gf is not None:
        return pf == gf

    # Optional SymPy fallback for algebraic equivalence (e.g. 1/2*sqrt(2) vs sqrt(2)/2)
    try:
        import sympy  # type: ignore
        from sympy.parsing.latex import parse_latex  # type: ignore

        p_expr = parse_latex(p)
        g_expr = parse_latex(g)
        if sympy.simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass

    return False


def verify_math_rlvr(task: VerifiableTask, response: str) -> VerificationResult:
    """Deterministic verifier for open-source RLVR math tasks (DAPO-Math / MATH-500 / GSM8K)."""
    t0 = time.perf_counter()
    gold = str(task.verifier_spec["answer"])

    extracted = extract_boxed_answer(response)
    has_boxed = extracted is not None

    if extracted is None:
        m_hash = _HASH_ANS_RE.findall(response)
        if m_hash:
            extracted = m_hash[-1].strip()
        else:
            m_ans = _ANSWER_LINE_RE.findall(response)
            if m_ans:
                extracted = m_ans[-1].strip()

    if not extracted:
        return VerificationResult(
            passed=False,
            reward=0.0,
            format_reward=0.0,
            correctness_reward=0.0,
            extracted_answer=None,
            reason="No \\boxed{...} or final answer found",
            execution_ms=(time.perf_counter() - t0) * 1000.0,
        )

    matched = answers_equivalent(extracted, gold)
    elapsed = (time.perf_counter() - t0) * 1000.0
    if matched:
        return VerificationResult(
            passed=True,
            reward=1.0,
            format_reward=1.0 if has_boxed else 0.5,
            correctness_reward=1.0,
            extracted_answer=extracted,
            reason=f"Exact match ({normalize_math_answer(extracted)} == {normalize_math_answer(gold)})",
            execution_ms=elapsed,
        )

    return VerificationResult(
        passed=False,
        reward=0.0,
        format_reward=0.15 if has_boxed else 0.0,
        correctness_reward=0.0,
        extracted_answer=extracted,
        reason=f"Mismatch: pred={extracted!r}, gold={gold!r}",
        execution_ms=elapsed,
    )


def load_dapo_math_tasks(
    *,
    train_limit: int = 120,
    val_limit: int = 30,
    seed: int = 42,
) -> list[VerifiableTask]:
    """Load strictly disjoint train & val slices from `open-r1/DAPO-Math-17k-Processed`."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    ds = ds.shuffle(seed=seed)

    tasks: list[VerifiableTask] = []
    total_needed = train_limit + val_limit
    for idx, row in enumerate(ds):
        if len(tasks) >= total_needed:
            break
        prompt_text = str(row["prompt"]).strip()
        gold = str(row.get("solution") or row.get("reward_model", {}).get("ground_truth", "")).strip()
        if not prompt_text or not gold or len(prompt_text) > 1200:
            continue

        split: SplitName = "train" if len(tasks) < train_limit else "val"
        tasks.append(
            VerifiableTask(
                task_id=f"dapo_math_{split}_{idx:05d}",
                domain="math_rlvr",
                split=split,
                difficulty=3,
                system_prompt=MATH_RLVR_SYSTEM_PROMPT,
                prompt=f"{prompt_text}\n\nSolve step by step and put your final answer inside \\boxed{{}}.",
                verifier_spec={"answer": gold},
                metadata={"source": "open-r1/DAPO-Math-17k-Processed"},
            )
        )
    return tasks


def load_math500_held_out(
    *,
    limit: int = 60,
    max_level: int = 4,
    seed: int = 42,
) -> list[VerifiableTask]:
    """Load official `HuggingFaceH4/MATH-500` held-out benchmark tasks."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=seed)

    tasks: list[VerifiableTask] = []
    for idx, row in enumerate(ds):
        if len(tasks) >= limit:
            break
        level = int(row.get("level", 3))
        if level > max_level:
            continue
        problem = str(row["problem"]).strip()
        gold = str(row["answer"]).strip()
        if not problem or not gold:
            continue
        tasks.append(
            VerifiableTask(
                task_id=f"math500_ho_l{level}_{idx:03d}",
                domain="math_rlvr",
                split="held_out",
                difficulty=min(5, max(1, level)),
                system_prompt=MATH_RLVR_SYSTEM_PROMPT,
                prompt=f"{problem}\n\nSolve step by step and put your final answer inside \\boxed{{}}.",
                verifier_spec={"answer": gold},
                metadata={
                    "source": "HuggingFaceH4/MATH-500",
                    "subject": row.get("subject", ""),
                    "level": level,
                },
            )
        )
    return tasks
