"""General-Capability Regression Suite (Anti-Catastrophic-Forgetting Gate).

Evaluates whether recursive self-improvement on specialized reasoning/coding tasks
preserved general instruction following, structured output, basic logic, and QA.
"""

from __future__ import annotations

import json
import re
import time

from model_rsi.protocol import VerifiableTask, VerificationResult


def get_regression_suite() -> list[VerifiableTask]:
    """Return a locked general-capability regression benchmark."""
    raw_items = [
        (
            "reg_001_json",
            "Return ONLY a valid JSON object with keys 'language' set to 'Python' and 'year' set to 1991. No extra text.",
            {"type": "json_kv", "expected": {"language": "Python", "year": 1991}},
        ),
        (
            "reg_002_capital",
            "What is the capital city of Japan? Answer with the city name.",
            {"type": "contains_any", "keywords": ["tokyo"]},
        ),
        (
            "reg_003_logic",
            "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer Yes or No.",
            {"type": "contains_any", "keywords": ["yes"]},
        ),
        (
            "reg_004_list_count",
            "List exactly 3 primary colors of light (additive RGB), separated by commas.",
            {"type": "contains_all", "keywords": ["red", "green", "blue"]},
        ),
        (
            "reg_005_reverse",
            "Reverse the word 'algorithm' letter by letter.",
            {"type": "contains_any", "keywords": ["mhtirogla"]},
        ),
        (
            "reg_006_math_word",
            "A train travels 60 km/h for 2.5 hours. How many kilometers does it travel? Give the number.",
            {"type": "contains_any", "keywords": ["150"]},
        ),
        (
            "reg_007_format_upper",
            "Reply with the single word HELLO in all uppercase letters.",
            {"type": "exact_contains_case", "substring": "HELLO"},
        ),
        (
            "reg_008_cs_complexity",
            "What is the average time complexity of binary search on a sorted array of n elements?",
            {"type": "contains_any", "keywords": ["o(log n)", "o(logn)", "log(n)", "log n"]},
        ),
        (
            "reg_009_negation",
            "Which of these is NOT a prime number: 7, 11, 13, 15, 17? Answer with the number.",
            {"type": "contains_any", "keywords": ["15"]},
        ),
        (
            "reg_010_summarize_constraint",
            "Write a single sentence about the ocean that contains the word 'waves' and is under 25 words.",
            {"type": "word_and_max_len", "keyword": "waves", "max_words": 25},
        ),
    ]
    return [
        VerifiableTask(
            task_id=tid,
            domain="regression",
            split="regression",
            difficulty=1,
            system_prompt="You are a helpful, concise assistant. Follow instructions carefully.",
            prompt=prompt,
            verifier_spec=spec,
        )
        for tid, prompt, spec in raw_items
    ]


def verify_regression(task: VerifiableTask, response: str) -> VerificationResult:
    """Deterministic verifier for general-capability regression checks."""
    t0 = time.perf_counter()
    spec = task.verifier_spec
    check_type = spec["type"]
    text = response.strip()
    low = text.lower()
    passed = False
    reason = ""

    if check_type == "json_kv":
        # Strip optional markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        try:
            obj = json.loads(cleaned)
            passed = all(obj.get(k) == v for k, v in spec["expected"].items())
            reason = "Valid JSON match" if passed else f"JSON mismatch: {obj}"
        except Exception as exc:
            passed = False
            reason = f"Invalid JSON: {exc}"
    elif check_type == "contains_any":
        passed = any(k.lower() in low for k in spec["keywords"])
        reason = "Keyword matched" if passed else f"Missing any of {spec['keywords']}"
    elif check_type == "contains_all":
        passed = all(k.lower() in low for k in spec["keywords"])
        reason = "All keywords matched" if passed else f"Missing some of {spec['keywords']}"
    elif check_type == "exact_contains_case":
        passed = spec["substring"] in text
        reason = "Exact case matched" if passed else f"Missing {spec['substring']}"
    elif check_type == "word_and_max_len":
        words = text.split()
        passed = spec["keyword"].lower() in low and 3 <= len(words) <= spec["max_words"]
        reason = (
            f"Valid ({len(words)} words)"
            if passed
            else f"Constraint failed (words={len(words)}, has_kw={spec['keyword'].lower() in low})"
        )
    else:
        raise ValueError(f"Unknown regression check type: {check_type}")

    return VerificationResult(
        passed=passed,
        reward=1.0 if passed else 0.0,
        format_reward=1.0 if passed else 0.0,
        correctness_reward=1.0 if passed else 0.0,
        extracted_answer=text[:120],
        reason=reason,
        execution_ms=(time.perf_counter() - t0) * 1000.0,
    )
