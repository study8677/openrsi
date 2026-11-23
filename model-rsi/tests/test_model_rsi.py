"""Comprehensive tests for OpenRSI Model-RSI core engine."""

from __future__ import annotations

import json
from pathlib import Path

from model_rsi.curriculum import FrontierCurriculum
from model_rsi.data_builder import (
    build_dpo_pairs,
    build_star_sft_examples,
    compute_diversity_ratio,
)
from model_rsi.flywheel import import_harness_evolution_traces
from model_rsi.gating import GatePolicy, evaluate_promotion_gate
from model_rsi.protocol import (
    GenerationMetrics,
    Trajectory,
    TrajectoryStore,
    VerifiableTask,
    VerificationResult,
)
from model_rsi.reporting import (
    load_evolution_history,
    plot_evolution_curve,
    save_evolution_history,
)
from model_rsi.tasks.code_exec import builtin_code_tasks, verify_code_exec
from model_rsi.tasks.countdown import generate_countdown_suite, verify_countdown
from model_rsi.tasks.regression import get_regression_suite, verify_regression


def test_countdown_disjoint_splits_and_ast_verifier() -> None:
    tasks = generate_countdown_suite(
        seed=123,
        train_per_level={1: 10, 2: 10},
        val_per_level={1: 5, 2: 5},
        held_out_per_level={1: 5, 2: 5},
    )
    sigs_by_split: dict[str, set[tuple[tuple[int, ...], int]]] = {}
    for t in tasks:
        sig = (tuple(sorted(t.verifier_spec["numbers"])), int(t.verifier_spec["target"]))
        sigs_by_split.setdefault(t.split, set()).add(sig)

    assert sigs_by_split["train"].isdisjoint(sigs_by_split["val"])
    assert sigs_by_split["train"].isdisjoint(sigs_by_split["held_out"])
    assert sigs_by_split["val"].isdisjoint(sigs_by_split["held_out"])

    # Every generated reference solution must pass the deterministic AST verifier
    for t in tasks[:15]:
        ref = t.metadata["reference_solution"]
        resp = f"<think>Let us combine the given numbers carefully to hit the target.</think>\n<answer>{ref}</answer>"
        ver = verify_countdown(t, resp)
        assert ver.passed, f"Ref failed: {t.task_id} -> {ver.reason}"
        assert ver.reward == 1.0

    # Reward-hacking resistance: wrong multiset, unsafe code, division by zero
    sample = tasks[0]
    target = sample.verifier_spec["target"]
    hack1 = verify_countdown(sample, f"<think>cheat</think><answer>{target}</answer>")
    assert not hack1.passed
    hack2 = verify_countdown(sample, "<think>inject</think><answer>__import__('os').system('echo hi')</answer>")
    assert not hack2.passed
    hack3 = verify_countdown(sample, "<think>div0</think><answer>1 / (1 - 1)</answer>")
    assert not hack3.passed


def test_code_exec_sandbox_verifier() -> None:
    tasks = builtin_code_tasks()
    rle_task = tasks[0]
    good = (
        "<think>Iterate through chars and count consecutive runs.</think>\n"
        "```python\n"
        "def rle_encode(s: str) -> str:\n"
        "    if not s:\n"
        "        return ''\n"
        "    out, ch, cnt = [], s[0], 1\n"
        "    for c in s[1:]:\n"
        "        if c == ch:\n"
        "            cnt += 1\n"
        "        else:\n"
        "            out.append(f'{ch}{cnt}')\n"
        "            ch, cnt = c, 1\n"
        "    out.append(f'{ch}{cnt}')\n"
        "    return ''.join(out)\n"
        "```\n"
    )
    ver_good = verify_code_exec(rle_task, good)
    assert ver_good.passed
    assert ver_good.reward == 1.0

    bad = "```python\ndef rle_encode(s):\n    return s\n```"
    ver_bad = verify_code_exec(rle_task, bad)
    assert not ver_bad.passed

    timeout_code = "```python\ndef rle_encode(s):\n    while True: pass\n```"
    ver_to = verify_code_exec(rle_task, timeout_code, timeout_sec=0.4)
    assert not ver_to.passed
    assert "timed out" in ver_to.reason.lower()


def test_regression_verifier() -> None:
    reg = get_regression_suite()
    assert len(reg) == 10
    v_json = verify_regression(reg[0], '{"language": "Python", "year": 1991}')
    assert v_json.passed
    v_json_bad = verify_regression(reg[0], '{"language": "Rust", "year": 2010}')
    assert not v_json_bad.passed
    v_cap = verify_regression(reg[1], "The capital of Japan is Tokyo.")
    assert v_cap.passed


def test_curriculum_data_builder_and_store(tmp_path: Path) -> None:
    tasks = generate_countdown_suite(
        seed=7,
        train_per_level={1: 4, 2: 4},
        val_per_level={1: 2},
        held_out_per_level={1: 2},
    )
    train_tasks = [t for t in tasks if t.split == "train"]
    curr = FrontierCurriculum(train_tasks, seed=7)

    # Create mixed pass/fail trajectories for first task (frontier), all-pass for second (mastered)
    t0, t1 = train_tasks[0], train_tasks[1]
    trajs = [
        Trajectory(
            trajectory_id="tr1",
            task_id=t0.task_id,
            domain="countdown",
            split="train",
            difficulty=t0.difficulty,
            generation=0,
            model_id="base",
            prompt=t0.prompt,
            system_prompt=t0.system_prompt,
            response=f"<think>good reasoning here</think><answer>{t0.metadata['reference_solution']}</answer>",
            verification=VerificationResult(passed=True, reward=1.0, format_reward=1.0, correctness_reward=1.0, extracted_answer="ok"),
        ),
        Trajectory(
            trajectory_id="tr2",
            task_id=t0.task_id,
            domain="countdown",
            split="train",
            difficulty=t0.difficulty,
            generation=0,
            model_id="base",
            prompt=t0.prompt,
            system_prompt=t0.system_prompt,
            response="<think>wrong reasoning here</think><answer>1 + 1</answer>",
            verification=VerificationResult(passed=False, reward=0.15, format_reward=0.15, correctness_reward=0.0, extracted_answer="1 + 1"),
        ),
        Trajectory(
            trajectory_id="tr3",
            task_id=t1.task_id,
            domain="countdown",
            split="train",
            difficulty=t1.difficulty,
            generation=0,
            model_id="base",
            prompt=t1.prompt,
            system_prompt=t1.system_prompt,
            response=f"<think>good reasoning</think><answer>{t1.metadata['reference_solution']}</answer>",
            verification=VerificationResult(passed=True, reward=1.0, format_reward=1.0, correctness_reward=1.0, extracted_answer="ok2"),
        ),
    ]

    store = TrajectoryStore(tmp_path / "trajs.jsonl")
    assert store.append(trajs) == 3
    assert len(store.load_filtered(passed_only=True)) == 2

    curr.update(trajs)
    report = curr.mastery_report()
    assert report[1]["frontier"] == 1.0
    assert report[1]["mastered"] == 1.0

    batch = curr.select_batch(4, max_difficulty=2)
    assert len(batch) == 4
    assert t0.task_id in [b.task_id for b in batch]

    sft = build_star_sft_examples(trajs)
    assert len(sft) == 2

    dpo = build_dpo_pairs(trajs)
    assert len(dpo) == 1
    assert dpo[0]["task_id"] == t0.task_id
    assert compute_diversity_ratio(trajs) == 1.0


def test_promotion_gate_and_reporting(tmp_path: Path) -> None:
    base = GenerationMetrics(
        generation=0, paradigm="baseline", model_id="base", status="baseline",
        held_out_pass_at_1=0.25, val_pass_at_1=0.28, regression_score=0.90,
        domain_scores={"countdown_L1": 0.40, "countdown_L2": 0.15},
    )
    good = GenerationMetrics(
        generation=1, paradigm="star_sft", parent_generation=0, model_id="gen1",
        held_out_pass_at_1=0.45, val_pass_at_1=0.48, regression_score=0.90,
        domain_scores={"countdown_L1": 0.65, "countdown_L2": 0.30},
    )
    regressed = GenerationMetrics(
        generation=2, paradigm="grpo_rlvr", parent_generation=1, model_id="gen2",
        held_out_pass_at_1=0.60, val_pass_at_1=0.62, regression_score=0.60,
        domain_scores={"countdown_L1": 0.80, "countdown_L2": 0.45},
    )

    d1 = evaluate_promotion_gate(good, base, base, GatePolicy())
    assert d1.promoted
    good.status = "promoted"

    d2 = evaluate_promotion_gate(regressed, good, base, GatePolicy())
    assert not d2.promoted
    assert "floor" in d2.reason.lower()
    regressed.status = "rejected"

    hist_path = save_evolution_history([base, good, regressed], tmp_path / "hist.json")
    loaded = load_evolution_history(hist_path)
    assert len(loaded) == 3

    curve = plot_evolution_curve(loaded, tmp_path / "curve.png")
    assert curve.exists() and curve.stat().st_size > 1000


def test_flywheel_harness_import(tmp_path: Path) -> None:
    reports = tmp_path / "harness_reports"
    archive = reports / "archive"
    archive.mkdir(parents=True)
    (archive / "gen_03_compactor.py").write_text("def compact(messages, max_tokens, llm_call=None):\n    return 'ok'\n")
    (reports / "evolution_history.json").write_text(
        json.dumps([
            {"generation": 0, "status": "baseline", "train_score": 0.5, "notes": "base"},
            {"generation": 3, "status": "kept", "train_score": 0.92, "notes": "improved"},
        ])
    )
    store = TrajectoryStore(tmp_path / "store.jsonl")
    n = import_harness_evolution_traces(reports, store, min_score=0.80)
    assert n == 1
    loaded = store.load_all()
    assert loaded[0].source == "harness_rsi"


def test_math_rlvr_and_kodcode_verifiers() -> None:
    from model_rsi.tasks.math_rlvr import verify_math_rlvr

    math_task = VerifiableTask(
        task_id="m_test_1",
        domain="math_rlvr",
        split="held_out",
        difficulty=2,
        prompt="Compute 3/4.",
        verifier_spec={"answer": r"\frac{3}{4}"},
    )
    v1 = verify_math_rlvr(math_task, r"Step by step: $6/8 = 3/4$. Final: $\boxed{\frac{3}{4}}$")
    assert v1.passed and v1.reward == 1.0
    v2 = verify_math_rlvr(math_task, r"Therefore $\boxed{0.75}$.")
    assert v2.passed and v2.reward == 1.0
    v3 = verify_math_rlvr(math_task, r"Therefore $\boxed{\frac{2}{3}}$.")
    assert not v3.passed and v3.reward == 0.0

    # Test KodCode multi-file sandbox mode
    kc_task = VerifiableTask(
        task_id="kc_test_1",
        domain="code_exec",
        split="train",
        difficulty=2,
        prompt="Write add(a, b).",
        verifier_spec={
            "test_mode": "kodcode",
            "test_code": "from solution import add\n\ndef test_pos():\n    assert add(2, 3) == 5\n\ndef test_neg():\n    assert add(-1, 1) == 0\n",
        },
    )
    vc = verify_code_exec(kc_task, "```python\ndef add(a, b):\n    return a + b\n```")
    assert vc.passed
    vc_bad = verify_code_exec(kc_task, "```python\ndef add(a, b):\n    return a - b\n```")
    assert not vc_bad.passed

