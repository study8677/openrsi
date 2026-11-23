"""End-to-end Google Colab GPU runner for OpenRSI Model-RSI.

Runs multi-generation Recursive Self-Improvement on an open-weight model
(default: Qwen/Qwen2.5-1.5B-Instruct) across:
  - Gen 0: Baseline evaluation (Countdown L1/L2/L3 + MBPP Code Exec + Regression Gate)
  - Gen 1: M2 Verified SFT / Rejection Sampling (STaR) + Anchor Replay
  - Gen 2: Iterative DPO on same-task verified (pass, fail) pairs
  - Gen 3: M3 Online RLVR / GRPO with deterministic AST/sandbox verifiers
  - Gen 4: M4 Frontier-Curriculum scaled GRPO / STaR iteration

Every candidate checkpoint is gated by strict Held-out Pass@1 and
General-Capability Regression checks before promotion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from model_rsi.curriculum import FrontierCurriculum
from model_rsi.data_builder import (
    build_dpo_pairs,
    build_star_sft_examples,
    compute_diversity_ratio,
)
from model_rsi.gating import GatePolicy, evaluate_promotion_gate
from model_rsi.protocol import GenerationMetrics, TrajectoryStore, VerifiableTask
from model_rsi.reporting import plot_evolution_curve, save_evolution_history
from model_rsi.tasks.code_exec import load_mbpp_tasks
from model_rsi.tasks.countdown import generate_countdown_suite
from model_rsi.tasks.regression import get_regression_suite
from model_rsi.trainers.grpo_rlvr import train_grpo_rlvr
from model_rsi.trainers.iterative_dpo import train_iterative_dpo
from model_rsi.trainers.rollout import (
    load_model_and_tokenizer,
    release_gpu_memory,
    run_batched_rollouts,
    summarize_trajectories,
)
from model_rsi.trainers.star_sft import train_star_sft


def evaluate_checkpoint(
    model_path: str,
    *,
    generation: int,
    paradigm: str,
    parent_generation: int | None,
    val_tasks: list[VerifiableTask],
    held_out_tasks: list[VerifiableTask],
    regression_tasks: list[VerifiableTask],
    batch_size: int = 24,
    max_new_tokens: int = 384,
) -> tuple[GenerationMetrics, dict[str, Any]]:
    """Run deterministic greedy Pass@1 evaluation on Val, Held-out, and Regression splits."""
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_path, bf16=True)
    try:
        val_trajs = run_batched_rollouts(
            model, tokenizer, val_tasks,
            generation=generation, model_id=model_path,
            num_samples=1, temperature=0.0,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        ho_trajs = run_batched_rollouts(
            model, tokenizer, held_out_tasks,
            generation=generation, model_id=model_path,
            num_samples=1, temperature=0.0,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        reg_trajs = run_batched_rollouts(
            model, tokenizer, regression_tasks,
            generation=generation, model_id=model_path,
            num_samples=1, temperature=0.0,
            max_new_tokens=160, batch_size=batch_size,
        )
    finally:
        release_gpu_memory(model)

    val_sum = summarize_trajectories(val_trajs)
    ho_sum = summarize_trajectories(ho_trajs)
    reg_sum = summarize_trajectories(reg_trajs)

    metrics = GenerationMetrics(
        generation=generation,
        paradigm=paradigm,
        parent_generation=parent_generation,
        model_id=model_path,
        checkpoint_path=model_path,
        val_pass_at_1=val_sum["pass_at_1"],
        held_out_pass_at_1=ho_sum["pass_at_1"],
        held_out_format_rate=ho_sum["format_rate"],
        domain_scores=ho_sum["breakdown"],
        regression_score=reg_sum["pass_at_1"],
        avg_response_tokens=round(ho_sum["avg_chars"] / 4.0, 1),
        wall_time_sec=round(time.time() - t0, 1),
    )
    diagnostics = {
        "val": val_sum,
        "held_out": ho_sum,
        "regression": reg_sum,
        "failed_regression": [
            {"task_id": t.task_id, "reason": t.verification.reason}
            for t in reg_trajs
            if not t.verification.passed
        ],
    }
    return metrics, diagnostics


def run_full_rsi_experiment(
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    work_dir: str = "/content/openrsi_model_rsi",
    seed: int = 42,
    include_mbpp: bool = True,
) -> list[GenerationMetrics]:
    """Run the complete multi-generation Model-RSI loop on Colab GPU."""
    root = Path(work_dir)
    ckpt_root = root / "checkpoints"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    store = TrajectoryStore(root / "trajectories.jsonl")

    # 1. Build strictly disjoint task suites
    countdown_tasks = generate_countdown_suite(
        seed=seed,
        train_per_level={1: 48, 2: 64, 3: 32},
        val_per_level={1: 12, 2: 16, 3: 8},
        held_out_per_level={1: 16, 2: 24, 3: 16},
    )
    code_tasks = (
        load_mbpp_tasks(train_limit=48, val_limit=12, held_out_limit=24)
        if include_mbpp
        else []
    )
    all_tasks = countdown_tasks + code_tasks
    train_tasks = [t for t in all_tasks if t.split == "train"]
    val_tasks = [t for t in all_tasks if t.split == "val"]
    held_out_tasks = [t for t in all_tasks if t.split == "held_out"]
    reg_tasks = get_regression_suite()

    curriculum = FrontierCurriculum(train_tasks, seed=seed)
    policy = GatePolicy()
    history: list[GenerationMetrics] = []

    # --- Gen 0: Baseline ---
    print(f"[Gen 0] Evaluating baseline: {base_model}")
    baseline, _ = evaluate_checkpoint(
        base_model,
        generation=0,
        paradigm="baseline",
        parent_generation=None,
        val_tasks=val_tasks,
        held_out_tasks=held_out_tasks,
        regression_tasks=reg_tasks,
    )
    baseline.status = "baseline"
    baseline.notes = f"Base open-weight model ({base_model})"
    history.append(baseline)
    current_best = baseline
    save_evolution_history(history, reports_dir / "evolution_history.json")
    print(
        f"  -> Held-out={baseline.held_out_pass_at_1:.2%}, "
        f"Val={baseline.val_pass_at_1:.2%}, Reg={baseline.regression_score:.2%}, "
        f"Breakdown={baseline.domain_scores}"
    )

    # Helper to sample rollouts from current_best
    def _sample_frontier(gen_idx: int, batch_n: int, k: int, max_diff: int) -> list[Any]:
        batch = curriculum.select_batch(batch_n, max_difficulty=max_diff)
        model, tok = load_model_and_tokenizer(current_best.checkpoint_path, bf16=True)
        try:
            trajs = run_batched_rollouts(
                model, tok, batch,
                generation=gen_idx, model_id=current_best.checkpoint_path,
                num_samples=k, temperature=0.8, max_new_tokens=384, batch_size=24,
            )
        finally:
            release_gpu_memory(model)
        store.append(trajs)
        curriculum.update(trajs)
        return trajs

    # --- Gen 1: STaR / Verified SFT ---
    print("[Gen 1] Sampling rollouts + Verified SFT (STaR)...")
    t0 = time.time()
    g1_trajs = _sample_frontier(1, batch_n=120, k=6, max_diff=2)
    g1_sum = summarize_trajectories(g1_trajs)
    sft_examples = build_star_sft_examples(g1_trajs, max_per_task=2, seed=seed + 1)
    ckpt1 = ckpt_root / "gen1_star_sft"
    tr1 = train_star_sft(
        current_best.checkpoint_path, sft_examples, ckpt1,
        num_train_epochs=2.0, learning_rate=1.5e-4, seed=seed + 1,
    )
    release_gpu_memory()
    m1, _ = evaluate_checkpoint(
        str(ckpt1), generation=1, paradigm="star_sft",
        parent_generation=current_best.generation,
        val_tasks=val_tasks, held_out_tasks=held_out_tasks, regression_tasks=reg_tasks,
    )
    m1.train_pass_at_1 = g1_sum["pass_at_1"]
    m1.train_pass_at_k = g1_sum["pass_at_k"]
    m1.unique_solution_ratio = compute_diversity_ratio(g1_trajs)
    m1.num_train_examples = tr1["num_examples"]
    m1.train_loss = round(tr1["train_loss"], 4)
    m1.wall_time_sec = round(time.time() - t0, 1)
    d1 = evaluate_promotion_gate(m1, current_best, baseline, policy)
    m1.status = "promoted" if d1.promoted else "rejected"
    m1.notes = d1.reason
    history.append(m1)
    if d1.promoted:
        current_best = m1
    save_evolution_history(history, reports_dir / "evolution_history.json")
    print(f"  -> {d1.reason}")

    # --- Gen 2: Iterative DPO ---
    print("[Gen 2] Frontier sampling + Iterative DPO...")
    t0 = time.time()
    g2_trajs = _sample_frontier(2, batch_n=120, k=6, max_diff=3)
    g2_sum = summarize_trajectories(g2_trajs)
    dpo_pairs = build_dpo_pairs(store.load_all(), max_pairs_per_task=2, seed=seed + 2)
    ckpt2 = ckpt_root / "gen2_iterative_dpo"
    tr2 = train_iterative_dpo(
        current_best.checkpoint_path, dpo_pairs, ckpt2,
        num_train_epochs=1.0, learning_rate=4e-5, beta=0.1, seed=seed + 2,
    )
    release_gpu_memory()
    m2, _ = evaluate_checkpoint(
        str(ckpt2), generation=2, paradigm="iterative_dpo",
        parent_generation=current_best.generation,
        val_tasks=val_tasks, held_out_tasks=held_out_tasks, regression_tasks=reg_tasks,
    )
    m2.train_pass_at_1 = g2_sum["pass_at_1"]
    m2.train_pass_at_k = g2_sum["pass_at_k"]
    m2.unique_solution_ratio = compute_diversity_ratio(g2_trajs)
    m2.num_train_examples = tr2["num_examples"]
    m2.train_loss = round(tr2["train_loss"], 4)
    m2.wall_time_sec = round(time.time() - t0, 1)
    d2 = evaluate_promotion_gate(m2, current_best, baseline, policy)
    m2.status = "promoted" if d2.promoted else "rejected"
    m2.notes = d2.reason
    history.append(m2)
    if d2.promoted:
        current_best = m2
    save_evolution_history(history, reports_dir / "evolution_history.json")
    print(f"  -> {d2.reason}")

    # --- Gen 3: Online GRPO / RLVR ---
    print("[Gen 3] Online GRPO / RLVR on Frontier Curriculum...")
    t0 = time.time()
    grpo_tasks = curriculum.select_batch(96, max_difficulty=3)
    ckpt3 = ckpt_root / "gen3_grpo_rlvr"
    tr3 = train_grpo_rlvr(
        current_best.checkpoint_path, grpo_tasks, ckpt3,
        max_steps=45, num_generations=4, learning_rate=5e-5, beta=0.04, seed=seed + 3,
    )
    release_gpu_memory()
    m3, _ = evaluate_checkpoint(
        str(ckpt3), generation=3, paradigm="grpo_rlvr",
        parent_generation=current_best.generation,
        val_tasks=val_tasks, held_out_tasks=held_out_tasks, regression_tasks=reg_tasks,
    )
    m3.num_train_examples = tr3["num_examples"]
    m3.train_loss = round(tr3["train_loss"], 4)
    m3.wall_time_sec = round(time.time() - t0, 1)
    d3 = evaluate_promotion_gate(m3, current_best, baseline, policy)
    m3.status = "promoted" if d3.promoted else "rejected"
    m3.notes = d3.reason
    history.append(m3)
    if d3.promoted:
        current_best = m3
    save_evolution_history(history, reports_dir / "evolution_history.json")
    plot_evolution_curve(history, reports_dir / "evolution_curve.png")
    print(f"  -> {d3.reason}")

    return history


if __name__ == "__main__":
    hist = run_full_rsi_experiment()
    print(json.dumps([h.model_dump() for h in hist], indent=2))
