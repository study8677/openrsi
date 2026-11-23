"""Promotion Gate & Lineage Archive for Model-RSI.

Enforces OpenRSI design principles:
1. Only trust deterministic held-out verification, never training loss alone.
2. Guard against catastrophic forgetting via general-capability regression floor.
3. Maintain full lineage archive so any regressed checkpoint is cleanly rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_rsi.protocol import GenerationMetrics


@dataclass
class GatePolicy:
    """Strict criteria for promoting Checkpoint N -> Checkpoint N+1."""

    min_held_out_gain: float = 0.01  # At least +1% on held-out OR equal held-out with val gain
    allow_held_out_tie_with_val_gain: bool = True
    min_val_gain_on_tie: float = 0.02
    max_regression_drop: float = 0.10  # Max allowed drop vs baseline regression score
    min_regression_floor: float = 0.70  # Hard floor on general capability regression


@dataclass
class GateDecision:
    promoted: bool
    reason: str
    held_out_delta: float
    val_delta: float
    regression_delta: float


def evaluate_promotion_gate(
    candidate: GenerationMetrics,
    current_best: GenerationMetrics,
    baseline: GenerationMetrics,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Decide whether a candidate checkpoint should be promoted to the new base model."""
    pol = policy or GatePolicy()

    held_out_delta = round(candidate.held_out_pass_at_1 - current_best.held_out_pass_at_1, 4)
    val_delta = round(candidate.val_pass_at_1 - current_best.val_pass_at_1, 4)
    regression_delta = round(candidate.regression_score - baseline.regression_score, 4)

    # 1. Anti-catastrophic-forgetting check
    if candidate.regression_score < pol.min_regression_floor:
        return GateDecision(
            promoted=False,
            reason=(
                f"REJECTED: Regression score {candidate.regression_score:.2%} fell below "
                f"hard floor {pol.min_regression_floor:.2%}"
            ),
            held_out_delta=held_out_delta,
            val_delta=val_delta,
            regression_delta=regression_delta,
        )

    if regression_delta < -pol.max_regression_drop - 1e-9:
        return GateDecision(
            promoted=False,
            reason=(
                f"REJECTED: Regression dropped by {regression_delta:+.2%} vs baseline "
                f"(max allowed: -{pol.max_regression_drop:.2%})"
            ),
            held_out_delta=held_out_delta,
            val_delta=val_delta,
            regression_delta=regression_delta,
        )

    # 2. Held-out generalization check
    if held_out_delta >= pol.min_held_out_gain - 1e-9:
        return GateDecision(
            promoted=True,
            reason=(
                f"PROMOTED: Held-out Pass@1 improved {held_out_delta:+.2%} "
                f"({current_best.held_out_pass_at_1:.2%} -> {candidate.held_out_pass_at_1:.2%}), "
                f"regression={candidate.regression_score:.2%}"
            ),
            held_out_delta=held_out_delta,
            val_delta=val_delta,
            regression_delta=regression_delta,
        )

    if (
        pol.allow_held_out_tie_with_val_gain
        and held_out_delta >= -1e-9
        and val_delta >= pol.min_val_gain_on_tie - 1e-9
    ):
        return GateDecision(
            promoted=True,
            reason=(
                f"PROMOTED: Held-out held steady ({candidate.held_out_pass_at_1:.2%}) "
                f"while Val Pass@1 improved {val_delta:+.2%}, "
                f"regression={candidate.regression_score:.2%}"
            ),
            held_out_delta=held_out_delta,
            val_delta=val_delta,
            regression_delta=regression_delta,
        )

    return GateDecision(
        promoted=False,
        reason=(
            f"REJECTED: Insufficient held-out improvement ({held_out_delta:+.2%}; "
            f"val_delta={val_delta:+.2%})"
        ),
        held_out_delta=held_out_delta,
        val_delta=val_delta,
        regression_delta=regression_delta,
    )
