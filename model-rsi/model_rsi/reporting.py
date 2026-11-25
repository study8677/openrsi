"""Reproducible Reporting & Visualization for Model-RSI Evolution Curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_rsi.protocol import GenerationMetrics


def save_evolution_history(
    history: list[GenerationMetrics],
    out_path: str | Path,
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [h.model_dump() for h in history]
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load_evolution_history(path: str | Path) -> list[GenerationMetrics]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [GenerationMetrics.model_validate(item) for item in raw]


def plot_evolution_curve(
    history: list[GenerationMetrics],
    out_path: str | Path,
    *,
    title: str = "OpenRSI Model-RSI — Recursive Weight Evolution",
) -> Path:
    """Plot Held-out Pass@1, Val Pass@1, Regression Score, and Difficulty Breakdown."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return p

    gens = [h.generation for h in history]
    labels = [f"G{h.generation}: {h.paradigm}" for h in history]
    held_out = [h.held_out_pass_at_1 * 100.0 for h in history]
    val_scores = [h.val_pass_at_1 * 100.0 for h in history]
    reg_scores = [h.regression_score * 100.0 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=150)

    # Panel 1: Main RSI Curve + Anti-Forgetting Regression Gate
    ax1.plot(gens, held_out, marker="o", linewidth=2.5, markersize=8, color="#2563eb", label="Held-out Pass@1 (%)")
    ax1.plot(gens, val_scores, marker="s", linewidth=2.0, linestyle="--", color="#10b981", label="Val Pass@1 (%)")
    ax1.plot(gens, reg_scores, marker="^", linewidth=2.0, linestyle=":", color="#f59e0b", label="General Regression (%)")

    for h, x, y in zip(history, gens, held_out):
        tag = "✓" if h.status == "promoted" else ("✗" if h.status == "rejected" else "base")
        color = "#16a34a" if h.status == "promoted" else ("#dc2626" if h.status == "rejected" else "#475569")
        ax1.annotate(
            f"{y:.1f}% ({tag})",
            (x, y),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color=color,
        )

    ax1.set_xticks(gens)
    ax1.set_xticklabels(labels, fontsize=8.0, rotation=22, ha="right")
    ax1.set_ylim(-5, 108)
    ax1.set_ylabel("Accuracy / Pass@1 (%)", fontsize=10)
    ax1.set_title("Held-out Generalization & Regression Gate", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.5)

    # Panel 2: Per-Domain / Per-Difficulty Held-out Breakdown
    all_keys: list[str] = []
    for h in history:
        for k in h.domain_scores:
            if k not in all_keys:
                all_keys.append(k)

    palette = ["#6366f1", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6"]
    if all_keys:
        for idx, key in enumerate(all_keys):
            vals = [h.domain_scores.get(key, 0.0) * 100.0 for h in history]
            ax2.plot(
                gens,
                vals,
                marker="o",
                linewidth=2.0,
                color=palette[idx % len(palette)],
                label=key,
            )
        ax2.set_xticks(gens)
        ax2.set_xticklabels(labels, fontsize=8.0, rotation=22, ha="right")
        ax2.set_ylim(-5, 108)
        ax2.set_ylabel("Held-out Pass@1 (%)", fontsize=10)
        ax2.set_title("Domain & Difficulty Frontier Breakdown", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=8.5)
    else:
        ax2.text(0.5, 0.5, "No domain breakdown recorded", ha="center", va="center")
        ax2.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
