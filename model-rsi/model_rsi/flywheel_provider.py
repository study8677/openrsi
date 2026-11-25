"""Phase 4 Reverse Bridge (Model → Harness): CompactBench HF Provider & 4-Way Ablation.

Implements official `compactbench.providers.Provider` backed directly by a local
or Colab HuggingFace causal LM (`Qwen/Qwen3.5-2B` or `OpenRSI-Qwen3.5-2B-Flywheel`),
enabling live end-to-end evaluation with `harness-rsi`'s evolved `OpenRSICompactor v2`
and 2x2 Dual-Loop Flywheel ablation benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
from compactbench.providers import CompletionRequest, CompletionResponse, Provider
from compactbench.runner import CaseCompleteEvent, RunArgs, RunEndEvent, iter_events, run_experiment

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode_ordinary(text))


class HFCompactBenchProvider(Provider):
    """Official CompactBench Provider backed by a local/Colab HuggingFace PyTorch model."""

    key: ClassVar[str] = "hf-openrsi"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str = "openrsi-qwen35-2b",
        default_max_new_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._default_max_new_tokens = default_max_new_tokens

    @property
    def endpoint_kind(self) -> str:
        return "custom"

    def _complete_sync(self, request: CompletionRequest) -> CompletionResponse:
        import torch

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system.strip()})
        user_parts: list[str] = []
        if request.cached_prefix:
            user_parts.append(request.cached_prefix.strip())
        user_parts.append(request.prompt.strip())
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        prompt_str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        max_new = min(
            int(request.max_tokens or self._default_max_new_tokens),
            self._default_max_new_tokens,
        )
        temp = float(request.temperature) if request.temperature is not None else 0.0
        do_sample = temp > 0.0

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        text = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        return CompletionResponse(
            text=text,
            prompt_tokens=_count_tokens(prompt_str),
            completion_tokens=_count_tokens(text),
            model=self._model_name,
            raw={"provider": "hf-openrsi"},
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await asyncio.to_thread(self._complete_sync, request)


def run_official_compactbench_eval(
    provider: Provider,
    *,
    method_spec: str,
    model_label: str,
    output_path: str | Path,
    suite: str = "starter",
    seed_group: str = "default",
    cases_per_family: int = 1,
    budget_tokens: int = 2048,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run official `compactbench.runner.run_experiment` with a local HF provider."""
    t0 = time.monotonic()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    args = RunArgs(
        method_spec=method_spec,
        suite=suite,
        provider_key=provider.key,
        model=model_label,
        output_path=out_p,
        reader_provider_key=provider.key,
        reader_model=model_label,
        cases_per_family=cases_per_family,
        seed_group=seed_group,
        budget_tokens=budget_tokens,
        concurrency=concurrency,
    )

    # Support both sync scripts and Jupyter/Colab running event loops
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result_file = pool.submit(
                lambda: asyncio.run(
                    run_experiment(args, provider=provider, reader_provider=provider)
                )
            ).result()
    except RuntimeError:
        result_file = asyncio.run(
            run_experiment(args, provider=provider, reader_provider=provider)
        )

    total_probes = 0
    full_credit = 0
    sum_score = 0.0
    by_family: dict[str, list[float]] = {}
    metrics_summary: dict[str, Any] = {}

    for ev in iter_events(result_file):
        if isinstance(ev, CaseCompleteEvent):
            cr = ev.case_result
            for cyc in getattr(cr, "cycles", []):
                for pr in getattr(cyc, "probe_results", []) or []:
                    sc = float(getattr(pr, "score", 0.0) or 0.0)
                    fam = str(getattr(pr, "family", "unknown") or "unknown")
                    total_probes += 1
                    sum_score += sc
                    if sc >= 0.999:
                        full_credit += 1
                    by_family.setdefault(fam, []).append(sc)
        elif isinstance(ev, RunEndEvent):
            m = getattr(ev, "metrics", None)
            if m is not None:
                metrics_summary = {
                    "official_score": float(getattr(m, "score", 0.0) or 0.0),
                    "drift_resistance": float(getattr(m, "drift_resistance", 0.0) or 0.0),
                    "decay": float(getattr(m, "decay", 0.0) or 0.0),
                    "stale_rate": float(getattr(m, "stale_rate", 0.0) or 0.0),
                    "trap_rate": float(getattr(m, "trap_rate", 0.0) or 0.0),
                    "hallucination_rate": float(getattr(m, "hallucination_rate", 0.0) or 0.0),
                    "mean_artifact_tokens": float(getattr(m, "mean_artifact_tokens", 0.0) or 0.0),
                    "budget_violations": int(getattr(m, "budget_violations", 0) or 0),
                }

    mean_probe_score = round(sum_score / max(1, total_probes), 4)
    fam_avg = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_family.items())}

    return {
        "model_label": model_label,
        "method_spec": method_spec,
        "suite": suite,
        "seed_group": seed_group,
        "total_probes": total_probes,
        "full_credit_probes": full_credit,
        "mean_probe_score": mean_probe_score,
        "official_score": round(metrics_summary.get("official_score", mean_probe_score), 4),
        "drift_resistance": round(metrics_summary.get("drift_resistance", 0.0), 4),
        "decay": round(metrics_summary.get("decay", 0.0), 4),
        "stale_rate": round(metrics_summary.get("stale_rate", 0.0), 4),
        "trap_rate": round(metrics_summary.get("trap_rate", 0.0), 4),
        "hallucination_rate": round(metrics_summary.get("hallucination_rate", 0.0), 4),
        "mean_artifact_tokens": round(metrics_summary.get("mean_artifact_tokens", 0.0), 1),
        "budget_violations": metrics_summary.get("budget_violations", 0),
        "by_family": fam_avg,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }


def plot_flywheel_ablation(
    ablation_rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "OpenRSI Phase 4 — Dual-Loop Flywheel 2×2 Ablation (CompactBench Official)",
) -> Path:
    """Plot the 4-way Dual-Loop Flywheel ablation bar chart + drift resistance comparison."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not ablation_rows:
        return p

    labels = [r["label"] for r in ablation_rows]
    scores = [r["official_score"] * 100.0 for r in ablation_rows]
    drifts = [r["drift_resistance"] * 100.0 for r in ablation_rows]
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#7c3aed"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)
    x = list(range(len(labels)))

    bars1 = ax1.bar(x, scores, color=colors[: len(x)], width=0.58, edgecolor="#1e293b", linewidth=0.8)
    for bar, val in zip(bars1, scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Official CompactBench Score (%)", fontsize=10)
    ax1.set_title("Overall Probe Accuracy (2×2 Ablation)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, drifts, color=colors[: len(x)], width=0.58, edgecolor="#1e293b", linewidth=0.8)
    for bar, val in zip(bars2, drifts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Drift Resistance (%)", fontsize=10)
    ax2.set_title("Multi-Cycle Drift Resistance", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
