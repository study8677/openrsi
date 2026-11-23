"""M4: Harness-to-Model Flywheel Bridge.

Imports verified breakthrough trajectories produced by `harness-rsi` (or
test-time search / multi-turn scaffolds) into the unified Model-RSI
TrajectoryStore for distillation into model weights.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from model_rsi.protocol import Trajectory, TrajectoryStore, VerificationResult


def import_harness_evolution_traces(
    harness_reports_dir: str | Path,
    store: TrajectoryStore,
    *,
    min_score: float = 0.80,
    generation: int = 0,
) -> int:
    """Import kept, high-scoring Harness-RSI artifacts into the Model-RSI TrajectoryStore.

    Reads `evolution_history.json` + archived `gen_XX_*.py` files from `harness-rsi/reports/`
    without modifying anything inside `harness-rsi/`.
    """
    reports_path = Path(harness_reports_dir)
    history_file = reports_path / "evolution_history.json"
    archive_dir = reports_path / "archive"
    if not history_file.exists():
        return 0

    history = json.loads(history_file.read_text(encoding="utf-8"))
    imported: list[Trajectory] = []

    for entry in history:
        status = entry.get("status", "")
        score = float(entry.get("train_score", 0.0))
        gen = int(entry.get("generation", 0))
        if status not in ("kept", "promoted") or score < min_score:
            continue

        # Find archived artifact for this generation
        candidates = sorted(archive_dir.glob(f"gen_{gen:02d}_*.py")) if archive_dir.exists() else []
        if not candidates:
            continue

        code = candidates[0].read_text(encoding="utf-8")
        notes = entry.get("notes", "")
        prompt = (
            "Design a high-precision, token-budgeted Python context compactor module "
            "`compact(messages, max_tokens, llm_call=None)` that preserves critical "
            "constraints, requirement overrides, negative-memory pitfalls, and buried log warnings."
        )
        response = f"<think>Synthesizing verified Harness-RSI generation {gen} architecture (score={score:.4f}): {notes}</think>\n```python\n{code.strip()}\n```"

        imported.append(
            Trajectory(
                trajectory_id=f"harness_distill_g{gen}_{uuid.uuid4().hex[:8]}",
                task_id=f"harness_compactor_gen{gen:02d}",
                domain="harness_distill",
                split="train",
                difficulty=3,
                generation=generation,
                model_id="harness-rsi",
                source="harness_rsi",
                prompt=prompt,
                system_prompt="You are an expert AI systems engineer.",
                response=response,
                verification=VerificationResult(
                    passed=True,
                    reward=min(1.0, max(0.0, score)),
                    format_reward=1.0,
                    correctness_reward=min(1.0, max(0.0, score)),
                    extracted_answer=candidates[0].name,
                    reason=f"Imported verified Harness-RSI gen {gen} (score={score:.4f})",
                ),
            )
        )

    if imported:
        store.append(imported)
    return len(imported)
