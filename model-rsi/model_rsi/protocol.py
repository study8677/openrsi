"""M1: Unified Verifiable Trajectory & Verifier Protocol.

Standardizes tasks, rollouts, deterministic verification results, and dataset
serialization across direct model sampling and Harness-RSI distillation traces.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SplitName = Literal["train", "val", "held_out", "regression"]
TaskDomain = Literal["countdown", "code_exec", "regression", "harness_distill"]
TrajectorySource = Literal["model_rollout", "harness_rsi", "anchor_replay"]


class VerifiableTask(BaseModel):
    """A single verifiable task with strict split isolation."""

    task_id: str
    domain: TaskDomain
    split: SplitName
    difficulty: int = Field(ge=1, le=5, default=1)
    prompt: str
    system_prompt: str = ""
    verifier_spec: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Deterministic, auditable verdict from an immutable sandbox verifier."""

    passed: bool
    reward: float = Field(ge=0.0, le=1.0)
    format_reward: float = Field(ge=0.0, le=1.0, default=0.0)
    correctness_reward: float = Field(ge=0.0, le=1.0, default=0.0)
    extracted_answer: str | None = None
    reason: str = ""
    execution_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """A single sampled solution trajectory paired with its verified reward."""

    trajectory_id: str
    task_id: str
    domain: TaskDomain
    split: SplitName
    difficulty: int = 1
    generation: int = 0
    model_id: str
    source: TrajectorySource = "model_rollout"
    prompt: str
    system_prompt: str = ""
    response: str
    verification: VerificationResult
    temperature: float = 0.7
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def response_hash(self) -> str:
        return hashlib.sha256(
            f"{self.task_id}::{self.response.strip()}".encode("utf-8")
        ).hexdigest()[:16]


class GenerationMetrics(BaseModel):
    """Comprehensive evaluation snapshot for a single Model-RSI generation."""

    generation: int
    paradigm: str  # "baseline" | "star_sft" | "iterative_dpo" | "grpo_rlvr" | "curriculum_grpo"
    parent_generation: int | None = None
    model_id: str
    checkpoint_path: str = ""
    status: Literal["baseline", "promoted", "rejected"] = "baseline"

    # Primary verifiable metrics
    train_pass_at_1: float = 0.0
    train_pass_at_k: float = 0.0
    val_pass_at_1: float = 0.0
    held_out_pass_at_1: float = 0.0
    held_out_format_rate: float = 0.0

    # Multi-domain / difficulty breakdown (e.g. countdown_l1, countdown_l2, code_exec)
    domain_scores: dict[str, float] = Field(default_factory=dict)

    # Anti-forgetting & anti-collapse guardrails
    regression_score: float = 1.0
    unique_solution_ratio: float = 1.0
    avg_response_tokens: float = 0.0

    # Training telemetry
    num_train_examples: int = 0
    train_loss: float | None = None
    wall_time_sec: float = 0.0
    notes: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TrajectoryStore:
    """Append-only JSONL store for verifiable trajectories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trajectories: list[Trajectory]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")
        return len(trajectories)

    def load_all(self) -> list[Trajectory]:
        if not self.path.exists():
            return []
        items: list[Trajectory] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(Trajectory.model_validate_json(line))
        return items

    def load_filtered(
        self,
        *,
        generation: int | None = None,
        split: SplitName | None = None,
        domain: TaskDomain | None = None,
        passed_only: bool = False,
    ) -> list[Trajectory]:
        out: list[Trajectory] = []
        for t in self.load_all():
            if generation is not None and t.generation != generation:
                continue
            if split is not None and t.split != split:
                continue
            if domain is not None and t.domain != domain:
                continue
            if passed_only and not t.verification.passed:
                continue
            out.append(t)
        return out


def save_tasks_jsonl(tasks: list[VerifiableTask], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")


def load_tasks_jsonl(path: str | Path) -> list[VerifiableTask]:
    p = Path(path)
    if not p.exists():
        return []
    tasks: list[VerifiableTask] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(VerifiableTask.model_validate_json(line))
    return tasks
