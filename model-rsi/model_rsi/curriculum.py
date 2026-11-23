"""M4: Frontier Curriculum & Anti-Collapse Replay Manager.

Focuses sampling and post-training compute on the model's capability frontier
(tasks where 0 < pass_rate < 1, i.e. solvable with search but not yet mastered),
while mixing anchor/replay samples to prevent model collapse and forgetting.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from model_rsi.protocol import Trajectory, VerifiableTask


@dataclass
class TaskMastery:
    task_id: str
    difficulty: int
    attempts: int
    passes: int

    @property
    def pass_rate(self) -> float:
        return self.passes / self.attempts if self.attempts > 0 else 0.0

    @property
    def is_frontier(self) -> bool:
        return self.attempts > 0 and 0.0 < self.pass_rate < 1.0


class FrontierCurriculum:
    """Tracks per-task mastery across generations and schedules frontier batches."""

    def __init__(self, tasks: list[VerifiableTask], *, seed: int = 42) -> None:
        self.tasks = {t.task_id: t for t in tasks if t.split == "train"}
        self.rng = random.Random(seed)
        self.attempts: dict[str, int] = defaultdict(int)
        self.passes: dict[str, int] = defaultdict(int)

    def update(self, trajectories: list[Trajectory]) -> None:
        for tr in trajectories:
            if tr.task_id in self.tasks:
                self.attempts[tr.task_id] += 1
                if tr.verification.passed:
                    self.passes[tr.task_id] += 1

    def mastery_report(self) -> dict[int, dict[str, float]]:
        by_diff: dict[int, list[TaskMastery]] = defaultdict(list)
        for tid, task in self.tasks.items():
            m = TaskMastery(
                task_id=tid,
                difficulty=task.difficulty,
                attempts=self.attempts[tid],
                passes=self.passes[tid],
            )
            by_diff[task.difficulty].append(m)

        report: dict[int, dict[str, float]] = {}
        for diff, items in sorted(by_diff.items()):
            attempted = [m for m in items if m.attempts > 0]
            frontier = [m for m in items if m.is_frontier]
            mastered = [m for m in items if m.attempts > 0 and m.pass_rate >= 0.99]
            avg_pr = (
                sum(m.pass_rate for m in attempted) / len(attempted)
                if attempted
                else 0.0
            )
            report[diff] = {
                "total": float(len(items)),
                "attempted": float(len(attempted)),
                "frontier": float(len(frontier)),
                "mastered": float(len(mastered)),
                "mean_pass_rate": round(avg_pr, 4),
            }
        return report

    def select_batch(
        self,
        batch_size: int,
        *,
        max_difficulty: int = 3,
        frontier_weight: float = 0.55,
        unsolved_weight: float = 0.30,
        replay_weight: float = 0.15,
    ) -> list[VerifiableTask]:
        """Select a balanced curriculum batch prioritizing the capability frontier.

        - Frontier tasks (0 < pass_rate < 1): highest learning signal for STaR / DPO / GRPO
        - Unsolved / unseen tasks (pass_rate == 0 or attempts == 0): exploration
        - Mastered tasks (pass_rate == 1): replay anchor to prevent forgetting
        """
        eligible = [t for t in self.tasks.values() if t.difficulty <= max_difficulty]
        if not eligible:
            eligible = list(self.tasks.values())

        frontier: list[VerifiableTask] = []
        unsolved: list[VerifiableTask] = []
        mastered: list[VerifiableTask] = []

        for t in eligible:
            att = self.attempts[t.task_id]
            pas = self.passes[t.task_id]
            if att == 0 or pas == 0:
                unsolved.append(t)
            elif pas < att:
                frontier.append(t)
            else:
                mastered.append(t)

        # Sort frontier tasks by proximity to 0.5 pass rate (maximum variance / GRPO advantage)
        frontier.sort(
            key=lambda t: abs((self.passes[t.task_id] / self.attempts[t.task_id]) - 0.5)
        )
        self.rng.shuffle(unsolved)
        self.rng.shuffle(mastered)

        n_frontier = int(round(batch_size * frontier_weight))
        n_unsolved = int(round(batch_size * unsolved_weight))
        n_replay = max(0, batch_size - n_frontier - n_unsolved)

        selected: list[VerifiableTask] = []
        selected.extend(frontier[:n_frontier])
        selected.extend(unsolved[:n_unsolved])
        selected.extend(mastered[:n_replay])

        # Backfill if any bucket was smaller than target
        if len(selected) < batch_size:
            selected_ids = {t.task_id for t in selected}
            remaining = [t for t in eligible if t.task_id not in selected_ids]
            self.rng.shuffle(remaining)
            selected.extend(remaining[: batch_size - len(selected)])

        return selected[:batch_size]
