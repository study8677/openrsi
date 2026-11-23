"""Training Signal Builders: Verified SFT (STaR), Iterative DPO, and GRPO Datasets.

Enforces deduplication, max-per-task capping (anti-collapse), format filtering,
and optional anchor-replay mixing.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from model_rsi.protocol import Trajectory


def build_star_sft_examples(
    trajectories: list[Trajectory],
    *,
    max_per_task: int = 2,
    min_reward: float = 0.85,
    anchor_examples: list[dict[str, Any]] | None = None,
    anchor_ratio: float = 0.10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build verified SFT (STaR / Rejection Sampling) chat examples from passed trajectories.

    Caps samples per task to avoid overfitting easy tasks and preserve diversity.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    seen_hashes: set[str] = set()

    for tr in trajectories:
        if tr.split != "train":
            continue
        if not tr.verification.passed or tr.verification.reward < min_reward:
            continue
        h = tr.response_hash
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        by_task[tr.task_id].append(tr)

    examples: list[dict[str, Any]] = []
    for task_id, group in sorted(by_task.items()):
        # Prefer higher reward (clean <think> + exact answer) and concise reasoning
        group.sort(key=lambda t: (-t.verification.reward, len(t.response)))
        for tr in group[:max_per_task]:
            messages = []
            if tr.system_prompt:
                messages.append({"role": "system", "content": tr.system_prompt})
            messages.append({"role": "user", "content": tr.prompt})
            messages.append({"role": "assistant", "content": tr.response})
            examples.append(
                {
                    "task_id": task_id,
                    "domain": tr.domain,
                    "difficulty": tr.difficulty,
                    "reward": tr.verification.reward,
                    "source": tr.source,
                    "messages": messages,
                }
            )

    if anchor_examples and examples and anchor_ratio > 0:
        n_anchor = max(1, int(len(examples) * anchor_ratio))
        chosen_anchors = rng.sample(
            anchor_examples, k=min(n_anchor, len(anchor_examples))
        )
        examples.extend(chosen_anchors)

    rng.shuffle(examples)
    return examples


def build_dpo_pairs(
    trajectories: list[Trajectory],
    *,
    max_pairs_per_task: int = 2,
    min_reward_margin: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build Iterative DPO preference pairs (chosen=verified pass, rejected=verified fail).

    Pairs come strictly from the SAME task so the model learns exact reasoning
    corrections rather than superficial length or topic biases.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for tr in trajectories:
        if tr.split == "train":
            by_task[tr.task_id].append(tr)

    pairs: list[dict[str, Any]] = []
    for task_id, group in sorted(by_task.items()):
        passed = [t for t in group if t.verification.passed]
        failed = [t for t in group if not t.verification.passed]
        if not passed or not failed:
            continue

        # Sort best passed first; hardest/closest-format failed first (informative negatives)
        passed.sort(key=lambda t: (-t.verification.reward, len(t.response)))
        failed.sort(key=lambda t: (-t.verification.format_reward, len(t.response)))

        seen_pair: set[tuple[str, str]] = set()
        added = 0
        for pos in passed:
            for neg in failed:
                if added >= max_pairs_per_task:
                    break
                if pos.verification.reward - neg.verification.reward < min_reward_margin:
                    continue
                key = (pos.response_hash, neg.response_hash)
                if key in seen_pair:
                    continue
                seen_pair.add(key)

                prompt_msgs = []
                if pos.system_prompt:
                    prompt_msgs.append({"role": "system", "content": pos.system_prompt})
                prompt_msgs.append({"role": "user", "content": pos.prompt})

                pairs.append(
                    {
                        "task_id": task_id,
                        "domain": pos.domain,
                        "difficulty": pos.difficulty,
                        "prompt": prompt_msgs,
                        "chosen": [{"role": "assistant", "content": pos.response}],
                        "rejected": [{"role": "assistant", "content": neg.response}],
                    }
                )
                added += 1

    rng.shuffle(pairs)
    return pairs


def compute_diversity_ratio(trajectories: list[Trajectory]) -> float:
    """Measure unique response diversity among passed trajectories (anti-collapse signal)."""
    passed = [t for t in trajectories if t.verification.passed]
    if not passed:
        return 0.0
    unique_answers = {
        (t.task_id, (t.verification.extracted_answer or t.response.strip()))
        for t in passed
    }
    return round(len(unique_answers) / len(passed), 4)
