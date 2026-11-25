"""Phase 5: Multi-Turn Agentic Self-Repair RLVR (`Write -> Test -> Traceback -> Repair`).

Executes a real 2-turn deterministic sandbox loop over Python coding tasks
(`KodCode-V1`, `MBPP`, `HumanEval+`), capturing structured execution tracebacks
when Turn 1 fails, prompting the model to diagnose and repair its own bug in
Turn 2, and verifying the repaired code in the exact same isolated `python -I`
sandbox.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from model_rsi.protocol import VerifiableTask, VerificationResult
from model_rsi.tasks.code_exec import verify_code_exec

AGENTIC_CODE_SYSTEM_PROMPT = (
    "You are an expert Python software engineer and debugging agent. "
    "Write clean, self-contained Python solutions inside a single ```python ... ``` code block. "
    "When given a test failure traceback, carefully diagnose the root cause and output the "
    "complete corrected Python implementation inside a ```python ... ``` code block."
)

REPAIR_TURN_TEMPLATE = (
    "Your previous Python solution failed automated unit-test verification.\n\n"
    "Execution feedback:\n"
    "```\n{feedback}\n```\n\n"
    "Identify the exact bug (edge case, off-by-one, wrong return type, logic error, or syntax error) "
    "and output the complete corrected Python implementation inside a single ```python ... ``` block."
)


@dataclass
class AgenticEpisode:
    """One complete 2-turn (Write -> Verify -> Repair -> Re-Verify) episode."""

    task_id: str
    split: str
    difficulty: int
    prompt: str
    system_prompt: str
    turn1_response: str
    turn1_verification: VerificationResult
    repair_prompt: str | None = None
    turn2_response: str | None = None
    turn2_verification: VerificationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn1_passed(self) -> bool:
        return bool(self.turn1_verification.passed)

    @property
    def final_passed(self) -> bool:
        if self.turn1_passed:
            return True
        return bool(self.turn2_verification and self.turn2_verification.passed)

    @property
    def repaired(self) -> bool:
        return (not self.turn1_passed) and self.final_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "difficulty": self.difficulty,
            "turn1_passed": self.turn1_passed,
            "final_passed": self.final_passed,
            "repaired": self.repaired,
            "turn1_reason": self.turn1_verification.reason,
            "turn2_reason": (
                self.turn2_verification.reason if self.turn2_verification else None
            ),
            "turn1_response": self.turn1_response,
            "repair_prompt": self.repair_prompt,
            "turn2_response": self.turn2_response,
            "metadata": self.metadata,
        }


def sanitize_traceback_feedback(ver: VerificationResult, *, max_chars: int = 650) -> str:
    """Format concise, deterministic sandbox error feedback without leaking temp paths."""
    raw = str(ver.details.get("stderr") or ver.reason or "Verification failed").strip()
    # Strip temporary sandbox paths so feedback is clean and portable
    cleaned = re.sub(r'File "[^"]*?([^/\\"]+\.py)"', r'File "\1"', raw)
    cleaned = re.sub(r"/tmp/[^\s\"']+", "<sandbox>", cleaned)
    cleaned = re.sub(r"/var/folders/[^\s\"']+", "<sandbox>", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[-max_chars:]
    return cleaned or "AssertionError: unit test failed."


def build_repair_prompt(ver: VerificationResult) -> str:
    """Build the deterministic Turn-2 debugging prompt from a Turn-1 sandbox failure."""
    return REPAIR_TURN_TEMPLATE.format(feedback=sanitize_traceback_feedback(ver))


def verify_agentic_repair(task: VerifiableTask, response: str) -> VerificationResult:
    """Deterministic verifier for an agentic_debug task (delegates to isolated code_exec)."""
    return verify_code_exec(task, response)


def summarize_agentic_episodes(episodes: list[AgenticEpisode]) -> dict[str, Any]:
    """Compute Turn-1 Pass@1, Turn-2 Final Pass@1, and Self-Repair Conversion Rate."""
    n = len(episodes)
    if n == 0:
        return {
            "total": 0,
            "turn1_pass_at_1": 0.0,
            "turn2_pass_at_1": 0.0,
            "repair_gain": 0.0,
            "turn1_failures": 0,
            "repaired_count": 0,
            "repair_conversion_rate": 0.0,
            "by_source": {},
        }

    t1_pass = sum(1 for e in episodes if e.turn1_passed)
    final_pass = sum(1 for e in episodes if e.final_passed)
    t1_fail = n - t1_pass
    repaired = sum(1 for e in episodes if e.repaired)

    by_src: dict[str, list[AgenticEpisode]] = {}
    for e in episodes:
        src = str(e.metadata.get("dataset") or e.metadata.get("source") or "code_exec")
        by_src.setdefault(src, []).append(e)

    src_summary: dict[str, dict[str, float]] = {}
    for src, grp in sorted(by_src.items()):
        m = len(grp)
        g_t1 = sum(1 for x in grp if x.turn1_passed)
        g_fin = sum(1 for x in grp if x.final_passed)
        g_fail = m - g_t1
        g_rep = sum(1 for x in grp if x.repaired)
        src_summary[src] = {
            "count": m,
            "turn1_pass_at_1": round(g_t1 / m, 4),
            "turn2_pass_at_1": round(g_fin / m, 4),
            "repair_conversion_rate": round(g_rep / g_fail, 4) if g_fail > 0 else 0.0,
        }

    return {
        "total": n,
        "turn1_pass_at_1": round(t1_pass / n, 4),
        "turn2_pass_at_1": round(final_pass / n, 4),
        "repair_gain": round((final_pass - t1_pass) / n, 4),
        "turn1_failures": t1_fail,
        "repaired_count": repaired,
        "repair_conversion_rate": round(repaired / t1_fail, 4) if t1_fail > 0 else 0.0,
        "by_source": src_summary,
    }


def build_agentic_repair_sft_examples(
    episodes: list[AgenticEpisode],
    *,
    include_turn1_passes: bool = True,
    repair_upweight: int = 2,
) -> list[dict[str, Any]]:
    """Convert verified 2-turn episodes into completion-only SFT training examples.

    - Every verified self-repair `(prompt -> buggy_t1 -> traceback -> fixed_t2)` teaches
      true multi-turn debugging (upweighted by `repair_upweight`).
    - Verified Turn-1 passes preserve strong first-attempt coding capability.
    """
    examples: list[dict[str, Any]] = []
    seen_repairs: set[tuple[str, str]] = set()
    seen_t1: set[tuple[str, str]] = set()

    for ep in episodes:
        if ep.split != "train":
            continue
        sys_msg = ep.system_prompt or AGENTIC_CODE_SYSTEM_PROMPT

        if ep.repaired and ep.repair_prompt and ep.turn2_response:
            key = (ep.task_id, ep.turn2_response.strip()[:200])
            if key in seen_repairs:
                continue
            seen_repairs.add(key)
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ep.prompt},
                {"role": "assistant", "content": ep.turn1_response},
                {"role": "user", "content": ep.repair_prompt},
                {"role": "assistant", "content": ep.turn2_response},
            ]
            for rep_i in range(max(1, repair_upweight)):
                examples.append(
                    {
                        "task_id": f"{ep.task_id}_repair_{rep_i}",
                        "domain": "agentic_debug",
                        "difficulty": ep.difficulty,
                        "reward": 1.0,
                        "source": "agentic_self_repair",
                        "messages": msgs,
                    }
                )
        elif include_turn1_passes and ep.turn1_passed:
            key = (ep.task_id, ep.turn1_response.strip()[:200])
            if key in seen_t1:
                continue
            seen_t1.add(key)
            examples.append(
                {
                    "task_id": f"{ep.task_id}_t1",
                    "domain": "code_exec",
                    "difficulty": ep.difficulty,
                    "reward": 1.0,
                    "source": "agentic_turn1_pass",
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": ep.prompt},
                        {"role": "assistant", "content": ep.turn1_response},
                    ],
                }
            )

    return examples


def build_agentic_repair_dpo_pairs(
    episodes: list[AgenticEpisode],
) -> list[dict[str, Any]]:
    """Build multi-turn DPO preference pairs `(prompt_with_traceback, fixed_t2, failed_t2)`.

    Pairs a verified Turn-2 repair (`chosen`) against a failed Turn-2 attempt (`rejected`)
    from the same failing Turn-1 task, directly reinforcing traceback-grounded debugging.
    """
    by_task: dict[str, list[AgenticEpisode]] = {}
    for ep in episodes:
        if ep.split != "train" or ep.turn1_passed:
            continue
        by_task.setdefault(ep.task_id, []).append(ep)

    pairs: list[dict[str, Any]] = []
    for task_id, grp in sorted(by_task.items()):
        wins = [e for e in grp if e.repaired and e.repair_prompt and e.turn2_response]
        losses = [
            e
            for e in grp
            if (not e.final_passed) and e.repair_prompt and e.turn2_response
        ]
        if not wins or not losses:
            continue
        win = wins[0]
        loss = losses[0]
        sys_msg = win.system_prompt or AGENTIC_CODE_SYSTEM_PROMPT
        prompt_msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": win.prompt},
            {"role": "assistant", "content": win.turn1_response},
            {"role": "user", "content": win.repair_prompt},
        ]
        pairs.append(
            {
                "task_id": task_id,
                "domain": "agentic_debug",
                "difficulty": win.difficulty,
                "prompt_messages": prompt_msgs,
                "chosen": win.turn2_response,
                "rejected": loss.turn2_response,
            }
        )
    return pairs


def run_two_turn_agentic_eval(
    model: Any,
    tokenizer: Any,
    tasks: list[VerifiableTask],
    *,
    num_samples: int = 1,
    turn1_temperature: float = 0.0,
    turn2_temperature: float = 0.0,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> list[AgenticEpisode]:
    """Run batched 2-turn Write -> Sandbox Verify -> Traceback -> Self-Repair on GPU."""
    import torch

    def _apply_chat(msgs: list[dict[str, str]]) -> str:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def _gen_batch(prompts: list[str], temp: float) -> list[str]:
        outs: list[str] = []
        do_sample = temp > 0.0
        for s in range(0, len(prompts), batch_size):
            batch_p = prompts[s : s + batch_size]
            inputs = tokenizer(
                batch_p,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2560,
            ).to(model.device)
            plen = inputs["input_ids"].shape[1]
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = temp
                kwargs["top_p"] = 0.95
            with torch.no_grad():
                gen = model.generate(**inputs, **kwargs)
            for idx in range(len(batch_p)):
                txt = tokenizer.decode(gen[idx][plen:], skip_special_tokens=True).strip()
                txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
                outs.append(txt)
        return outs

    expanded: list[VerifiableTask] = []
    for t in tasks:
        for _ in range(num_samples):
            expanded.append(t)

    # Turn 1: Initial code generation + deterministic sandbox execution
    t1_prompts = [
        _apply_chat(
            [
                {"role": "system", "content": t.system_prompt or AGENTIC_CODE_SYSTEM_PROMPT},
                {"role": "user", "content": t.prompt},
            ]
        )
        for t in expanded
    ]
    t1_resps = _gen_batch(t1_prompts, turn1_temperature)

    episodes: list[AgenticEpisode] = []
    need_repair_idx: list[int] = []
    t2_prompts: list[str] = []

    for idx, (task, resp1) in enumerate(zip(expanded, t1_resps)):
        ver1 = verify_code_exec(task, resp1)
        ep = AgenticEpisode(
            task_id=task.task_id,
            split=task.split,
            difficulty=task.difficulty,
            prompt=task.prompt,
            system_prompt=task.system_prompt or AGENTIC_CODE_SYSTEM_PROMPT,
            turn1_response=resp1,
            turn1_verification=ver1,
            metadata=dict(task.metadata),
        )
        episodes.append(ep)
        if not ver1.passed:
            r_prompt = build_repair_prompt(ver1)
            ep.repair_prompt = r_prompt
            need_repair_idx.append(idx)
            t2_prompts.append(
                _apply_chat(
                    [
                        {"role": "system", "content": ep.system_prompt},
                        {"role": "user", "content": ep.prompt},
                        {"role": "assistant", "content": ep.turn1_response},
                        {"role": "user", "content": r_prompt},
                    ]
                )
            )

    # Turn 2: Self-Repair on failed episodes + deterministic re-verification
    if t2_prompts:
        t2_resps = _gen_batch(t2_prompts, turn2_temperature)
        for ep_idx, resp2 in zip(need_repair_idx, t2_resps):
            task = expanded[ep_idx]
            ep = episodes[ep_idx]
            ep.turn2_response = resp2
            ep.turn2_verification = verify_code_exec(task, resp2)

    return episodes
