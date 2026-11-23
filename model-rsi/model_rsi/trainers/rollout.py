"""Batched GPU Rollout & Evaluation Engine for Model-RSI."""

from __future__ import annotations

import gc
import uuid
from collections import defaultdict
from typing import Any

from model_rsi.protocol import Trajectory, VerifiableTask
from model_rsi.tasks import verify_response


def format_chat_prompt(tokenizer: Any, task: VerifiableTask) -> str:
    messages = []
    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})
    messages.append({"role": "user", "content": task.prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model_and_tokenizer(model_path: str, *, bf16: bool = True) -> tuple[Any, Any]:
    """Load an open-weight HF model (or merged checkpoint) on GPU with left-padding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def release_gpu_memory(model: Any = None) -> None:
    import torch

    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_batched_rollouts(
    model: Any,
    tokenizer: Any,
    tasks: list[VerifiableTask],
    *,
    generation: int,
    model_id: str,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 384,
    batch_size: int = 16,
) -> list[Trajectory]:
    """Sample `num_samples` trajectories per task and score each with its deterministic verifier."""
    import torch

    expanded: list[VerifiableTask] = []
    for t in tasks:
        for _ in range(num_samples):
            expanded.append(t)

    trajectories: list[Trajectory] = []
    do_sample = temperature > 0.0

    for start in range(0, len(expanded), batch_size):
        batch_tasks = expanded[start : start + batch_size]
        prompts = [format_chat_prompt(tokenizer, t) for t in batch_tasks]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for idx, task in enumerate(batch_tasks):
            gen_ids = outputs[idx][prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            ver = verify_response(task, response)
            trajectories.append(
                Trajectory(
                    trajectory_id=f"g{generation}_{task.task_id}_{uuid.uuid4().hex[:8]}",
                    task_id=task.task_id,
                    domain=task.domain,
                    split=task.split,
                    difficulty=task.difficulty,
                    generation=generation,
                    model_id=model_id,
                    source="model_rollout",
                    prompt=task.prompt,
                    system_prompt=task.system_prompt,
                    response=response,
                    verification=ver,
                    temperature=temperature,
                )
            )

    return trajectories


def summarize_trajectories(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Compute pass@1, pass@k (any sample passed), format rate, and per-difficulty breakdown."""
    if not trajectories:
        return {
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "format_rate": 0.0,
            "avg_chars": 0.0,
            "breakdown": {},
        }

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    by_bucket: dict[str, list[Trajectory]] = defaultdict(list)
    for tr in trajectories:
        by_task[tr.task_id].append(tr)
        bucket = f"{tr.domain}_L{tr.difficulty}" if tr.domain == "countdown" else tr.domain
        by_bucket[bucket].append(tr)

    pass_at_1 = sum(1 for t in trajectories if t.verification.passed) / len(trajectories)
    pass_at_k = sum(
        1 for grp in by_task.values() if any(t.verification.passed for t in grp)
    ) / len(by_task)
    format_rate = (
        sum(1 for t in trajectories if t.verification.format_reward >= 0.15)
        / len(trajectories)
    )
    avg_chars = sum(len(t.response) for t in trajectories) / len(trajectories)

    breakdown = {
        k: round(sum(1 for t in v if t.verification.passed) / len(v), 4)
        for k, v in sorted(by_bucket.items())
    }
    return {
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_k": round(pass_at_k, 4),
        "format_rate": round(format_rate, 4),
        "avg_chars": round(avg_chars, 1),
        "breakdown": breakdown,
    }
