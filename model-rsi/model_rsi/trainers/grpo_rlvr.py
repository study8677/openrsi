"""M3: Online RLVR / GRPO Post-Training Runner with Immutable Sandbox Rewards.

Uses TRL GRPOTrainer with our deterministic AST / subprocess sandbox verifiers
as online reward functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_rsi.protocol import VerifiableTask
from model_rsi.tasks import verify_response


def train_grpo_rlvr(
    base_model_path: str,
    tasks: list[VerifiableTask],
    output_dir: str | Path,
    *,
    max_steps: int = 40,
    num_generations: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.04,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    max_completion_length: int = 384,
    temperature: float = 0.85,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pure_binary_reward: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run online GRPO (RLVR) using deterministic sandbox verification rewards."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_path / "_grpo_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    task_map = {t.task_id: t for t in tasks}

    rows = []
    for t in tasks:
        msgs = []
        if t.system_prompt:
            msgs.append({"role": "system", "content": t.system_prompt})
        msgs.append({"role": "user", "content": t.prompt})
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        rows.append({"prompt": prompt_str, "task_id": t.task_id})
    ds = Dataset.from_list(rows)

    def verifier_reward_func(completions: list[str], task_id: list[str], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for comp, tid in zip(completions, task_id):
            text = comp if isinstance(comp, str) else str(comp)
            ver = verify_response(task_map[tid], text)
            if pure_binary_reward:
                # Anti-reward-hacking: reward ONLY deterministic verification pass (no format shortcut)
                rewards.append(1.0 if ver.passed else 0.0)
            else:
                rewards.append(float(ver.reward))
        return rewards

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Keep Gated DeltaNet recurrence gates (in_proj_a/b) frozen on Qwen3.5 to preserve state stability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    grpo_args = GRPOConfig(
        output_dir=str(adapter_dir),
        max_steps=max_steps,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        temperature=temperature,
        max_completion_length=max_completion_length,
        lr_scheduler_type="cosine",
        warmup_steps=max(2, max_steps // 10),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[verifier_reward_func],
        args=grpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    return {
        "checkpoint_path": str(out_path),
        "train_loss": float(train_result.training_loss),
        "num_examples": len(tasks),
        "max_steps": max_steps,
    }
