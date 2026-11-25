"""Iterative DPO Post-Training Runner using Verified (Pass, Fail) Trajectory Pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_iterative_dpo(
    base_model_path: str,
    pairs: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    num_train_epochs: float = 1.0,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Run LoRA DPO on same-task verified (chosen=pass, rejected=fail) pairs."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_path / "_dpo_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for p in pairs:
        try:
            prompt_str = tokenizer.apply_chat_template(
                p["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_str = tokenizer.apply_chat_template(
                p["prompt"], tokenize=False, add_generation_prompt=True
            )
        rows.append(
            {
                "prompt": prompt_str,
                "chosen": p["chosen"][0]["content"],
                "rejected": p["rejected"][0]["content"],
            }
        )
    ds = Dataset.from_list(rows)

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = has_cuda and not use_bf16
    dev_map: Any = {"": 0} if (has_cuda and torch.cuda.device_count() == 1) else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32),
        device_map=dev_map,
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

    dpo_args = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        lr_scheduler_type="cosine",
        warmup_steps=max(2, len(pairs) // 20),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        max_length=max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT implicitly uses base model as reference!
        args=dpo_args,
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
        "num_examples": len(pairs),
    }
