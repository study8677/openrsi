"""M2: Verified SFT / Rejection Sampling (STaR) Post-Training Runner.

Trains LoRA adapters on deterministically verified positive trajectories via TRL
SFTTrainer, then merges adapters into a standalone Checkpoint N+1 directory for
clean recursive multi-generation chaining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_star_sft(
    base_model_path: str,
    examples: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    num_train_epochs: float = 2.0,
    learning_rate: float = 1.5e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Run LoRA SFT on verified STaR examples and export a merged checkpoint."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_path / "_lora_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use prompt/completion columns with completion_only_loss=True so SFT never overfits prompts
    prompts: list[str] = []
    completions: list[str] = []
    for ex in examples:
        msgs = ex["messages"]
        try:
            prompt_str = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_str = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
        comp_str = msgs[-1]["content"] + (tokenizer.eos_token or "")
        prompts.append(prompt_str)
        completions.append(comp_str)
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

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

    sft_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(2, len(examples) // 20),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        completion_only_loss=True,
        max_length=max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()

    # Merge LoRA into base weights so next generation can cleanly stack or run GRPO/DPO
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    return {
        "checkpoint_path": str(out_path),
        "train_loss": float(train_result.training_loss),
        "num_examples": len(examples),
    }
