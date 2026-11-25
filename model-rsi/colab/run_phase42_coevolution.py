"""Phase 4.2 Dual-Loop Co-Evolution Runner (Google Colab GPU: T4 or L4).

Evolves `OpenRSI-Qwen3.5-2B-Flywheel` from Gen 3 (`59.28%` Elite Held-out) into
Gen 4 (`compactor_reader_co_sft`) by jointly training on:
  1. 322 atomically verified Compactor Patch + Audit distillation examples
     (`data/compactbench_patch_distill_v1.jsonl.gz`, strictly from the 36 Train cases).
  2. Verified `buried_constraint` + `conditional_scope` + `reference_resolution`
     precision-calibrated Reader examples (curing multi-select over-selection).
  3. Anchor replay from Gen 1 verified teacher Reader batches (anti-forgetting).

Then evaluates on:
  - Locked Elite Held-out (249 unseen 32k probes) + Val (251 probes) + Regression (10 items)
  - Official `compactbench==0.3.0` `starter/default` (102 probes) Cell 3 & Cell 4 ablation
    with live `compactor_valid_patch_rate` telemetry!
"""

from __future__ import annotations

import gc
import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Any

# Patch peft/torchao version mismatch on Colab before importing peft/trl
import peft.import_utils
import peft.tuners.lora.torchao

peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from model_rsi.flywheel_provider import (
    HFCompactBenchProvider,
    plot_flywheel_ablation,
    run_official_compactbench_eval,
)
from model_rsi.gating import GatePolicy, evaluate_promotion_gate
from model_rsi.protocol import GenerationMetrics, VerifiableTask
from model_rsi.reporting import (
    load_evolution_history,
    plot_evolution_curve,
    save_evolution_history,
)
from model_rsi.tasks.compactbench_patch import (
    build_buried_constraint_precision_examples,
    load_patch_distill_gz,
)
from model_rsi.tasks.compactbench_rlvr import (
    build_harness_distill_sft_examples,
    verify_compactbench_batch,
)
from model_rsi.tasks.regression import get_regression_suite, verify_regression


def _hardware_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def load_gpu_model(model_path: str) -> tuple[Any, Any]:
    """Load model strictly onto cuda:0 (NEVER CPU offload) in native bf16 (L4) or fp16 (T4)."""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if _hardware_bf16() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def free_gpu(model: Any = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_prompt_safe(tok: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def eval_cb_reader_tasks(
    model: Any,
    tok: Any,
    tasks: list[VerifiableTask],
    *,
    batch_size: int = 4,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    import re

    total_probes = 0
    full_credit = 0
    sum_score = 0.0
    json_ok = 0
    by_fam: dict[str, list[float]] = {}

    for s in range(0, len(tasks), batch_size):
        batch = tasks[s : s + batch_size]
        prompts = [
            _format_prompt_safe(
                tok,
                [
                    {"role": "system", "content": t.system_prompt},
                    {"role": "user", "content": t.prompt},
                ],
            )
            for t in batch
        ]
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2560,
        ).to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for i, t in enumerate(batch):
            raw = tok.decode(out[i][plen:], skip_special_tokens=True).strip()
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            ver = verify_compactbench_batch(t, raw)
            if ver.format_reward >= 0.99:
                json_ok += 1
            n_p = int(ver.details.get("probe_count", 0))
            total_probes += n_p
            full_credit += int(ver.details.get("full_credit_probes", 0))
            sum_score += float(ver.details.get("mean_score", 0.0)) * n_p
            for fam, sc in (ver.details.get("by_family") or {}).items():
                by_fam.setdefault(fam, []).append(float(sc))

    return {
        "probe_score": round(sum_score / max(1, total_probes), 4),
        "full_credit": full_credit,
        "total_probes": total_probes,
        "json_rate": round(json_ok / max(1, len(tasks)), 4),
        "by_family": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_fam.items())
        },
    }


def eval_regression(model: Any, tok: Any) -> float:
    import re

    reg_tasks = get_regression_suite()
    passed = 0
    for t in reg_tasks:
        p = _format_prompt_safe(
            tok,
            [
                {"role": "system", "content": t.system_prompt},
                {"role": "user", "content": t.prompt},
            ],
        )
        inp = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device
        )
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        txt = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
        txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
        if verify_regression(t, txt).passed:
            passed += 1
    return round(passed / len(reg_tasks), 4)


def run_phase42(
    *,
    repo_root: str = "/content/openrsi",
    work_dir: str = "/content/openrsi_flywheel",
    parent_ckpt: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(repo_root)
    work = Path(work_dir)
    ckpt_dir = work / "checkpoints"
    rep_dir = work / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen Phase 4 flywheel tasks
    fw_gz = root / "model-rsi" / "data" / "compactbench_flywheel_v1.jsonl.gz"
    all_tasks = [
        VerifiableTask.model_validate(json.loads(ln))
        for ln in gzip.open(fw_gz, "rt", encoding="utf-8")
        if ln.strip()
    ]
    train_tasks = [t for t in all_tasks if t.split == "train"]
    val_tasks = [t for t in all_tasks if t.split == "val"]
    ho_tasks = [t for t in all_tasks if t.split == "held_out"]

    # Deterministic eval subsets matching Phase 4A exactly (40 held-out = 249 probes, 40 val = 251 probes)
    rng_eval = random.Random(seed)
    ho_eval = list(ho_tasks)
    rng_eval.shuffle(ho_eval)
    ho_eval = ho_eval[:40]

    val_eval = list(val_tasks)
    rng_eval.shuffle(val_eval)
    val_eval = val_eval[:40]

    # Build Phase 4.2 Co-Evolution SFT dataset:
    #   (A) 322 verified Compactor Patch + Audit examples (strictly train-only 36 cases)
    #   (B) Precision-calibrated buried_constraint / conditional_scope / reference_resolution batches
    #   (C) Anchor replay of general high-scoring teacher Reader batches
    patch_gz = root / "model-rsi" / "data" / "compactbench_patch_distill_v1.jsonl.gz"
    patch_ex = load_patch_distill_gz(patch_gz)
    buried_ex = build_buried_constraint_precision_examples(train_tasks)
    anchor_ex = build_harness_distill_sft_examples(
        train_tasks, min_teacher_score=0.90, use_verified_teacher=True
    )

    rng = random.Random(seed + 42)
    rng.shuffle(patch_ex)
    rng.shuffle(buried_ex)
    rng.shuffle(anchor_ex)

    # Balanced co-evolution mixture (~420 high-signal examples -> ~5-7 min on T4, ~2.5 min on L4)
    mix = patch_ex[:220] + buried_ex[:140] + anchor_ex[:60]
    rng.shuffle(mix)
    print(
        f"[Phase 4.2] Co-Evolution SFT mix: total={len(mix)} "
        f"(compactor_patch=220, buried_precision=140, reader_anchor=60)"
    )

    # Determine starting checkpoint (prefer Gen 3 Flywheel champion if present on disk, else Qwen3.5-2B)
    gen3_path = ckpt_dir / "gen3_bottleneck_star"
    if parent_ckpt:
        start_ckpt = parent_ckpt
    elif gen3_path.exists():
        start_ckpt = str(gen3_path)
    else:
        start_ckpt = "Qwen/Qwen3.5-2B"
    print(f"[Phase 4.2] Starting from checkpoint: {start_ckpt} | bf16={_hardware_bf16()}")

    # Train Gen 4: Compactor + Reader Co-Evolution SFT (DeltaNet-Safe LoRA, completion_only_loss=True)
    gen4_out = ckpt_dir / "gen4_compactor_reader_co_sft"
    tok = AutoTokenizer.from_pretrained(start_ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts: list[str] = []
    completions: list[str] = []
    for ex in mix:
        msgs = ex["messages"]
        prompts.append(_format_prompt_safe(tok, msgs[:-1]))
        completions.append(msgs[-1]["content"] + (tok.eos_token or ""))
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

    use_bf16 = _hardware_bf16()
    model = AutoModelForCausalLM.from_pretrained(
        start_ckpt,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Gated DeltaNet Freeze Rule: NEVER touch in_proj_a / in_proj_b
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    sft_cfg = SFTConfig(
        output_dir=str(gen4_out / "_lora"),
        num_train_epochs=2.0,
        per_device_train_batch_size=2 if use_bf16 else 1,
        gradient_accumulation_steps=4 if use_bf16 else 8,
        learning_rate=6e-5 if "gen3" in start_ckpt else 1.2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=8,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed + 42,
        completion_only_loss=True,
        max_length=2560,
    )
    t0 = time.monotonic()
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    train_res = trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(gen4_out))
    tok.save_pretrained(str(gen4_out))
    train_sec = round(time.monotonic() - t0, 1)
    print(
        f"[Phase 4.2] Gen 4 training done in {train_sec}s | loss={train_res.training_loss:.4f}"
    )
    free_gpu(merged)
    del trainer, model
    free_gpu()

    # Evaluate Gen 4 on locked Elite Held-out, Val, Regression, and Official CompactBench Ablation
    model, tok = load_gpu_model(str(gen4_out))
    ho_res = eval_cb_reader_tasks(model, tok, ho_eval, batch_size=4)
    val_res = eval_cb_reader_tasks(model, tok, val_eval, batch_size=4)
    reg_score = eval_regression(model, tok)
    print(
        f"[Phase 4.2] Gen 4 Elite Held-out={ho_res['probe_score']:.4f} "
        f"({ho_res['full_credit']}/{ho_res['total_probes']}), "
        f"Val={val_res['probe_score']:.4f}, Reg={reg_score:.4f}, JSON={ho_res['json_rate']:.4f}"
    )
    print("[Phase 4.2] By family:", ho_res["by_family"])

    # Run Official CompactBench `starter/default` (102 probes) for Cell 3 & Cell 4 with Gen 4!
    harness_spec = f"{root / 'harness-rsi' / 'harness' / 'cb_compactor.py'}:OpenRSICompactor"
    provider_trunc = HFCompactBenchProvider(
        model, tok, model_name="openrsi-flywheel-gen4", default_max_new_tokens=1024
    )
    cell3_g4 = run_official_compactbench_eval(
        provider_trunc,
        method_spec="truncate-last-n",
        model_label="openrsi-flywheel-gen4",
        output_path=rep_dir / "ablation_cell3_gen4_trunc.jsonl",
        suite="starter",
        seed_group="default",
        cases_per_family=1,
    )
    provider_h = HFCompactBenchProvider(
        model, tok, model_name="openrsi-flywheel-gen4", default_max_new_tokens=1024
    )
    cell4_g4 = run_official_compactbench_eval(
        provider_h,
        method_spec=harness_spec,
        model_label="openrsi-flywheel-gen4",
        output_path=rep_dir / "ablation_cell4_gen4_harness.jsonl",
        suite="starter",
        seed_group="default",
        cases_per_family=1,
    )
    free_gpu(model)

    print("\n=== Phase 4.2 Official CompactBench Starter Ablation (Gen 4) ===")
    print("Cell 3 (Gen4 + Trunc):", cell3_g4["official_score"], "drift:", cell3_g4["drift_resistance"])
    print(
        "Cell 4 (Gen4 + OpenRSICompactor):",
        cell4_g4["official_score"],
        "drift:",
        cell4_g4["drift_resistance"],
        "stale:",
        cell4_g4["stale_rate"],
        "tokens:",
        cell4_g4["mean_artifact_tokens"],
        "provider_stats:",
        cell4_g4.get("provider_stats"),
    )

    out_summary = {
        "gen4_ckpt": str(gen4_out),
        "parent_ckpt": start_ckpt,
        "train_loss": round(float(train_res.training_loss), 4),
        "train_sec": train_sec,
        "held_out": ho_res,
        "val": val_res,
        "regression_score": reg_score,
        "cell3_gen4_trunc": cell3_g4,
        "cell4_gen4_harness": cell4_g4,
    }
    (rep_dir / "phase42_summary.json").write_text(
        json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_summary


if __name__ == "__main__":
    run_phase42()
