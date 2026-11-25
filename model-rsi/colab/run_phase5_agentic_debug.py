"""Phase 5: Multi-Turn Agentic Self-Repair RLVR Runner (Google Colab GPU: L4 / T4).

Evolves `Qwen/Qwen3.5-2B` on a genuine 2-turn sandbox execution loop:
  Turn 1 (Write) -> Isolated `python -I` Unit Tests -> Traceback Feedback
  -> Turn 2 (Self-Repair) -> Re-Verify

Generations:
  - Gen 0: Baseline `Qwen/Qwen3.5-2B` on locked Held-out (40 HumanEval+ + 24 MBPP) + Regression
  - Gen 1: Verified Multi-Turn Self-Repair STaR SFT (collected from KodCode-V1 + MBPP train ONLY)
  - Gen 2: Repair-Focused DeltaNet-Safe Iterative DPO (paired winning vs failing Turn-2 repairs)
"""

from __future__ import annotations

import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import peft.import_utils
import peft.tuners.lora.torchao

peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from model_rsi.gating import GatePolicy, evaluate_promotion_gate
from model_rsi.protocol import GenerationMetrics, VerifiableTask
from model_rsi.reporting import plot_evolution_curve, save_evolution_history
from model_rsi.tasks.agentic_debug import (
    AGENTIC_CODE_SYSTEM_PROMPT,
    build_agentic_repair_dpo_pairs,
    build_agentic_repair_sft_examples,
    run_two_turn_agentic_eval,
    summarize_agentic_episodes,
)
from model_rsi.tasks.code_exec import (
    load_humanevalplus_tasks,
    load_kodcode_tasks,
    load_mbpp_tasks,
)
from model_rsi.tasks.regression import get_regression_suite, verify_regression


def _hardware_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _format_chat(tok: Any, msgs: list[dict[str, str]]) -> str:
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_gpu_model(path: str) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if _hardware_bf16() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    return model, tok


def free_gpu(model: Any = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_regression(model: Any, tok: Any) -> float:
    reg_tasks = get_regression_suite()
    passed = 0
    for t in reg_tasks:
        p = _format_chat(
            tok,
            [
                {"role": "system", "content": t.system_prompt},
                {"role": "user", "content": t.prompt},
            ],
        )
        inp = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
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


def run_phase5(
    *,
    base_model: str = "Qwen/Qwen3.5-2B",
    work_dir: str = "/content/openrsi_phase5_debug",
    seed: int = 42,
) -> dict[str, Any]:
    work = Path(work_dir)
    ckpt_dir = work / "checkpoints"
    rep_dir = work / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load strictly isolated train / val / held_out coding benchmarks
    kod_train = [t for t in load_kodcode_tasks(train_limit=100, val_limit=0, seed=seed) if t.split == "train"]
    mbpp_all = load_mbpp_tasks(train_limit=80, val_limit=24, held_out_limit=24)
    mbpp_train = [t for t in mbpp_all if t.split == "train"]
    mbpp_val = [t for t in mbpp_all if t.split == "val"]
    mbpp_ho = [t for t in mbpp_all if t.split == "held_out"]
    he_ho = load_humanevalplus_tasks(limit=40, seed=seed)

    for t in kod_train + mbpp_train + mbpp_val + mbpp_ho + he_ho:
        t.system_prompt = AGENTIC_CODE_SYSTEM_PROMPT

    train_pool = kod_train + mbpp_train
    val_tasks = mbpp_val
    ho_tasks = he_ho + mbpp_ho
    print(
        f"[Phase 5] Loaded train={len(train_pool)} (KodCode={len(kod_train)}, MBPP={len(mbpp_train)}), "
        f"val={len(val_tasks)}, held_out={len(ho_tasks)} (HumanEval+={len(he_ho)}, MBPP={len(mbpp_ho)})"
    )

    policy = GatePolicy(min_held_out_gain=0.005, max_regression_drop=0.05, min_regression_floor=0.80)
    history: list[GenerationMetrics] = []
    detailed_metrics: list[dict[str, Any]] = []

    def evaluate_ckpt(ckpt_path: str, gen: int, paradigm: str, parent_gen: int | None) -> tuple[GenerationMetrics, dict[str, Any]]:
        model, tok = load_gpu_model(ckpt_path)
        ho_eps = run_two_turn_agentic_eval(
            model, tok, ho_tasks, num_samples=1, turn1_temperature=0.0, turn2_temperature=0.0, batch_size=8
        )
        val_eps = run_two_turn_agentic_eval(
            model, tok, val_tasks, num_samples=1, turn1_temperature=0.0, turn2_temperature=0.0, batch_size=8
        )
        reg_sc = eval_regression(model, tok)
        free_gpu(model)

        ho_sum = summarize_agentic_episodes(ho_eps)
        val_sum = summarize_agentic_episodes(val_eps)
        gm = GenerationMetrics(
            generation=gen,
            paradigm=paradigm,
            parent_generation=parent_gen,
            model_id=Path(ckpt_path).name if Path(ckpt_path).exists() else ckpt_path,
            checkpoint_path=ckpt_path,
            held_out_pass_at_1=ho_sum["turn2_pass_at_1"],
            val_pass_at_1=val_sum["turn2_pass_at_1"],
            regression_score=reg_sc,
            domain_scores={
                "turn1_pass_at_1": ho_sum["turn1_pass_at_1"],
                "turn2_pass_at_1": ho_sum["turn2_pass_at_1"],
                "repair_conv_rate": ho_sum["repair_conversion_rate"],
                "humaneval_plus_t2": ho_sum["by_source"].get("humaneval_plus", {}).get("turn2_pass_at_1", 0.0),
                "mbpp_t2": ho_sum["by_source"].get("mbpp", {}).get("turn2_pass_at_1", 0.0),
            },
            notes=(
                f"T1={ho_sum['turn1_pass_at_1']:.2%} -> T2={ho_sum['turn2_pass_at_1']:.2%} "
                f"(repaired {ho_sum['repaired_count']}/{ho_sum['turn1_failures']} = {ho_sum['repair_conversion_rate']:.2%})"
            ),
        )
        return gm, {"generation": gen, "paradigm": paradigm, "held_out": ho_sum, "val": val_sum, "regression": reg_sc}

    # --- Gen 0: Baseline ---
    g0, d0 = evaluate_ckpt(base_model, 0, "baseline", None)
    g0.status = "baseline"
    history.append(g0)
    detailed_metrics.append(d0)
    best_gm = g0
    print(f"[Phase 5 Gen 0] {g0.notes} | Reg={g0.regression_score:.2%}")

    # --- Gen 1: Rollout 2-turn episodes on Train -> Multi-Turn Self-Repair STaR SFT ---
    rng = random.Random(seed)
    rollout_subset = list(train_pool)
    rng.shuffle(rollout_subset)
    rollout_subset = rollout_subset[:120]

    model, tok = load_gpu_model(best_gm.checkpoint_path)
    t_eps_g1 = run_two_turn_agentic_eval(
        model,
        tok,
        rollout_subset,
        num_samples=2,
        turn1_temperature=0.6,
        turn2_temperature=0.7,
        batch_size=8,
    )
    free_gpu(model)
    tr_sum = summarize_agentic_episodes(t_eps_g1)
    print(
        f"[Phase 5 Gen 1 Rollout] Train T1={tr_sum['turn1_pass_at_1']:.2%}, "
        f"T2={tr_sum['turn2_pass_at_1']:.2%}, repaired={tr_sum['repaired_count']}/{tr_sum['turn1_failures']}"
    )

    sft_ex = build_agentic_repair_sft_examples(t_eps_g1, include_turn1_passes=True, repair_upweight=3)
    print(f"[Phase 5 Gen 1 SFT] Verified examples={len(sft_ex)}")
    gen1_dir = ckpt_dir / "gen1_agentic_repair_star"
    use_bf16 = _hardware_bf16()
    tok = AutoTokenizer.from_pretrained(best_gm.checkpoint_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_dict(
        {
            "prompt": [_format_chat(tok, ex["messages"][:-1]) for ex in sft_ex],
            "completion": [ex["messages"][-1]["content"] + (tok.eos_token or "") for ex in sft_ex],
        }
    )
    model = AutoModelForCausalLM.from_pretrained(
        best_gm.checkpoint_path,
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=str(gen1_dir / "_lora"),
        num_train_epochs=2.0,
        per_device_train_batch_size=2 if use_bf16 else 1,
        gradient_accumulation_steps=4 if use_bf16 else 8,
        learning_rate=1.0e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed + 1,
        completion_only_loss=True,
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(gen1_dir))
    tok.save_pretrained(str(gen1_dir))
    free_gpu(merged)
    del trainer, model
    free_gpu()

    g1, d1 = evaluate_ckpt(str(gen1_dir), 1, "agentic_repair_star", best_gm.generation)
    dec1 = evaluate_promotion_gate(g1, best_gm, g0, policy)
    g1.status = "promoted" if dec1.promoted else "rejected"
    history.append(g1)
    detailed_metrics.append(d1)
    if dec1.promoted:
        best_gm = g1
    print(f"[Phase 5 Gen 1] {g1.status.upper()} | {g1.notes} | Reg={g1.regression_score:.2%}")

    # --- Gen 2: Self-Repair Preference DPO (only on bf16 hardware or fallback to Hard-Repair STaR) ---
    model, tok = load_gpu_model(best_gm.checkpoint_path)
    t_eps_g2 = run_two_turn_agentic_eval(
        model,
        tok,
        rollout_subset,
        num_samples=3,
        turn1_temperature=0.7,
        turn2_temperature=0.8,
        batch_size=8,
    )
    free_gpu(model)
    dpo_pairs = build_agentic_repair_dpo_pairs(t_eps_g2)
    print(f"[Phase 5 Gen 2] Harvested {len(dpo_pairs)} multi-turn self-repair DPO pairs")

    if len(dpo_pairs) >= 10 and use_bf16:
        gen2_dir = ckpt_dir / "gen2_agentic_repair_dpo"
        tok = AutoTokenizer.from_pretrained(best_gm.checkpoint_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dpo_ds = Dataset.from_dict(
            {
                "prompt": [_format_chat(tok, p["prompt_messages"]) for p in dpo_pairs],
                "chosen": [p["chosen"] + (tok.eos_token or "") for p in dpo_pairs],
                "rejected": [p["rejected"] + (tok.eos_token or "") for p in dpo_pairs],
            }
        )
        model = AutoModelForCausalLM.from_pretrained(
            best_gm.checkpoint_path, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
        )
        dpo_cfg = DPOConfig(
            output_dir=str(gen2_dir / "_lora"),
            num_train_epochs=2.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=3e-5,
            beta=0.1,
            lr_scheduler_type="cosine",
            warmup_steps=3,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            seed=seed + 2,
            max_length=2048,
            max_prompt_length=1536,
        )
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_cfg,
            train_dataset=dpo_ds,
            processing_class=tok,
            peft_config=peft_cfg,
        )
        dpo_trainer.train()
        merged = dpo_trainer.model.merge_and_unload()
        merged.save_pretrained(str(gen2_dir))
        tok.save_pretrained(str(gen2_dir))
        free_gpu(merged)
        del dpo_trainer, model
        free_gpu()

        g2, d2 = evaluate_ckpt(str(gen2_dir), 2, "agentic_repair_dpo", best_gm.generation)
        dec2 = evaluate_promotion_gate(g2, best_gm, g0, policy)
        g2.status = "promoted" if dec2.promoted else "rejected"
        history.append(g2)
        detailed_metrics.append(d2)
        if dec2.promoted:
            best_gm = g2
        print(f"[Phase 5 Gen 2] {g2.status.upper()} | {g2.notes} | Reg={g2.regression_score:.2%}")

    save_evolution_history(history, rep_dir / "phase5_agentic_debug_history.json")
    plot_evolution_curve(
        history,
        rep_dir / "phase5_agentic_debug_curve.png",
        title="OpenRSI Phase 5 — Multi-Turn Agentic Self-Repair RLVR (Qwen3.5-2B)",
    )
    summary = {
        "best_generation": best_gm.generation,
        "best_checkpoint": best_gm.checkpoint_path,
        "details": detailed_metrics,
    }
    (rep_dir / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    run_phase5()
