"""Phase 4 Dual-Loop Flywheel Runner (Harness-RSI <-> Model-RSI on CompactBench).

Works seamlessly on Google Colab GPUs (NVIDIA L4/A100/T4) and Apple Silicon MPS.

Pipeline:
  - Phase 4A (Harness -> Model):
      Gen 0: Baseline Raw Qwen/Qwen3.5-2B on unseen CompactBench Elite Held-out + Regression
      Gen 1: Harness-to-Model Distillation (DeltaNet-Safe Completion-Only LoRA SFT + Anchor Replay)
      Gen 2: Probe-Verified Iterative DPO on P0/P1 Bottlenecks (buried_constraint, conditional_scope, reference_resolution)
  - Phase 4B (Model -> Harness Reverse Closing Loop):
      Official `compactbench==0.3.0` 2x2 Ablation Matrix on `starter` suite (8 families x 3 cycles = 102 probes):
        (1) Raw Qwen3.5-2B             + built-in:truncate-last-n  (No Harness, No Model-RSI)
        (2) Raw Qwen3.5-2B             + OpenRSICompactor v2       (Harness-RSI only)
        (3) OpenRSI-Qwen3.5-2B-Flywheel + built-in:truncate-last-n (Model-RSI only)
        (4) OpenRSI-Qwen3.5-2B-Flywheel + OpenRSICompactor v2      (FULL DUAL-LOOP FLYWHEEL!)
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

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
from model_rsi.tasks import verify_response
from model_rsi.tasks.compactbench_rlvr import (
    build_harness_distill_sft_examples,
    verify_compactbench_batch,
)
from model_rsi.tasks.regression import get_regression_suite

MODEL_RSI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_RSI_ROOT.parent
DEFAULT_DATA_GZ = MODEL_RSI_ROOT / "data" / "compactbench_flywheel_v1.jsonl.gz"
DEFAULT_HARNESS_COMPACTOR = REPO_ROOT / "harness-rsi" / "harness" / "cb_compactor.py"

# Keep Gated DeltaNet recurrence gates (in_proj_a/b) frozen on Qwen3.5!
DELTANET_SAFE_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ANCHOR_REPLAY = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "Return ONLY a valid JSON object with keys 'planet' set to 'Mars' and 'moons' set to 2."},
            {"role": "assistant", "content": '{"planet": "Mars", "moons": 2}'},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "What is the capital city of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "If all Cats are Mammals and all Mammals are Animals, are all Cats Animals? Answer Yes or No."},
            {"role": "assistant", "content": "Yes."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "What is the time complexity of looking up a key in a hash table on average?"},
            {"role": "assistant", "content": "O(1)."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "Reply with the single word READY in all uppercase letters."},
            {"role": "assistant", "content": "READY"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully."},
            {"role": "user", "content": "A cyclist rides at 20 km/h for 3 hours. How many kilometers do they ride?"},
            {"role": "assistant", "content": "60 kilometers."},
        ]
    },
]


def detect_runtime() -> tuple[str, torch.dtype, bool, bool]:
    """Return (device_str, dtype, use_bf16, use_fp16)."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:  # L4 / A100 / H100
            return "cuda", torch.bfloat16, True, False
        return "cuda", torch.float16, False, True  # T4
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16, False, False
    return "cpu", torch.float32, False, False


def free_memory(model: Any = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model(model_path: str, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if device == "cuda":
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto", trust_remote_code=True
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        ).to(device)
    m.eval()
    return m, tok


def load_flywheel_splits(data_gz: Path) -> tuple[list[VerifiableTask], list[VerifiableTask], list[VerifiableTask]]:
    all_cb: list[VerifiableTask] = []
    with gzip.open(data_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_cb.append(VerifiableTask.model_validate_json(line))
    cb_train = [t for t in all_cb if t.split == "train"]
    cb_val = [t for t in all_cb if t.split == "val"]
    cb_ho = [t for t in all_cb if t.split == "held_out"]
    return cb_train, cb_val, cb_ho


def eval_cb_suite(
    model: Any,
    tok: Any,
    tasks: list[VerifiableTask],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> dict[str, Any]:
    total_probes = 0
    sum_probe_score = 0.0
    full_credit = 0
    json_ok = 0
    stale_total = 0
    trap_total = 0
    fam_scores: dict[str, list[float]] = defaultdict(list)
    total_chars = 0

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        prompts = []
        for t in batch:
            msgs = []
            if t.system_prompt:
                msgs.append({"role": "system", "content": t.system_prompt})
            msgs.append({"role": "user", "content": t.prompt})
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3200).to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for i, t in enumerate(batch):
            resp = tok.decode(outs[i][plen:], skip_special_tokens=True).strip()
            total_chars += len(resp)
            ver = verify_response(t, resp)
            if t.domain == "compactbench_flywheel":
                d = ver.details
                n = int(d.get("probe_count", 0))
                total_probes += n
                sum_probe_score += float(d.get("mean_score", 0.0)) * n
                full_credit += int(d.get("full_credit_probes", 0))
                stale_total += int(d.get("stale_count", 0))
                trap_total += int(d.get("trap_count", 0))
                if ver.format_reward >= 0.99:
                    json_ok += 1
                for fam, sc in (d.get("by_family") or {}).items():
                    fam_scores[fam].append(sc)
            else:
                total_probes += 1
                sum_probe_score += float(ver.reward)
                if ver.passed:
                    full_credit += 1

    mean_score = round(sum_probe_score / max(1, total_probes), 4)
    by_fam = {k: round(sum(v) / len(v), 4) for k, v in sorted(fam_scores.items())}
    return {
        "score": mean_score,
        "full_credit": full_credit,
        "total_probes": total_probes,
        "json_rate": round(json_ok / max(1, len(tasks)), 4),
        "stale": stale_total,
        "trap": trap_total,
        "by_family": by_fam,
        "avg_tokens": round((total_chars / max(1, len(tasks))) / 4.0, 1),
    }


def evaluate_checkpoint_elite(
    model_path: str,
    *,
    generation: int,
    paradigm: str,
    parent_gen: int | None,
    ho_tasks: list[VerifiableTask],
    val_tasks: list[VerifiableTask],
    reg_tasks: list[VerifiableTask],
    device: str,
    dtype: torch.dtype,
    batch_size: int = 2,
) -> tuple[GenerationMetrics, dict[str, Any]]:
    t0 = time.time()
    m, tok = load_model(model_path, device, dtype)
    try:
        ho = eval_cb_suite(m, tok, ho_tasks, max_new_tokens=256, batch_size=batch_size)
        val = eval_cb_suite(m, tok, val_tasks, max_new_tokens=256, batch_size=batch_size)
        reg = eval_cb_suite(m, tok, reg_tasks, max_new_tokens=160, batch_size=min(5, batch_size))
    finally:
        free_memory(m)

    metrics = GenerationMetrics(
        generation=generation,
        paradigm=paradigm,
        parent_generation=parent_gen,
        model_id=model_path,
        checkpoint_path=model_path,
        val_pass_at_1=val["score"],
        held_out_pass_at_1=ho["score"],
        held_out_format_rate=ho["json_rate"],
        domain_scores=ho["by_family"],
        regression_score=reg["score"],
        avg_response_tokens=ho["avg_tokens"],
        wall_time_sec=round(time.time() - t0, 1),
    )
    return metrics, {"ho": ho, "val": val, "reg": reg}


def run_distill_sft(
    base_model: str,
    cb_train: list[VerifiableTask],
    out_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
    use_bf16: bool,
    use_fp16: bool,
    max_examples: int = 120,
    epochs: float = 2.0,
    lr: float = 6e-5,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.time()
    distill_all = build_harness_distill_sft_examples(cb_train, min_teacher_score=0.80, use_verified_teacher=True)
    distill_all.sort(key=lambda ex: (-ex["difficulty"], -ex["reward"]))
    distill_sel = distill_all[:max_examples]
    train_pool = distill_sel + ANCHOR_REPLAY * 3
    random.Random(seed + 1).shuffle(train_pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts, completions = [], []
    for ex in train_pool:
        msgs = ex["messages"]
        prompts.append(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True))
        completions.append(msgs[-1]["content"] + (tok.eos_token or ""))
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

    m, _ = load_model(base_model, device, dtype)
    m.train()
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=DELTANET_SAFE_TARGETS,
    )
    sft_args = SFTConfig(
        output_dir=str(out_dir / "_lora"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed + 1,
        completion_only_loss=True,
        max_length=2560,
    )
    trainer = SFTTrainer(
        model=m, args=sft_args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg,
    )
    res = trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    free_memory(merged)
    free_memory(m)
    return {
        "checkpoint_path": str(out_dir),
        "train_loss": round(float(res.training_loss), 4),
        "num_examples": len(train_pool),
        "wall_time_sec": round(time.time() - t0, 1),
    }


def run_probe_dpo(
    parent_model: str,
    cb_train: list[VerifiableTask],
    out_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
    use_bf16: bool,
    use_fp16: bool,
    sample_pool_size: int = 64,
    lr: float = 2e-5,
    beta: float = 0.12,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.time()
    bottleneck_fams = {"buried_constraint", "conditional_scope", "reference_resolution"}
    bn_train = [
        t for t in cb_train if bottleneck_fams.intersection(set(t.metadata.get("families") or []))
    ]
    random.Random(seed + 2).shuffle(bn_train)
    pool = bn_train[:sample_pool_size]

    # Sample parent model mistakes on P0/P1 bottlenecks
    m, tok = load_model(parent_model, device, dtype)
    dpo_rows: list[dict[str, str]] = []
    try:
        bs = 4 if device == "cuda" else 2
        for start in range(0, len(pool), bs):
            batch = pool[start : start + bs]
            prompts = []
            for t in batch:
                msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.prompt}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3200).to(m.device)
            plen = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outs = m.generate(
                    **inputs, max_new_tokens=256, do_sample=True, temperature=0.65, top_p=0.95,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            for i, t in enumerate(batch):
                resp = tok.decode(outs[i][plen:], skip_special_tokens=True).strip()
                ver_neg = verify_compactbench_batch(t, resp)
                gold_resp = str(t.metadata["gold_response"])
                ver_pos = verify_compactbench_batch(t, gold_resp)
                if ver_pos.reward - ver_neg.reward >= 0.15 and resp:
                    dpo_rows.append({"prompt": prompts[i], "chosen": gold_resp, "rejected": resp})
    finally:
        free_memory(m)

    print(f"Harvested {len(dpo_rows)} on-policy Probe-Verified DPO pairs from {len(pool)} bottleneck batches")
    out_dir.mkdir(parents=True, exist_ok=True)
    m, tok = load_model(parent_model, device, dtype)
    m.train()
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=DELTANET_SAFE_TARGETS,
    )
    dpo_args = DPOConfig(
        output_dir=str(out_dir / "_dpo"),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        beta=beta,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_steps=3,
        save_strategy="no",
        report_to="none",
        seed=seed + 2,
        max_length=2560,
    )
    trainer = DPOTrainer(
        model=m, ref_model=None, args=dpo_args,
        train_dataset=Dataset.from_list(dpo_rows), processing_class=tok, peft_config=peft_cfg,
    )
    res = trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    free_memory(merged)
    free_memory(m)
    return {
        "checkpoint_path": str(out_dir),
        "train_loss": round(float(res.training_loss), 4),
        "num_examples": len(dpo_rows),
        "wall_time_sec": round(time.time() - t0, 1),
    }


def run_4way_ablation(
    raw_model_path: str,
    flywheel_model_path: str,
    work_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
    suite: str = "starter",
    seed_group: str = "default",
    cases_per_family: int = 1,
    budget_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """Run the official `compactbench==0.3.0` 2x2 Dual-Loop Flywheel Ablation Matrix."""
    ablation_dir = work_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    harness_spec = f"{DEFAULT_HARNESS_COMPACTOR}:OpenRSICompactor"
    rows: list[dict[str, Any]] = []

    configs = [
        ("Raw 2B + Truncate\n(No RSI)", raw_model_path, "built-in:truncate-last-n", "cell1_raw_trunc"),
        ("Raw 2B + Harness v2\n(Harness-RSI only)", raw_model_path, harness_spec, "cell2_raw_harness"),
        ("Flywheel 2B + Truncate\n(Model-RSI only)", flywheel_model_path, "built-in:truncate-last-n", "cell3_fw_trunc"),
        ("Flywheel 2B + Harness v2\n(Full Dual-Loop!)", flywheel_model_path, harness_spec, "cell4_dual_loop"),
    ]

    # Group by model path so each 2B model is loaded only once!
    by_model: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for cfg in configs:
        by_model[cfg[1]].append(cfg)

    results_by_tag: dict[str, dict[str, Any]] = {}
    for mpath, cfgs in by_model.items():
        print(f"\n[4-Way Ablation] Loading model: {mpath}")
        m, tok = load_model(mpath, device, dtype)
        provider = HFCompactBenchProvider(
            m, tok, model_name=Path(mpath).name, default_max_new_tokens=768
        )
        try:
            for label, _, method_spec, tag in cfgs:
                print(f"  -> Running {tag} ({method_spec}) on CompactBench `{suite}`...")
                res = run_official_compactbench_eval(
                    provider,
                    method_spec=method_spec,
                    model_label=tag,
                    output_path=ablation_dir / f"{tag}.jsonl",
                    suite=suite,
                    seed_group=seed_group,
                    cases_per_family=cases_per_family,
                    budget_tokens=budget_tokens,
                    concurrency=1,
                )
                res["label"] = label
                res["tag"] = tag
                results_by_tag[tag] = res
                print(
                    f"     official_score={res['official_score']:.2%} "
                    f"({res['full_credit_probes']}/{res['total_probes']}), "
                    f"drift_resistance={res['drift_resistance']:.2%}, "
                    f"stale={res['stale_rate']:.2%}, violations={res['budget_violations']}, "
                    f"time={res['elapsed_sec']}s"
                )
        finally:
            free_memory(m)

    for _, _, _, tag in configs:
        rows.append(results_by_tag[tag])

    out_json = work_dir / "flywheel_ablation_4way.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_flywheel_ablation(rows, work_dir / "flywheel_ablation_4way.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRSI Phase 4 Dual-Loop Flywheel Runner")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--work-dir", default="/tmp/openrsi_phase4_flywheel")
    parser.add_argument("--stage", choices=["all", "phase4a", "phase4b"], default="all")
    parser.add_argument("--flywheel-ckpt", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device, dtype, use_bf16, use_fp16 = detect_runtime()
    print(f"Runtime: device={device}, dtype={dtype}, bf16={use_bf16}, fp16={use_fp16}")

    work = Path(args.work_dir)
    ckpt_root = work / "checkpoints"
    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    cb_train, cb_val, cb_ho = load_flywheel_splits(DEFAULT_DATA_GZ)
    ho_eval = sorted(cb_ho, key=lambda t: (t.metadata["case_id"], t.metadata["cycle"], t.task_id))[::2][:40]
    val_eval = sorted(cb_val, key=lambda t: (t.metadata["case_id"], t.metadata["cycle"], t.task_id))[::4][:20]
    reg_tasks = get_regression_suite()

    eval_bs = 4 if device == "cuda" else 2
    hist_path = reports / "flywheel_evolution_history.json"
    history = load_evolution_history(hist_path)
    policy = GatePolicy()

    best_ckpt = args.flywheel_ckpt or args.base_model

    if args.stage in ("all", "phase4a"):
        if not history:
            print("\n=== Phase 4A Gen 0: Baseline ===")
            base_m, base_d = evaluate_checkpoint_elite(
                args.base_model, generation=0, paradigm="baseline", parent_gen=None,
                ho_tasks=ho_eval, val_tasks=val_eval, reg_tasks=reg_tasks,
                device=device, dtype=dtype, batch_size=eval_bs,
            )
            base_m.status = "baseline"
            base_m.notes = f"Raw {args.base_model} ({base_d['ho']['full_credit']}/{base_d['ho']['total_probes']} full-credit, stale={base_d['ho']['stale']})"
            history.append(base_m)
            save_evolution_history(history, hist_path)
            print(f"Gen 0 -> Held-out={base_m.held_out_pass_at_1:.2%}, Val={base_m.val_pass_at_1:.2%}, Reg={base_m.regression_score:.2%}")

        baseline = history[0]
        promoted_list = [h for h in history if h.status in ("baseline", "promoted")]
        current_best = promoted_list[-1]

        if len(history) < 2:
            print("\n=== Phase 4A Gen 1: Harness -> Model Distillation SFT ===")
            ckpt1 = ckpt_root / "gen1_harness_distill_sft"
            tr1 = run_distill_sft(
                current_best.checkpoint_path, cb_train, ckpt1,
                device=device, dtype=dtype, use_bf16=use_bf16, use_fp16=use_fp16,
                max_examples=120, epochs=2.0, lr=6e-5, seed=args.seed,
            )
            m1, d1_diag = evaluate_checkpoint_elite(
                str(ckpt1), generation=1, paradigm="harness_distill_sft", parent_gen=current_best.generation,
                ho_tasks=ho_eval, val_tasks=val_eval, reg_tasks=reg_tasks,
                device=device, dtype=dtype, batch_size=eval_bs,
            )
            m1.num_train_examples = tr1["num_examples"]
            m1.train_loss = tr1["train_loss"]
            m1.wall_time_sec = round(tr1["wall_time_sec"] + m1.wall_time_sec, 1)
            dec1 = evaluate_promotion_gate(m1, current_best, baseline, policy)
            m1.status = "promoted" if dec1.promoted else "rejected"
            m1.notes = dec1.reason + f" | ({d1_diag['ho']['full_credit']}/{d1_diag['ho']['total_probes']}, stale={d1_diag['ho']['stale']})"
            history.append(m1)
            if dec1.promoted:
                current_best = m1
            save_evolution_history(history, hist_path)
            print("Gen 1 ->", m1.notes)

        if len(history) < 3:
            print("\n=== Phase 4A Gen 2: Probe-Verified Bottleneck DPO ===")
            ckpt2 = ckpt_root / "gen2_probe_verified_dpo"
            tr2 = run_probe_dpo(
                current_best.checkpoint_path, cb_train, ckpt2,
                device=device, dtype=dtype, use_bf16=use_bf16, use_fp16=use_fp16,
                sample_pool_size=64, lr=2e-5, beta=0.12, seed=args.seed,
            )
            m2, d2_diag = evaluate_checkpoint_elite(
                str(ckpt2), generation=2, paradigm="probe_verified_dpo", parent_gen=current_best.generation,
                ho_tasks=ho_eval, val_tasks=val_eval, reg_tasks=reg_tasks,
                device=device, dtype=dtype, batch_size=eval_bs,
            )
            m2.num_train_examples = tr2["num_examples"]
            m2.train_loss = tr2["train_loss"]
            m2.wall_time_sec = round(tr2["wall_time_sec"] + m2.wall_time_sec, 1)
            dec2 = evaluate_promotion_gate(m2, current_best, baseline, policy)
            m2.status = "promoted" if dec2.promoted else "rejected"
            m2.notes = dec2.reason + f" | ({d2_diag['ho']['full_credit']}/{d2_diag['ho']['total_probes']}, stale={d2_diag['ho']['stale']})"
            history.append(m2)
            if dec2.promoted:
                current_best = m2
            save_evolution_history(history, hist_path)
            print("Gen 2 ->", m2.notes)

        plot_evolution_curve(
            history,
            reports / "flywheel_evolution_curve.png",
            title="OpenRSI Phase 4 — Harness → Model Flywheel Evolution (Qwen3.5-2B)",
        )
        best_ckpt = current_best.checkpoint_path

    if args.stage in ("all", "phase4b"):
        print(f"\n=== Phase 4B: Official CompactBench 2x2 Ablation (best={best_ckpt}) ===")
        run_4way_ablation(
            args.base_model,
            best_ckpt,
            reports,
            device=device,
            dtype=dtype,
            suite="starter",
            seed_group="default",
            cases_per_family=1,
            budget_tokens=2048,
        )


if __name__ == "__main__":
    main()
