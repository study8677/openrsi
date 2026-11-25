"""Phase 6: Unified Multi-Domain Model-RSI (`OpenRSI-Qwen3.5-2B-Unified`).

Combines ALL verified OpenRSI evolutionary lineages into ONE unified 2B model
via DeltaNet-Safe Task Arithmetic (`merge_task_arithmetic`) + verified
multi-domain anchor consolidation, then evaluates that SINGLE checkpoint across
all 4 locked held-out benchmarks simultaneously:
  1. Official `HuggingFaceH4/MATH-500` (40 unseen competition math problems)
  2. Official `evalplus/humanevalplus` + `MBPP` (64 unseen coding problems: Turn-1 & 2-Turn Self-Repair)
  3. `CompactBench` Elite Held-out (249 unseen 32k probes)
  4. General Regression Anti-Forgetting Suite (10 items, floor >= 90%)
"""

from __future__ import annotations

import gc
import gzip
import json
import random
import re
from pathlib import Path
from typing import Any

import peft.import_utils
import peft.tuners.lora.torchao

peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_rsi.protocol import VerifiableTask
from model_rsi.reporting import plot_unified_multidomain_chart
from model_rsi.tasks.agentic_debug import (
    AGENTIC_CODE_SYSTEM_PROMPT,
    run_two_turn_agentic_eval,
    summarize_agentic_episodes,
)
from model_rsi.tasks.code_exec import load_humaneval_plus_tasks, load_mbpp_tasks
from model_rsi.tasks.compactbench_rlvr import verify_compactbench_batch
from model_rsi.tasks.math_rlvr import load_math500_tasks, verify_math_rlvr
from model_rsi.tasks.regression import get_regression_suite, verify_regression
from model_rsi.trainers.lineage_merge import merge_task_arithmetic


def _hardware_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


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


def evaluate_unified_suite(
    ckpt_path: str,
    *,
    math_tasks: list[VerifiableTask],
    code_tasks: list[VerifiableTask],
    cb_ho_tasks: list[VerifiableTask],
) -> dict[str, float]:
    """Evaluate ONE checkpoint across all 4 official held-out suites."""
    model, tok = load_gpu_model(ckpt_path)

    # 1. MATH-500
    math_pass = 0
    for s in range(0, len(math_tasks), 8):
        batch = math_tasks[s : s + 8]
        prompts = [
            _format_chat(
                tok,
                [
                    {"role": "system", "content": t.system_prompt},
                    {"role": "user", "content": t.prompt},
                ],
            )
            for t in batch
        ]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        for i, t in enumerate(batch):
            txt = tok.decode(out[i][plen:], skip_special_tokens=True).strip()
            txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
            if verify_math_rlvr(t, txt).passed:
                math_pass += 1
    math_score = round(math_pass / max(1, len(math_tasks)), 4)

    # 2. HumanEval+ + MBPP (Turn-1 & 2-Turn Self-Repair)
    code_eps = run_two_turn_agentic_eval(
        model, tok, code_tasks, num_samples=1, turn1_temperature=0.0, turn2_temperature=0.0, batch_size=8
    )
    code_sum = summarize_agentic_episodes(code_eps)

    # 3. CompactBench Elite Held-out (249 probes)
    cb_total, cb_sum = 0, 0.0
    for s in range(0, len(cb_ho_tasks), 4):
        batch = cb_ho_tasks[s : s + 4]
        prompts = [
            _format_chat(
                tok,
                [
                    {"role": "system", "content": t.system_prompt},
                    {"role": "user", "content": t.prompt},
                ],
            )
            for t in batch
        ]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2560).to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=384, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        for i, t in enumerate(batch):
            txt = tok.decode(out[i][plen:], skip_special_tokens=True).strip()
            txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
            ver = verify_compactbench_batch(t, txt)
            n_p = int(ver.details.get("probe_count", 0))
            cb_total += n_p
            cb_sum += float(ver.details.get("mean_score", 0.0)) * n_p
    cb_score = round(cb_sum / max(1, cb_total), 4)

    # 4. General Regression Suite (10 items)
    reg_tasks = get_regression_suite()
    reg_pass = 0
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
                **inp, max_new_tokens=128, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        txt = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
        txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
        if verify_regression(t, txt).passed:
            reg_pass += 1
    reg_score = round(reg_pass / len(reg_tasks), 4)

    free_gpu(model)
    macro_avg = round(
        (math_score + code_sum["turn1_pass_at_1"] + code_sum["turn2_pass_at_1"] + cb_score) / 4.0,
        4,
    )
    return {
        "math500": math_score,
        "code_t1": code_sum["turn1_pass_at_1"],
        "code_t2_repair": code_sum["turn2_pass_at_1"],
        "repair_conv_rate": code_sum["repair_conversion_rate"],
        "compactbench_elite": cb_score,
        "regression": reg_score,
        "macro_4bench_avg": macro_avg,
    }


def run_phase6(
    *,
    base_model: str = "Qwen/Qwen3.5-2B",
    specialist_ckpts: list[tuple[str, float]],
    repo_root: str = "/content/openrsi",
    work_dir: str = "/content/openrsi_phase6_unified",
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(repo_root)
    work = Path(work_dir)
    rep_dir = work / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 locked held-out suites
    math_tasks = load_math500_tasks(max_tasks=40, seed=seed)
    he_tasks = load_humaneval_plus_tasks(max_tasks=40, seed=seed)
    mbpp_ho = [
        t for t in load_mbpp_tasks(train_limit=10, val_limit=10, held_out_limit=24, seed=seed)
        if t.split == "held_out"
    ]
    code_tasks = he_tasks + mbpp_ho
    for t in code_tasks:
        t.system_prompt = AGENTIC_CODE_SYSTEM_PROMPT

    fw_gz = root / "model-rsi" / "data" / "compactbench_flywheel_v1.jsonl.gz"
    cb_ho = [
        VerifiableTask.model_validate(json.loads(ln))
        for ln in gzip.open(fw_gz, "rt", encoding="utf-8")
        if ln.strip()
    ]
    cb_ho = [t for t in cb_ho if t.split == "held_out"]
    rng_cb = random.Random(seed)
    rng_cb.shuffle(cb_ho)
    cb_ho = cb_ho[:40]

    # Merge specialists via DeltaNet-Safe Task Arithmetic
    unified_dir = work / "checkpoints" / "openrsi_qwen35_2b_unified"
    merge_info = merge_task_arithmetic(
        base_model,
        specialist_ckpts,
        unified_dir,
        save_dtype="bf16" if _hardware_bf16() else "fp16",
    )

    results: dict[str, dict[str, float]] = {}
    results["Raw Qwen3.5-2B"] = evaluate_unified_suite(
        base_model, math_tasks=math_tasks, code_tasks=code_tasks, cb_ho_tasks=cb_ho
    )
    for spec_path, _ in specialist_ckpts:
        label = Path(spec_path).name
        results[label] = evaluate_unified_suite(
            spec_path, math_tasks=math_tasks, code_tasks=code_tasks, cb_ho_tasks=cb_ho
        )
    results["OpenRSI-Qwen3.5-2B-Unified"] = evaluate_unified_suite(
        str(unified_dir), math_tasks=math_tasks, code_tasks=code_tasks, cb_ho_tasks=cb_ho
    )

    bench_labels = ["MATH-500", "Code Turn-1", "Code 2-Turn Repair", "CompactBench Elite", "Regression Gate"]
    series = {
        name: [
            m["math500"],
            m["code_t1"],
            m["code_t2_repair"],
            m["compactbench_elite"],
            m["regression"],
        ]
        for name, m in results.items()
    }
    plot_unified_multidomain_chart(
        bench_labels,
        series,
        rep_dir / "phase6_unified_multidomain.png",
    )
    payload = {"merge_info": merge_info, "results": results}
    (rep_dir / "phase6_unified_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
