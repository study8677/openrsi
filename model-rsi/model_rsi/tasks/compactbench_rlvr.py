"""Phase 4 Dual-Loop Flywheel: CompactBench Harness-to-Model Tasks & Official Verifier.

Reads verified CompactBench traces produced by `harness-rsi` (STRICTLY READ-ONLY),
enforces strict case_id-level train/val/held_out isolation, and scores model
completions using the official `compactbench==0.3.0` deterministic `score_probe`.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from compactbench.contracts import Probe, ProbeFormat
from compactbench.jsonutil import load_json_object
from compactbench.runner.reader import (
    DEFAULT_CHUNK_SIZE,
    READER_SYSTEM,
    context_prefix,
    questions_prompt,
)
from compactbench.scoring import NOT_STATED_TOKEN, option_letter, score_probe

from model_rsi.protocol import SplitName, VerifiableTask, VerificationResult


def _probe_from_dict(raw: dict[str, Any]) -> Probe:
    return Probe.model_validate(raw)


def _gold_answer_for_probe(probe: Probe) -> Any:
    """Convert a CompactBench Probe's ground-truth answer into official JSON format."""
    expects_abstention = probe.answer in ([NOT_STATED_TOKEN], [])
    if probe.format == ProbeFormat.SINGLE:
        if expects_abstention:
            for i, opt in enumerate(probe.options):
                if "not stated" in opt.lower():
                    return option_letter(i)
            return NOT_STATED_TOKEN
        target = probe.answer[0] if probe.answer else ""
        for i, opt in enumerate(probe.options):
            if opt.strip().lower() == target.strip().lower():
                return option_letter(i)
        return target
    if probe.format == ProbeFormat.MULTI:
        if expects_abstention:
            return []
        targets = {a.strip().lower() for a in probe.answer}
        letters = [
            option_letter(i)
            for i, opt in enumerate(probe.options)
            if opt.strip().lower() in targets
        ]
        return letters
    if probe.format == ProbeFormat.VALUE:
        if expects_abstention:
            return NOT_STATED_TOKEN
        return probe.answer[0] if probe.answer else NOT_STATED_TOKEN
    # LIST
    if expects_abstention:
        return [NOT_STATED_TOKEN]
    return list(probe.answer)


def verify_compactbench_batch(
    task: VerifiableTask,
    response: str,
) -> VerificationResult:
    """Score a model's JSON response against official `compactbench.scoring.score_probe`."""
    t0 = time.perf_counter()
    probes_spec: list[dict[str, Any]] = task.verifier_spec["probes"]
    qids: list[str] = task.verifier_spec["qids"]

    parsed: dict[str, Any] | None = None
    try:
        obj = load_json_object(response)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        parsed = None

    if parsed is None:
        return VerificationResult(
            passed=False,
            reward=0.0,
            format_reward=0.0,
            correctness_reward=0.0,
            extracted_answer=None,
            reason="Failed to parse valid JSON object for CompactBench probe batch",
            execution_ms=(time.perf_counter() - t0) * 1000.0,
            details={
                "probe_count": len(probes_spec),
                "full_credit_probes": 0,
                "mean_score": 0.0,
                "stale_count": 0,
                "trap_count": 0,
                "by_family": {},
            },
        )

    total_score = 0.0
    full_credit = 0
    stale_count = 0
    trap_count = 0
    fam_scores: dict[str, list[float]] = defaultdict(list)

    for qid, p_dict in zip(qids, probes_spec):
        probe = _probe_from_dict(p_dict)
        raw_ans = parsed.get(qid)
        pr = score_probe(probe, raw_ans)
        total_score += float(pr.score)
        if pr.score >= 0.999:
            full_credit += 1
        if pr.stale:
            stale_count += 1
        if pr.trapped:
            trap_count += 1
        fam_scores[pr.family or "unknown"].append(float(pr.score))

    n = max(1, len(probes_spec))
    mean_score = round(total_score / n, 4)
    by_family = {
        fam: round(sum(vals) / len(vals), 4)
        for fam, vals in sorted(fam_scores.items())
    }
    elapsed = (time.perf_counter() - t0) * 1000.0

    return VerificationResult(
        passed=mean_score >= 0.80,
        reward=mean_score,
        format_reward=1.0,
        correctness_reward=mean_score,
        extracted_answer=json.dumps(parsed, ensure_ascii=False)[:240],
        reason=(
            f"Official CompactBench batch score={mean_score:.4f} "
            f"({full_credit}/{n} full-credit, stale={stale_count}, trap={trap_count})"
        ),
        execution_ms=elapsed,
        details={
            "probe_count": n,
            "full_credit_probes": full_credit,
            "mean_score": mean_score,
            "stale_count": stale_count,
            "trap_count": trap_count,
            "by_family": by_family,
        },
    )


def load_compactbench_flywheel_tasks(
    traces_path: str | Path,
    *,
    train_cases: int = 36,
    val_cases: int = 6,
    held_out_cases: int = 6,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = 42,
) -> list[VerifiableTask]:
    """Load `harness-rsi` CompactBench traces with strict `case_id` split isolation.

    Returns exact official CompactBench Reader tasks (chunked by `chunk_size`)
    paired with verified teacher responses (from Harness-RSI's OpenRSICompactor
    runs) for Harness-to-Model distillation and online RLVR.
    """
    p = Path(traces_path)
    if not p.exists():
        raise FileNotFoundError(f"CompactBench traces file not found: {p}")

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_case[str(row["case_id"])].append(row)

    all_case_ids = sorted(rows_by_case.keys())
    rng = random.Random(seed)
    rng.shuffle(all_case_ids)

    split_map: dict[str, SplitName] = {}
    for cid in all_case_ids[:train_cases]:
        split_map[cid] = "train"
    for cid in all_case_ids[train_cases : train_cases + val_cases]:
        split_map[cid] = "val"
    for cid in all_case_ids[train_cases + val_cases : train_cases + val_cases + held_out_cases]:
        split_map[cid] = "held_out"

    tasks: list[VerifiableTask] = []
    for cid, split in split_map.items():
        case_rows = sorted(rows_by_case[cid], key=lambda r: int(r["cycle"]))
        for row in case_rows:
            cycle = int(row["cycle"])
            visible_context = str(row.get("visible_context") or "")
            raw_probes: list[dict[str, Any]] = list(row.get("probes") or [])
            teacher_answers: dict[str, Any] = dict(row.get("answers") or {})
            probe_results: list[dict[str, Any]] = list(
                (row.get("result") or {}).get("probe_results") or []
            )
            score_by_key = {
                str(pr["key"]): float(pr.get("score", 0.0)) for pr in probe_results
            }

            ctx_prefix = context_prefix(visible_context)
            for b_idx, start in enumerate(range(0, len(raw_probes), chunk_size)):
                chunk = raw_probes[start : start + chunk_size]
                if not chunk:
                    continue
                qids = [f"Q{i + 1}" for i in range(len(chunk))]
                probe_objs = [_probe_from_dict(pd) for pd in chunk]
                batch_pairs = list(zip(qids, probe_objs))
                prompt_text = ctx_prefix + questions_prompt(batch_pairs)

                # Build teacher JSON (from Harness-RSI run) & exact gold JSON
                teacher_json_obj: dict[str, Any] = {}
                gold_json_obj: dict[str, Any] = {}
                teacher_scores: list[float] = []
                families: list[str] = []

                for qid, pd, pobj in zip(qids, chunk, probe_objs):
                    key = str(pd["key"])
                    teacher_json_obj[qid] = teacher_answers.get(key)
                    gold_json_obj[qid] = _gold_answer_for_probe(pobj)
                    teacher_scores.append(score_by_key.get(key, 0.0))
                    families.append(str(pd.get("family") or "unknown"))

                teacher_batch_score = (
                    sum(teacher_scores) / len(teacher_scores) if teacher_scores else 0.0
                )
                # Higher difficulty for later drift cycles & bottleneck families
                has_bottleneck = any(
                    fam in {"reference_resolution", "buried_constraint", "conditional_scope"}
                    for fam in families
                )
                diff = min(5, 1 + cycle + (1 if has_bottleneck else 0))

                tasks.append(
                    VerifiableTask(
                        task_id=f"cb_{split}_{cid[-8:]}_c{cycle}_b{b_idx}",
                        domain="compactbench_flywheel",
                        split=split,
                        difficulty=diff,
                        system_prompt=READER_SYSTEM,
                        prompt=prompt_text,
                        verifier_spec={
                            "qids": qids,
                            "probes": chunk,
                            "case_id": cid,
                            "cycle": cycle,
                        },
                        metadata={
                            "source": "harness_rsi_compactbench_traces",
                            "case_id": cid,
                            "cycle": cycle,
                            "teacher_score": round(teacher_batch_score, 4),
                            "teacher_response": json.dumps(
                                teacher_json_obj, indent=2, ensure_ascii=False
                            ),
                            "gold_response": json.dumps(
                                gold_json_obj, indent=2, ensure_ascii=False
                            ),
                            "families": sorted(set(families)),
                        },
                    )
                )

    return tasks


def build_harness_distill_sft_examples(
    train_tasks: list[VerifiableTask],
    *,
    min_teacher_score: float = 0.85,
    use_verified_teacher: bool = True,
) -> list[dict[str, Any]]:
    """Build Harness-to-Model distillation SFT examples from verified CompactBench traces.

    Only retains train batches where the Harness-RSI teacher achieved
    `teacher_score >= min_teacher_score` (and verifies the response through
    `verify_compactbench_batch` before inclusion!).
    """
    examples: list[dict[str, Any]] = []
    for task in train_tasks:
        if task.split != "train" or task.domain != "compactbench_flywheel":
            continue
        t_score = float(task.metadata.get("teacher_score", 0.0))
        if t_score < min_teacher_score:
            continue
        target_resp = (
            str(task.metadata["teacher_response"])
            if use_verified_teacher
            else str(task.metadata["gold_response"])
        )
        ver = verify_compactbench_batch(task, target_resp)
        if ver.reward < min_teacher_score:
            continue

        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": target_resp},
        ]
        examples.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "reward": ver.reward,
                "source": "harness_rsi",
                "messages": messages,
            }
        )
    return examples
