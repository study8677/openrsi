"""Lineage Archive Weight-Space Merging (Model Soup) for OpenRSI Model-RSI.

Implements OpenRSI Principle #4 (Lineage Archive Evolution): combines multiple
promoted checkpoints from different evolutionary niches (e.g. an RLVR/GRPO
coding specialist and an Iterative-DPO / STaR reasoning specialist) via
weighted parameter interpolation, then submits the merged candidate to the
immutable Held-out & Regression Promotion Gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_lineage_checkpoints(
    checkpoint_weights: list[tuple[str, float]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Interpolate model weights across promoted lineage checkpoints (Model Soup)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not checkpoint_weights:
        raise ValueError("checkpoint_weights must be non-empty")

    total_w = sum(w for _, w in checkpoint_weights)
    if total_w <= 0:
        raise ValueError("Total merge weight must be positive")
    norm_pairs = [(p, w / total_w) for p, w in checkpoint_weights]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load first checkpoint as accumulator on CPU to avoid GPU OOM
    base_path, w0 = norm_pairs[0]
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    acc_model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    acc_state = acc_model.state_dict()
    for k in acc_state:
        acc_state[k].mul_(w0)

    for ckpt_path, weight in norm_pairs[1:]:
        other = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
        other_state = other.state_dict()
        for k in acc_state:
            acc_state[k].add_(other_state[k], alpha=weight)
        del other, other_state

    # Cast back to bf16 and save
    for k in acc_state:
        acc_state[k] = acc_state[k].to(torch.bfloat16)
    acc_model.load_state_dict(acc_state)
    acc_model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    del acc_model, acc_state

    return {
        "checkpoint_path": str(out_path),
        "parents": [{"path": p, "weight": round(w, 4)} for p, w in norm_pairs],
    }


def merge_task_arithmetic(
    base_model_path: str,
    specialist_deltas: list[tuple[str, float]],
    output_dir: str | Path,
    *,
    save_dtype: str = "bf16",
) -> dict[str, Any]:
    """DeltaNet-safe Task Arithmetic merge: `theta_0 + sum(alpha_i * (theta_i - theta_0))`.

    Preserves untouched Gated DeltaNet recurrence layers (`in_proj_a`, `in_proj_b`)
    bit-identically to `base_model_path` while combining multiple specialized RSI
    lineages (e.g., Math+Code/Agentic-Debug + CompactBench Flywheel) without 0.5x
    weight dilution.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not specialist_deltas:
        raise ValueError("specialist_deltas must be non-empty")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    merged_state = {k: v.clone() for k, v in base_state.items()}

    for ckpt_path, alpha in specialist_deltas:
        spec_model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
        spec_state = spec_model.state_dict()
        for k in merged_state:
            if k in spec_state and merged_state[k].is_floating_point():
                delta = spec_state[k] - base_state[k]
                merged_state[k].add_(delta, alpha=float(alpha))
        del spec_model, spec_state

    target_dtype = torch.bfloat16 if save_dtype == "bf16" else torch.float16
    for k in merged_state:
        if merged_state[k].is_floating_point():
            merged_state[k] = merged_state[k].to(target_dtype)

    base_model.load_state_dict(merged_state)
    base_model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    del base_model, base_state, merged_state

    return {
        "checkpoint_path": str(out_path),
        "base_model": base_model_path,
        "method": "task_arithmetic",
        "specialists": [{"path": p, "alpha": round(float(a), 4)} for p, a in specialist_deltas],
    }
