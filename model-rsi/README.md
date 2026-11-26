# Model-RSI

**English** | [中文](./README_zh.md) | [Technical Report (ZH)](./docs/TECHNICAL_REPORT_zh.md) | [← Back to OpenRSI](../README.md)

**Model-RSI** improves the AI model's *internal weights* across successive generations. Instead of relying solely on human-labeled datasets, the model generates its own solutions and tasks, filters or scores them through automated verifiers in sandboxes, trains on the resulting signal, and promotes a stronger checkpoint only after passing strict held-out gates.

---

## Core Architecture

```mermaid
flowchart LR
    M["Current Policy<br/>(Checkpoint N)"] --> R["Sandbox Rollouts &<br/>Task Exploration"]
    H["Harness-RSI<br/>Verified Traces"] -. "Optional distillation" .-> V
    R --> V["Automated Verifier<br/>(Tests / Execution / Formal Checks)"]
    V --> D["Training Signal<br/>(Verified SFT / Preferences / Rewards)"]
    D --> T["Post-Training<br/>(SFT / DPO / GRPO / RLVR)"]
    T --> C["Candidate Checkpoint<br/>(N + 1)"]
    C --> G{"Held-Out Eval &<br/>Regression Gate"}
    G -->|Pass| M
    G -->|Fail| X["Discard & Update<br/>Curriculum / Hyperparams"]
    X -.-> R
```

### Key Components

1. **Rollout & Self-Play Engine**: Samples diverse trajectories from the current checkpoint (optionally amplified by a strong harness or test-time search).
2. **Automated Verifier**: Produces objective, ground-truth rewards—such as unit test pass/fail, compilation success, mathematical equivalence, or sandbox execution checks.
3. **Data & Curriculum Manager**: Balances difficulty (targeting the frontier where pass rate is neither 0% nor 100%), deduplicates trajectories, and mixes anchor data to prevent forgetting.
4. **Post-Training Runner**: Pluggable training backend supporting LoRA or full-parameter updates via SFT, DPO, or online RL (GRPO / PPO / RLVR).
5. **Promotion Gate**: Evaluates candidate checkpoints on unseen held-out benchmarks plus general capability anchors before promoting Checkpoint `N + 1`.

---

## Core Evolution Paradigms

| Paradigm | How It Works | Trade-offs |
| :--- | :--- | :--- |
| **Rejection Sampling / STaR (Verified SFT)** | Sample many solutions per problem, keep only verifier-approved traces, fine-tune on them | Simple, stable, works well with small compute; weaker on learning from negative examples |
| **RLVR (GRPO / PPO)** | Optimize policy directly against verifiable sandbox rewards over grouped rollouts | Strongest ceiling on reasoning & coding; requires careful reward design and RL stability |
| **Preference Optimization (DPO / Iterative DPO)** | Pair verified winning traces against failing traces from the same prompt | Uses negative signal cleanly without full online RL complexity |
| **Harness-to-Model Distillation** | Run expensive `harness-rsi` search/tooling to solve hard tasks, then distill the final verified solution into model weights | Turns slow inference-time compute into fast single-pass model intuition |
| **Self-Play & Frontier Curriculum** | Co-evolve a task generator and solver so training tasks stay just above the model's current skill level | Prevents dataset saturation; hardest to keep grounded and non-degenerate |

---

## Key Risks & Guardrails

Recursive training on model-generated data introduces distinct failure modes that must be guarded against:

- **Model Collapse & Diversity Loss**: Training purely on self-generated text can narrow the distribution. Mitigate with strict external verifiers, temperature/search diversity, and mixing verified anchor datasets.
- **Reward Hacking**: Models may exploit weak verifiers (e.g., trivializing tests or gaming heuristics). Keep verifiers deterministic, sandboxed, and audited against held-out checks.
- **Catastrophic Forgetting**: Narrow domain self-improvement can degrade general skills. Always include a general regression suite in the promotion gate.
- **Benchmark Contamination**: Strictly separate rollout task pools from held-out evaluation benchmarks.

---

## Experimental Results (Colab NVIDIA GPU)

### Run 3 & 4 (Phase 4 & 4.2 Dual-Loop Flywheel + Co-Evolution): `Harness-RSI ↔ Model-RSI` on Official `CompactBench`

Closed the full Dual-Loop Flywheel using `harness-rsi`'s 48 Elite cases (4,659 official deterministic probes, strictly split 36/6/6 by `case_id` with zero leakage, auditable via `uv run model-rsi audit-splits`) and distilled both **Reader QA** and **`OpenRSICompactor v0.2.4` Atomic Patch Protocol** (`A | H | KIND:` / `R n | H | KIND:` / `NO CHANGE` + Audit Pass) into `Qwen/Qwen3.5-2B` (DeltaNet-Safe), then evaluated back inside official `compactbench==0.3.0` via [HFCompactBenchProvider](./model_rsi/flywheel_provider.py):

![Phase 4A & 4.2 Flywheel Evolution Curve](./reports/flywheel_evolution_curve.png)
![Phase 4B & 4.2 Official CompactBench Ablation](./reports/flywheel_ablation_4way.png)

- **Phase 4A (`Harness → Model`, 249 unseen 32k Elite probes)**: `53.15% (Gen 0) → 56.97% (Gen 1 Distill) → 59.28% (Gen 3 Hard-Neg STaR)` (**`+6.13%`**, full-credit `127/249 → 141/249`, JSON `100%`, regression locked at `90.00%`; P0 `reference_resolution` **`52.78% → 67.50%` (+14.72%)**, P1 `conditional_scope` **`32.81% → 46.25%` (+13.44%)**).
- **Phase 4.2 (`Compactor + Reader` Co-Evolution, Gen 4)**: Added 322 atomically verified Compactor Diff + Audit examples ([`compactbench_patch.py`](./model_rsi/tasks/compactbench_patch.py)) + 743 `buried_constraint` precision-calibrated batches from the same 36 train cases:
  - **Compactor Atomic Patch Validity (`compactor_valid_patch_rate`)**: surged from **`0.00%`** (Raw 2B / Gen 3 regex fallback) to **`90.62%` (29/32 real atomic patches accepted by `_apply_delta`)**!
  - **Official `starter/default` (102 probes) Cell 3 (`Gen 4 + truncate-last-n`)**: **`56.20% → 58.26% → 61.55%` (`+5.35%` vs Raw 2B)**, drift resistance **`54.17% → 64.87%` (`+10.70%`)**!
  - **Official `starter/default` Cell 4 (`Gen 4 + OpenRSICompactor v0.2.4`)**: **`45.95% → 49.65% → 52.78%` (`+6.83%` vs Raw 2B + Harness)**, stale-memory rate slashed **`-20.00%`** (`53.33% → 33.33%`)!
  - **Locked 32k Elite Held-out (260 probes, `max_length=3584`)**: single-stage from raw `Qwen3.5-2B` reached **`58.44%` (`+5.29%`, JSON `100%`, Regression `90.00%`)**, boosting bottleneck **`buried_constraint` from `18.23% → 26.00%` (`+7.77%`)**, `decision_override` to **`84.48%`**, and `provenance_authority` to **`84.78%`**.
  - Full analysis in [`reports/phase4_flywheel_report.md`](./reports/phase4_flywheel_report.md).

### Run 2 (Flagship): `Qwen/Qwen3.5-2B` on Official Open-Source RLVR Benchmarks (`MATH-500` + `HumanEval+`)

Trained on `open-r1/Big-Math-RL-Verified-Processed` (L1–L2) + `KodCode/KodCode-V1` + `MBPP`, gated on locked external **`HuggingFaceH4/MATH-500`** (40 tasks) + **`evalplus/humanevalplus`** (40 tasks) + 10 general regression checks:

![Qwen3.5-2B Model-RSI Evolution Curve](./reports/qwen35_2b_evolution_curve.png)

| Gen | Paradigm | Gate Status | Held-out Pass@1 | MATH-500 (HO) | HumanEval+ (HO) | Hard Val | Regression |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | `baseline` (`Qwen/Qwen3.5-2B`) | baseline | 42.50% | 30.00% | 55.00% | 10.00% | 90.00% |
| 1 | `pure_rlvr_grpo` (`all-linear`) | rejected | 36.25% | 25.00% | 47.50% | 12.50% | 90.00% |
| 2 | `deltanet_safe_dpo` (freeze `in_proj_a/b`) | **promoted** | 46.25% | 32.50% | **60.00%** | 10.00% | 90.00% |
| 3 | `deltanet_safe_grpo` (strict frontier) | **promoted** | **48.75%** | **37.50%** | **60.00%** | **15.00%** | **90.00%** |

Key finding: freezing Qwen3.5's sensitive Gated DeltaNet recurrence gates (`in_proj_a`/`in_proj_b`) and running DeltaNet-safe Iterative DPO → Strict-Frontier Pure-Binary RLVR GRPO boosted official Held-out Pass@1 by **`+6.25%`** (`42.50% → 48.75%`, with `MATH-500` `+7.50%` and `HumanEval+` `+5.00%`) at **`90.00%` regression (zero forgetting)**. See [`reports/qwen35_2b_experiment_summary.md`](./reports/qwen35_2b_experiment_summary.md).

### Run 1: `Qwen/Qwen2.5-1.5B-Instruct` (8-Gen Countdown + MBPP)

We ran an end-to-end 8-generation (`Gen 0 → Gen 7`) Model-RSI loop on Google Colab L4 GPU across strictly disjoint splits of **Countdown Arithmetic Reasoning** (deterministic AST + exact `Fraction` verifier), **MBPP Sanitized Python Coding** (isolated `python -I` subprocess sandbox), and a **10-item General Regression Gate**:

![Model-RSI Evolution Curve](./reports/evolution_curve.png)

| Gen | Paradigm | Gate Status | Held-out Pass@1 | Val Pass@1 | MBPP Code (HO) | Countdown L1 (HO) | Regression |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | `baseline` | baseline | 15.00% | 18.75% | 50.00% | 0.00% | 90.00% |
| 1 | `star_sft` (full-seq) | rejected | 15.00% | 18.75% | 37.50% | 12.50% | 90.00% |
| 2 | `iterative_dpo` | **promoted** | 17.50% | 22.92% | 54.17% | 6.25% | 90.00% |
| 3 | `grpo_rlvr` | **promoted** | 17.50% | 25.00% | **58.33%** | 0.00% | 90.00% |
| 4 | `star_sft_comp_only` | **promoted** | **18.75%** | 20.83% | 45.83% | **18.75%** | 80.00% |
| 5 | `frontier_dpo_v2` | **promoted** | **18.75%** | **31.25%** | 54.17% | 12.50% | **90.00%** |
| 6 | `lineage_soup` | rejected | 17.50% | 25.00% | 50.00% | 12.50% | 90.00% |
| 7 | `pure_rlvr_grpo_v2` | rejected | 17.50% | 25.00% | 50.00% | 12.50% | 90.00% |

Full diagnostics (including the Gen 3 length-collapse reward-hacking diagnosis and pure-binary RLVR cure) are archived in [`reports/experiment_summary.md`](./reports/experiment_summary.md) and [`reports/evolution_history.json`](./reports/evolution_history.json).

---

## Quickstart

```bash
cd model-rsi
uv run pytest -v                        # Run 11 deterministic verifier, curriculum, gating, patch & agentic tests
uv run model-rsi audit-splits           # Verify zero case_id leakage across train / val / held_out / starter
uv run model-rsi export-patch-dataset   # Rebuild verified Compactor patch distillation dataset (322 examples)
uv run model-rsi status                 # Inspect multi-generation evolution ledger
uv run model-rsi plot                   # Render reports/evolution_curve.png
uv run model-rsi generate-tasks         # Export disjoint train/val/held_out/regression JSONL splits
uv run model-rsi import-harness         # Import verified Harness-RSI traces (M4 flywheel)
```

For full GPU post-training on Google Colab (T4 / L4 / A100), use our end-to-end runners:
- `colab/run_model_rsi_colab.py` — Multi-gen Countdown / Math / Code RLVR
- `colab/run_phase4_flywheel.py` & `colab/run_phase42_coevolution.py` — Dual-Loop Flywheel + Compactor/Reader Co-Evolution
- `colab/run_phase5_agentic_debug.py` — Multi-Turn Agentic Self-Repair RLVR (`Write → Sandbox Tests → Traceback → Turn-2 Repair`)
- `colab/run_phase6_unified_rsi.py` — DeltaNet-Safe Task Arithmetic Unified Multi-Domain Model-RSI

---

## Roadmap

- [x] **M1 — Unified Trajectory & Verifier Schema**: Standardize how verified rollouts (from direct model sampling or `harness-rsi`) are stored, scored, and filtered (`model_rsi/protocol.py`, `model_rsi/tasks/`).
- [x] **M2 — Verified SFT / Rejection Sampling Loop**: End-to-end iterative STaR (completion-only LoRA SFT) & Iterative DPO pipelines with automated held-out & regression gating (`model_rsi/trainers/star_sft.py`, `iterative_dpo.py`, `gating.py`).
- [x] **M3 — RLVR (GRPO) Integration**: Pluggable sandboxed online GRPO training loop with anti-reward-hacking pure-binary verifiable rewards (`model_rsi/trainers/grpo_rlvr.py`).
- [x] **M4 — Frontier Curriculum & Dual-Loop Flywheel (Phase 4 & 4.2)**: Frontier mastery scheduler (`model_rsi/curriculum.py`), Compactor+Reader co-evolution (`compactbench_patch.py`, `90.62%` atomic patch validity), official `HFCompactBenchProvider` 2×2 ablation, and closed-loop distillation from [`harness-rsi`](../harness-rsi/README.md).
- [x] **M5 — Multi-Turn Agentic Self-Repair & Unified Task Arithmetic Infrastructure (Phase 5 & 6)**: 2-turn sandbox debugging & traceback self-repair RLVR engine (`model_rsi/tasks/agentic_debug.py`), DeltaNet-Safe Task Arithmetic merger (`merge_task_arithmetic`), and reproducible Colab GPU runners.
