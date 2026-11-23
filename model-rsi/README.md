# Model-RSI

**English** | [中文](./README_zh.md) | [← Back to OpenRSI](../README.md)

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

## Experimental Results (Colab NVIDIA L4 GPU, `Qwen/Qwen2.5-1.5B-Instruct`)

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
uv run pytest -v                        # Run deterministic verifier, curriculum, gating & flywheel tests
uv run model-rsi status                 # Inspect multi-generation evolution ledger
uv run model-rsi plot                   # Render reports/evolution_curve.png
uv run model-rsi generate-tasks         # Export disjoint train/val/held_out/regression JSONL splits
uv run model-rsi import-harness         # Import verified Harness-RSI traces (M4 flywheel)
```

For full GPU post-training (STaR SFT, Iterative DPO, Online GRPO/RLVR, and Lineage Model Soup) on Google Colab (L4 / A100 / H100), install `requirements-colab.txt` and run `colab/run_model_rsi_colab.py`.

---

## Roadmap

- [x] **M1 — Unified Trajectory & Verifier Schema**: Standardize how verified rollouts (from direct model sampling or `harness-rsi`) are stored, scored, and filtered (`model_rsi/protocol.py`, `model_rsi/tasks/`).
- [x] **M2 — Verified SFT / Rejection Sampling Loop**: End-to-end iterative STaR (completion-only LoRA SFT) & Iterative DPO pipelines with automated held-out & regression gating (`model_rsi/trainers/star_sft.py`, `iterative_dpo.py`, `gating.py`).
- [x] **M3 — RLVR (GRPO) Integration**: Pluggable sandboxed online GRPO training loop with anti-reward-hacking pure-binary verifiable rewards (`model_rsi/trainers/grpo_rlvr.py`).
- [x] **M4 — Frontier Curriculum & Dual-Loop Flywheel**: Frontier mastery scheduler (`model_rsi/curriculum.py`), lineage weight-space merging (`model_rsi/trainers/lineage_merge.py`), and closed-loop distillation from [`harness-rsi`](../harness-rsi/README.md) (`model_rsi/flywheel.py`).
