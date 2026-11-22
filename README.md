# OpenRSI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/study8677/openrsi?style=social)](https://github.com/study8677/openrsi/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/study8677/openrsi/pulls)

**English** | [中文](./README_zh.md)

> An open-source **Recursive Self-Improvement (RSI)** engine — let your AI Agent rewrite its own code, verify in sandboxes, and evolve automatically.

OpenRSI explores how AI systems can improve themselves systematically, verifiably, and safely. Instead of treating "self-improvement" as a single vague concept, OpenRSI separates it into two complementary tracks:

1. **[Harness-RSI](./harness-rsi/README.md)** — Evolving everything *around* the model (agent code, prompts, tools, memory, workflows, and search strategies) while keeping model weights frozen.
2. **[Model-RSI](./model-rsi/README.md)** — Evolving the model's *internal weights* through self-generated trajectories, verifiable rewards, and automated post-training loops.

---

## Architecture Overview

Both tracks share the same core scientific loop: **Propose → Execute in Sandbox → Verify & Score → Gate & Archive**. Together, they form a dual-loop flywheel:

```mermaid
flowchart TB
    subgraph Outer["Outer Loop: Harness-RSI (Fast, Low-Cost)"]
        H1["Current Harness<br/>(Code / Prompts / Tools / Workflows)"] --> H2["Diagnose Failures &<br/>Propose Patch"]
        H2 --> H3["Isolated Sandbox<br/>Run & Evaluate"]
        H3 --> H4{"Score Improved &<br/>Checks Pass?"}
        H4 -- Yes --> H5["Commit & Archive<br/>New Lineage"]
        H4 -- No --> H6["Rollback & Log<br/>Failure Trace"]
        H5 --> H1
        H6 --> H1
    end

    subgraph Inner["Inner Loop: Model-RSI (Deep, High-Ceiling)"]
        M1["Current Model Weights"] --> M2["Rollout & Self-Play<br/>in Sandbox"]
        M2 --> M3["Automated Verifier<br/>(Tests / Execution / Rubrics)"]
        M3 --> M4["Post-Training<br/>(SFT / DPO / GRPO / RLVR)"]
        M4 --> M5{"Held-Out Eval<br/>Gate Pass?"}
        M5 -- Yes --> M6["Promote Checkpoint"]
        M5 -- No --> M7["Discard & Adjust<br/>Curriculum"]
        M6 --> M1
        M7 --> M1
    end

    H5 -. "Verified trajectories &<br/>distillation data" .-> M3
    M6 -. "Stronger base model<br/>proposes better patches" .-> H2
```

### Why the Dual-Loop Flywheel Matters

- **Harness-RSI feeds Model-RSI**: A stronger harness (better search, verification, and tool use) solves harder problems and produces high-signal, verified trajectories that can be distilled into model weights.
- **Model-RSI feeds Harness-RSI**: A stronger base model diagnoses subtle harness bugs more accurately and writes higher-quality code/prompt mutations in fewer iterations.

---

## Harness-RSI vs. Model-RSI

| Dimension | [Harness-RSI](./harness-rsi/README.md) | [Model-RSI](./model-rsi/README.md) |
| :--- | :--- | :--- |
| **What Evolves** | Agent source code, prompts, tools, memory, routing, search & planning logic | Neural network parameters (LoRA / full weights) |
| **Model Weights** | Frozen (works with closed APIs or open weights) | Updated via automated training loops (requires open weights) |
| **Iteration Cycle** | Minutes to hours | Hours to days |
| **Compute & Cost** | Low — API / inference cost + lightweight sandboxes | High — GPU cluster for rollouts + RL/SFT training |
| **Primary Methods** | Code self-editing, prompt optimization, tool synthesis, evolutionary archive (e.g., MAP-Elites) | Rejection sampling (STaR), RLVR (GRPO/PPO), DPO, self-play curriculum |
| **Rollback & Auditability** | Trivial — `git revert` and human-readable diffs | Heavier — checkpoint versioning and benchmark regression suites |
| **Main Failure Modes** | Overfitting to benchmark tests, reward hacking, brittle patches | Model collapse, catastrophic forgetting, reward hacking |
| **Best Starting Point** | **Start here** — fast feedback, transparent diffs, immediate utility | Build next — once verified tasks and harness trajectories exist |

---

## Design Principles

1. **Verifiable Progress Over Vibes**: Every self-modification must be judged by deterministic execution, unit tests, or strictly isolated evaluation suites—never by an ungrounded self-score alone.
2. **Immutable Evaluator, Mutable Target**: The agent is allowed to edit the target harness or generate training data, never the grading harness or held-out test set.
3. **Sandbox-First Safety**: All candidate code changes and model rollouts execute inside isolated sandboxes with strict resource, network, and time limits.
4. **Lineage & Archive Over Blind Greedy Hill-Climbing**: Keep a versioned tree of candidates so evolution can branch, backtrack, and preserve diverse solutions rather than collapsing into a local optimum.
5. **Reproducible & Open**: Every evolution run should emit inspectable traces, diffs, scores, and seeds.

---

## Repository Structure

```text
openrsi/
├── README.md               # Project overview, architecture, comparison & roadmap (EN)
├── README_zh.md            # Project overview (ZH)
├── harness-rsi/
│   ├── README.md           # Harness-RSI architecture, evolvable surface & roadmap (EN)
│   └── README_zh.md        # Harness-RSI documentation (ZH)
└── model-rsi/
    ├── README.md           # Model-RSI architecture, training paradigms & roadmap (EN)
    └── README_zh.md        # Model-RSI documentation (ZH)
```

---

## Roadmap

- **Phase 0 — Specification & Architecture (Current)**
  - Define the dual-track `harness-rsi` and `model-rsi` design, interfaces, and safety boundaries.
- **Phase 1 — Harness-RSI MVP**
  - Minimal self-modifying agent loop with git-backed lineage, Docker/sandbox runner, immutable verifier, and automatic keep/rollback gating.
- **Phase 2 — Benchmarks, Archives & Anti-Hacking**
  - Train/validation/held-out benchmark split, population archive (branching evolution), cost-aware scoring, and rich trace inspection.
- **Phase 3 — Model-RSI Pipeline**
  - Automated rollout harvesting, verifier filtering, and pluggable SFT / GRPO / RLVR training recipes on open-weight models.
- **Phase 4 — Dual-Loop Co-Evolution**
  - Close the flywheel: distill Harness-RSI breakthroughs into Model-RSI checkpoints, and feed promoted checkpoints back into Harness-RSI.

---

## Dive Deeper

- Read **[Harness-RSI](./harness-rsi/README.md)** to see how an agent rewrites and verifies its own code and workflows.
- Read **[Model-RSI](./model-rsi/README.md)** to see how models self-improve their weights via verified rollouts and post-training.
