# OpenRSI × CompactBench Leaderboard

Evaluated with `compactbench==0.3.0` via local `CodexExecProvider` (`codex exec -c model_reasoning_effort="low"`, ChatGPT OAuth, \$0 extra API cost).

## Official Full Suite: `practice` (Elite Difficulty — 48 cases, ~32k opening tokens, 8 mixed families, 4 drift cycles, 4,546 probes, budget = 2048 tokens)

| Method | Cases | Full-Credit Probes | Official Score (95% CI) | Drift Resistance | Decay (C0 - C4) | Stale / Trap / Halluc | Mean Tokens / Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OpenRSICompactor` v1 (RSI Delta Evidence Ledger)** | **48 / 48** | **3,535 / 4,546** | **`0.762 [0.740, 0.784]`** | **`0.815`** | **`+0.164`** (`0.90 → 0.69`) | `3.3% / 0.0% / 0.0%` | `1,874 / 0` |

### 48-Case Diagnostic Breakdown (Sorted Weak → Strong — Roadmap for Next Evolution)

| Family | 48-Case Score | Status / Evolution Priority |
| :--- | :---: | :--- |
| `reference_resolution` | **0.48** | **P0 Bottleneck** — ordinal/superlative/recommender shortlists lost under 32k compaction |
| `buried_constraint` | **0.65** | **P1 Bottleneck** — rules buried among look-alike sibling projects |
| `conditional_scope` | **0.66** | **P1 Bottleneck** — scoped/time-bounded rule widening & narrowing |
| `numeric_precision` | **0.72** | P2 — exact figures/dates/limits shadowed by look-alike projects |
| `task_continuity` | **0.72** | P2 — ticket board state transitions (`in_review` vs `done`) |
| `entity_confusion` | **0.92** | Strong (protect against regression) |
| `decision_override` | **0.95** | Strong (protect against regression) |
| `provenance_authority` | **0.98** | Near-perfect (protect against regression) |

- **By Probe Category (weak → strong)**: `reference_resolution: 0.54` | `constraint_application: 0.60` | `fact_recall: 0.61` | `task_state: 0.70` | `constraint_recall: 0.72` | `entity_attribution: 0.91` | `decision_state: 0.96` | `abstention: 1.00`.
- **Full JSON summary**: [`cb_practice48_openrsi_v1.summary.json`](./cb_practice48_openrsi_v1.summary.json)

---

## Quick Comparison Subsets (Early Evolution Checks)

### `starter` (`seed_group="default"`, 8 families × 3 cycles = 102 probes, budget = 2048)

| Method | Official Score | Drift Resistance | Decay (C0 - C2) | Mean Tokens | Budget Violations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `OpenRSICompactor` v0 (Baseline truncation) | 0.623 | 0.542 | +0.450 | 1,980 | 18 |
| `built-in:codex-cli` (Codex CLI default) | 0.896 | 1.000 | +0.000 | 1,074 | 0 |
| `OpenRSICompactor` Gen 3 (RSI on `starter`) | 0.896 | 0.958 | +0.042 | 1,438 | 0 |

### `practice` 2-Case Pilot (`seed_group="practice_eval"`, 191 probes, budget = 2048)

| Method | Official Score | Full-Credit Probes | Drift Resistance | Decay (C0 - C4) | Cycle 4 Score | Budget Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `built-in:codex-cli` (Codex CLI default) | 0.697 | 131 / 191 | 0.557 | +0.403 | 0.55 | 0 |
| `OpenRSICompactor` v1 (RSI Delta Evidence Ledger) | **0.707** | **140 / 191** | **0.783** | **+0.191** | **0.67** | **0** |
