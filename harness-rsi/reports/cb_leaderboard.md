# OpenRSI × CompactBench Leaderboard

Evaluated with `compactbench==0.3.0` via local `CodexExecProvider` (`codex exec -c model_reasoning_effort="low"`, ChatGPT OAuth, \$0 extra API cost).

## Suite 1: `starter` (`seed_group="default"` — unseen during evolution, 8 families × 3 cycles = 102 probes, budget = 2048 tokens)

| Method | Official Score | Drift Resistance | Decay (C0 - C2) | Mean Tokens | Budget Violations | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `OpenRSICompactor` v0 (Baseline truncation) | **0.623** | 0.542 | +0.450 | 1,980 | 18 | Drops early rules (`provenance_authority`: 0.33, `conditional_scope`: 0.50) |
| `built-in:codex-cli` (Codex CLI default) | **0.896** | 1.000 | +0.000 | 1,074 | 0 | `numeric_precision`: 0.67, `conditional_scope`: 0.50 |
| `OpenRSICompactor` Gen 3 (RSI on `starter`) | **0.896** | 0.958 | +0.042 | 1,438 | 0 | **+0.273 over v0**, tied `codex-cli` overall & beat on `numeric_precision` (0.78 vs 0.67) |

## Suite 2: `practice` (Elite Difficulty — ~32k opening tokens, all 8 families mixed, 4 drift cycles, `seed_group="practice_eval"` = 191 probes, budget = 2048 tokens)

| Method | Official Score | Full-Credit Probes | Drift Resistance | Decay (C0 - C4) | Cycle 4 Score | Mean Tokens | Budget Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `built-in:codex-cli` (Codex CLI default) | **0.697** | 131 / 191 | 0.557 | +0.403 | 0.55 | 1,997 | 0 |
| `OpenRSICompactor` Gen 3 (evolved on `starter`) | **0.628** | 126 / 191 | 0.767 | +0.181 | 0.54 | 1,838 | 0 |
| `OpenRSICompactor` Elite Round 1 (RSI on `practice`) | **0.690** | 138 / 191 | **0.839** | **+0.125** | 0.65 | 1,882 | 0 |
| **`OpenRSICompactor` Elite Round 2 (RSI on `practice`)** | **`0.707`** | **`140 / 191`** | **0.783** | **+0.191** | **`0.67`** | **`1,881`** | **`0`** |

### Per-Family Breakdown on Unseen Elite `practice_eval` (`OpenRSICompactor` Elite Round 2 vs `built-in:codex-cli`)

| Family | `built-in:codex-cli` | **`OpenRSICompactor` (RSI)** | Delta |
| :--- | :---: | :---: | :---: |
| `decision_override` | 0.73 | **1.00** | **+0.27** |
| `entity_confusion` | 0.60 | **1.00** | **+0.40** |
| `provenance_authority` | 0.80 | **1.00** | **+0.20** |
| `conditional_scope` | 0.61 | **0.67** | **+0.06** |
| `numeric_precision` | 0.70 | **0.75** | **+0.05** |
| `buried_constraint` | 0.88 | 0.59 | -0.29 |
| `reference_resolution` | 0.45 | 0.30 | -0.15 |
| `task_continuity` | 0.80 | 0.35 | -0.45 |
| **Overall Official Score** | **0.697** | **`0.707`** | **+0.010** |
