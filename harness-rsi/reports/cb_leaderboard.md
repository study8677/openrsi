# OpenRSI × CompactBench Leaderboard

Evaluated with `compactbench==0.3.0` via local `CodexExecProvider` (`codex exec -c model_reasoning_effort="low"`, ChatGPT OAuth, \$0 extra API cost).

## Suite 1: `starter` (`seed_group="default"` — unseen during evolution, 8 families × 3 cycles = 102 probes, budget = 2048 tokens)

| Method | Official Score | Drift Resistance | Decay (C0 - C2) | Mean Tokens | Budget Violations | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `OpenRSICompactor` v0 (Baseline truncation) | **0.623** | 0.542 | +0.450 | 1,980 | 18 | Drops early rules (`provenance_authority`: 0.33, `conditional_scope`: 0.50) |
| `built-in:codex-cli` (Codex CLI default) | **0.896** | 1.000 | +0.000 | 1,074 | 0 | `numeric_precision`: 0.67, `conditional_scope`: 0.50 |
| `OpenRSICompactor` Gen 3 (RSI on `starter`) | **0.896** | 0.958 | +0.042 | 1,438 | 0 | **+0.273 over v0**, tied `codex-cli` overall & beat on `numeric_precision` (0.78 vs 0.67) |

## Suite 2: `practice` (Elite Difficulty — ~32k opening tokens, all 8 families mixed, 4 drift cycles, `seed_group="practice_eval"` = 191 probes, budget = 2048 tokens)

| Method | Official Score | Drift Resistance | Decay (C0 - C4) | Mean Tokens | Budget Violations | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `built-in:codex-cli` (Codex CLI default) | **0.697** | 0.557 | +0.403 | 1,997 | 0 | Cycle 0→4: `0.91 → 0.55`; weak on `reference_resolution` (0.45), `entity_confusion` (0.60) |
| `OpenRSICompactor` Gen 3 (evolved on `starter` only) | **0.628** | **0.767** | **+0.181** | 1,838 | 0 | **Beat `codex-cli` on Drift Resistance (+0.210), `decision_override` (1.00 vs 0.73), `entity_confusion` (0.67 vs 0.60), `numeric_precision` (0.75 vs 0.70)**; weak on `task_continuity` (0.20) & `reference_resolution` (0.20) |
