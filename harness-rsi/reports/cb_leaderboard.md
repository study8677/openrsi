# OpenRSI × CompactBench Leaderboard

Evaluated with `compactbench==0.3.0` via local `CodexExecProvider` (`codex exec -c model_reasoning_effort="low"`, ChatGPT OAuth, \$0 extra API cost).

## Suite: `starter` (`seed_group="default"`, 8 families × 3 cycles = 102 probes, budget = 2048 tokens)

| Method | Official Score | Drift Resistance | Decay (C0 - C2) | Mean Tokens | Budget Violations | Weakest Families |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `OpenRSICompactor` v0 (Baseline truncation) | **0.623** | 0.542 | +0.450 | 1,980 | 18 | `provenance_authority` (0.33), `conditional_scope` (0.50), `decision_override` (0.56) |
| `built-in:codex-cli` (Codex CLI default) | **0.896** | 1.000 | +0.000 | 1,074 | 0 | `conditional_scope` (0.50), `numeric_precision` (0.67) |
