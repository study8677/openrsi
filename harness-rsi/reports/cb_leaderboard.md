# OpenRSI × CompactBench Leaderboard

Evaluated with `compactbench==0.3.0` via local `CodexExecProvider` (`codex exec -c model_reasoning_effort="low"`, ChatGPT OAuth, \$0 extra API cost).

## Suite: `starter` (`seed_group="default"` — unseen during evolution, 8 families × 3 cycles = 102 probes, budget = 2048 tokens)

| Method | Official Score | Drift Resistance | Decay (C0 - C2) | Mean Tokens | Budget Violations | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `OpenRSICompactor` v0 (Baseline truncation) | **0.623** | 0.542 | +0.450 | 1,980 | 18 | Drops early rules (`provenance_authority`: 0.33, `conditional_scope`: 0.50) |
| `built-in:codex-cli` (Codex CLI default) | **0.896** | 1.000 | +0.000 | 1,074 | 0 | `numeric_precision`: 0.67, `conditional_scope`: 0.50 |
| `OpenRSICompactor` Gen 3 (RSI-evolved on `openrsi_train`) | **0.896** | 0.958 | +0.042 | 1,438 | 0 | **+0.273 over v0**, tied `codex-cli` overall & beat on `numeric_precision` (0.78 vs 0.67) |

### Evolution Trace (`starter`, `seed_group="openrsi_train"`)
- **Gen 0 (Baseline)**: `0.537` (drift resistance `0.266`, `19` budget violations)
- **Gen 1–2**: Caught by compile guardrail → automatically rolled back
- **Gen 3 (Promoted)**: **`0.938`** (drift resistance **`1.000`**, **`0`** budget violations, 7/8 families at `1.00`)
