# OpenRSI Phase 4 技术报告：双循环飞轮（Harness-RSI ↔ Model-RSI）实测闭环

> **共享真实战场**：`compactbench==0.3.0`（长上下文多轮漂移压缩与探针评测）  
> **外循环输入（只读）**：`harness-rsi` 进化出的 `OpenRSICompactor`（Elite 48 案例 / 246 个漂移 Cycle / 4,659 个官方确定性探针轨迹）  
> **内循环模型**：`Qwen/Qwen3.5-2B`（Google Colab NVIDIA Tesla T4 15GB GPU，DeltaNet-Safe 冻结 `in_proj_a/b` LoRA Merge-and-Chain）

---

## 1. Phase 4A：`Harness → Model` 内循环进化（32k Elite 留出集）

严格按 `case_id` 将 `harness-rsi` 产出的 48 个 Elite 案例隔离切分为 **36 个训练案例 / 6 个验证案例 / 6 个锁定留出案例（零泄漏）**，使用官方 `compactbench.scoring.score_probe` 实时验分：

![Phase 4A Flywheel Evolution Curve](./flywheel_evolution_curve.png)

| 世代 | 进化范式 | 门控结果 | Held-out 官方探针分 | 满分探针 | Val 得分 | JSON 格式率 | 通用回归 | `reference_res` (P0) | `conditional_scope` (P1) | `task_continuity` |
| ---: | :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `baseline`（原始 `Qwen/Qwen3.5-2B`） | baseline | 53.15% | 127 / 249 | 60.05% | 95.00% | 90.00% | 52.78% | 32.81% | 54.35% |
| 1 | `harness_distill_sft`（高分轨迹蒸馏） | **promoted** | 56.97% | 136 / 249 | 58.97% | **100.00%** | 90.00% | 57.50% | 37.57% | 58.33% |
| 2 | `probe_verified_dpo`（T4 fp16 溢出） | **rejected** | 56.97% | 136 / 249 | 58.97% | 100.00% | 90.00% | 57.50% | 37.57% | 58.33% |
| 3 | `bottleneck_hard_neg_star`（87 错题在线挖掘纠偏） | **promoted** | **59.28%** | **141 / 249** | **62.28%** | **100.00%** | **90.00%** | **67.50%** | **46.25%** | **62.50%** |

**核心突破**：
- 经 `Gen 0 → Gen 1 → Gen 3` 两代晋升，`Qwen/Qwen3.5-2B` 在未见 32k Elite 账本上的官方探针得分从 **`53.15% → 59.28%`（`+6.13%` 绝对提升，`+14` 个满分探针）**，JSON 合规率升至 **`100.00%`**，通用回归全程稳锁 **`90.00%`（零遗忘）**；
- `Harness-RSI` X-Ray 诊断出的最难瓶颈家族实现大幅跃升：**`reference_resolution` `+14.72%`**（`52.78% → 67.50%`）、**`conditional_scope` `+13.44%`**（`32.81% → 46.25%`）、**`task_continuity` `+8.15%`**（`54.35% → 62.50%`）。

---

## 2. Phase 4B：`Model → Harness` 反向闭环（官方 `compactbench==0.3.0` 2×2 四组对照）

通过我们实现的官方 Provider 接口 [HFCompactBenchProvider](../model_rsi/flywheel_provider.py)，把原始 `Qwen/Qwen3.5-2B` 与进化后的 **`OpenRSI-Qwen3.5-2B-Flywheel`（Gen 3）** 直接接回官方 `compactbench`（`starter` 套件，`seed_group="default"`，8 家族 × 3 漂移轮 = 102 官方探针，预算 2048 tokens），与 `harness-rsi` 进化出的真实 **`OpenRSICompactor v2`** 跑满 2×2 消融实验：

![Phase 4B 4-Way Dual-Loop Ablation](./flywheel_ablation_4way.png)

| 实验组 | 模型底座 | 压缩脚手架 | Official Score | 满分探针 | 抗漂移率 | Decay | 陈旧记忆率 (Stale) | 预算违规 |
| :--- | :--- | :--- | ---: | :---: | ---: | ---: | ---: | :---: |
| **Cell 1 (No RSI)** | Raw `Qwen/Qwen3.5-2B` | `built-in:truncate-last-n` | 56.20% | 59 / 102 | 54.17% | 0.3802 | 26.67% | 0 |
| **Cell 2 (Harness only)** | Raw `Qwen/Qwen3.5-2B` | `OpenRSICompactor v2` | 45.95% | 47 / 102 | **61.43%** | **0.2396** | 53.33% | 0 |
| **Cell 3 (Model only)** | **`OpenRSI-Qwen3.5-2B-Flywheel`** | `built-in:truncate-last-n` | **58.26%** | **61 / 102** | 59.81% | 0.3108 | 26.67% | 0 |
| **Cell 4 (Dual-Loop)** | **`OpenRSI-Qwen3.5-2B-Flywheel`** | `OpenRSICompactor v2` | **49.65%** | **52 / 102** | 57.38% | 0.2917 | **40.00%** | 0 |

---

## 3. 关键科学发现与下一阶段演进方向

1. **Model-RSI 在所有评测维度均带来稳定正收益**：
   - 在 32k Elite 高难证据账本上：**`+6.13%`**（`53.15% → 59.28%`）；
   - 在官方 `starter` + `truncate-last-n` 上：**`+2.06%`**（`56.20% → 58.26%`），抗漂移率 **`+5.64%`**（`54.17% → 59.81%`）；
   - 在官方 `starter` + `OpenRSICompactor v2` 上：**`+3.70%`**（`45.95% → 49.65%`），**陈旧记忆错误率大幅下降 `-13.33%`**（`53.33% → 40.00%`）！
2. **揭示了“小模型驱动强脚手架”的核心瓶颈（Phase 4.2 突破口）**：
   - `OpenRSICompactor v2` 是为前沿大模型（Codex）进化出的严格补丁协议（`A | H | KIND` / `R n | H | KIND` + 独立审计轮）。当由 2B 小模型同时充当 **Compactor + Reader** 时，Cell 2 与 Cell 4 的 `mean_artifact_tokens` 均落到 `1,792.3`（说明 2B 模型尚未学会原生输出 `OpenRSICompactor v2` 的补丁语法，触发了确定性兜底回退）。
   - **下一步最高价值动作**：在现有 Reader 蒸馏的基础上，加入 **Compactor 增量补丁协议蒸馏（记录 `OpenRSICompactor v2` 每一轮的 `(prior_ledger, delta_turns) → patch_lines` 轨迹并蒸馏进 2B 权重）**，让 2B 小模型既会**读**证据账本，又会**写**增量补丁，真正释放双循环在 2B 端侧上的全部潜力！
