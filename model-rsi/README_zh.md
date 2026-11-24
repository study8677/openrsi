# Model-RSI（模型层自我进化）

[English](./README.md) | **中文** | [技术报告](./docs/TECHNICAL_REPORT_zh.md) | [← 返回 OpenRSI 首页](../README_zh.md)

**Model-RSI** 通过多代自动化闭环持续提升 AI 模型*自身的参数权重*。它不再单纯依赖人工标注数据，而是让模型自主采样解题或生成任务，通过沙盒中的自动验证器筛选出客观高质量信号，执行自动化后训练，并在严格通过留出测试集门控后晋升为新一代基座权重。

---

## 核心架构

```mermaid
flowchart LR
    M["当前策略模型<br/>（Checkpoint N）"] --> R["沙盒采样与<br/>任务探索"]
    H["Harness-RSI<br/>已验证轨迹"] -. "可选蒸馏输入" .-> V
    R --> V["自动验证器<br/>（单测 / 执行 / 形式化检查）"]
    V --> D["训练信号构建<br/>（验证 SFT / 偏好对 / 奖励）"]
    D --> T["自动化后训练<br/>（SFT / DPO / GRPO / RLVR）"]
    T --> C["候选权重<br/>（Checkpoint N + 1）"]
    C --> G{"留出集评测与<br/>防退化门控"}
    G -->|通过| M
    G -->|未通过| X["丢弃并调整<br/>课程与超参"]
    X -.-> R
```

### 核心模块

1. **采样与自博弈引擎（Rollout & Self-Play）**：基于当前 Checkpoint 生成多样化解题轨迹（也可叠加更强的 Harness 或推理时搜索来放大探索能力）。
2. **自动验证器（Automated Verifier）**：给出客观可验证的真实奖励，例如单元测试通过、代码编译执行成功、数学等价性校验等。
3. **数据与课程管理器（Data & Curriculum）**：动态控制任务难度（聚焦模型当前通过率既不是 0% 也不是 100% 的能力边界），去重并混入锚点数据防止遗忘。
4. **后训练执行器（Post-Training Runner）**：可插拔的训练后端，支持 LoRA 或全量参数的 SFT、DPO 以及在线强化学习（GRPO / PPO / RLVR）。
5. **晋升门控（Promotion Gate）**：在不可见的留出评测集与通用能力回归集上全面验分，只有真正变强才晋升为 `Checkpoint N + 1`。

---

## 核心进化范式

| 范式 | 工作方式 | 特点与权衡 |
| :--- | :--- | :--- |
| **拒绝采样 / STaR（验证 SFT）** | 对同一题采样多条轨迹，只保留验证通过的样本进行微调 | 实现简单、训练稳定、算力友好；对负样本利用不足 |
| **可验证强化学习（RLVR / GRPO / PPO）** | 直接依据沙盒验证奖励对分组采样轨迹进行策略优化 | 推理与编程能力上限最高；对奖励设计与训练稳定性要求高 |
| **迭代偏好优化（Iterative DPO）** | 将同一题下验证通过与失败的轨迹配成偏好对训练 | 能充分利用负样本信号，复杂度低于在线 RL |
| **Harness-to-Model 蒸馏** | 先用高算力的 `harness-rsi` 搜索与工具链攻克难题，再把验证通过的解蒸馏进模型权重 | 将昂贵的推理时搜索转化为模型自身的单步直觉 |
| **自博弈与前沿课程（Self-Play）** | 协同进化“出题器”与“解题器”，让训练任务始终紧贴模型当前能力边界 | 解决静态数据集见顶问题；但最难防止题目退化或失真 |

---

## 核心风险与护栏

用模型自生成数据进行多代递归训练，必须防范以下典型风险：

- **模型坍塌与多样性退化（Model Collapse）**：长期只在自产数据上训练会导致分布变窄。需要结合严格外部验证器、高多样性采样以及高质量锚点数据混合来缓解。
- **奖励黑客（Reward Hacking）**：模型可能学会钻验证器空子（如敷衍测试或迎合启发式规则）。验证器必须保持确定性、沙盒隔离，并接受留出集审计。
- **灾难性遗忘（Catastrophic Forgetting）**：窄领域自我强化可能损伤通用能力。晋升门控必须包含通用能力回归测试。
- **评测集污染（Contamination）**：采样任务池必须与留出评测集严格隔离。

---

## 实测结果（Colab NVIDIA L4 GPU）

### 实验 2（主力基座）：`Qwen/Qwen3.5-2B` × 开源成熟 RLVR 基准（`MATH-500` + `HumanEval+`）

训练集采用 `open-r1/Big-Math-RL-Verified-Processed`（L1–L2）+ `KodCode/KodCode-V1` + `MBPP`，门控评测严格锁定外部官方基准 **`HuggingFaceH4/MATH-500`**（40 题）+ **`evalplus/humanevalplus`**（40 题）+ 10 题通用防遗忘回归集：

![Qwen3.5-2B Model-RSI 进化曲线](./reports/qwen35_2b_evolution_curve.png)

| 世代 | 进化范式 | 门控结果 | Held-out Pass@1 | MATH-500 (留出) | HumanEval+ (留出) | Hard Val | 通用回归 |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | `baseline`（`Qwen/Qwen3.5-2B`） | baseline | 42.50% | 30.00% | 55.00% | 10.00% | 90.00% |
| 1 | `pure_rlvr_grpo`（`all-linear`） | rejected | 36.25% | 25.00% | 47.50% | 12.50% | 90.00% |
| 2 | `deltanet_safe_dpo`（冻结 `in_proj_a/b`） | **promoted** | 46.25% | 32.50% | **60.00%** | 10.00% | 90.00% |
| 3 | `deltanet_safe_grpo`（严格前沿 RLVR） | **promoted** | **48.75%** | **37.50%** | **60.00%** | **15.00%** | **90.00%** |

关键发现：冻结 Qwen3.5 Gated DeltaNet 的敏感递归状态门（`in_proj_a`/`in_proj_b`），仅在 Attention + MLP 层执行 **DeltaNet-Safe Iterative DPO → 严格前沿纯二值 RLVR（GRPO）**，让官方留出集总分提升 **`+6.25%`**（`42.50% → 48.75%`，其中 `MATH-500` `+7.50%`、`HumanEval+` `+5.00%`），同时通用回归锁定 **`90.00%`（零遗忘）**。详见 [`reports/qwen35_2b_experiment_summary.md`](./reports/qwen35_2b_experiment_summary.md)。

### 实验 1：`Qwen/Qwen2.5-1.5B-Instruct`（8 代 Countdown + MBPP）

我们在 Google Colab NVIDIA L4 GPU 上对开源权重 `Qwen/Qwen2.5-1.5B-Instruct` 完整跑通了 8 代（`Gen 0 → Gen 7`）Model-RSI 递归进化闭环，评测覆盖严格隔离的 **Countdown 可验证算术推理**（确定性 AST + 精确分数验证器）、**MBPP Sanitized Python 编程**（独立 `python -I` 子进程沙盒）以及 **10 题通用能力防遗忘回归门控**：

![Model-RSI 进化曲线](./reports/evolution_curve.png)

| 世代 | 进化范式 | 门控结果 | Held-out Pass@1 | Val Pass@1 | MBPP 代码 (留出) | Countdown L1 (留出) | 通用回归 |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | `baseline` | baseline | 15.00% | 18.75% | 50.00% | 0.00% | 90.00% |
| 1 | `star_sft`（全序列） | rejected | 15.00% | 18.75% | 37.50% | 12.50% | 90.00% |
| 2 | `iterative_dpo` | **promoted** | 17.50% | 22.92% | 54.17% | 6.25% | 90.00% |
| 3 | `grpo_rlvr` | **promoted** | 17.50% | 25.00% | **58.33%** | 0.00% | 90.00% |
| 4 | `star_sft_comp_only` | **promoted** | **18.75%** | 20.83% | 45.83% | **18.75%** | 80.00% |
| 5 | `frontier_dpo_v2` | **promoted** | **18.75%** | **31.25%** | 54.17% | 12.50% | **90.00%** |
| 6 | `lineage_soup` | rejected | 17.50% | 25.00% | 50.00% | 12.50% | 90.00% |
| 7 | `pure_rlvr_grpo_v2` | rejected | 17.50% | 25.00% | 50.00% | 12.50% | 90.00% |

完整诊断分析（包含 Gen 3 奖励黑客/长度坍塌发现与纯二值 RLVR 修复）见 [`reports/experiment_summary.md`](./reports/experiment_summary.md) 与 [`reports/evolution_history.json`](./reports/evolution_history.json)。

---

## 快速开始

```bash
cd model-rsi
uv run pytest -v                        # 运行确定性验证器、课程、门控与飞轮单测
uv run model-rsi status                 # 查看多代模型进化账本
uv run model-rsi plot                   # 生成 reports/evolution_curve.png
uv run model-rsi generate-tasks         # 导出严格隔离的 train/val/held_out/regression 任务集
uv run model-rsi import-harness         # 从 harness-rsi 导入已验证轨迹（M4 飞轮）
```

在 Google Colab GPU（L4 / A100 / H100）上复现完整多代后训练（STaR SFT、Iterative DPO、在线 GRPO/RLVR、谱系权重融合），安装 `requirements-colab.txt` 后运行 `colab/run_model_rsi_colab.py`。

---

## 路线图（Roadmap）

- [x] **M1 — 统一轨迹与验证器协议**：标准化可验证轨迹（来自模型直采或 `harness-rsi`）的存储、打分与过滤格式（`model_rsi/protocol.py`、`model_rsi/tasks/`）。
- [x] **M2 — 验证 SFT / 拒绝采样闭环**：在开源模型上跑通端到端 Completion-Only STaR SFT 与 Iterative DPO 管线及自动留出集/回归门控（`model_rsi/trainers/star_sft.py`、`iterative_dpo.py`、`gating.py`）。
- [x] **M3 — 接入 RLVR（GRPO）**：提供面向代码与可验证推理任务的可插拔沙盒在线强化学习训练流，内置防奖励黑客纯二值验证奖励（`model_rsi/trainers/grpo_rlvr.py`）。
- [x] **M4 — 前沿课程与双循环飞轮**：实现前沿难度调度（`model_rsi/curriculum.py`）、谱系权重融合（`model_rsi/trainers/lineage_merge.py`），并与 [`harness-rsi`](../harness-rsi/README_zh.md) 打通闭环蒸馏（`model_rsi/flywheel.py`）。
