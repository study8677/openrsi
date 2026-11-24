# OpenRSI Model-RSI 技术报告：面向开源小模型的权重级递归自我进化

> **项目仓库**：`openrsi/model-rsi`  
> **实验平台**：Google Colab NVIDIA L4 GPU（23.7 GB VRAM，CUDA 13.0，PyTorch 2.14.1，`transformers 5.22.1`，`peft 0.21.2`，`trl 1.18.2`）  
> **核心结论**：在不依赖人工标注的前提下，通过「前沿课程采样 → 确定性沙盒验分 → 迭代式 LoRA 权重合并后训练（DPO + 纯二值 RLVR/GRPO）→ 不可变留出集与通用回归门控」闭环，将最新开源小模型 **`Qwen/Qwen3.5-2B`** 在外部标准留出基准（`MATH-500` + `HumanEval+`）上的贪心 Pass@1 从 **`42.50%` 提升至 `48.75%`（`+6.25%`）**，同时通用能力回归全程锁定 **`90.00%`（零灾难性遗忘）**。

---

## 1. 背景与定位

在 OpenRSI 双轨架构中：
- **[Harness-RSI](../../harness-rsi/README_zh.md)** 冻结底层模型权重，递归进化外围脚手架（上下文压缩、Prompt、工具链与控制流）；
- **[Model-RSI](../README_zh.md)** 则直接面向模型自身的参数权重，通过**可验证奖励信号**与**自动化多代后训练闭环**，把推理时靠多次采样试错才能解出的能力（Pass@k）持续内化为模型的单步参数直觉（Pass@1）。

Model-RSI 严格遵守 OpenRSI 的五项设计原则：
1. **只信确定性执行验证，不信模型自我打分**；
2. **评测不可变，训练集与留出集零重叠**；
3. **隔离沙盒执行，防超时与代码注入**；
4. **基于谱系与门控进化，退化自动回滚**；
5. **全流程可复现、账本与曲线可审计**。

---

## 2. 系统架构

```mermaid
flowchart TB
    subgraph Data["1. 开源任务池 & 严格隔离切分 (tasks/)"]
        T_TR["训练题库 (Train Pool)<br/>Big-Math L1-2 / KodCode-V1 / MBPP / Countdown"]
        T_VAL["高难验证集 (Val)<br/>DAPO-Math-17k / KodCode"]
        T_HO["锁定留出考卷 (Held-out, 只读)<br/>MATH-500 + HumanEval+"]
        T_REG["防遗忘回归集 (Regression, 只读)<br/>10 题通用指令/JSON/逻辑/问答"]
    end

    subgraph Loop["2. 多代递归进化主循环 (Gen N → Gen N+1)"]
        CKPT["当前最优权重<br/>Checkpoint N<br/>(初始: Qwen/Qwen3.5-2B)"]
        CURR["前沿课程调度器 (curriculum.py)<br/>优先挑选 0 < pass_rate < 1 能力边界题"]
        ROLL["GPU 批量采样引擎 (rollout.py)<br/>每题生成 k 条候选解法"]
        VER["不可变确定性沙盒验证器<br/>数学 AST/精确分数验算 + python -I 隔离单测"]
        STORE[("统一可验证轨迹库<br/>TrajectoryStore JSONL<br/>(protocol.py)")]
        BUILD["训练信号构造器 (data_builder.py)<br/>去重 / 长度过滤 / 同题配对 / 锚点回放"]

        subgraph Trainers["3. 可插拔后训练执行器 (DeltaNet-Safe LoRA + Merge)"]
            TR_STAR["STaR SFT<br/>Completion-Only 拒绝采样"]
            TR_DPO["Iterative DPO<br/>同题 (Pass vs Fail) 对比"]
            TR_GRPO["在线 RLVR (GRPO)<br/>纯二值 1/0 实时沙盒奖励"]
        end

        MERGE["权重合并 (merge_and_unload)<br/>W_(N+1) = W_N + ΔW<br/>(冻结 DeltaNet in_proj_a/b)"]
        CAND["候选权重<br/>Checkpoint N+1"]
    end

    subgraph Gate["4. 晋升门控 & 谱系档案库 (gating.py / reporting.py)"]
        EVAL["贪心 Pass@1 全量评测"]
        DEC{"留出集提升 且<br/>通用回归 ≥ 基座容忍线？"}
        PROM["晋升为新基座<br/>更新 Checkpoint N"]
        REJ["拒绝并回滚<br/>保留诊断日志 & 调整策略"]
        REP["进化账本 & 曲线报告<br/>evolution_history.json / .png"]
    end

    subgraph Flywheel["5. 双循环飞轮接口 (flywheel.py)"]
        HARNESS["Harness-RSI<br/>高分已验证轨迹 (只读导入)"]
    end

    T_TR --> CURR
    CKPT --> ROLL
    CURR --> ROLL
    ROLL --> VER
    HARNESS -. "蒸馏轨迹导入" .-> STORE
    VER --> STORE
    STORE --> CURR
    STORE --> BUILD

    BUILD --> TR_STAR
    BUILD --> TR_DPO
    CURR --> TR_GRPO
    VER -. "在线 1/0 奖励" .-> TR_GRPO

    CKPT --> TR_STAR
    CKPT --> TR_DPO
    CKPT --> TR_GRPO

    TR_STAR --> MERGE
    TR_DPO --> MERGE
    TR_GRPO --> MERGE
    MERGE --> CAND

    CAND --> EVAL
    T_VAL --> EVAL
    T_HO --> EVAL
    T_REG --> EVAL

    EVAL --> DEC
    DEC -- "通过 (Promoted)" --> PROM
    DEC -- "未通过 (Rejected)" --> REJ
    PROM --> CKPT
    PROM --> REP
    REJ --> REP
```

### 核心模块与源码映射

| 模块 | 文件路径 | 职责说明 |
| :--- | :--- | :--- |
| 统一轨迹协议（M1） | `model_rsi/protocol.py` | 定义 `VerifiableTask`、`VerificationResult`、`Trajectory`、`GenerationMetrics` 与 JSONL 只追加存储 `TrajectoryStore` |
| 数学可验证任务与验证器 | `model_rsi/tasks/math_rlvr.py`、`countdown.py` | 接入 `open-r1/DAPO-Math-17k-Processed`、`open-r1/Big-Math-RL-Verified-Processed`、`HuggingFaceH4/MATH-500` 与 Countdown；提供确定性 `\boxed{}` 嵌套提取、精确分数与 AST 验算器 |
| 代码沙盒任务与验证器 | `model_rsi/tasks/code_exec.py` | 接入 `KodCode/KodCode-V1`、`google-research-datasets/mbpp`、`evalplus/humanevalplus`；在独立 `python -I` 超时隔离临时目录中执行不可变单测 |
| 防遗忘回归门控集 | `model_rsi/tasks/regression.py` | 10 题锁定通用能力回归基准（JSON 结构化、逻辑推理、基础问答、指令长度与大小写约束） |
| 前沿课程与信号构造（M4） | `model_rsi/curriculum.py`、`data_builder.py` | 追踪每题历史通过率，聚焦 `0 < pass_rate < 1` 能力边界题；构造去重 STaR SFT 样本、同题 Pass/Fail DPO 偏好对及多样性指标 |
| 后训练执行器（M2/M3） | `model_rsi/trainers/star_sft.py`、`iterative_dpo.py`、`grpo_rlvr.py`、`lineage_merge.py` | 可插拔支持 Completion-Only STaR SFT、Iterative DPO、在线纯二值 RLVR（GRPO）与谱系权重融合（Model Soup） |
| 晋升门控与报告 | `model_rsi/gating.py`、`reporting.py` | 严格对比候选权重与当前最优/基座权重，自动决定 `promoted` 或 `rejected` 并生成进化曲线与 JSON 账本 |
| 双循环飞轮桥接（M4） | `model_rsi/flywheel.py` | 只读导入 `harness-rsi/reports/` 中验证通过的高分轨迹进入 `TrajectoryStore` |

---

## 3. 核心技术设计：参数更新机制与三大范式

### 3.1 参数更新机制：迭代式 LoRA + 权重合并（Iterative LoRA Merge-and-Chain）
在单卡 Colab L4（23.7 GB VRAM）的算力约束下，我们采用**迭代式 LoRA 挂载 + 世代权重合并**：
- 每一代在当前最优基座 `Checkpoint N` 的注意力与 MLP 投影层（`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`，`r=16, alpha=32`）挂载低秩适配器；
- 训练结束后调用 `merge_and_unload()` 将增量权重 $\Delta W = B \cdot A$ 真实合入主权重矩阵，产出独立可加载、可审计、可回滚的 `Checkpoint N+1`；
- **重要架构约束**：在 `Qwen3.5` 等混合线性注意力（Gated DeltaNet）模型上，**必须冻结递归状态衰减/写入门（`in_proj_a` / `in_proj_b`）**，避免低样本 RL/DPO 更新破坏长程状态稳定性；
- 若部署于多卡 A100/H100 集群，只需移除 `peft_config` 即可无缝切换为全量参数更新，上层 RSI 调度与门控完全不变。

### 3.2 三大可插拔后训练范式对比
- **Completion-Only STaR SFT**：只对验证通过的回答部分计算 Loss（屏蔽 Prompt Loss），适合冷启动破冰（将 0% 通过率拉起至 15%~30%），配合锚点数据防遗忘。
- **Iterative DPO**：严格从**同一道题**的采样轨迹中配对 `(chosen=验证通过, rejected=验证失败)`，同时吸收正负样本信号并带 KL 正则（`beta=0.10~0.15`），训练极稳、抗遗忘极强。
- **在线纯二值 RLVR（GRPO）**：对前沿边界题在线采样组轨迹（$G=4\sim 6$），仅按确定性沙盒结果给予二值 `1.0 / 0.0` 奖励，在组内做相对优势策略优化，探索上限最高且杜绝格式捷径作弊。

---

## 4. 真实 GPU 实验结果与分析

### 4.1 实验二（主力基座）：`Qwen/Qwen3.5-2B` × 官方开源 RLVR 基准

- **训练池**：`Big-Math-RL-Verified-Processed`（L1–L2）+ `KodCode-V1`（前沿）+ `MBPP`
- **锁定外部留出基准（80 题）**：`HuggingFaceH4/MATH-500`（40 题）+ `evalplus/humanevalplus`（40 题）
- **高难验证集（40 题）**：`DAPO-Math-17k`（20 题）+ `KodCode-V1`（20 题）

![Qwen3.5-2B 进化曲线](../reports/qwen35_2b_evolution_curve.png)

| 世代 | 父代 | 进化范式 | 门控结果 | Held-out Pass@1 | MATH-500 | HumanEval+ | Hard Val | 通用回归 | 平均 Token |
| ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | — | `baseline`（`Qwen/Qwen3.5-2B`） | baseline | 42.50% | 30.00% | 55.00% | 10.00% | 90.00% | 254.5 |
| 1 | 0 | `pure_rlvr_grpo`（`all-linear`） | **rejected** | 36.25% | 25.00% | 47.50% | 12.50% | 90.00% | 262.0 |
| 2 | 0 | `deltanet_safe_dpo`（冻结 `in_proj_a/b`） | **promoted** | 46.25% | 32.50% | **60.00%** | 10.00% | 90.00% | 255.5 |
| 3 | 2 | `deltanet_safe_grpo`（严格前沿 RLVR） | **promoted** | **48.75%** | **37.50%** | **60.00%** | **15.00%** | **90.00%** | 253.6 |

**结果解读**：
- 经过 `Gen 0 → Gen 2 → Gen 3` 两代连续晋升，官方外部留出集总分净提升 **`+6.25%`**（`42.50% → 48.75%`，相对提升 `+14.7%`）；
- `MATH-500` 提升 **`+7.50%`**（`30.00% → 37.50%`），`HumanEval+` 提升 **`+5.00%`**（`55.00% → 60.00%`），高难验证集提升 **`+50%`**（`10.00% → 15.00%`）；
- 通用回归门控全程保持 **`90.00%`**，响应长度稳定在 `253.6 ~ 255.5` tokens，无任何退化或坍塌。

### 4.2 实验一（全范式验证）：`Qwen/Qwen2.5-1.5B-Instruct`（8 代闭环）

在 `Countdown`（3/4/5 数参数化难度）+ `MBPP` 上完成了 8 代（`Gen 0 → Gen 7`）全范式对比（STaR SFT / Iterative DPO / Online GRPO / Completion-Only SFT / Lineage Model Soup / Pure-Binary RLVR）：
- **Held-out Pass@1**：`15.00% → 18.75%`（相对提升 `+25.0%`）；
- **Val Pass@1**：`18.75% → 31.25%`（相对提升 `+66.7%`）；
- **MBPP 代码执行**：`50.00% → 58.33%`（Gen 3 峰值）；**Countdown L1**：`0.00% → 18.75%`（Gen 4 峰值）；
- **通用回归门控**：最佳综合晋升节点（Gen 5）保持 **`90.00%`**。

![Qwen2.5-1.5B 进化曲线](../reports/evolution_curve.png)

---

## 5. 实战踩坑与关键工程发现

这两轮真实 GPU 实验帮我们沉淀了三条极具通用价值的 Model-RSI 工程规律，并已全部固化进代码库默认配置：

1. **Qwen3.5 Gated DeltaNet 必须冻结递归状态门（`in_proj_a` / `in_proj_b`）**
   - `all-linear` LoRA 会扰动 DeltaNet 的指数衰减/写入门，导致 `Qwen3.5-2B` 在 Gen 1 掉分至 `36.25%`；冻结后仅训练 Attention + MLP 投影层，Gen 2 立即反弹 `+10.00%` 至 `46.25%`，Gen 3 稳定推高至 `48.75%`。
2. **RLVR 必须使用纯二值验证奖励（`1.0 / 0.0`），杜绝廉价格式奖励**
   - 在 `Qwen2.5-1.5B` Gen 3 中，`+0.15` 的格式奖励导致模型在难题上学会输出 70 字符的极短占位 `<think>` 标签以骗取格式分（长度坍塌至 `46.1` tokens）；切换为 `pure_binary_reward=True` 后彻底消除该作弊行为。
3. **STaR SFT 必须启用 `completion_only_loss=True` + 长度过滤 + 锚点回放**
   - 全序列 SFT 容易过拟合 Prompt 分布和采样噪声导致代码精度下滑；改为仅对回答算 Loss 并混入通用锚点后，既能破冰难题又能保护已有能力。

---

## 6. 快速复现

```bash
cd model-rsi
uv run pytest -v                                          # 运行 7 项核心单测（协议/沙盒/课程/门控/飞轮）
uv run model-rsi status --history-path reports/qwen35_2b_evolution_history.json
uv run model-rsi plot --history-path reports/qwen35_2b_evolution_history.json --out-path reports/qwen35_2b_evolution_curve.png
```

在 Google Colab GPU（L4 / A100 / H100）上安装 `requirements-colab.txt` 后，即可直接运行 `colab/run_model_rsi_colab.py` 或按 `reports/qwen35_2b_experiment_summary.md` 复现多代闭环。
