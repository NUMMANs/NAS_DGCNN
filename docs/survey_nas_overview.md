# NAS 领域权威调研：综述、基准、2023–2026 趋势与关键争议

> 调研目的：为一份"NAS 跨领域结合（跨域点云分割）"综述提供骨架与大方向判断依据。
> 背景锚点：作者在 DGCNN 结构族内实证发现——"按密度/房间/域条件化架构"的 headroom = +0.0000 mIoU（拿不到收益），唯一显著主效应是池化 max vs mean 差 0.07 mIoU。本报告据此对照领域公认结论。
> 标注约定：【读】=检索到的原文结论；【推】=基于证据的推断（非原文明示）。全部链接均来自检索结果，可核实。

---

## 1. 权威综述（标题 / 期刊会议 / 年份 / 链接）

| 综述 | 出处 | 年份 | 链接 | 要点 |
|---|---|---|---|---|
| Neural Architecture Search: A Survey | JMLR 20:1–21 (Elsken, Metzen, Hutter) | 2019 | https://arxiv.org/abs/1808.05377 | 奠基性三元框架：搜索空间 / 搜索策略 / 性能估计策略。被引约 4600+ |
| Neural Architecture Search: Insights from 1000 Papers | arXiv:2301.08727 (White, Safari, Sukthanker, Ru, Elsken, Zela, Dey, Hutter) | 2023 | https://arxiv.org/abs/2301.08727 | 号称"1000 篇论文"的系统分类，覆盖搜索空间/算法/加速技术、基准与最佳实践；公认最新全景综述 |
| Automated Machine Learning: Methods, Systems, Challenges (AutoML Book) | Hutter, Kotthoff, Vanschoren (Eds.), Springer | 2019 | https://www.automl.org/book/ | 含 NAS 章节，定义 NAS 为 AutoML 子领域；方法/系统/挑战三视角 |
| Neural Architecture Search Survey: A Computer Vision Perspective | PMCID PMC9920395 | 2022 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9920395/ | 偏 CV，系统梳理 DARTS 系改进（DARTS+/FairDARTS/P-DARTS/PC-DARTS）与评估策略 |
| Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap | arXiv:2401.10034 (Wu, Wu, Wu, Tan 等) | 2024 | https://arxiv.org/abs/2401.10034 | LLM+进化计算融合路线图，含 IV-A "NAS"专节（GPT-NAS、LLMatic 等） |
| Multi-objective deep learning: Taxonomy and survey of the state of the art | ScienceDirect (Elsevier) | 2025 | https://www.sciencedirect.com/science/article/pii/S2666827025000830 | 多目标深度学习（含 MO-NAS）最新分类与训练算法税 |
| Foundation-model / LLM for NAS 相关工作（非综述，散见 2023–2026） | 见第 3 节 | — | — | "NAS for LLM / LLM for NAS"多为论文而非综述，综述尚在形成中 |

> 【读】"Insights from 1000 Papers"（2023）是目前最常被引用的"新综述"；它明确承接 Elsken 2019 的三元框架并扩展到 1000+ 篇。【推】截至 2026，尚未出现公认取代它的"Foundation-Model-NAS 综述"，相关讨论分散在 LLM×进化计算综述与个别论文中。

---

## 2. 基准与可复现性（含代理基准与排序相关性约定）

### 2.1 主流基准一览

| 基准 | 规模 / 任务 | 类型 | 意义与结论 | 链接 |
|---|---|---|---|---|
| NAS-Bench-101 | 423,624 个架构（CIFAR-10，DAG 节点取操作） | Tabular | 首个表格化基准，可查询，开启可复现 NAS | https://github.com/google-research/nasbench |
| NAS-Bench-201 | 15,625 / 6,466 非同构（CIFAR-10/100/ImageNet-16-120，边上取操作） | Tabular | 扩展可复现性至多数据集；10 种算法对评测 | https://arxiv.org/abs/2001.00326 |
| NAS-Bench-301 | DARTS 空间（~10^18），训练 60k 架构拟合代理 | Surrogate（首个） | 把代理基准推到 10^18 量级 | https://arxiv.org/abs/2008.09777 |
| NATS-Bench | 基于 NAS-Bench-201，额外变化 macro 架构 | Tabular | 区分 micro/macro 贡献 | https://arxiv.org/abs/2102.11535 |
| TransNAS-Bench | cell/macro 两空间 + 7 个 Taskonomy 任务 | Tabular | 跨任务（像素级/回归/自监督）NAS | https://arxiv.org/abs/2102.11315 |
| NAS-Bench-360 | 多异构任务（医疗、传感器等） | Tabular/Surrogate | 把 NAS 推向非 CV 科学任务 | https://arxiv.org/abs/2203.05356 |
| NAS-Bench-Suite | 统一接口集成 25 个搜索空间×数据集组合 | 框架 | 结论：在 NB-101/201 上强 ≠ 在别处强；超参不可跨空间迁移 | https://arxiv.org/abs/2201.13396 |

### 2.2 代理基准（surrogate）与排序相关性（Kendall τ / Spearman ρ）

【读】"How Powerful are Performance Predictors in Neural Architecture Search?"（White, Zela, Ru, Liu, Hutter, **NeurIPS 2021**, https://arxiv.org/abs/2102.09565 / https://papers.neurips.cc/paper_files/paper/2021/file/ef575e8837d065a1683c022d2077d342-Paper.pdf）：
- 第一次大规模比较 31 个性能预测器，覆盖 NAS-Bench-101/201/301/NLP，指标含 **Pearson、Spearman、Kendall τ、sparse Kendall τ**。
- 关键发现：不同家族（model-based / learning-curve / zero-cost / weight-sharing）**互补**，组合（OMNI）可比单预测器高至多 ~30% Kendall τ；但在"mutation-based（近邻）测试集"上，所有预测器都**难以区分非常相似的架构**——这与你"条件化架构 headroom = 0"高度呼应：架构差异若落在噪声地板之下，预测/排序都无能为力。
- 【读】该文把 Kendall τ 作为主排序指标，但**未给出统一"通过阈值"约定**；社区惯例是把 τ>0.6–0.7 视为"可用"，τ<0.3 视为"近乎随机"，但这是隐性约定而非标准。

【读】NAS-Bench-Suite（Mehta, Zela, White, Hutter, **ICLR 2022**, https://arxiv.org/abs/2201.13396）：
- 用 **Spearman 秩相关**评估黑盒预测器对留出 200 架构的排序能力；结论：RF 默认最佳（avg rank 1.57），XGBoost 经每基准 HPO 最佳（1.23）；NAO 在 NB-101/201 最佳却在他处很差。
- 核心警示：**"基于少数基准得出的结论不可泛化"**——这直接支撑"单点云数据集上的微小收益不足以证明条件化架构有效"。

> 关于你点名的 "How Reliable is Your NAS Benchmark?"：【查无此精确标题】。检索到的同类"可靠性"论著有：① NAS-Bench-Suite（上述"结论不泛化"）；② "NAS evaluation is frustratingly hard"（Yang et al., **ICLR 2020**, https://arxiv.org/abs/1912.12522，见第 4 节）；③ NAS-Bench-201 原文对"验证/测试相关性"的讨论。若你指的是其中某一篇，结论方向一致。

---

## 3. 2023–2026 最新趋势（重点）

### 3.1 NAS + LLM / Foundation Model
【读】LLM 作架构**生成器**：
- **LLMatic**（Nasir et al., **GECCO 2024**, https://arxiv.org/abs/2306.01102）：用 CodeGen 在 QD（质量-多样性）框架内生成架构变体，CIFAR-10/NAS-Bench-201 上仅评估 2000 个候选即达 SOTA 邻域。**原文明确：LLM 单独通过 prompt 无法直接做 NAS，必须嵌入搜索循环（QD）**。这本身就是"LLM 不是银弹"的证据。
- **GPT-NAS**（Yu et al.，载于 EC×LLM 综述 arXiv:2401.10034）：GPT 预测优良架构以引导 EA 搜索。
- **LEMONADE**（Rahman et al., **Scientific Reports 15:16871, 2025**, https://www.nature.com/articles/s41598-025-97378-5）：ChatGPT-4o 作后端，无需预定义搜索空间，CIFAR-10 95.54% / CIFAR-100 79.43%。
- **From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures**（arXiv:2601.02997, 2026）：评估 LLM 是否真能"创造"而非记忆架构。
【推】"Are LLMs good at architecture design?" 目前答案偏向：**LLM 能提供有知识的变异/初始化，但离"端到端替代搜索"尚早**；收益主要来自把先验知识注入搜索，而非架构本身更优。

### 3.2 NAS for LLM / 高效 LLM 的自动结构搜索
【读】
- **LLaMA-NAS**（arXiv:2405.18377, https://arxiv.org/abs/2405.18377）：对 LLaMA2-7B 做 one-shot NAS，在 Alpaca 上微调后进化搜索层数/MLP 中间维度，得到 **1.5× 更小、1.3× 加速、精度近乎无损**的子网络；证明"预训练 LLM 对某任务参数过剩"。
- 进化计算综述（arXiv:2401.10034）列 **AutoTinyBERT、LiteTransformerSearch**：在 Transformer 每层独立搜 hidden/heads/FFN 维度，建立 decoder 参数量↔困惑度的免训练代理。
- 【读】资源类综述（rework.com "What is NAS?"）直言：**"对语言任务，微调预训练基础模型几乎总是胜过从零搜 NAS 架构"**——NAS 的价值转向专用域/边缘部署/新硬件。
【推】"NAS for LLM"主要收益是**压缩与硬件适配（层数/宽度/MoE 路由）**，而非发现全新拓扑；更接近超网剪枝/结构化搜索，而非经典 cell 搜索。

### 3.3 对"NAS 被 scaling law 取代"的批评
【读】
- NAS 1000 论文综述与 theneuralbase 等资料一致指出：**最优架构是"算力预算相关"的**——Chinchilla 最优 ≠ 边缘设备最优，搜索边界必须把延迟/内存作为硬约束。
- 【读】rework.com 明确：在语言任务上"fine-tuning a foundation model beats a NAS-optimized architecture from scratch"。
【推】在 scaling law 框架下，**规模（参数/数据/算力）是主导杠杆，架构搜索的边际价值集中在"为该预算找 compute-optimal 的 macro 配置"（深度/宽度/扩张比/层数/注意力变体）**——这恰好再次印证"macro > micro"。所谓"NAS is dead"并非字面成立，而是"从零搜 cell 已无意义，架构搜索退化为预算约束下的 macro 配置搜索"。

### 3.4 多目标 / 多任务 / 迁移 NAS 新进展
【读】
- **MO-EMT-NAS**（跨数据集多任务多目标进化框架，https://chatpaper.com/zh-CN/paper/98461）：用超网络权重共享 + 辅助目标缓解"小模型陷阱"，在 2–4 任务组合上比单目标 MT-NAS 省时 59.7%–77.7%。
- **LibMOON**（NeurIPS D&B 2024, https://github.com/Baijiong-Lin/Awesome-Multi-Objective-Deep-Learning）等多目标库成熟，MO-NAS 进入可复现阶段。
- 【推】对跨域点云分割，最契合的方向是**多目标/多任务 NAS（精度 vs 计算 vs 跨域鲁棒）**，而非单目标"条件化 cell"。

### 3.5 AutoML 与 NAS 融合 / AI-Generated AI
【读】
- **AutoML-Zero**（Real, Liang, So, Le, **ICML 2020**, https://arxiv.org/abs/2003.03384）：从基本数学运算进化出完整学习算法，在 CIFAR-10 变体上**重新发现 SGD/ReLU/梯度归一化/权重平均**，证明"最小化人类偏置的搜索可发现已知技术"。这是"AI 生成 AI"的极端形式。
- 【推】AutoML-Zero 与你的观察形成有趣对照：当搜索空间足够"开放"，架构/算法差异才有大 headroom；而**在高度工程化的封闭空间（如 DGCNN 族）内做微条件化，headroom 天然趋近于 0**。

---

## 4. 关键争议（与你的实证直接对照）

### 4.1 "搜索空间比搜索算法更重要"
【读】强证据来自 **Yang, Esperança, Carlucci, "NAS evaluation is frustratingly hard", ICLR 2020, https://arxiv.org/abs/1912.12522**：
- 提出以"相对随机平均架构的改进（RI）"做公平比较，**发现许多 NAS 方法仅比随机平均架构高 0.3%–0.7%，部分数据集甚至不如随机采样**。
- 在 DARTS 空间采样 200+ 架构全训练，**top-1 准确率全落在约 1 个百分点之内**（均值 97.03%，std 0.23）。
- 四条结论直击要害：**(i) 训练协议（trick）比架构选择影响大得多；(ii) 同空间架构彼此极相似；(iii) 种子/层数等超参对排名影响显著；(iv) 手工设计的 macro 结构比搜出的 micro 操作更重要。**
【读】White 等（1000 论文综述）同样强调：**随机搜索/局部搜索在高度工程化空间里是极强基线**，很多"先进算法"优越性存疑。
→ **对你：你的"条件化架构 = +0.0000"正是 (ii)(iv) 的实例——在已精心设计的 DGCNN 族里，micro 级（条件化）差异小于评测噪声；而 max vs mean 池化作为 macro/聚合选择，恰是少数能撬动 0.07 的主效应。完全符合"macro > micro"。**

### 4.2 DARTS 鲁棒性 / NAS 不可复现
【读】代表文献：
- **Zela et al., "Understanding and Robustifying Differentiable Architecture Search", ICLR 2020**（https://arxiv.org/abs/1909.11926）：揭示 DARTS 在深超网中偏向 skip-connection/池化导致**性能崩塌（performance collapse）**，且**架构参数与最终子网重要性不一致**。
- **DARTS-**（Chu et al., ICLR 2021, https://openreview.net/forum?id=KLH36ELmwIB）：加辅助跳跃连接消除不公平优势，并指出 Hessian 特征根作崩塌指标存在反例。
- FairDARTS (ECCV 2020)、P-DARTS (ICCV 2019)、PC-DARTS (ICLR 2020)、RobustDARTS (ICLR 2020)、SDARTS (ICML 2020) 等构成"救火"谱系。
【读】不可复现根因：**离散化偏差（discretization gap）+ 搜索/评估阶段深度差（depth-gap）+ 超参敏感**。

### 4.3 输入条件化 / 动态架构（dynamic, input-adaptive, MoD/MoE）收益是否被高估
这是你最关心的问题。检索到的系统评估如下：
- **McGill & Perona, "Deciding How to Decide: Dynamic Routing in ANNs", ICML 2017, https://proceedings.mlr.press/v70/mcgill17a.html**：系统评测动态路由，**结论：动态路由仅在"大量低难度决策"任务上更划算，在 CIFAR-10 上"仅略有优势（only a slight advantage）"，在 CIFAR-2 上以 1/3 算力达更高峰值精度**——收益是**效率导向而非精度导向**，且任务依赖。
- **Mixture-of-Depths**（Raposo et al., DeepMind, 2024, https://arxiv.org/abs/2404.02546 类）：在 ~50% 算力匹配稠密 transformer 损失；但**谨慎提示：自回归解码时路由预测器质量会限制训练期增益，推理增益未必在真实负载上复现**。
- 【读】性能预测文献（第 2.2 节）表明：**非常相似的架构在排序上不可区分**——这意味着"按密度/房间/域条件化"若只产生微小结构差异，其真实收益会淹没在训练方差里，天然难以被可靠测出。
【推】**目前没有检索到一篇以"跨域点云分割 + 条件化架构"为题并报告 headroom≈0 的论文**，但上述三股证据共同支持一个稳健判断：**在强基础结构（如 DGCNN/Transformer）内做输入/域条件化的 micro 级架构改动，其精度 headroom 通常 ≤ 噪声地板，而它的收益主要体现在"动态算力分配（效率）"而非"精度"**。你的 +0.0000 因此不是异常，而是该范式预期的常态。

---

## 5. Scaling Law vs NAS 的关系定位
- 【读】Chinchilla（Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models"）确立"参数/数据近似等比"最优比，使"从零为任务搜架构"让位于"在预算下配 compute-optimal 结构"。
- 【读】rework.com 明确"语言任务上微调基础模型 > 从零 NAS"。
- 【推】综合定位：**Scaling law 把 NAS 从"发现新 cell"降级为"预算约束下的 macro 配置搜索 + 压缩/硬件适配"**。在 CV/点云等非基础模型主导、或资源受限/跨域鲁棒的场景，NAS 仍有价值，但价值集中在 (a) macro 与聚合方式（呼应你的 max/mean 池化主效应），(b) 多目标 Pareto（精度/算力/跨域），而非 micro 条件化。

---

## 6. 给你的"大方向"判断（基于上述读到的结论）

1. **你的实证与领域共识一致，不是负结果而是预期内**：在高度工程化结构族中，micro 级（按密度/房间/域条件化）架构 headroom 趋零；macro/聚合选择（max vs mean 池化）才是可测主效应。引用 Yang 2020 + White 2023 + 性能预测文献即可确立这一论点。
2. **"条件化架构收益有限"有间接但系统性的文献支撑**（McGill&Perona 2017、MoD 2024 的效率导向结论、性能预测器的近邻不可分性），但直接以"跨域点云分割"为题的报告尚缺——这是你综述可填补的空白。
3. **综述骨架建议**：权威综述（§1）→ 基准与排序指标（§2）→ 趋势（§3：LLM×NAS、NAS-for-LLM、scaling-law 批评、MO/迁移 NAS）→ 争议（§4：macro>micro、DARTS 不可复现、动态架构收益被高估）→ scaling-law 定位（§5）→ 落到跨域点云分割的启示。
4. **方法学提醒**：跨域点云分割的 NAS 论文应强制报告 (a) 相对随机/固定架构基线的 RI，(b) 多随机种子与多域的 Kendall τ/Spearman 排序一致性，(c) 在 ≥2 个基准/域上验证，避免被 NB-101/201 式"局部结论"误导（NAS-Bench-Suite 2022 教训）。

---

### 附：未能核实项
- "How Reliable is Your NAS Benchmark?" 这一精确标题**未检索到**，疑似与 "NAS-Bench-Suite"(ICLR 2022) 或 "NAS evaluation is frustratingly hard"(ICLR 2020) 混淆；已按相近可靠文献给出。
- "NAS is dead?" 类口号性讨论**未找到正式论文**，仅在行业博客/综述推论中出现；本报告以"scaling law 下 NAS 价值重定位"表述，避免夸张断言。
- 部分 2025–2026 论文（如 LEMONADE、From Memorization to Creativity、MoED SSRN）来自预印本/期刊，引用前建议再到 arXiv/期刊页核对最终版本号。
