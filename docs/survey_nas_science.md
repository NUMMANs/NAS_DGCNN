# NAS 与科学计算 / 物理 / 工程结合调研报告（"NAS + Science" 脉络）

> 调研时间：2026-09-24 ｜ 方法：WebSearch + WebFetch 核实标题/会议/年份/arXiv 号
> 标记约定：**[已核实]** = 已读到可核实的论文元信息；**[推测/未找到]** = 检索未命中或仅为作者推断，请勿当作事实引用。

---

## 0. 结论速览

"NAS for Science / Physics-informed NAS" 并非一个统一子领域，而是沿三条主轴展开：
1. **把 NAS 用于 PDE 求解器/神经算子架构**（最贴近"Neural Operator NAS"）—— 代表是 XD-Operations、Auto-PINN / NAS-PINN / Auto-PICNN。
2. **把物理定律/PDE 残差作为 NAS 的搜索信号或约束**（Physics-informed NAS）—— 多出现在 PINN 谱系。
3. **用对称性（等变）作为架构归纳偏置**—— 主体是等变网络（NequIP/MACE/EGNN），但**严格意义的"E(3)-equivariant NAS"我未检索到主导性同行评审论文**，属于开放/新兴小方向。

一个关键事实：在科学任务上，SOTA NAS 经常**打不过手工设计的专家架构**（如 Darcy Flow 上 FNO 击败 NAS），且"搜索空间比搜索算法更重要"在科学场景同样成立（见第 5 节）。

---

## 1. 代表性论文总表

| 论文 | 年份/会议(期刊) | arXiv | 搜索空间 | 物理注入方式 | 关键结论 |
|---|---|---|---|---|---|
| **Rethinking Neural Operations for Diverse Tasks**（XD-Operations）Roberts, Khodak, Dao, Li, Ré, Talwalkar | 2021 NeurIPS | [2103.15798](https://arxiv.org/abs/2103.15798) | 操作空间：用 Kaleidoscope 矩阵把卷积、FNO、图卷积连续化插值 | 把 FNO/卷积当作操作空间的**特例**（归纳偏置写入操作） | 在 Burgers/Darcy/NS、蛋白折叠距离预测上优于基线且参更少；可 warm-start 卷积 |
| **Auto-PINN: Understanding and Optimizing Physics-Informed Neural Architecture** Wang et al. | 2022 preprint / 2023 NeurIPS ML4PS Workshop | [2205.13748](https://arxiv.org/abs/2205.13748) | MLP 超参（深度/宽度/激活/连接方式/残差） | PINN 训练损失作为搜索目标；PDE 残差进 loss | 首个系统化 PINN 超参优化；发现各超参可解耦 |
| **NAS-PINN: NAS-guided PINN for solving PDEs** Wang Yifan, Zhong Linlin | 2024 J. Comput. Phys. 496:112603 | [2305.10127](https://arxiv.org/abs/2305.10127) | DARTS 式连续松弛 + mask（不同形状张量相加） | 双层优化（内：权重；外：架构参数）；PDE 残差进 loss | Poisson/Burgers/Advection 及不规则域优于手工；浅而宽网络有时更好 |
| **Auto-PICNN: AutoML for physics-informed CNNs** Zhou, Chu | 2024 preprint / 2025 Commun. Phys. (Nature) s42005-025-02414-5 | [2407.06151](https://arxiv.org/abs/2407.06151) | 两阶段：损失函数空间（operator-infused）+ CNN 架构（标准/UNet 整结构） | **算子注入的损失搜索空间**；PDE 驱动 loss + 验证集相对 L2 作目标 | 6 类 PDE（热/流/多孔介质），最高 59.8× 误差下降；约 2 天内自动发现 |
| **FourCastNet**（Adaptive FNO，天气）Pathak et al. | 2022 arXiv（后成期刊） | [2202.11214](https://arxiv.org/abs/2202.11214) | **非 NAS**；架构 = AFNO-in-ViT | 球面等变（几何信息写入架构） | 全球 0.25° 天气，比 IFS 快~4.5 万倍；[推测]天气领域 NAS 应用很少，主流是手工基础模型 |
| **NAS-Bench-360** Tu, Roberts, Khodak, Shen, Sala, Talwalkar | 2022 NeurIPS D&B | [项目页](https://nb360.ml.cmu.edu/) | NAS-Bench-201 的 CNN cell 空间 | 无（基准） | 含 Darcy Flow PDE、PSICOV 蛋白折叠；SOTA NAS 在科学任务上**不稳定**，专家 FNO 击败 NAS |
| **Evaluating the Search Phase of NAS** Yu, Sciuto, Jaggi, Musat, Salzmann | 2020 ICLR | — | DARTS/ENAS/NAO 空间 | 无 | 随机搜索可匹敌 SOTA；权重共享破坏候选排序 |
| **Multi-Fidelity Multi-Objective NAS** Wei, Ooi, Ong | 2025 arXiv | [2509.01943](https://arxiv.org/abs/2509.01943) | cell-based U-Net 连续编码 | 无（但用于 Darcy flow / 城市风场） | Co-Kriging 代理；精度 vs 复杂度多目标；用于城市风速回归 |
| **PEDS: Physics-enhanced deep surrogates** Pestourie et al. | 2023 Nat. Mach. Intell. | [2111.05841](https://arxiv.org/abs/2111.05841) | 非 NAS，但给出廉价代理范式 | 低保真物理模拟器 + NN 生成器 | 训练数据少 100×；可作为 NAS 的廉价代理 |
| **Agentic AI Scientific Community for Automated Neural Operator Discovery** | 2026 arXiv（近期，[推测]未正式发表） | [2607.12122](https://arxiv.org/abs/2607.12122) | operator-block genome，LLM agent 进化/粒子群 | 适应度 = 确定性训练误差 | LLM planner 提出多族混合架构；"无免费午餐"算子结论 |

**补充（等变网络，非 NAS，作为物理约束"写进架构"的对照）：** TFN (Thomas et al. 2018)、SE(3)-Transformer (Fuchs et al. 2020)、EGNN (Satorras et al. 2021)、NequIP (Batzner et al., Nat. Commun. 2022)、MACE (Batatia et al., NeurIPS 2022)、EquiformerV2。

> **[推测/注意]** 检索中出现的 "Neural operator search"（Pattern Recognition 2023）是**视觉**（CIFAR/ImageNet）论文，与 PDE 算子无关，属命名撞车，请勿误引为科学 NAS。

---

## 2. 物理约束是怎么注入搜索的？

分三档，科学 NAS 比 CV NAS 多了一层"物理信号"：

1. **只当 loss 惩罚（最普遍）**：PINN 谱系把 PDE 残差、边界/初值残差写进损失；Auto-PINN 进一步用"训练损失"本身作为 NAS 的搜索目标，NAS-PINN 用双层优化让架构参数在外循环被更新。
2. **写入搜索空间/操作（改搜索算子）**：XD-Operations 是典型——它把卷积与 FNO 都定义为同一族"Expressive Diagonalization"操作的特例，于是搜索是在"操作族"上连续插值，而非从离散候选里挑；这等于把"谱方法归纳偏置"编进了搜索空间。FourCastNet/AFNO 把**球面等变**写进架构（几何信息编入），使长时 rollout 不发散。
3. **写入损失搜索空间**：Auto-PICNN 显式构造 "operator-infused loss function search space"（含残差调节操作），把物理量（守恒/加权策略）做成可搜的超参。

关于你特别问的两点：
- **物理引导的搜索空间剪枝**：**[推测/未找到强证据]** 我在同行评审文献里没看到命名明确的"physics-guided search-space pruning"论文；仅有博客/工程叙述提到对 PINN 用剪枝降搜索代价（如固定其他超参、一次搜一个），以及 Auto-PICNN 选用"整结构搜索空间而非 cell-based"来容纳物理结构。属新兴实践，需谨慎引用。
- **E(3)-equivariant NAS**：**[未找到]** 我没有检索到一篇以 "E(3)-equivariant neural architecture search" 命名、且被同行评审的工作。等变**架构**已成熟（上表补充），但"在等变操作集合上跑 NAS"基本是空白——合理推断这是你项目的潜在差异化切入点。

---

## 3. 搜索目标（fitness）的特点

科学计算的 fitness 确实常**不是精度**，而是"物理一致性 + 外推/泛化"：

- **Auto-PICNN 的关键发现**：直接用 PDE 驱动 loss 作搜索目标效果差，改用**验证集相对 L2 误差**更稳，因为物理 loss 与预测误差在不同 PDE 实例间不均匀。
- **外推/长时稳定性**：FourCastNet 的设计目标之一是"长时 rollout 不发散"，靠球面等变实现（而非在 loss 里加惩罚）；天气/气候 NAS 的隐式 fitness 是"对未见参数与长时程的稳定"。
- **多目标 NAS（精度 vs 物理一致性 vs 效率）**：**[已核实部分]** Wei et al. (2025, arXiv:2509.01943) 做了精度 vs 计算复杂度的多目标+多保真 Co-Kriging 搜索，并用于 Darcy flow 与城市风场；物理一致性在该工作中折叠进损失而非独立目标。**纯"精度 vs 物理一致性 vs 效率"三目标**的专门论文我未检索到，属可研究空白。

---

## 4. 训练代价与廉价代理

PDE 求解训练贵，科学 NAS 用以下代理：

- **低保真物理模拟器**：PEDS（Nat. Mach. Intell. 2023）用粗网格/可解释低保真求解器作瓶颈，NN 只做残差修正，训练数据少 100×——可直接当 NAS 的廉价 evaluator。
- **Co-Kriging / 多保真代理**：Wei et al. 用聚类局部多保真填充的代理模型加速多目标 NAS。
- **表格化基准**：NAS-Bench-360 对 Darcy Flow 预计算 15,625 个架构性能，搜索算法可在"秒级查表"而非训练，是科学 NAS 最干净的低成本评测范式。
- **零成本代理（NASWOT/TE-NAS 等）**：在通用 NAS 里成熟，**[推测]** 在 PDE 算子搜索上尚未成体系应用（因初始化激活的统计量与物理 loss 关联弱）。
- **predictor-based / 性能外推**：普遍存在于 NAS 综述，但科学算子场景多为上述"低保真仿真 + 表格基准"替代，专门的性能预测器论文较少。

---

## 5. 公认痛点与批评

1. **"搜索空间比搜索算法重要"**：Yu et al. (ICLR 2020) 证明在同一空间里随机搜索可匹敌 DARTS/ENAS/NAO，权重共享严重破坏排序。XD / FNO 的强表现也主要来自"人设计的好操作空间"。
2. **科学任务上 NAS 经常输给专家架构**：NAS-Bench-360 显示，在 Darcy Flow 上专家设计的 **FNO 显著击败 NAS**；且在低算力下"买现成 WRN 比做 NAS 更划算"（buy-vs-build 困境）。这直接质疑"科学计算里 NAS 收益被高估"。
3. **物理 loss 地形病态**：PINN 的多目标 loss（数据+PDE+边界）梯度量级差数阶，标准 NAS 依赖干净验证信号，易选到"零力平凡解"；单次候选评估还需高阶自动微分（如 3D 板弯曲需 4 阶导数），搜索代价可能超过有限元本身。
4. **可复现性**：NAS 通病（单次运行、跨种子波动大）在科学场景叠加"数据由昂贵仿真生成"，复现更难。

---

## 6. 对你（跨域点云分割 NAS）的启示

- 你"跨领域结合"的科学支线，核心是**把领域归纳偏置（对称性/守恒/几何）写进搜索空间或操作族**，而不是只做 loss 惩罚——XD-Operations 与 AFNO 的等变设计是范本。
- 点云具 SE(3)/E(3) 对称，而"E(3)-equivariant NAS"近乎空白 → 这是高价值切入点（在 EGNN/NequIP 类操作集合上跑 NAS）。
- 评测务必设"跨域外推/长时稳定"为目标，并优先用低保真代理（如粗网格求解器、表格基准）压低搜索代价；同时以"随机搜索/固定强基线"作对照，避免重蹈"搜索算法无效却靠空间取胜"的坑。
- 教训：先投入精力设计**好的跨域搜索空间**（含等变原语、跨域可迁移 cell），其收益大概率高于打磨搜索策略。

---

## 参考文献（可核实链接）

- XD-Operations: https://arxiv.org/abs/2103.15798
- Auto-PINN: https://arxiv.org/abs/2205.13748
- NAS-PINN: https://arxiv.org/abs/2305.10127 （J. Comput. Phys. 496:112603, 2024）
- Auto-PICNN: https://arxiv.org/abs/2407.06151 （Commun. Phys. s42005-025-02414-5, 2025）
- FourCastNet: https://arxiv.org/abs/2202.11214
- NAS-Bench-360: https://nb360.ml.cmu.edu/ （NeurIPS 2022 D&B）
- Evaluating the Search Phase of NAS: ICLR 2020（Yu, Sciuto, Jaggi, Musat, Salzmann）
- Multi-Fidelity Multi-Objective NAS: https://arxiv.org/abs/2509.01943
- PEDS: https://arxiv.org/abs/2111.05841 （Nat. Mach. Intell. 2023）
- Agentic Neural Operator Discovery: https://arxiv.org/abs/2607.12122
- 等变架构：TFN 2018；SE(3)-Transformer (Fuchs et al. 2020)；EGNN (Satorras et al. 2021)；NequIP (Batzner et al., Nat. Commun. 2022)；MACE (Batatia et al., NeurIPS 2022)
