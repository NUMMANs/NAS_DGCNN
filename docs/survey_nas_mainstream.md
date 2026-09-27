# NAS 跨领域结合：主流范式与脉络综述（主干章节）

> 调研时间：2026-09-24｜方法：WebSearch / WebFetch 交叉核实｜用途：为"NAS 跨域结合"综述提供主干章节
> 标注约定：**【读到的】** = 来自可核实论文/检索结果；**【推测的】** = 基于已有事实的外推，未找到原文直接佐证。

---

## 0. 一句话脉络

NAS 的十年主线是从"暴力试错"（RL/EA，数千 GPU-day）走向"可微分 + 权重共享超网"（DARTS、One-shot，单卡数 GPU-day），再走向"训练无关代理"（zero-cost proxies）与"LLM 生成式搜索"。但与此同时，大量批评文献指出：**在受限/同质的搜索空间里，NAS 找到的架构往往与随机搜索或手工设计差距很小，搜索阶段本身提供的增量信号有限**——这对"跨域点云分割 NAS"意味着：搜索策略不是瓶颈，**搜索空间设计与可信的性能评估（排序可靠性）才是关键**。

---

## 1. 三大主流搜索范式 + 较新方向

| 范式 | 代表作（会议/年份） | 核心思想 | 成本 | 链接 |
|---|---|---|---|---|
| 强化学习 RL | **NASNet** (Zoph et al., CVPR 2018)；**NAS with RL** (Zoph & Le, ICLR 2017) | RNN 控制器生成架构，验证精度作 reward（REINFORCE/PPO） | ~2000 GPU-day（CIFAR-10） | https://arxiv.org/abs/1707.07012 |
| RL + 权重共享 | **ENAS** (Pham et al., ICML 2018) | 单一超网共享权重，控制器采样子网络 | ~0.5 GPU-day | https://arxiv.org/abs/1802.03268 |
| 进化型 EA | **AmoebaNet / Regularized Evolution** (Real et al., AAAI 2019)；**NSGA-Net** (Lu et al., 2019) | 年龄正则锦标赛变异 / 多目标帕累托 | ~3150 GPU-day | https://arxiv.org/abs/1802.01548；https://arxiv.org/abs/1810.03522 |
| 可微分 | **DARTS** (Liu, Simonyan, Yang, ICLR 2019)；后续 P-DARTS/PC-DARTS/FairDARTS | 把离散边松弛为 softmax 混合操作，双级梯度优化 | ~1–4 GPU-day | https://arxiv.org/abs/1806.09055 |
| One-shot / 权重共享超网 | **SMASH** (Brock et al., ICLR 2018)、**Once-for-All** (Cai et al., 2019)、**SPOS/FairNAS** | 训一次超网，子网络继承权重即评估 | 一次性训练 | https://arxiv.org/abs/1708.05344 |
| 训练无关代理（zero-cost） | **NASWOT** (Mellor et al., ICLR 2021)、**SynFlow/Zero-Cost Proxies** (Abdelfattah et al., ICLR 2021)、**Zen-NAS** (ICCV 2021)、**TE-NAS** (Chen et al., CVPR 2021) | 用随机初始化网络统计量（梯度范数、线性区域数、NTK）排序，免训练 | 秒级/千架构 | https://arxiv.org/abs/2006.16966；https://arxiv.org/abs/2101.08134；https://arxiv.org/abs/2102.01063；https://arxiv.org/abs/2104.07857 |
| LLM 驱动 NAS | **EvoPrompting** (Chen, Dohan, So, NeurIPS 2023) | 用 LLM 作进化 NAS 的"变异/交叉"算子生成代码级架构 | 低–中 | https://arxiv.org/abs/2302.14838 |

**【读到的】** DARTS 把搜索成本从千级 GPU-day 降到个位数；ENAS 通过权重共享带来约 1000× 加速。**【推测的】** 严格地说，DARTS 类方法在跨域（如 3D 点云）上常出现"skip-connect 坍塌"与泛化差，需要早停/正则（见 §4、§6）。

---

## 2. 搜索空间的演化与"是否难超手工设计"

| 阶段 | 空间形态 | 代表 |
|---|---|---|
| Cell-based（微观） | 固定 normal/reduction cell 的 DAG，仅搜内部操作；再堆叠 | NASNet、DARTS、NAS-Bench-101/201 |
| Macro / 分层 | 同时搜主干宽度、深度、block 组合 | MnasNet、Auto-DeepLab、分级 NAS |
| CNN → Transformer | 把 depth/embedding/heads/MLP-ratio 作为弹性维度 | **AutoFormer** (Chen et al., ICCV 2021, weight-entanglement) https://arxiv.org/abs/2107.00651；ViT-NAS |
| CNN → GNN | 搜聚合函数(SUM/MEAN/MAX)、层数、hidden size、pooling | **GraphNAS** (Gao et al., IJCAI 2020) https://arxiv.org/abs/1904.09981；**Auto-GNN** (Zhou et al., 2021/2022)；**SANE** (arXiv:2104.06608) |
| 多模态 / 多任务 | 统一编码图像/文本/表格，搜融合策略与骨干 | AutoMM (AutoGluon-Multimodal, 2024) https://arxiv.org/abs/2404.16233 |

**各领域搜索空间差异（读到的）：**
- **GNN**：维度是"聚合函数/层数/隐藏维/pooling"，早期 GraphNAS、Auto-GNN 直接套用 RL+参数共享，但搜索空间过于受限（SANE 论文指出其无法覆盖 GeniePath、JK-Net 等），故转向 one-shot 可微分（SANE 为首个可微分 GNN-NAS）。
- **Transformer**：块同构性强，AutoFormer 用"权重纠缠"让大块权重被小块复用，使超网子网络无需重训即可直接用——这是 CV CNN 空间里少见的"搜索即部署"成功案例。
- **【推测的】** 你做**点云分割**，空间更接近 GNN（采样/分组/聚合/注意力）而非规则网格 CNN；建议自定义搜索维度（采样半径、邻域大小、是否用最远点采样、Transformer 层位置），而非直接套 DARTS cell。

**"手工设计的好架构已经很难被超过？" —— 是的，证据充分【读到的】：**
- Yang et al., **"NAS Evaluation is Frustratingly Hard"** (ICLR 2020, arXiv:1912.12522)：8 种 NAS 在 5 个数据集上，**多数未能显著超越"随机采样平均架构"**；cell 空间精度区间极窄，手设计的 macro 结构比搜出的 micro 操作更重要。
- **NAS-Bench-360** (Tu et al., ICLR 2022, arXiv:2110.05668)：在 16 个非图像任务上，**专家手工架构在多数任务上显著优于 NAS 找到的架构**（如 Darcy Flow：专家误差 0.008 vs 最佳 NAS 0.026）。
- 综合综述 Ren et al. **"A Comprehensive Survey of NAS"** (arXiv:2006.02903) 明言："*the performance gains obtained using NAS are limited compared to manually designed networks*"。
- 反向证据（说明 NAS 仍有价值）：AutoFormer 在 ImageNet 上确实超越 ViT/DeiT；EfficientNet/NAS-FPN 等也在其目标上超过手工基线——**结论应是"在同质小空间里收益有限，在搜索空间本身设计得好时仍有收益"**。

---

## 3. 跨领域迁移的 NAS

| 工作 | 做法 | 结论 |
|---|---|---|
| **Transfer Neural AutoML** (Wong et al., NeurIPS 2018, arXiv:1803.02780) | 多任务并行训 RL 控制器，再把控制器迁移到新任务 | 收敛时间降低一个数量级以上 |
| **MetaNAS** (Elsken et al., 2019, arXiv:1911.11090) | DARTS + REPTILE 联合元学习，新任务几步适配架构 | few-shot 分类 SOTA |
| **T-NAS / Warm-starting DARTS via meta-learning** (arXiv:2205.06355) | P-DARTS 搜"可迁移架构"按任务相似度热启动 | 减少新任务搜索成本 |
| **AT-NAS / H-Meta-NAS / EAT-NAS** | 进化+元学习或弹性架构迁移（小→大任务） | 跨任务架构可快速适配 |

**迁移结论【读到的/推测混合】：**
- NASNet 的 cell 在小数据（CIFAR-10）搜出后"堆叠迁移"到 ImageNet，是 NAS 早期标准范式，证明 **cell 级结构具有跨数据集可迁移性**。
- 但**架构的绝对性能（权重）不可直接迁移**，迁移的是"拓扑/连接方式"而非参数；且**同一搜出的 cell 在不同任务的相对排序并不稳定**（见 §5 的排序可靠性问题）。
- **【推测的】** 对你"跨域点云分割"：若源域（如 ScanNet）与目标域（如 KITTI/室外 LiDAR）分布差异大，cell 拓扑可能仍可迁移，但**必须用目标域的评测来重排**，不能信任源域 ranking。

---

## 4. 稳健性/鲁棒性 NAS 与输入条件化架构

**鲁棒性搜索（读到的）：**
- **When NAS Meets Robustness (RobNet)** (Guo et al., CVPR 2020)：用 one-shot 搜对抗鲁棒架构，发现"密集连接 + 在直连边加卷积"更抗攻击；RobNet 在白/黑盒攻击下比 ResNet/DenseNet 高约 5% 鲁棒精度。https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_When_NAS_Meets_Robustness_In_Search_of_Robust_Architectures_Against_CVPR_2020_paper.pdf
- **NAS-OoD** (Bai et al., 2021, arXiv:2109.02038)：用条件生成器合成分布偏移样本，联合最小化 OoD 损失来搜架构。
- **ARNAS** (Ou et al., CVPR 2024, arXiv:2405.05502)：为对抗训练搜"准确且鲁棒"架构，发现输入/输出附近应部署不同结构。

**输入条件化 / 动态 / 输入自适应架构（读到的）：**
- **Dynamic Routing Networks** (Cai et al., WACV 2021)：用轻量 hypernetwork（RouterNet）+ Gumbel-Softmax 为**每个输入实例**选择变换分支，推理 FLOPs 大幅下降且精度持平。
- **Learning Dynamic Routing for Semantic Segmentation** (arXiv:2003.10401)：分割中按输入尺度动态选分辨率路径（与 Auto-DeepLab 空间相关）。
- **MoE 路由**：Switch/GSparse-MoE 用门控网络按 token 动态激活专家（2020+），本质即"输入条件化容量"。

**收益争议【读到的/推测】：**
- **【读到的】** 条件化/动态架构在"实例难度差异大"时收益明显（简单样本走浅路、难样本走深路），**但若任务难度均匀，增益很小甚至因路由不稳定而下降**。动态路由引入的路由崩溃、训练不稳定是被反复讨论的问题。
- **【推测的】** 点云分割里"近密远疏、遮挡不均"，输入条件化（如按局部密度选聚合半径/专家）理论上很契合，但**目前缺乏在点云上的 NAS+动态路由实证**，收益需你自己验证，别预设它一定有效。

---

## 5. 预算与排序可靠性（含"短预算排序是否可靠"）

**基准与相关性度量（读到的）：**
- **NAS-Bench-101** (arXiv:1902.09635, 423K cell 全训练)、**NAS-Bench-201** (Dong & Yang, ICLR 2020, arXiv:2001.00326, 15,625 架构×3 数据集)、**NAS-Bench-301** (surrogate, arXiv:2008.09777, ~60K 架构 + GIN/XGBoost 代理)、**NAS-Bench-Suite** (White et al., ICLR 2022, arXiv:2201.13396, 25 基准统一评测)。
- 排序相关性主流用 **Kendall τ / Spearman ρ / S-KdT（稀疏 Kendall）**；训练无关代理论文报告 SynFlow 在 NAS-Bench-201 上 Spearman≈0.82（vs EcoNAS 0.61）。

**关键可靠性结论（读到的）：**
- **Sciuto et al., "Evaluating the Search Phase of NAS"** (arXiv:1902.08142)：**权重共享使搜索期 ranking 与真实 ranking 的 Kendall τ "close to zero"**——即搜索阶段几乎不提供排序信号；且 SOTA NAS 平均不比随机好。
- **NAS-Bench-1Shot1** (Bender et al., ICLR 2020, arXiv:1902.09587)：one-shot 验证误差与真实测试误差**相关性弱**，DARTS 甚至"test regret 随搜索 epoch 发散"。
- **"Evaluating Efficient Performance Estimators"** (Ning et al.)：系统揭示 one-shot / zero-cost 估计器存在**偏差与方差**，OSE 排序可能与独立训练排序不一致。

**"短预算训练得到的排序与完整训练排序不一致甚至负相关"——（重要澄清）：**
- **【读到的】** 我**未检索到**任何论文用"负相关（negative correlation）"作为结论来报告；可核实的最强表述是 **Kendall τ ≈ 0（无相关）**，以及 DARTS "diverge / 弱相关"。
- **【推测的】** "接近零相关"在实践上**等价于搜索信号不可靠**：排序既不正也不负，意味着短预算/权重共享给出的排序对选优没有预测力。若要严格证明"负相关"是否存在，建议你在自己的点云搜索空间里跑 NAS-Bench 风格的相关性实验（Spearman 随训练 epoch 变化曲线）——这是你项目里一个**可发表的小贡献点**。

---

## 6. 批评与反思

| 论文 | 核心批判 | 链接 |
|---|---|---|
| **Evaluating the Search Phase of NAS** (Sciuto et al., 2019) | SOTA NAS 平均不优于随机；权重共享破坏排序 | https://arxiv.org/abs/1902.08142 |
| **NAS Evaluation is Frustratingly Hard** (Yang et al., ICLR 2020) | 训练 trick 比架构更重要；cell 空间精度区间极窄；手设计 macro > 搜出 micro | https://arxiv.org/abs/1912.12522 |
| **Understanding and Robustifying DARTS** (Zela et al., ICLR 2020, arXiv:2002.05283) | DARTS 在新空间会坍塌为 skip；验证损失 Hessian 主特征值与泛化误差强相关，需早停/正则 | https://arxiv.org/abs/2002.05283 |
| **On the Importance of Search Space** (即 Yang 2020 脉络 + NAS-Bench-360) | 搜索空间与评估协议主导结果，搜索算法贡献有限 | arXiv:1912.12522 / 2110.05668 |
| **A Comprehensive Survey of NAS: Challenges & Solutions** (Ren et al., 2020) | "相比手工网络，NAS 收益有限"；呼吁强 baseline 与更多消融 | https://arxiv.org/abs/2006.02903 |
| **NAS: Insights from 1000 Papers** (White et al., 2023, arXiv:2301.08727) | 综述：DARTS 空间方差小，复杂搜索策略相对随机仅边际提升 | https://arxiv.org/abs/2301.08727 |

**"NAS is not a silver bullet"** 并非单一论文标题，而是上述综述与分析的共识：**NAS 是强大的自动化工具，但在受限/同质空间里收益有限、可复现性差、排序不可靠**。

---

## 7. 对你"跨域点云分割 NAS"项目的启示（综合）

1. **不要押注于"搜索算法"本身**：DARTS/RL/EA 在可控空间里差距很小（§1、§6）。优先投入**点云专属搜索空间**的定义（采样、分组、聚合、注意力、动态路由的候选操作）。
2. **用 zero-cost proxy 做初筛、用 NAS-Bench 风格代理做排序**：NASWOT/SynFlow/Zen-NAS 可秒级排千架构；但**必须先在目标域验证代理与真实精度的相关性**（§5 的"负相关"空白正是你的验证机会）。
3. **跨域迁移只迁拓扑、不迁权重，且用目标域重排**：源域 ranking 不可信（§3、§5）。
4. **稳健性/动态架构值得尝试但需设对照**：点云密度不均天然适合输入条件化（MoE/动态路由），但务必与固定架构做公平对比，防止"收益很小"的尴尬（§4）。
5. **报告时主动回应批评**：给出"相对随机/手工基线的提升"与"搜索期 vs 评估期 Kendall τ"，可一次性覆盖审稿人对 NAS 可复现性与排序可靠性的两大质疑。

---
*说明：GraphNAS-v2 / SAN 等名称在本次检索中未找到独立可核实论文，已并入 GraphNAS / SANE 脉络，未单列以免编造。*
