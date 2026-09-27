# NAS+ 跨领域结合调研总报告

> 调研日期：2026-09-24
> 目的：考察 NAS（Neural Architecture Search）如何跨领域结合、大方向是什么、都是怎样实现的
> 方法：5 路并行文献检索（WebSearch / WebFetch），所有论文均给可核实标题与链接
> **证据标记**：**【读到】**= 检索到可核实原文/元信息；**【推测】**= 基于事实的外推；**【未找到】**= 明确检索但无可靠来源

---

## 0. 一句话结论

NAS 的跨领域扩散遵循一条清晰规律：**领域知识注入的深浅，而不是搜索算法本身，决定了 NAS 在该领域能不能拿到收益**。

- 只在 **loss 上加领域惩罚** → 收益最小，常被随机搜索追平；
- 把领域结构写进 **搜索空间/搜索算子** → 这是唯一被反复验证有效的路径（XD-Operations 把 FNO 与卷积放进同一操作族、AFNO 把球面等变写进架构、Auto-PICNN 把算子塞进 loss 搜索空间）；
- 把领域约束做成 **多目标中的一等目标**（延迟、显存、物理一致性、域偏移） → 硬件感知 NAS 是 NAS 十个子方向里**收益最可靠的一支**。

而 **"输入条件化 / 动态架构"（按输入难度、密度、域动态切换结构）这一路，文献支撑最弱**：系统评估显示其收益是**效率导向而非精度导向**，且强任务依赖。这与本项目实测「按密度/房间/域条件化架构 headroom = +0.0000」高度自洽。

---

## 1. NAS 跨领域结合的全景图

```
                        ┌─────────────────────────────────────────┐
                        │        NAS 内核（三大范式已收敛）        │
                        │  RL → 进化 → 可微分/one-shot → 零成本代理 │
                        │            → LLM 生成式搜索              │
                        └────────────────┬────────────────────────┘
                                         │
        ┌────────────────┬───────────────┼───────────────┬────────────────┐
        │                │               │               │                │
   【科学计算/物理】  【CV/NLP/图】   【3D点云/LiDAR】 【垂直行业】    【系统/硬件】
        │                │               │               │                │
   • PINN-NAS        • AutoFormer      • Point-NAS     • 医疗 NAS-Unet  • MnasNet
   • NAS-PINN        • GraphNAS        • Auto-Points   • MM-NAS(多模态) • FBNet
   • XD-Operations   • SANE            • 3D-NAS(SPVConv)• 生信 BioNAS   • ProxylessNAS
   • Auto-PICNN      • LLMatic         • PointSeaConv  • 遥感 UDA-NAS★ • Once-for-All
   • FourCastNet     • AutoML-Zero     • LidarNAS      • 语音 Conformer • NSGA-Net
        │                │               │               │                │
   物理→搜索空间      空间设计主导     几何基元搜索    域偏移=一等目标    延迟=一等目标
        └────────────────┴───────────────┴───────────────┴────────────────┘
                                         │
                          【共同的批评与共识】
              • 搜索空间比搜索算法重要（Yang ICLR'20）
              • 手工设计难被超过（NAS-Bench-360、Ren 综述）
              • 权重共享破坏排序（Sciuto'19，Kendall τ≈0）
              • 动态/条件化架构：效率导向 > 精度导向
```

★ = 与"跨域"最直接相关，本项目最应精读

---

## 2. 三大主流范式（各领域共用的技术底座）

| 范式 | 代表作 | 会议/年份 | 链接 | 成本量级 |
|---|---|---|---|---|
| RL 控制器 | NASNet | CVPR 2018 | arXiv:1707.07012 | ~2000 GPU-day |
| RL + 权重共享 | ENAS | ICML 2018 | arXiv:1802.03268 | ~0.5 GPU-day |
| 进化（年龄正则） | AmoebaNet / Regularized Evolution | AAAI 2019 | arXiv:1802.01548 | ~3150 GPU-day |
| 进化 + 多目标 | NSGA-Net | GECCO 2019 | arXiv:1810.03522 | 8 GPU-day |
| 可微分 | DARTS | ICLR 2019 | arXiv:1806.09055 | 1–4 GPU-day |
| One-shot 超网 | SMASH / SPOS / Once-for-All | ICLR'18 / ECCV'20 / CVPR'20 | arXiv:1908.09791 | 一次性训练 |
| 训练无关代理 | NASWOT / Zero-Cost / Zen-NAS / TE-NAS | ICLR'21 / ICCV'21 / CVPR'21 | arXiv:2006.16966, 2101.08134, 2102.01063, 2104.07857 | 秒级 |
| LLM 驱动 | EvoPrompting / LLMatic / GPT-NAS | NeurIPS'23 / GECCO'24 | arXiv:2302.14838, 2306.01102 | 低–中 |
| 从零进化出算法 | AutoML-Zero | ICML 2020 | arXiv:2003.03384 | 高（但发现 SGD/ReLU） |

**收敛趋势【读到】**：RL/进化退居特定场景 → **DARTS/one-shot 为主流** → **zero-cost proxy 加速** → **LLM 生成式搜索兴起**。

**【读到】** DARTS 把搜索成本从千级 GPU-day 降到个位数；ENAS 带来约 1000× 加速；Zero-Cost Proxies 在 NAS-Bench-201 上 **Spearman 0.82**（vs EcoNAS 0.61）。

---

## 3. 跨领域结合的四种实现模式（核心）

这是本次调研最有价值的归纳：**领域知识注入 NAS 的深度有四个层次**。

### 模式 A：领域约束只进 loss（最浅，收益最小）

| 领域 | 论文 | 做法 | 结果 |
|---|---|---|---|
| PINN | **Auto-PINN** (arXiv:2205.13748, 2022/NeurIPS ML4PS'23) | PDE 残差进 loss，用训练损失当搜索目标 | 首个系统化 PINN 超参优化；各超参可解耦 |
| PINN | **NAS-PINN** (J. Comput. Phys. 496:112603, 2024; arXiv:2305.10127) | DARTS 式软松弛 + 双层优化 | Poisson/Burgers 优于手工；浅而宽有时更好 |

**【读到】** Auto-PICNN 的关键反面发现：**直接把 PDE 驱动 loss 当搜索目标效果差**，改用验证集相对 L2 误差才稳。→ **领域 loss 不是好的搜索信号。**

### 模式 B：领域结构写进搜索空间 / 搜索算子（最有效）

| 领域 | 论文 | 把什么写进空间 |
|---|---|---|
| 神经算子 | **XD-Operations** (NeurIPS 2021; arXiv:2103.15798) | 用 Kaleidoscope 矩阵把**卷积、FNO、图卷积**连续插值成同一操作族；FNO 成为特例 |
| 神经算子 | **Auto-PICNN** (arXiv:2407.06151; Commun. Phys. 2025) | 显式构造 **operator-infused loss search space**（含残差调节算子） |
| 天气 | **FourCastNet/AFNO** (arXiv:2202.11214) | **球面等变**写进架构 → 长时 rollout 不发散 |
| 点云 | **PointSeaConv** (ICCV 2021; arXiv:2108.12856) | 卷积 = "基本关联" DAG（欧氏距离、相对位置、全局上下文）；**搜几何关系组合**；S3DIS 71.9 mIoU，0.25 GPU-day；PointNet++/DGCNN/RS-CNN 都是其特例 |
| 点云 | **3D-NAS / SPVConv** (NeurIPS 2020; arXiv:2007.16100) | point↔sparse-voxel 融合算子 + 弹性通道/深度/分辨率；LiDAR 分割 1.8–23.7× 加速 |
| 点云 | **LidarNAS** (arXiv:2210.05018) | 宏观搜**视图/格式变换**（voxel/pillar/point/range/fusion 怎么串） |
| 遥感 | **UDA-NAS** (CVPR 2024) | NAS + **无监督域适应**（师生自训练 + MRF 搜索）；1.28M 参数 59.38 mIoU 超 DAFormer |
| 遥感 | **SGNAS** (Remote Sensing 2021; DOI 10.3390/rs13173460) | **空间广义化**搜索空间，专攻跨数据集场景分类 |

**共同点【读到】**：都是**先有人类专家设计的强操作/强基元**，NAS 在其上插值/组合。收益来自"空间里有好东西"，不是"搜索算法聪明"。

### 模式 C：把领域约束做成多目标一等目标（最可靠）

| 领域 | 论文 | 第二目标 | 做法 |
|---|---|---|---|
| 移动端 | **MnasNet** (arXiv:1807.11626) | 真实像素延迟 | 精度×(延迟/目标)^γ 的 RL 多目标 |
| 移动端 | **FBNet** (arXiv:1812.03443) | 延迟（LUT 查表） | DNAS + Gumbel-Softmax，延迟可微 |
| 通用 | **ProxylessNAS** (arXiv:1812.00332) | 目标硬件延迟 | 延迟作连续函数，省显存 |
| 通用 | **Once-for-All** (arXiv:1908.09791) | 多硬件 | 训一超网按深度/宽/核/分辨率提取子网 |
| 多目标 | **NSGA-Net** (arXiv:1810.03522) | FLOPs | NSGA-II 非支配排序 + 拥挤度，输出 Pareto 前沿 |
| 医疗边缘 | **NanoCNN** (MDPI 2026) | 少数类召回 + MCU 硬件 | 8-bit 量化算子 |
| 科学 | **Wei et al.** (arXiv:2509.01943, 2025) | 计算复杂度 | 多保真 Co-Kriging 代理；Darcy flow / 城市风场 |

**【读到】** 这是 NAS 十个子方向中**收益最可靠、最被工业界接受的一支** —— 因为"给定延迟下精度更高"是可量化、无争议的。

### 模式 D：输入条件化 / 动态架构（文献支撑最弱，需警惕）

| 工作 | 做法 | 关键结论 |
|---|---|---|
| **Dynamic Routing Networks** (Cai et al., WACV 2021) | 轻量 hypernetwork + Gumbel-Softmax 为**每个输入实例**选分支 | 推理 FLOPs 大降，精度持平 |
| **Learning Dynamic Routing** (arXiv:2003.10401) | 分割按输入尺度动态选分辨率路径 | 尺度自适应 |
| **Mixture-of-Depths** (DeepMind 2024) | 动态跳过 Transformer 块 | ~50% 算力匹配稠密损失；**但自回归解码时路由预测器质量会限制增益，推理增益未必在真实负载复现** |
| **McGill & Perona, "Deciding How to Decide"** (ICML 2017) | 系统评测动态路由 | **动态路由只在"大量低难度决策"任务上划算；CIFAR-10 上"仅略有优势"；CIFAR-2 上以 1/3 算力达更高峰值精度** |
| MoE / Switch-MoE | 按 token 路由专家 | 本质即"输入条件化容量"，但**路由崩溃/训练不稳定被反复讨论** |
| **RobNet** (CVPR 2020) | one-shot 搜对抗鲁棒架构 | 密集连接 + 直连边加卷积更抗攻击，白/黑盒高约 5% |
| **RobNAS** (ICASSP 2025) | 搜索期注入 PGD 对抗训练 | 点云分类对抗防御 SOTA；**针对对抗扰动，非密度/稀疏** |

**【读到】系统结论**：动态/条件化架构的收益**主要是"动态算力分配（效率）"，不是"精度"**，且强任务依赖 —— 任务难度均匀时增益很小甚至因路由不稳定而下降。

**【未找到】** 以"跨域点云分割 + 条件化架构"为题、报告 headroom≈0 的直接论文；**点云的密度/尺度条件化 NAS、非对抗鲁棒 NAS、E(3)-等变 NAS 均为文献空白**。

---

## 4. 分领域横向对比表

| 领域 | 代表论文 | 搜索什么 | 领域知识注入方式 | 报告收益 | 收益性质 |
|---|---|---|---|---|---|
| **3D 点云-分类/分割** | Point-NAS (TIP'23)、Auto-Points (TMM'24) | 弹性宽度/深度/采样点数（one-shot supernet） | 无特殊注入 | S3DIS 68.6 / 69.1 mIoU | 主效应型 |
| **3D 点云-几何基元** | PointSeaConv (ICCV'21) | 卷积内部几何关联 DAG | **几何关系写进算子** | S3DIS 71.9 mIoU，0.25 GPU-day | 空间设计型 |
| **LiDAR/自动驾驶** | 3D-NAS (NeurIPS'20)、LidarNAS、PVNAS | 通道+深度+分辨率 / 视图格式变换 | point-voxel 稀疏算子 | 1.8–23.7× 加速 | 部署优化型 |
| **点云-鲁棒** | RobNAS (ICASSP'25) | 架构级对抗鲁棒 | PGD 进搜索 | 对抗防御 SOTA | 单点方向 |
| **PDE/神经算子** | XD-Operations (NeurIPS'21)、NAS-PINN (JCP'24)、Auto-PICNN (2024) | 操作族连续插值 / loss 搜索空间 | **FNO/卷积进同一操作族** | 算子任务优于基线且参数更少；PICNN 最高 59.8× 误差下降 | 空间设计型 |
| **天气/气候** | FourCastNet (arXiv:2202.11214) | **非 NAS**，手工 AFNO | 球面等变 | 比 IFS 快 4.5 万倍 | — |
| **图神经网络** | GraphNAS (IJCAI'20)、Auto-GNN、SANE (arXiv:2104.06608) | 聚合函数/层数/隐藏维/pooling | 图聚合原语 | 早期空间过窄；SANE 首个可微分 GNN-NAS | 空间设计型 |
| **Transformer** | AutoFormer (ICCV'21)、ViT-NAS | depth/embedding/heads/MLP-ratio | 权重纠缠 | 超 ViT/DeiT | 空间设计型 |
| **医疗-分割** | NAS-Unet (IEEE Access'19)、Auto-DeepLab (CVPR'19) | U 形 DownSC/UpSC cell | 无 | 0.8M 参数超 U-Net | 自动化型 |
| **医疗-多模态** | MM-NAS (MICCAI'20) | 双流 PET/CT 融合模块 | **融合方式进空间** | 肉瘤转移预测超 SOTA | 空间设计型 |
| **生信** | BioNAS (arXiv:1909.00337)、GenomeNet-Architect (Commun. Biol. 2024) | 1D 序列 CNN cell | 可解释性进目标 | 误分 -19%、参数 -83% | 自动化型 |
| **遥感-跨域** | **UDA-NAS (CVPR'24)、SGNAS (RS'21)** | MRF + 师生自训练 / 广义化 cell | **域偏移=一等目标** | 59.38 mIoU 超 DAFormer | **跨域型 ★** |
| **语音** | NAS for SR (arXiv:2007.08818)、Conformer NAS (arXiv:2104.05390) | TDNN-F 瓶颈 / Conformer 模块 | 无 | WER -1.0% abs，模型 -28% | 主效应型 |
| **硬件感知** | MnasNet/FBNet/ProxylessNAS/Once-for-All/NSGA-Net | 深度宽核分辨率 + 延迟 | **延迟进目标** | 帕累托前沿 | **部署优化型（最可靠）** |
| **EEG/ECG** | CTNAS-EEG (arXiv:2210.06298)、NanoCNN (2026) | 跨任务 1D 空间 / 量化算子 | 跨人泛化 / 硬件 | 跨人跨任务 SOTA | 跨域 + 部署 |

---

## 5. 领域共识与"负面结论"（务必知道）

### 5.1 搜索空间 > 搜索算法（强证据）

| 论文 | 出处 | 证据 |
|---|---|---|
| **NAS evaluation is frustratingly hard** (Yang et al.) | ICLR 2020, arXiv:1912.12522 | 以"相对随机平均架构的改进 RI"公平比较：**许多 NAS 仅比随机高 0.3%–0.7%，部分数据集不如随机**；DARTS 空间采 200+ 架构全训练，**top-1 全落在约 1 个百分点内**（均值 97.03%，std 0.23） |
| **Evaluating the Search Phase of NAS** (Sciuto et al.) | arXiv:1902.08142 | **SOTA NAS 平均不优于随机**；**权重共享使 Kendall τ ≈ 0** |
| **Random Search and Reproducibility for NAS** (Li & Talwalkar) | UAI 2020, arXiv:1902.07638 | 随机搜索 + 早停 ≈ ENAS |
| **NAS-Bench-1Shot1** (Bender et al.) | ICLR 2020, arXiv:1902.09587 | one-shot 验证误差与真实测试误差相关性弱；DARTS 的 test regret 随搜索 epoch 发散 |
| **NAS: Insights from 1000 Papers** (White et al.) | arXiv:2301.08727 | "哪怕最好的搜索算法，也只能在搜索空间内最好的候选上产出结果" |
| **A Comprehensive Survey of NAS** (Ren et al.) | arXiv:2006.02903 | "NAS 相对手工网络的收益有限" |

**四条被广泛引用的结论（Yang 2020）**：(i) **训练 trick 比架构选择影响大得多**；(ii) 同空间架构彼此极相似；(iii) 种子/层数等超参对排名影响显著；(iv) **手工设计的 macro 结构比搜出的 micro 操作更重要**。

### 5.2 手工设计难被超过

| 证据 | 出处 | 数字 |
|---|---|---|
| **NAS-Bench-360** (Tu et al.) | ICLR 2022 D&B, arXiv:2110.05668 | 16 个非图像任务上，专家手工架构多数显著优于 NAS；**Darcy Flow：专家 FNO 0.008 vs 最佳 NAS 0.026** |
| 语言任务 | 多来源综述 | **"微调预训练基础模型几乎总是胜过从零搜 NAS 架构"** |

### 5.3 排序可靠性 —— 与本项目最相关的痛点

| 结论 | 出处 | 内容 |
|---|---|---|
| 性能预测器系统评测 | **White et al., NeurIPS 2021, arXiv:2102.09565** | 31 个预测器大规模对比；不同家族**互补**，组合(OMNI)可提升 Kendall τ 至多 ~30%；**但"mutation-based（近邻）"测试集上所有预测器都难以区分非常相似的架构** |
| 代理基准泛化性 | **NAS-Bench-Suite, ICLR 2022, arXiv:2201.13396** | 用 **Spearman** 评估；**"基于少数基准得出的结论不可泛化"**；超参不可跨空间迁移 |
| Kendall τ 约定 | 社区惯例【推测，非标准】 | τ>0.6–0.7 视为可用，τ<0.3 视为近乎随机 |

⚠ **重要澄清【读到】**：检索**未找到**任何论文以"**负相关（negative correlation）**"作为短预算排序的结论。可核实的最强表述是 **Kendall τ ≈ 0（无相关）**。

→ 本项目实测的 **ρ(25ep, 300ep) = −0.825**（显著负相关）**是文献中少见的结果**，属于**可发表的独立贡献点**：文献普遍只报到"无相关"，本项目给出了"短预算会系统性反转排序"的完整曲线（−0.825 → +0.943 → +0.993）。

### 5.4 DARTS 鲁棒性与不可复现

- **Zela et al., ICLR 2020, arXiv:1909.11926**：DARTS 在深超网中偏向 skip-connection 导致**性能崩塌**，架构参数与最终子网重要性不一致。
- 根因【读到】：**离散化偏差（discretization gap）+ 搜索/评估深度差（depth-gap）+ 超参敏感**。
- 救火谱系：DARTS- (ICLR'21)、FairDARTS (ECCV'20)、P-DARTS (ICCV'19)、PC-DARTS (ICLR'20)、RobustDARTS、SDARTS (ICML'20)。

### 5.5 Scaling Law 对 NAS 的降维

- **Chinchilla**（Hoffmann et al., 2022）确立参数/数据近似等比最优 → "从零为任务搜架构"让位于"**在预算下配 compute-optimal 结构**"。
- **【推测】定位**：Scaling law 把 NAS 从"发现新 cell"降级为"**预算约束下的 macro 配置搜索 + 压缩/硬件适配**"。
- 所谓"NAS is dead"**不是字面成立**，而是：**从零搜 micro cell 已无意义，架构搜索退化为 macro 配置 + 多目标 Pareto**。

---

## 6. 大方向判断（五个层面）

### 6.1 时间维度：从"找架构"到"找约束下的架构"

```
2017-2019          2019-2021           2021-2023          2023-2026
RL/进化             可微分/one-shot      零成本代理           LLM 生成 + 多目标
"能不能搜"          "搜得快不快"         "不用搜能不能排"     "搜什么才有意义"
数千 GPU-day        1-4 GPU-day         秒级/千架构          预算约束 + Pareto
```

### 6.2 空间维度：micro → macro

**【读到】** macro（深度、宽度、分辨率、聚合方式、视图/格式变换）的收益**系统性高于 micro（cell 内算子替换/条件化）**。LidarNAS 甚至发现车/行人两类分别搜出的宏观结构**竟一致**（说明宏观选择是主导的、稳定的）。

### 6.3 知识注入维度：loss → 空间 → 目标

**收益排序（本次调研最核心的判断）**：
```
目标注入（多目标 Pareto，延迟/显存/物理一致性）  ★★★ 最可靠
空间注入（把领域原语写进搜索算子/基元）          ★★  有效，需专家设计
loss 注入（仅加领域惩罚项）                      ★   常被随机搜索追平
条件化架构（按输入动态切换结构）                 ★   效率导向，精度 headroom 常 ≈ 0
```

### 6.4 任务维度：从"通用"到"跨域/鲁棒/部署"

**【读到】** 新兴且**真正有需求**的三个方向：
1. **跨域/领域泛化 NAS** —— UDA-NAS、SGNAS、AdaptNAS（NeurIPS'20）、NASDA（arXiv:2008.05706）
2. **鲁棒 NAS** —— RobNet (CVPR'20)、NAS-OoD (arXiv:2109.02038)、ARNAS (CVPR'24)
3. **硬件/资源感知 NAS** —— MnasNet/FBNet/ProxylessNAS/OFA（最成熟）

### 6.5 方法维度：多任务 / 多目标 / 迁移

- **迁移 NAS**：Transfer Neural AutoML (NeurIPS'18, arXiv:1803.02780)、MetaNAS (arXiv:1911.11090)、T-NAS (arXiv:2205.06355)
  **【读到】迁移结论**：cell 级**拓扑可迁移**，**权重不可迁移**；且**同一 cell 在不同任务的相对排序不稳定** → 目标域必须重排。
- **多任务多目标**：MO-EMT-NAS（比单目标 MT-NAS 省时 59.7%–77.7%）
- **NAS for LLM**：LLaMA-NAS (arXiv:2405.18377) 得到 1.5× 更小、1.3× 加速、精度近乎无损的子网；AutoTinyBERT、LiteTransformerSearch
  **【推测】** 收益是**压缩与硬件适配**，不是发现新拓扑。

---

## 7. 对本项目（跨域点云分割 NAS）的直接映射

### 7.1 本项目实测 vs 文献共识 —— 逐条对照

| 本项目实测结果 | 文献对照 | 判定 |
|---|---|---|
| 按密度/房间/域条件化架构 headroom = **+0.0000** | McGill&Perona ICML'17：动态路由"仅略有优势"；MoD 2024：效率导向；White NeurIPS'21：**近邻架构排序不可区分** | ✅ **与共识完全一致，不是异常** |
| 池化 **max vs mean 差 0.07 mIoU** 是最大主效应 | Yang ICLR'20 结论 (iv)：**macro 选择 > micro 操作**；LidarNAS 宏观结构稳定 | ✅ **正是文献预测的主效应** |
| **feature 加宽 +0.02** | Point-NAS / Auto-Points / 3D-NAS 的主收益**全部来自弹性宽度/深度** | ✅ **同源** |
| ρ(25ep, 300ep) = **−0.825** | 文献最强只到 **Kendall τ ≈ 0**（Sciuto'19） | ⚠ **本项目结果超出文献范围** |
| ρ(50ep, 300ep) = +0.943；100ep = +0.993 | NAS-Bench-Suite：**结论不可泛化**；需多基准验证 | ⚠ **需外部验证才能宣称普适** |
| 结构族**未饱和**（r(cheap,Δ)=+0.809） | XD-Operations/AutoFormer：好空间仍有收益 | ✅ 自洽 |

### 7.2 可以立论的两个空白

1. **"短预算排序负相关"** —— 文献只报到 τ≈0，本项目给出完整翻转曲线（−0.825 → +0.943 → +0.993）与机制（欠拟合惩罚随 k 变化 → seed 噪声从 0.0069 掉到 0.0003）。**这是可发表的独立贡献。**
2. **"跨域点云分割的条件化架构 headroom=0"** —— 检索**未找到**同题论文，但有系统性间接支撑（性能预测器的近邻不可分性 + 动态路由的效率导向结论）。**这是可立论的缺口。**

### 7.3 下一步方向建议（按预期收益排序）

| 优先级 | 方向 | 依据 | 具体做法 |
|---|---|---|---|
| **P0** | **转 macro 轴搜索** | Yang ICLR'20 (iv)；LidarNAS | 搜**视图/表示/融合方式**（point / pillar / voxel / 多分支 late-early fuse），而非条件化。已有多分支基础设施 (`multibranch.py`) |
| **P0** | **多目标 Pareto（精度 × 显存/延迟）** | MnasNet/FBNet/NSGA-Net 最可靠 | 点云显存是真实瓶颈（18 万点每 epoch 15s+）→ 显存约束下的 Pareto 前沿 |
| **P1** | **几何基元搜索（PointSeaConv 路线）** | PointSeaConv ICCV'21：S3DIS 71.9 mIoU | 搜"用哪些几何关系（欧氏距离/相对位置/全局上下文）组合"，而非搜 k |
| **P1** | **域适应与搜索同步（UDA-NAS 路线）** | UDA-NAS CVPR'24 | 把"域偏移"当一等目标；师生自训练 + NAS 联合 |
| **P2** | **等变原语 NAS** | **文献空白**（E(3)-equivariant NAS 未找到） | 在 EGNN/NequIP 类操作集合上跑 NAS |
| **P2** | **用 zero-cost proxy 替代短预算** | Zero-Cost ICLR'21：Spearman 0.82 | 用 NASWOT/SynFlow 替代 50ep 粗筛，需先验证在点云上的相关性 |
| **✗** | ~~条件化架构（按密度/房间/域）~~ | headroom +0.0000；文献亦不支持 | **已否证，不再投入** |

### 7.4 报告规范（规避审稿人质疑）

【读到】必须报告：
- (a) 相对**随机搜索/固定架构基线**的改进（RI），而非仅绝对值 —— Yang ICLR'20
- (b) **多随机种子 + 多域的 Kendall τ / Spearman** 排序一致性 —— White NeurIPS'21
- (c) 在 **≥2 个基准/域**上验证 —— NAS-Bench-Suite ICLR'22 教训
- (d) 搜索期 vs 评估期的排序相关性

---

## 8. 未找到 / 需澄清项（诚实声明）

| 项 | 状态 |
|---|---|
| "AutoBEV"、"SparseBackbone" 作为 NAS 论文 | **未找到**（检索只返回 BEV 综述与稀疏卷积检测器 SAFDNet/FSTR） |
| "How Reliable is Your NAS Benchmark?" 精确标题 | **未找到**，疑似与 NAS-Bench-Suite (ICLR'22) 或 Yang (ICLR'20) 混淆 |
| 点云 NAS **直接搜 k / 半径 / 图构建方式** | **未找到** —— 文献空白 |
| 点云**域适应 NAS** | **未找到** —— 文献空白 |
| 点云**密度/尺度条件化 NAS** | **未找到** —— 文献空白 |
| 点云**非对抗鲁棒 NAS** | **未找到** —— 文献空白 |
| **E(3)-equivariant NAS** | **未找到**同行评审主导论文 —— 开放方向 |
| NAS 改进 **AlphaFold/蛋白结构** | **未找到**（AlphaFold2/3 用手工 Evoformer + 扩散） |
| SAR 多光谱融合 NAS 专文 | **未找到** |
| "NAS is dead?" 正式论文 | **未找到**（仅行业博客/综述推论） |
| 天气/气候 NAS 命名论文 | **未找到**（FourCastNet 是手工设计，非 NAS） |
| **"短预算排序负相关"** | **未找到**任何论文报告 —— **本项目独有** |

**已纠正的常见误用**：
- **时序 Autoformer** (NeurIPS'21, arXiv:2106.13008) **不是 NAS**（分解 + 自相关架构）；被误认为 NAS。
- **W-Net 不是 NAS**（手工 U 形网络）。
- "Neural operator search" (Pattern Recognition 2023) 是**视觉**论文，与 PDE 算子无关（命名撞车）。
- **"NasSeg"、"NAC"** 命名论文未找到。

---

## 9. 五份章节报告索引

| 文件 | 覆盖范围 | 字数 |
|---|---|---|
| `docs/survey_nas_pointcloud.md` | NAS × 3D 点云 / LiDAR / 自动驾驶 | ~2500 |
| `docs/survey_nas_science.md` | NAS × 科学计算 / PINN / 神经算子 | ~3000 |
| `docs/survey_nas_mainstream.md` | NAS × CV / NLP / GNN / 主流范式 | ~3500 |
| `docs/survey_nas_applied.md` | NAS × 医疗 / 生信 / 遥感 / 信号 / 边缘 | ~3000 |
| `docs/survey_nas_overview.md` | 权威综述 / 基准 / 2023-26 趋势 / 争议 | ~3500 |

---

## 10. 关键文献速查（按重要性）

**必读（与本项目直接相关）**
1. Yang et al., *NAS evaluation is frustratingly hard*, ICLR 2020 — arXiv:1912.12522
2. White et al., *How Powerful are Performance Predictors in NAS?*, NeurIPS 2021 — arXiv:2102.09565
3. Sciuto et al., *Evaluating the Search Phase of NAS*, arXiv:1902.08142
4. White et al., *NAS: Insights from 1000 Papers*, arXiv:2301.08727
5. McGill & Perona, *Deciding How to Decide: Dynamic Routing in ANNs*, ICML 2017
6. Nie et al., *Differentiable Convolution Search for Point Cloud (PointSeaConv)*, ICCV 2021 — arXiv:2108.12856
7. Broni-Bediako et al., *UDA-NAS*, CVPR 2024

**方法论支柱**
8. Elsken et al., *NAS: A Survey*, JMLR 2019 — arXiv:1808.05377
9. Liu et al., *DARTS*, ICLR 2019 — arXiv:1806.09055
10. Zela et al., *Understanding and Robustifying DARTS*, ICLR 2020 — arXiv:1909.11926

**趋势**
11. Roberts et al., *XD-Operations*, NeurIPS 2021 — arXiv:2103.15798
12. Nasir et al., *LLMatic*, GECCO 2024 — arXiv:2306.01102
13. Real et al., *AutoML-Zero*, ICML 2020 — arXiv:2003.03384
14. Tu et al., *NAS-Bench-360*, ICLR 2022 — arXiv:2110.05668
15. Mehta et al., *NAS-Bench-Suite*, ICLR 2022 — arXiv:2201.13396
