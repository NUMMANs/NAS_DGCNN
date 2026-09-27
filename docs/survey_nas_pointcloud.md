# NAS × 3D 点云 / LiDAR / 自动驾驶 文献脉络调研

> 调研目的：为"跨域点云语义分割的架构搜索"项目提供文献坐标。项目现状——以 **DGCNN 为 search substrate、S3DIS 室内数据**；实测发现"按密度 / 按房间 / 按域条件化架构"的 oracle headroom 为 **+0.0000**，仅主效应有收益（池化 max>mean 差 0.07 mIoU、feature 加宽 +0.02）。本文据此梳理"文献里 NAS+点云怎么做、搜什么、跨域/泛化如何处理、代价如何控制、有哪些负面结论"。
>
> **证据分级说明**：①"已核实"= 本会话通过 WebSearch/WebFetch 取到标题、会议/年份、arXiv 或 DOI、摘要；②"推测/通识"= 未在本次检索中直接核实，仅作背景；③"未找到"= 明确检索但未见可靠来源（如 AutoBEV、SparseBackbone 作为 NAS 工作均未找到）。

---

## 1. 代表性论文汇总表

| 论文 | 会议/年份 | 链接 | 搜索空间 | 跨域/鲁棒处理 | 关键结论 |
|---|---|---|---|---|---|
| **Point-NAS** (Xie et al.) | TIP 2023 | DOI 10.1109/TIP.2023.3331223 | One-shot supernet；弹性特征提取(EFE)模块，沿**宽度+深度**伸缩；权重耦合 sandwich 规则 + 梯度调整；资源约束下进化搜索 | 无专门跨域设计；多数据集(分类/分割/检测)验证 | S3DIS 68.6 mIoU；结论偏"弹性宽度/深度+supernet 权重共享"收益 |
| **Auto-Points** (Wang et al.) | TMM 2024 | DOI 10.1109/TMM.2023.3304892 | One-shot；Set Abstraction Mixer(SAM)层，沿**深度+宽度+每阶段采样点数**伸缩；weight-entwinement NAS | 无 | S3DIS 69.1 mIoU；ModelNet40 93.9 OA |
| **SGAS: Sequential Greedy Architecture Search** (Li et al.) | CVPR 2020 | arXiv:1912.00195 | 顺序贪心；在 DAG cell 内搜**算子类型**（图卷积/EdgeConv 类候选）；用于点云分类 | 无 | 点云分类 SOTA；强调"搜索阶段排名不稳定"，用贪心缓解 |
| **3D-NAS / Searching Efficient 3D Architectures with Sparse Point-Voxel Conv** (Liu, Tang, Lin, Han) | NeurIPS 2020 | arXiv:2007.16100 | One-shot supernet；**细粒度通道数 + 弹性网络深度 + 分辨率**；SPVConv(point+sparse voxel 融合)；MACs 硬约束 + 进化搜索 | 多场景：SemanticKITTI(LiDAR 分割)、ModelNet40 | 1.8–23.7× 实测加速；首次把"资源约束 NAS"用于大规模 LiDAR 场景 |
| **PVNAS: 3D Neural Architecture Search with Point-Voxel Conv** (Liu et al.) | arXiv 2022 (TPAMI 版) | arXiv:2204.11797 | 在 3D-NAS 基础上强化 PVConv 基元 + sparse conv | 部署到 MIT Driverless 自动驾驶赛车 | 更大检测范围、更高精度、更低延迟 |
| **Differentiable Convolution Search for Point Cloud Processing (PointSeaConv/PointSeaNet)** (Nie et al.) | ICCV 2021 | arXiv:2108.12856 | **搜卷积内部结构**：把卷积建模为"基本关联(Essential Associations)"的有向无环图（欧氏距离、相对位置、全局上下文等几何基元）；内外联合 + ε-greedy | 无；但搜的是**几何关系组合**而非 k/半径 | S3DIS 71.9 mIoU(6-fold)；0.25 GPU-day；指出 PointNet++/DGCNN/RS-CNN 都是其搜索空间的特例 |
| **LidarNAS: Unifying and Searching Neural Architectures for 3D Point Clouds** | arXiv 2022 | arXiv:2210.05018 | **微观(通道数) + 宏观(视图/格式变换：voxel/pillar/point/range/fusion 之间)**；正则化进化 | 在 Waymo Open Dataset 上做 3D 检测；跨车辆/行人两类 | 超过 RSN；发现"车/行人分别搜出的宏观结构竟一致"，NAS 修改巧合地复现人类设计 |
| **RobNAS: Robust NAS for Point Cloud Adversarial Defense** (Sun et al.) | ICASSP 2025 | DOI 10.1109/ICASSP49660.2025.10890087 | 在架构搜索阶段注入 **PGD 对抗训练**，搜"架构级鲁棒"网络 | 仅**对抗鲁棒**（非密度/稀疏/噪声） | 点云分类对抗防御 SOTA；首提"架构鲁棒性"视角 |
| **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah et al.) | ICLR 2021 | arXiv:2101.08134 | 代理指标（单 batch 零成本打分） | — | NAS-Bench-201 上与最终精度 **Spearman 0.82**（vs EcoNAS 0.61） |
| **Random Search and Reproducibility for NAS** (Li & Talwalkar) | UAI 2020 | arXiv:1902.07638 | 随机搜索 + 早停 / 权重共享 | — | 随机搜索 with 早停 **≈ ENAS**；权重共享随机搜索达 SOTA；点出 NAS 复现危机 |
| **Evaluating the Search Phase of NAS** (Sciuto, Yu, Jaggi et al.) | arXiv 2019 | arXiv:1902.08142 | 评估框架：对比 SOTA 搜索策略 vs 随机 | — | **SOTA NAS ≈ 随机**；**权重共享使 Kendall-Tau≈0**（搜索期排名与真实排名失联） |
| **Adapting Neural Architectures Between Domains (AdaptNAS)** (Cai et al.) | NeurIPS 2020 | proceedings.neurips.cc/paper/2020/hash/08f38e0434442128fab5ead6217ca759 | 代理任务→目标域的**域适应式架构迁移** | **跨域 NAS**（理论泛化界 + 数据分布距离约束） | 仅用少量 ImageNet 即可把架构优势迁移到全量 ImageNet |
| **Neural Architecture Search for Domain Adaptation (NASDA)** (Li & Peng) | arXiv 2020 | arXiv:2008.05706 | 可微 NAS + 多核 MMD 对齐 + 对抗训练 | **UDA 场景 NAS**（Office、VisDA 等图像域） | 域适应 benchmark SOTA |
| **Neural Architecture Search: Insights from 1000 Papers** (Wistuba et al.) | arXiv 2023 | arXiv:2301.08727 | 综述：搜索空间分类（macro/cell/hierarchical） | — | 明确："**哪怕最好的搜索算法，也只能在搜索空间内最好的候选上产出结果**" |

> 说明：**AutoBEV、SparseBackbone 作为"NAS 工作"在本次检索中未找到**（检索结果只返回 BEV 感知综述、稀疏卷积检测器 SAFDNet/FSTR 等，无对应 NAS 论文）。BEVFusion 本身是多模态 BEV 融合（非 NAS），本次也未发现其 NAS 变体。SPineNet（CVPR 2020, arXiv:1912.05027）是 2D 检测 backbone NAS（scale-permuted + 跨尺度连接），仅作**方法论类比**，非点云。

---

## 2. 搜索空间设计：他们到底搜什么？

按"搜的层次"归纳，现有点云 NAS 的搜索轴基本是**结构化超参**，而非几何本身：

1. **算子/基元类型**：SGAS 在 cell 内搜图卷积候选（EdgeConv 类）；3D-NAS 搜 point-voxel/sparse-conv 组合；Point-NAS/Auto-Points 在设定好的 SA/EFE 模块内搜子候选。
2. **通道宽度 / 网络深度**：几乎所有 one-shot 方法（Point-NAS、Auto-Points、3D-NAS、PVNAS）都把**宽度和深度做成弹性轴**——这与你项目里"feature 加宽 +0.02"的主效应完全对应：文献的主要收益正是这一轴。
3. **下采样率 / 每阶段采样点数 / 分辨率**：Auto-Points 显式搜"每阶段采样点数"；3D-NAS 搜输入分辨率与深度；这是点云特有的"稀疏度/尺度"轴。
4. **多分支融合方式**：3D-NAS/PVNAS 的 SPVConv 在 point 分支与 sparse-voxel 分支间融合；LidarNAS 在**宏观层面搜"视图/格式变换"**（voxel/pillar/point/range/fusion 怎么串）。
5. **图构建 / 邻域半径 / k 值（你最关心的轴）**：
   - **PointSeaConv (ICCV'21) 是最接近的**：它把卷积拆成"基本关联"的 DAG——欧氏距离、相对坐标、全局上下文等**几何关系基元**，搜的是"用哪些几何关系组合建模形状"。但它搜的是**关系类型**，不是直接的 k 或 radius 超参。
   - **直接搜邻域 k / 半径 / graph 构造方式**：本次检索**未找到**点云 NAS 论文把 k 或 radius 作为搜索空间节点的。图构建在 DGCNN/EdgeConv 里是"特征空间 k-NN（固定 k）"，属人工设定。
   - **推论（推测，非读到的结论）**：你项目里"按密度/按房间/按域条件化"本质上是在尝试一种**输入条件化的几何-架构映射**，而文献的搜索空间里连"静态的 k/radius"都没人搜，更没有"密度→架构"的条件轴——这解释了为什么你的 oracle 都给出 +0.0000：现有 search substrate 根本没有可让"条件化"产生差异的自由度。

---

## 3. 跨域 / 泛化 / 鲁棒 / 条件化架构

- **跨域 NAS 存在于图像域，点云域缺失**：
  - AdaptNAS (NeurIPS'20)、NASDA (arXiv:2008.05706)、DAMPC-NAS (arXiv:2109.06652)、DAP: Domain Adaptive Performance Predictor (IEEE Trans. Computers 2026) 都做"域适应/跨数据集 NAS"，但**全部在图像分类/域适应基准（ImageNet↔CIFAR、Office、VisDA）**，无一涉及点云。
  - 点云自身的跨域分割（xMUDA、GIPSO 等）是热门，但**没有 NAS 版本**。
- **鲁棒 NAS 在点云只有"对抗"一个方向**：RobNAS (ICASSP'25) 把 PGD 对抗训练塞进搜索阶段，搜架构级鲁棒性；它针对**对抗扰动**，而非你关心的**密度/稀疏度/采样噪声**鲁棒。后者的"鲁棒 NAS"在点云里同样**未找到**。
- **条件化 / 动态 / 输入自适应架构**：点云里 DGCNN 的"动态图"本质是**数据驱动、逐层重算的图**（非搜索所得）；但"**架构本身随输入密度/尺度变化**"的 NAS——即 hypernet 生成权重、或条件开关网络——**检索未见**。所以你实验里"按密度或尺度条件化结构搜索拿不到收益"，不仅不反常，反而是文献空白处的合理结果：没人证明过它能 work。

---

## 4. 搜索代价与排序可靠性（短预算）

- **权重共享 / one-shot supernet** 是主流降本手段：Point-NAS、Auto-Points、3D-NAS、PVNAS 均用 supernet 权重共享 + 进化/梯度搜索，把"训 N 个网络"降为"训 1 个超网"。
- **代理指标 / predictor**：Zero-Cost Proxies (ICLR'21) 用单 batch 零成本打分，NAS-Bench-201 上 **Spearman 0.82**；域适应场景用 DAP 这类域自适应性能预测器。
- **早停 / 预算约束**：Li & Talwalkar (UAI'20) 的"随机搜索+早停"即预算友好基线；另有工作显示"训 1 epoch 的排名 vs 训 50 epoch 真实排名" Spearman 仅 **0.65**，动态学习排名可升到 0.85——说明**短预算排序本身不可靠，需配合相对训练协议**。
- **排序相关性研究（直接呼应你的担忧）**：
  - Sciuto et al. (arXiv:1902.08142)：在受控实验中，**SOTA NAS 策略平均不优于随机**；更关键——**权重共享使 Kendall-Tau ≈ 0**，即 supernet 给出的候选排序与真实性能排序"失联"。这意味着 one-shot 的排序信号本就 noisy。
  - 引申到你的项目：你测的是"oracle headroom +0.0000"（即假设有完美条件选择器，收益也为 0），这与"搜索空间内本无实例级最优差异"一致；而 Sciuto 的工作进一步说明，即便用短预算代理去排，也未必排得动。

---

## 5. 失败/负面结论，以及对项目的启示

**文献中明确存在的负面结论**：
- "**搜索空间比搜索算法更重要**"：Insights-from-1000-Papers (arXiv:2301.08727) 原文"哪怕最好的搜索算法，也只能在搜索空间内最好的候选上产出结果"；TextNAS 等也强调搜索空间是人类先验。
- "**NAS ≈ 随机**"：Sciuto et al. 与 Li & Talwalkar 均显示，在充分约束的搜索空间里，SOTA 搜索策略与随机采样无统计显著差异；收益主要来源于搜索空间本身的良好先验（如弹性宽度/深度）。
- "**权重共享损害排名**"：Sciuto et al. 指出 one-shot 的 Kendall-Tau≈0。

**对你项目的解读（推测，非文献结论）**：
1. 你的主效应（pooling 0.07、width 0.02）正是文献反复验证的"宽度/聚合方式"轴——**这部分收益是真实且可解释的**，与 Point-NAS/Auto-Points/3D-NAS 的主收益同源。
2. 条件化（密度/房间/域）oracle headroom +0.0000，说明在 **DGCNN 这一族**中，不存在"随输入变化的更优架构"。这与"搜索空间决定上限"高度吻合：DGCNN 的搜索空间没有条件分支轴，自然搜不出条件收益。
3. 实务建议（推测）：若要继续，可考虑①**换更宽的 substrate**（如把 PointSeaConv 的几何关联基元作为搜索轴，而非仅在 DGCNN 上做条件化）；②放弃"条件化架构"，转向"条件化**训练/归一化/损失**"或"跨域训练策略"；③用零成本代理(Spearman 0.82)做短预算预筛，避免被 one-shot 的 Kendall≈0 噪声误导。

---

## 6. 未找到 / 需进一步核实

- **AutoBEV、SparseBackbone、BEVFusion-NAS**：本次检索未找到对应 NAS 论文（仅 BEV 感知综述与稀疏卷积检测器）。如确有其文，请补充线索后再核实。
- **点云 NAS 直接搜 k/radius/图构建**：未找到。
- **点云域适应 NAS / 密度条件化 NAS / 点云鲁棒(非对抗)NAS**：未找到。
- Auto-DeepLab (CVPR'19，2D 密集分割层次化 NAS) 作为方法学前身仅来自通识，本会话未直接核实，故不入表。
