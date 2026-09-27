# NAS 跨领域结合综述 · 应用章节调研报告

> 调研方式：WebSearch / WebFetch（2026-09-24）。本文区分两类表述：**【读到】**= 来自可核实论文/链接；**【推测】**= 基于上述文献的综合判断，非原文结论。
> 重要澄清（先纠错，避免以讹传讹）：
> 1. **W-Net 不是 NAS 方法**。W-net（Xia & Kulis 2017，无监督图像分割；Chen 等 2018 "Bridged U-net"）均为**手工设计**的 U 形网络；nnWNet（Zhou 等）是把手工 WNet 接进 nnU-Net 框架，也不是 NAS。
> 2. **"NasSeg" 作为命名论文未找到**（无权威可核实来源）。医疗分割 NAS 的奠基作是 **NAS-Unet** 与 **Auto-DeepLab** 的医学迁移。
> 3. **"NAC" 作为命名医疗 NAS 方法未找到**。真实对应的是 **ChaLearn AutoDL Challenge 2019** 与 **MedMNIST** 基准（AutoML/NAS 评测平台）。
> 4. **Autoformer（时序）≠ NAS**。**时序 Autoformer**（Wu 等, NeurIPS 2021）是"序列分解 + 自相关"的 Transformer 架构，**不是**架构搜索；真正做 Transformer 搜索的是同名近似词 **AutoFormer: Searching Transformers for Visual Recognition**（Chen 等, ICCV 2021），切勿混淆。

---

## 1. 医疗影像 NAS

### 1.1 分类
- **【读到】Evolutionary NAS for 2D and 3D Medical Image Classification**（Ali 等, ICCS 2024, DOI 10.1007/978-3-031-63751-3_9）：分别对 2D/3D 定义两套搜索空间，用进化算法 + **零成本代理（Zero-Cost proxy）** 估性能，并用**遗传算法自动搜数据增强（DA）策略**。在 MedMNIST 与 BreakHIS 上达到与现有 AutoML 可比的结果。链接：https://hal.science/hal-04976928v1/file/148330121.pdf
- **【读到】ChaLearn AutoDL Challenge 2019**（arXiv:2201.03801）：综合 AutoML 大赛，输入含图像/视频/文本/表格/时序。**原文明确写到："popular Neural Architecture Search (NAS) was impractical"**——在算力/时间受限的隐藏任务上，胜出方案依赖微调预训练网络而非 NAS。**【推测】**这侧面说明纯 NAS 在"小数据、限时"的医疗/通用分类场景性价比有限。

### 1.2 分割
- **【读到】NAS-Unet: Neural Architecture Search for Medical Image Segmentation**（Weng 等, IEEE Access 2019, DOI 10.1109/ACCESS.2019.2908991）：在 U 形骨架上搜两种 cell（DownSC/UpSC），微分架构搜索（DARTS 式）。在 Promise12(CT)、Chaos(MRI)、超声神经数据集上，以约 **0.8M 参数**超越 U-Net 及变体。链接：https://ieeexplore.ieee.org/abstract/document/8681706
- **【读到】Auto-DeepLab: Hierarchical NAS for Semantic Image Segmentation**（Liu 等, CVPR 2019, arXiv:1901.02985）：同时搜**网络级（分辨率/下采样路径）+ 单元级**结构，在 Cityscapes 等上不靠 ImageNet 预训练即逼近 DeepLabv3+，且更快。**【读到】**它是自然场景分割工作，但被医疗界广泛"迁移"为医学分割 NAS 的范式（医学分割综述 arxiv:2009.13120 专门引用 NAS-Unet 与 Auto-DeepLab 路线）。
- **【读到】Resource-optimized NAS for 3D Medical Image Segmentation**（Bae 等, MICCAI 2019）：针对 3D 体数据优化资源消耗的 NAS。

### 1.3 多模态医学融合 NAS
- **【读到】MM-NAS: Multi-Modality Information Fusion for Radiomics-based NAS**（Peng 等, MICCAI 2020, arXiv:2007.06002）：在 **PET-CT** 放射组学上自动搜"融合模块"（在哪一层、用何种算子融合 PET 与 CT），在软组织肉瘤远处转移预测上超过 SOTA 手工融合。链接：https://arxiv.org/abs/2007.06002

### 1.4 小样本 / 数据稀缺 / 3D 困难
- **【读到】**上述 Ali 2024 用 **Zero-Cost proxy** 解决医疗"评估昂贵"；用自动 DA 缓解过拟合——这是小样本 NAS 的主流对策。
- **【读到】**3D 医学影像（CT/MRI）特殊困难（来自 NAS-Unet、3D U-Net 综述 arxiv:2009.13120 与 Bae 2019）：(a) 体数据维度高、显存/计算开销大；(b) 标注稀缺、样本少；(c) 各模态（T1/T2/CT/超声）分布差异大。NAS 的处理思路是**缩小搜索空间 + 代理估性能 + 搜索空间面向 3D 卷积算子**，但尚无"根本性"突破。

---

## 2. 生物信息 / 基因组 / 蛋白 NAS

- **【读到】BioNAS: NAS for Interpretable DL in Genomics**（Zhang 等, arXiv:1909.00337）：面向**DNA/RNA 序列**的 CNN，引入"知识不相似度函数"，联合优化**预测力 + 生物学可解释性**（让第一层卷积滤波器富集已知 motif）。用于 eCLIP 蛋白-RNA 结合位点预测。**注意：BioNAS 不做蛋白结构预测**，是组学序列层面。链接：https://arxiv.org/abs/1909.00337
- **【读到】GenomeNet-Architect**（Gündüz 等, Communications Biology 2024, DOI 10.1038/s42003-024-06161-1）：专为基因组序列设计的搜索空间 + 贝叶斯优化（MBO），病毒分类任务上误分率降 **19%**、参数少 **83%**。链接：https://www.nature.com/articles/s42003-024-06161-1
- **【读到】Genome-DARTS / CWP-DARTS 等**（Scheppach 等, CIBCB 2023）：把 DARTS 家族适配到基因组序列数据，超越手工架构且参数量显著更少。
- **【读到/澄清】AlphaFold 类结构改进——未找到用 NAS 改进 AlphaFold 的论文。** AlphaFold2/3 用的是**手工设计的 Evoformer/Pairformer + 扩散**，并非 NAS 产物（来源：Jumper 等 Nature 2021; AlphaFold3 技术说明）。**【推测】**结构生物学领域更依赖"专家归纳偏置 + 大规模协同进化/MSA 数据"，NAS 在该子方向几乎缺位，是一个明确的空白而非"收益小"。

---

## 3. 遥感 / 高光谱 / SAR（与跨域最相关，重点）

- **【读到】SGNAS: Spatial Generalized NAS for Cross-Domain Scene Classification**（Chen 等, Remote Sensing 2021, DOI 10.3390/rs13173460）：**空间广义化 NAS**，专门面向高分辨率遥感图像**跨域场景分类**（训练/测试分布不同）。用 human-in-the-loop 把搜索空间泛化，使搜出的架构能迁移到新数据集/新任务。**这是"跨传感器/跨域泛化 NAS"的直接相关作**。链接：https://www.mdpi.com/2072-4292/13/17/3460
- **【读到】UDA-NAS: Unsupervised Domain Adaptation NAS with Self-Training for Land Cover Mapping**（Broni-Bediako 等, CVPR 2024）：把 NAS 与**无监督域适应（UDA）**结合（教师-学生自训练 + MRF 搜索引擎），搜轻量架构做卫星图像土地覆盖分割。在 OpenEarthMap 上以 **1.28M 参数**达 59.38% mIoU，超过 DAFormer 等大模型，且可迁移到 FLAIR#1。**【推测】这支与"跨域点云分割 NAS"方法论最贴近**——核心都是"在存在域偏移/传感器差异时，搜出对目标域鲁棒的轻量结构"。
- **【读到】Efficient Convolutional NAS for RS Scene Classification**（Peng 等, IEEE TGRS 2021, DOI 10.1109/TGRS.2020.3045823）：连续搜索空间 + 渐进贪婪搜索，构建大规模 RS 场景数据集，7 个基准验证泛化。
- **【读到】高光谱分类 NAS 综述 + 系列方法**（Wang 等, Remote Sensing 2025, DOI 10.3390/rs17152727；TR-NAS 2026 DOI 10.3390/rs18101586；RFSS-NAS 2024 DOI 10.3390/rs16173123）：高光谱 NAS 聚焦 1D/2D/3D 卷积与**空间-谱注意力**搜索；RFSS-NAS 用 Noisy-DARTS 解决 DARTS 性能崩塌。
- **【读到】SAR / 多光谱融合 NAS** 在本批检索中**未找到命名专文**（高光谱、场景分类、UDA 较多；SAR 特指与多光谱融合的 NAS 暂缺，标注为"未找到"）。

---

## 4. 信号 / 时序 / 语音 NAS

### 4.1 EEG / ECG
- **【读到】CTNAS-EEG: Cross Task NAS for EEG Signal Classification**（Duan 等, arXiv:2210.06298）：为 EEG 设计跨任务兼容搜索空间 + 约束搜索，在运动想象、情绪识别上达 SOTA，并首次按"人"定制结构。链接：https://arxiv.org/abs/2210.06298
- **【读到】NAS for 12-lead ECG**（Fayyazifar 等, CinC 2020）：用 DARTS（改造为 1D）搜 ECG 架构，10 折 CV 64.4% 优于手工 RCNN 61.9%。
- **【读到】Deep-PxAF**（Nature Scientific Reports 2023, DOI 10.1038/s41598-023-38541-8）：certified-GAN 造数据 + cell-based NAS 设计 CNN 检测阵发性房颤，准确率 ~99% 超手工。
- **【读到】NanoCNN**（MDPI Electronics 2026, DOI 10.3390/electronics15051044）：**少数类感知 + 硬件感知**多目标 NAS，专为 MCU 上的心律失常分类，目标含少数类召回、RAM/Flash、MACs。

### 4.2 语音识别
- **【读到】NAS for Speech Recognition**（Hu 等, arXiv:2007.08818）：用 DARTS/Gumbel-Softmax/Penalized-DARTS 自动配置 TDNN-F 的拼接上下文与瓶颈维度，在 300h Switchboard 上 WER 绝对降 **1.0%**、模型小 **28%**。链接：https://arxiv.org/abs/2007.08818
- **【读到】ST-NAS**（Zheng 等, IEEE SLT 2021, arXiv:2011.05649）：直通梯度高效 NAS，在 WSJ/Switchboard 上超手工架构。
- **【读到】Conformer-based NAS**（Liu 等, arXiv:2104.05390）：在 Conformer 骨架上做 DARTS 搜模块组合，AISHELL-1 上 CER 相对降 ~11%。

### 4.3 时序预测
- **【澄清/读到】时序 Autoformer（Wu 等, NeurIPS 2021, arXiv:2106.13008）不是 NAS**，是分解+自相关 Transformer，性能提升来自架构设计而非搜索。**【读到】**NAS 在时序的方向多为"搜时序/Transformer 结构"（如 AutoFormer-ViT），而非 Autoformer 本身。
- **【读到】**综述（Springer, 2026, DOI 10.1007/s10462-026-11550-5）指出 ASR-NAS 正从"单纯精度"转向**部署导向**（延迟感知、压缩、场景自适应），并明确警告"单场景优化的架构在多场景会掉点"——与跨域主题呼应。

---

## 5. 硬件感知 NAS（Edge / Hardware-aware）

- **【读到】MnasNet**（Tan 等, CVPR 2019, arXiv:1807.11626）：首个移动端感知 NAS，把**精度 × (延迟/目标)^γ** 作为 RL 多目标奖励，直接在 ImageNet + 真实 Pixel 1 延迟 LUT 上搜，比 NASNet 参数少 4.8×、MACs 少 10×。
- **【读到】FBNet**（Wu 等, CVPR 2019, arXiv:1812.03443）：DNAS + **Gumbel-Softmax** 把延迟做成可微目标，延迟查表（LUT）估计端到端延迟。
- **【读到】ProxylessNAS**（Cai 等, ICLR 2019, arXiv:1812.00332）：在**目标硬件 + 大规模数据**上直接搜，二值化路径省显存，把延迟作为架构参数的连续函数。
- **【读到】Once-for-All (OFA)**（Cai 等, ICLR 2020, arXiv:1908.09791）：训一个超网，按深度/宽度/核大小/分辨率采样子网络，**一次训练、按需提取**适配不同硬件；OFA 在 ImageNet 76.0% 且 58ms。
- **【读到】NSGA-Net: NAS using Multi-Objective Genetic Algorithm**（Lu 等, GECCO 2019, arXiv:1810.03522）：**用 NSGA-II 做多目标 NAS**，同时优化**误差 + FLOPs**，输出 Pareto 前沿；以 8 GPU-days 达 CIFAR-10 3.85% 错误率。是"NSGA-II + NAS + Pareto"最常被引用的具体论文。链接：https://arxiv.org/abs/1810.03522
- **【读到】**其他：SinglePathNAS、MobileNetV3（RL 多目标）、MCUNet/TinyNAS（MCU 内存约束协同设计）、NanoCNN（见 4.1，少数类+硬件多目标）。
- **【推测】**多目标 Pareto 的通用做法：目标= {精度, 延迟/FLOPs/能耗/内存}，NSGA-II 用"非支配排序 + 拥挤度"维持多样性；也可加权求和进 RL/梯度（MnasNet/FBNet/ProxylessNAS）。HW-NAS 综述（ACM Computing Surveys, DOI 10.1145/3524500）系统归纳了 RL/梯度/进化三类。

---

## 6. 共性问题：收益是"架构创新"还是"部署优化"？是否有论文说收益小？

- **【读到】"收益有限/不如随机搜索"的明确证据：**
  - Elsken 等《Neural Architecture Search: A Survey》(arXiv:1808.05377) 直言：很多架构源自 **NASNet 搜索空间**，**随机搜索也能得到不差甚至更好的结果**，且搜出的架构并不显著优于手工。
  - **Yang 等《NAS Evaluation is Frustratingly Hard》(arXiv:1912.12816)**：许多 NAS 方法难以显著战胜"随机采样平均架构"基线。
  - **ReNAS (Garg 等, arXiv:2010.05719)**：指出现有 NAS 搜索空间本就基于手工 SOTA 网络，导致与随机基线差异小。
  - **ChaLearn AutoDL 2019（见 1.1）**：明确"NAS was impractical"。
  - 2026 对照实验（Stanfordgroup, Zenodo）：在 NAS-Bench-201 上，**带早停的随机搜索在同等算力下胜过 RL 与进化 NAS**。
- **【读到/推测】**收益性质分化：
  - **部署优化型**（主流且更可靠）：MnasNet/FBNet/ProxylessNAS/OFA/NSGA-Net/MCUNet——收益是"在给定延迟/内存下精度更高"，属**帕累托/部署优化**，而非发现全新结构。
  - **架构创新型的真实突破较少**：医疗/遥感/生信里 NAS 多报告"超越手工 U-Net/手工融合/手工基因组 CNN 且参数更少"，但增量常来自**搜索空间本身借鉴了优良手工模块**，未必是结构性原创。
- **【读到】**医疗/生信/遥感里 NAS 的"主要卖点"其实是**自动化 + 数据/模态自适应**（为特定 PET-CT、特定高光谱传感器、特定基因组任务搜专用结构），而不是绝对超越 SOTA 很多。

---

## 汇总表（领域 / 论文 / 年份 / 搜索空间 / 搜索目标 / 收益）

| 领域 | 论文（可核实） | 年份 | 搜索空间 | 搜索目标 | 收益 / 备注 |
|---|---|---|---|---|---|
| 医疗-分类 | Evolutionary NAS 2D/3D Medical (Ali et al., ICCS) | 2024 | 2D/3D 两套 cell 空间 | 精度 + Zero-Cost 估性能 + 自动DA | MedMNIST/BreakHIS 可比 SOTA；小样本对策 |
| 医疗-分割 | NAS-Unet (Weng et al., IEEE Access) | 2019 | U 形骨架 + DownSC/UpSC cell | 分割精度（DARTS式） | 0.8M 参数超 U-Net；CT/MRI/超声 |
| 医疗-分割 | Auto-DeepLab (Liu et al., CVPR) | 2019 | 网络级+单元级格子空间 | 分割精度（可微） | 不预训练逼近 DeepLabv3+；医学迁移范式 |
| 医疗-融合 | MM-NAS (Peng et al., MICCAI) | 2020 | 双流 PET/CT + 融合 cell | 预测精度 + 融合方式 | 软组织肉瘤转移预测超 SOTA |
| 生信-组学 | BioNAS (Zhang et al.) | 2019 | 1D 序列 CNN cell | 预测力 + 生物学可解释性 | eCLIP 蛋白-RNA 结合；非蛋白结构 |
| 生信-组学 | GenomeNet-Architect (Gündüz et al.) | 2024 | 基因组专用层布局+超参 | 精度 + 参数量 + 推理速度 | 误分率-19%、参数-83% |
| 遥感-跨域 | SGNAS (Chen et al., Remote Sensing) | 2021 | 空间广义化 cell（human-in-loop） | 跨域分类精度 + 泛化 | 高分辨率遥感跨数据集迁移 |
| 遥感-跨域 | UDA-NAS (Broni-Bediako et al., CVPR) | 2024 | MRF 搜索 + 师生自训练 | 分割精度 + FLOP/参数预算（UDA） | 1.28M 参数达 59.38 mIoU，超 DAFormer |
| 遥感-高光谱 | RFSS-NAS / TR-NAS (Wang et al.) | 2024/2026 | 空间-谱注意力算子 | 分类精度（Noisy-DARTS） | 解决 DARTS 崩塌，鲁棒 |
| 信号-EEG | CTNAS-EEG (Duan et al.) | 2022 | 跨任务兼容 1D 空间 | 识别精度 + 跨任务/跨人 | MI/情绪 SOTA |
| 信号-ECG | NanoCNN (MDPI) | 2026 | 8-bit 量化兼容 1D 算子 | 少数类召回 + 硬件指标 | MCU 边缘心律失常分类 |
| 语音-ASR | NAS for Speech Recognition (Hu et al.) | 2020 | TDNN-F 上下文/瓶颈维度 | WER + 模型复杂度 | WER -1.0% abs，模型 -28% |
| 语音-ASR | Conformer NAS (Liu et al.) | 2021 | Conformer 模块组合 | CER（DARTS） | AISHELL-1 CER 相对 -11% |
| 硬件-NAS | MnasNet (Tan et al.) | 2019 | MobileNetV2 式 block + LUT | 精度 × 延迟^γ（RL） | 移动端精度/延迟帕累托 |
| 硬件-NAS | FBNet (Wu et al.) | 2019 | DNAS + Gumbel-Softmax | 精度 × log(延迟) | 延迟可微，查表估延迟 |
| 硬件-NAS | ProxylessNAS (Cai et al.) | 2019 | 目标硬件直接搜 | 精度 + 连续延迟项 | 大规模数据直接搜，省显存 |
| 硬件-NAS | Once-for-All (Cai et al.) | 2020 | 深度/宽/核/分辨率子网 | 精度 + 多硬件适配 | 一次训练按需提取子网络 |
| 硬件-多目标 | NSGA-Net (Lu et al., GECCO) | 2019 | 全网络结构 bit-string | 误差 + FLOPs（NSGA-II Pareto） | 8 GPU-days，CIFAR-10 3.85% |
| 时序-架构 | Autoformer (Wu et al., NeurIPS) | 2021 | —（非 NAS） | 长时序预测精度 | 非 NAS，分解+自相关；澄清误用 |
| 时序-NAS | AutoFormer: Searching Transformers (Chen et al., ICCV) | 2021 | ViT 模块（权纠缠超网） | ImageNet 精度 + 参数 | ViT/DeiT 被超越；与 Autoformer 同名不同物 |

---

## 结论（对跨域点云分割 NAS 的呼应）

- **【读到】**遥感里 **SGNAS（跨域分类）** 与 **UDA-NAS（无监督域适应分割）** 已经把"域偏移/传感器差异"作为 NAS 的一等目标——这与你的**跨域点云分割 NAS**在方法论上高度同构：不是搜一个"最强"结构，而是搜一个"对目标域/目标传感器鲁棒且轻量"的结构。
- **【推测】**点云跨域 NAS 可借鉴：(a) UDA-NAS 的"搜索-适应同步（search-while-adapting）"；(b) SGNAS 的"广义化搜索空间 + human-in-the-loop"；(c) HW-NAS 的多目标帕累托（点云常需显存/延迟约束）；(d) 医疗 NAS 的 Zero-Cost proxy 降本。
- **【读到】**务必警惕"NAS 收益被高估"的文献（随机搜索基线、AutoDL 大赛 NAS 不实用）：在做点云跨域 NAS 时，应报告**相对强基线（如固定结构的域适应 U-Net）的增益 + 部署指标**，而非只比随机/原始结构，否则结论说服力弱。

> 未找到（已声明）：命名 "NasSeg" 论文、命名 "NAC" 医疗 NAS 方法、NAS 改进 AlphaFold/蛋白结构的论文、SAR 特指多光谱融合 NAS 专文。W-Net 确认非 NAS。
