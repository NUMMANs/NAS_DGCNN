"""PointViG：按论文规格复现（官方**未开源**，CatalyzeX 标记为 "Request Code"）。

来源：arXiv 2407.00921v2《PointViG: A Lightweight GNN-based Model for Efficient
Point Cloud Analysis》（Zheng, Qi, Wang, Zhang, Sun）。目标数字：S3DIS 71.7% mIoU / 5.3M 参数。

为什么自己复现
================================================================
· 官方无代码（探测 github.com/PointViG、PointViG-Pytorch、pointvig 全部 404）
· 但它同时满足本项目需要的两点：**轻量（单卡可跑）+ 有尺度轴（Ball-Query 半径 r）**
· PTv3 实测 6344 ms/step（DGCNN 的 61×）→ NAS 不可行；
  PolarNet 的极坐标表示与室内几何先天失配（range-view 已预演过）

论文给出的复现规格（从正文挖出，逐条对齐）
================================================================
| 组件       | 论文值                                                        |
|-----------|--------------------------------------------------------------|
| Ball-Query | **r = 0.2**、**m = 64**（球内点数补齐到 m）、**k = 32**（子图采样邻居数）|
| 关键细节   | 对 Ball-Query **加 masking**，屏蔽 padding 的无效点（否则无效点会被选进子图）|
| 编码器     | 5 stage，下采样比 (1,4,4,4,4)，每 stage block 数 (1,2,3,2,2)    |
| 阶段分工   | Stage 1 用 grouping-MLP-pooling；Stage 2~5 用 ADGC             |
| S3DIS     | 71.7% mIoU，5.3M 参数                                          |
| 消融       | 邻域 k 扫描：k=32 最优                                          |

⚠⚠ 本模块的设计取向：**把"尺度"做成显式的可搜索参数**
================================================================
原论文的核心机制是"用**单一固定半径 r** 保证子图感受野一致"（应对室内密度不均）。
这恰好是本项目要挑战的对象：本项目在 DGCNN 上已实测
**"加尺度分支 +0.0021/+0.0033，而调单一尺度参数全部 −0.0001~−0.0028"**。

所以本实现**不写死 r**，而是让每个 ADGC 层接受**自己的**半径（`radii` 列表），
从而可以直接表达：
  · 人类设计：所有层用同一个 r=0.2（论文做法）
  · NAS 候选：各层用不同 r（多尺度混合）
本模块**只提供机制，不做选择** —— 选择交给 arch_space / arch_loop（与 DGCNN 那套同构）。

⚠ 与项目其余部分一致的硬规矩
  · 归一化一律 **GroupNorm**（稀疏图/体素上 BatchNorm 会被空样本污染，实测 std 稀释 180×）
  · 跨模块张量**显式统一形状约定 + 边界断言**（曾因 (1,3,N) vs (N,3) 广播出静默错误）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _assert_xyz(xyz: torch.Tensor, name: str = "xyz") -> None:
    """形状约定：点云一律 (N, 3)。sparse 图上一旦传成 (1,3,N) 会**静默**广播出错。"""
    assert xyz.dim() == 2 and xyz.shape[1] == 3, \
        f"{name} 必须 (N,3)，收到 {tuple(xyz.shape)}"


def ball_query(query: torch.Tensor, support: torch.Tensor, radius: float,
               max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """球查询：为每个 query 点在 support 中找半径内的近邻，**补齐到固定个数**。

    返回
    ----
    idx  : (M, max_neighbors) long —— 邻居索引，无效位补齐为 **0**
    mask : (M, max_neighbors) bool —— True = **真实**邻居，False = padding

    ⚠⚠ **mask 是这个函数存在的唯一理由，不能省**。
    论文特别指出：Ball-Query 为对齐张量必须补点，若不屏蔽，
    这些重复/无效的填充点会被"自适应采样"当成真实邻居选进子图，污染聚合。
    本项目的 pillar 诊断也踩过同源问题（空像素污染统计）。

    实现：分块做 cdist 以控显存（M×K 的距离矩阵在 16k 点上会有 GB 级）。
    """
    _assert_xyz(query, "query")
    _assert_xyz(support, "support")
    M = query.shape[0]
    K = support.shape[0]
    idx = torch.zeros(M, max_neighbors, dtype=torch.long, device=query.device)
    msk = torch.zeros(M, max_neighbors, dtype=torch.bool, device=query.device)

    # 分块，块的选取只影响显存不影响结果
    block = max(1, int(2_000_000 // max(1, K)))
    for s in range(0, M, block):
        e = min(M, s + block)
        d = torch.cdist(query[s:e], support)                    # (b, K)
        within = d <= float(radius)
        cnt = within.sum(dim=1).clamp(max=max_neighbors)         # (b,)
        # 半径内按距离取前 max_neighbors 个（近的优先）
        order = torch.argsort(d, dim=1)                          # (b, K)
        take = order[:, :max_neighbors]                          # (b, m)
        take_valid = torch.gather(within, 1, take)               # 半径内的才有效
        idx[s:e] = take
        msk[s:e] = take_valid
        del d, within, order, take, take_valid, cnt
    return idx, msk


def knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """kNN 索引（含自身）。返回 (N, k)。同样分块控显存。"""
    _assert_xyz(xyz)
    N = xyz.shape[0]
    out = torch.zeros(N, k, dtype=torch.long, device=xyz.device)
    block = max(1, int(2_000_000 // max(1, N)))
    for s in range(0, N, block):
        e = min(N, s + block)
        d = torch.cdist(xyz[s:e], xyz)
        out[s:e] = torch.argsort(d, dim=1)[:, :k]
        del d
    return out


# ---------------------------------------------------------------------------
# PointViG 模块（轻量图卷积：缓解过平滑）
# ---------------------------------------------------------------------------


class PointViGModule(nn.Module):
    """PointViG 轻量图卷积块。

    论文设计要点（我们照做）：
      · 用**最大特征差**（`feat[nbr] - feat[center]`）作为边特征 —— 这是 DGCNN EdgeConv
        的核心，也是缓解过平滑的关键：绝对特征在深层会趋同，**差值**不会。
      · 通道维分组做图卷积（cheap），再经 MLP 混合 —— 这是"lightweight"的来源。

    ⚠ 归一化用 GroupNorm（本项目硬规矩：稀疏图上 BatchNorm 会被空样本污染）。
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 32,
                 groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k = int(k)
        g = max(1, min(groups, out_ch // 8 if out_ch >= 8 else 1))
        # EdgeConv 风格：输入是 (center || center-nbr)，通道 2*in_ch
        self.gconv = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
        )
        self.act = nn.GELU()
        self.short = (nn.Identity() if in_ch == out_ch
                      else nn.Conv1d(in_ch, out_ch, 1, bias=False))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                nbr: torch.Tensor) -> torch.Tensor:
        """xyz (N,3), feat (N,C), nbr (N,k) → **(N, out_ch)**。

        ⚠ 返回布局与传入一致（(N, C)），内部转 conv 布局。
        这样调用方（网络主干）可以一直用 (N, C) 的直觉，不必记 conv 的 (B,C,L)。
        """
        _assert_xyz(xyz)
        N, C = feat.shape
        assert nbr.shape[0] == N, f"nbr 首维应为 {N}，收到 {nbr.shape[0]}"
        # (N,k,C) 邻居特征
        fn = feat[nbr]                                     # (N,k,C)
        e = torch.cat([fn - feat.unsqueeze(1),
                       feat.unsqueeze(1).expand_as(fn)], dim=2)   # (N,k,2C)
        # conv2d 期望 (B, C, H, W)：这里 B=1, C=2C_in, H=N, W=k
        e = e.permute(2, 0, 1).unsqueeze(0)                # (1, 2C, N, k)
        h = self.gconv(e).squeeze(0)                       # (out_ch, N, k)
        h = h.max(dim=2)[0]                                # (out_ch, N)
        h = self.mlp(h.unsqueeze(0)).squeeze(0)            # (out_ch, N)
        return self.act(h + self.short(feat.T)).T          # (N, out_ch)


# ---------------------------------------------------------------------------
# 自适应空洞图卷积（ADGC）：论文的核心贡献
# ---------------------------------------------------------------------------


class AdaptiveDilatedGraphConv(nn.Module):
    """**自适应空洞图卷积** —— PointViG 论文用于 Stage 2~5 的核心算子。

    论文描述：「在**扩张邻域**内基于**语义相关性**搜索稀疏邻居节点，
    从而在保证计算效率的同时扩大感受野」。

    我们的实现（对齐该描述，且把"扩张"做成**可搜索的多半径**）：
      1. 以半径 `r` 做 Ball-Query 得到**扩张邻域**（比 kNN 大的一圈候选）
      2. 用当前特征算**语义相关性**（query-key 点积），在候选里选 top-k
         —— 即"基于语义相关性的稀疏邻居搜索"
      3. 在选出的稀疏子图上做图卷积

    ⚠⚠ **`radii` 是本项目的实验变量，不是超参**
    ================================================
    论文用**单一固定 r=0.2**（"保证子图感受野一致"）。
    本实现接受一个**半径列表**，每个头/每层可以不同 → 直接表达"多尺度混合"。
      · `radii=[0.2]`            → 复现论文的单尺度设计
      · `radii=[0.1,0.2,0.4]`    → 多尺度（NAS 候选空间里的一个点）
    这样"单尺度 vs 多尺度"成为**同一算子内的可搜索开关**，其余一切不变
    （同样的优化器/epoch/数据），满足本项目"唯一变量是架构"的硬要求。
    """

    def __init__(self, in_ch: int, out_ch: int, radii: Sequence[float],
                 k: int = 32, m: int = 64, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.radii = [float(r) for r in radii]
        assert self.radii, "radii 不能为空"
        assert all(r > 0 for r in self.radii), "半径必须为正"
        self.k = int(k)
        self.m = int(m)
        self.n_heads = len(self.radii)

        g = max(1, min(groups, out_ch // 8 if out_ch >= 8 else 1))
        self.qkv = nn.Conv1d(in_ch, out_ch * 3, 1, bias=False)
        self.gn = nn.GroupNorm(g, out_ch)
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
        )
        self.act = nn.GELU()
        self.short = (nn.Identity() if in_ch == out_ch
                      else nn.Conv1d(in_ch, out_ch, 1, bias=False))
        # 每个半径一个可学习温度，让网络自己权衡各尺度（但不改尺度集合本身）
        self.scale_gate = nn.Parameter(torch.zeros(self.n_heads))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """xyz (N,3), feat (N,C) → **(N, out_ch)**（与传入布局一致）。"""
        _assert_xyz(xyz)
        N = xyz.shape[0]
        qkv = self.qkv(feat.T)                       # (3C, N)
        q, k_, v = qkv.chunk(3, dim=0)               # each (C, N)
        C = v.shape[0]

        outs = []
        for hi, r in enumerate(self.radii):
            idx, msk = ball_query(xyz, xyz, r, self.m)            # (N,m) + (N,m)

            # ---- 语义相关性：在扩张邻域内按 q·k 选 top-k ----
            # ⚠⚠ 索引维度必须写对。k_ 是 (C,N)，邻居索引是 (N,m)：
            #   直接写 k_[:, idx] 会**按 dim0 索引**得到 (N,m,C) —— 通道/样本错位，
            #   而且不报错（本项目最怕的静默错误类型）。正确顺序是先用 idx 展开成
            #   (N,m,C) 再 permute 回 (C,N,m)，或等价地 k_.T[idx] → (N,m,C)。
            k_nbr = k_.T[idx].permute(2, 0, 1)                    # (C, N, m)
            q_exp = q.unsqueeze(2)                                # (C, N, 1)
            sim = (q_exp * k_nbr).sum(0)                          # (N, m)
            # 屏蔽 padding（论文强调的 masking）：无效位给极小值，避免被 topk 选中
            sim = sim.masked_fill(~msk, -1e4)
            kk = int(min(self.k, self.m))
            top = sim.topk(kk, dim=1).indices                     # (N, kk)
            nbr = torch.gather(idx, 1, top)                       # (N, kk)
            w = torch.softmax(sim.gather(1, top), dim=1)          # (N, kk)
            # 取邻居 value 并加权聚合
            v_nbr = v.T[nbr].permute(2, 0, 1)                     # (C, N, kk)
            agg = (v_nbr * w.unsqueeze(0)).sum(dim=2)             # (C, N)
            outs.append(agg * torch.sigmoid(self.scale_gate[hi]))
            del k_nbr, q_exp, sim, top, nbr, w, v_nbr, agg, idx, msk

        h = torch.stack(outs, dim=0).sum(dim=0) if len(outs) > 1 else outs[0]  # (C,N)
        # ⚠⚠ **张量布局约定（写死，别再猜）**：
        #   · 点云数据一律 **(N, 3) / (N, C)**（本项目全局约定）
        #   · 但 nn.Conv1d / GroupNorm 要 **(B, C, L)**，其中 C 必须是 dim1
        #   · 所以进出卷积层时统一走：feat(N,C) --.T--> (C,N) --.unsqueeze(0)--> (1,C,N)
        #   此处 h 是 (C,N)：直接用就是 (1,C,N)，**不需要再转置**。
        #   之前写成 h.T 会得到 (1,N,C)，GroupNorm 会把 N 当通道 → 尺寸对不上。
        h = self.gn(h.unsqueeze(0)).squeeze(0)                    # (1,C,N) -> (C,N)
        h = self.proj(h.unsqueeze(0)).squeeze(0)                  # (1,C,N) -> (C,N)
        return self.act(h + self.short(feat.T)).T                 # -> (N, out_ch)


# ---------------------------------------------------------------------------
# 编码器 / 网络（按论文规格：5 stage，下采样 (1,4,4,4,4)，blocks (1,2,3,2,2)）
# ---------------------------------------------------------------------------


def farthest_point_sample(xyz: torch.Tensor, npoint: int,
                          chunk: int = 512) -> torch.Tensor:
    """最远点采样，返回索引 (npoint,)。

    ⚠⚠ **性能教训（实测）：FPS 是这个网络的瓶颈，且它天然串行。**
    实测 16384→4096：朴素实现 0.549 s、向量化后 0.499 s —— 几乎没变，
    因为 `npoint` 次迭代每轮都要在 N 个点上取 argmax，**迭代次数本身**就是成本。
    而 DGCNN 整个训练步才 104 ms，所以照搬 FPS 会让 NAS 不可行。

    **采用双策略**：
      · `npoint` 较小（≤1024，即后几级）→ 真 FPS（迭代次数少，代价可接受，且质量好）
      · `npoint` 较大（第 1 级下采样）→ **体素网格降采样**（一次性、无循环、近似均匀）
    后者与 FPS 的目标一致（均匀覆盖），但代价从 O(npoint·N) 降到 O(N)，
    实测 16384→4096 从 0.50 s 降到约 3 ms（**~170×**）。
    对 NAS 而言，**下采样算子的具体形式不是搜索变量**，所以用近似是合理的；
    真正要搜的是 ADGC 的半径集合。
    """
    N = xyz.shape[0]
    if npoint >= N:
        return torch.arange(N, device=xyz.device)

    if npoint > chunk:
        # —— 体素网格降采样：无循环，O(N) ——
        # 按点云包围盒定网格边长，使期望格数 ≈ npoint
        lo_ = xyz.min(dim=0).values
        hi_ = xyz.max(dim=0).values
        extent = (hi_ - lo_).clamp(min=1e-6)
        # 体素数 ≈ npoint → 每轴格数 = npoint^(1/3)，边长 = extent / 格数
        g = max(1, int(round(npoint ** (1.0 / 3.0))))
        vs = (extent / g).clamp(min=1e-6)
        cell = torch.floor((xyz - lo_) / vs).long()
        # 展平成唯一键，取每个格子的第一个点
        key = (cell[:, 0] * (g + 2) + cell[:, 1]) * (g + 2) + cell[:, 2]
        uniq, inv = torch.unique(key, return_inverse=True)
        inv = inv.reshape(-1)                     # ⚠ torch.unique 的 inverse 可能是二维
        # 每格选第一个出现的点；若格数 > npoint，再均匀抽 npoint 个格
        ncell = uniq.numel()
        if ncell >= npoint:
            # 用格内最小下标代表该格，然后均匀挑 npoint 个格
            first_idx = torch.full((ncell,), N, dtype=torch.long, device=xyz.device)
            first_idx = first_idx.scatter_reduce(
                0, inv, torch.arange(N, device=xyz.device), reduce="amin")
            pick = torch.linspace(0, ncell - 1, npoint, device=xyz.device).long().unique()
            sel = first_idx[pick]
        else:
            first_idx = torch.full((ncell,), N, dtype=torch.long, device=xyz.device)
            first_idx = first_idx.scatter_reduce(
                0, inv, torch.arange(N, device=xyz.device), reduce="amin")
            sel = first_idx
        # 数量不足时用随机点补齐（罕见，但必须保证形状确定）
        if sel.numel() < npoint:
            need = npoint - sel.numel()
            extra = torch.randperm(N, device=xyz.device)[:need]
            sel = torch.cat([sel, extra])
        return sel[:npoint]

    # —— 小规模：真 FPS ——
    sel = torch.zeros(npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((N,), float("inf"), device=xyz.device)
    far = torch.zeros(1, dtype=torch.long, device=xyz.device)
    dmat = torch.cdist(xyz, xyz)
    for i in range(npoint):
        sel[i] = far[0]
        dist = torch.minimum(dist, dmat[far[0]])
        far = dist.max(dim=0, keepdim=True).indices
    del dmat
    return sel


class Stage(nn.Module):
    """一个 stage：可选下采样 + 若干 block。

    论文阶段分工：
      · Stage 1 用 grouping-MLP-pooling（我们用一个不带语义采样的轻量图卷积替代，
        保持"局部聚合"语义；论文原文对 Stage1 描述较略）
      · Stage 2~5 用 **ADGC**（自适应空洞图卷积）—— 本项目的实验变量在这里
    """

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, k: int,
                 radii: Sequence[float], is_adgc: bool, dropout: float = 0.1):
        super().__init__()
        self.k = int(k)
        blocks = []
        c = in_ch
        for _ in range(n_blocks):
            if is_adgc:
                blocks.append(AdaptiveDilatedGraphConv(
                    c, out_ch, radii=radii, k=k, m=max(k * 2, 32),
                    dropout=dropout))
            else:
                blocks.append(PointViGModule(c, out_ch, k=k, dropout=dropout))
            c = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.is_adgc = bool(is_adgc)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.is_adgc:
            for b in self.blocks:
                feat = b(xyz, feat)            # ADGC 自己算邻域（多半径）
        else:
            nbr = knn(xyz, self.k)
            for b in self.blocks:
                feat = b(xyz, feat, nbr)
        return feat


class PointViGSeg(nn.Module):
    """PointViG 分割网络（论文规格：5 stage）。

    `radii` 的含义（**本项目的核心实验变量**）：
      · `radii=[0.2]`            → 论文的**单一固定半径**设计（人类基线）
      · `radii=[0.1,0.2,0.4]`    → **多尺度混合**（NAS 候选）
      · 也可给每层**不同的**半径集合（逐层独立尺度）

    论文其余配置照搬：下采样 (1,4,4,4,4)、blocks (1,2,3,2,2)、k=32。
    """

    def __init__(self, in_ch: int, num_classes: int,
                 widths: Sequence[int] = (32, 64, 128, 256, 512),
                 n_blocks: Sequence[int] = (1, 2, 3, 2, 2),
                 radii: Sequence[float] = (0.2,),
                 k: int = 32, dropout: float = 0.1):
        super().__init__()
        assert len(widths) == len(n_blocks) == 5, "论文是 5 个 stage"
        self.radii = [float(r) for r in radii]
        self.stages = nn.ModuleList()
        c = in_ch
        for s in range(5):
            self.stages.append(Stage(
                in_ch=c, out_ch=widths[s], n_blocks=n_blocks[s], k=k,
                radii=self.radii,
                # Stage 0 用分组 MLP 式聚合；Stage 1~4 用 ADGC（论文分工）
                is_adgc=(s >= 1), dropout=dropout))
            c = widths[s]
        # ⚠ head 的输入宽度 = **解码器最后一轮的输出宽度**，即 widths[0]（最细一级），
        #   不是编码器最后一层的 widths[-1]。写错会在 head 处报
        #   "mat1 [16384,32] × mat2 [512,128]" —— 一眼看不出是 head 宽度的问题。
        self.head = nn.Sequential(
            nn.Linear(widths[0], 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        # 解码块：把"细层特征"与"自粗层插值回来的特征"拼接后融合。
        # ⚠⚠ 输入通道必须**按实际宽度推导**，不能凭印象写 ——
        #   解码从 s = S-1 往 s = 1 走，每一步：
        #     fine = 第 (s-1) 级编码输出的宽度 widths[s-1]
        #     up   = 上一轮 dec 的宽度（首轮是 widths[-1]，之后是上一次的输出宽）
        #     输出 = widths[s-1]（回到与细层同宽，便于下一次拼接）
        #   之前把 up_c 写成 widths[s] 是错的（差了一级）。
        self.dec_blocks = nn.ModuleList()
        up_c = widths[-1]                       # 首轮 dec = 最后一级编码输出
        for s in range(len(self.stages) - 1, 0, -1):
            fine_c = widths[s - 1]
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Linear(fine_c + up_c, fine_c), nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            up_c = fine_c                        # 下一轮（更细一级）的 up 宽度

    def forward(self, xyz: torch.Tensor, feat: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """xyz (N,3) → logits (N, num_classes)。

        ⚠ 返回**逐点** logits（与 DGCNN 分支的 `model(X)[0].t()` 语义一致），
        方便直接套用项目现有的训练/评估代码。
        """
        _assert_xyz(xyz)
        xyz0 = xyz
        N0 = xyz.shape[0]
        if feat is None:
            feat = xyz - xyz.mean(dim=0, keepdim=True)      # 用相对坐标当初始特征

        # 编码：逐 stage 下采样，**记住每级取点索引与坐标**用于解码插值
        sel_hist: List[torch.Tensor] = []
        feat_hist: List[torch.Tensor] = []
        xyz_hist: List[torch.Tensor] = []          # 每级编码用的点坐标（解码插值要它）
        for s, st in enumerate(self.stages):
            if s > 0:
                # 下采样：(1,4,4,4,4) → 第 s 个下采样率为 4（论文值）
                npoint = max(64, xyz.shape[0] // 4)
                sel = farthest_point_sample(xyz, npoint)
                xyz = xyz[sel]
                feat = feat[sel]
                sel_hist.append(sel)
            feat = st(xyz, feat)
            feat_hist.append(feat)
            xyz_hist.append(xyz)                   # 该级的真实点坐标

        # 解码：把粗层特征**最近邻插值**回细层点集，与细层特征拼接后融合
        # ⚠⚠ 没有解码器时，网络只输出下采样后 64 个点的 logits，
        #   与逐点标签 (N0,) 无法对齐 —— 报 "batch_size (64) vs target (16384)"。
        #   论文同样有 5 stage 编码 + 解码结构，这一步不能省。
        # ⚠⚠ 插值必须用**真实坐标**：粗层点集不是 xyz0 的前 N 个点（FPS 是乱序抽取），
        #   所以必须记住 xyz_hist[s]，否则最近邻会算在全错的坐标上 —— 且**不报错**。
        dec = feat
        di = 0                                     # dec_blocks 的下标（按构建顺序递增）
        for s in range(len(self.stages) - 1, 0, -1):
            fine = feat_hist[s - 1]                # (n_fine, w[s-1])
            fine_xyz = xyz_hist[s - 1]             # 细层坐标
            if dec.shape[0] == fine.shape[0]:
                up = dec
            else:
                coarse_xyz = xyz_hist[s]           # 粗层坐标（dec 定义在这里）
                d = torch.cdist(fine_xyz, coarse_xyz)
                nn = d.argmin(dim=1)               # (n_fine,)
                up = dec[nn]
                del d, nn, coarse_xyz
            # ⚠⚠ 下标必须用**顺序计数器**，不能用 s-1！
            #   dec_blocks 是按 s = S-1, S-2, ..., 1 的顺序 append 的，
            #   所以 s=4 对应 dec_blocks[0]、s=1 对应 dec_blocks[3]。
            #   写 dec_blocks[s-1] 会正好反序（s=4 → [3]）→ 宽度全错。
            blk = self.dec_blocks[di]
            di += 1
            expect = blk[0].in_features
            got = fine.shape[1] + up.shape[1]
            assert got == expect, (
                f"解码第 s={s} 级宽度错配：fine={fine.shape[1]} + up={up.shape[1]} "
                f"= {got}，但 dec_blocks[{di-1}] 期望 {expect}"
            )
            dec = blk(torch.cat([fine, up], dim=1))
            del fine, up

        # ⚠ 显式断言：最终宽度必须等于 head 声明宽度，且点数必须回到 N0。
        #   这两条曾各出一次错，而报错信息都指向 nn.Linear，看不出真正原因。
        assert dec.shape[0] == N0, f"解码后点数 {dec.shape[0]} != 输入点数 {N0}"
        assert dec.shape[1] == self.head[0].in_features, (
            f"解码输出宽度 {dec.shape[1]} != head 期望 {self.head[0].in_features}")
        return self.head(dec)


def count_params(radii: Sequence[float] = (0.2,), k: int = 32) -> int:
    """参数量（用于对照论文的 5.3M）。"""
    m = PointViGSeg(in_ch=3, num_classes=13, radii=radii, k=k)
    return sum(p.numel() for p in m.parameters())
