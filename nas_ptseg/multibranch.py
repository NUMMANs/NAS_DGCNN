"""多分支架构：让**表示**成为可搜索的对象，而不只是调 DGCNN 的超参。

为什么做这个（决策依据，写清楚免得以后又被"归因"带回去）
================================================================
已实测的结论：
  · k 轴（邻域大小）—— 逐房间 oracle headroom 仅 +0.0017（< σ）→ 无空间
  · 架构轴（池化 / 图位置 / 容量）—— 主效应显著（max vs mean 差 0.07、feat128 +0.019），
    **但零交互**：6 个 Area 的最优配置完全相同
  · 该结构族**未饱和**（`r(cheap,Δ)=+0.809`，优势被训练拉大）→ 还有更好的结构存在

推论：**在同一族内换超参，收益封顶在主效应**。要拿到更大的效果，
需要换掉"族"——即换**几何表示**，因为不同表示对域差的响应机制在数学上就不同：
  · 笛卡尔点集 + kNN：感受野随密度漂移（本项目前几轮反复测的就是这件事）
  · 体素 / 柱状（BEV）：把点压成规则栅格 → 感受野由**栅格尺度**决定，与点数解耦
  · 多尺度柱状：同时用粗细两套栅格 → 粗糙尺度在低密度下不塌

本模块实现三个可组合的分支，并用 `MultiBranchSeg` 支持"分支存在性 + 融合位置"的组合：

    points 分支   : 现有 DGCNN EdgeConv（笛卡尔，kNN 动态图）
    pillar 分支   : 柱状体素化 → 逐柱 PointNet 式聚合 → BEV 伪图像 → 2D 卷积
                    （**感受野由柱宽与卷积核决定，与点数无关** —— 这是分支的关键性质）
    multiscale 分支: 同一 BEV 上跑两级栅格，粗尺度给低密度域兜底

所有分支都输出**逐点**特征（pillar 分支通过把 BEV 特征按柱索引回填到点上），
这样任意分支组合都能在**同一点集**上相加/拼接，融合位置（早期/晚期）成为可搜索量。

⚠ 重要：本模块**不引入任何额外训练技巧**（同样的 Adam/StepLR/300 epoch），
   唯一变量是架构。否则赢了也说不清是架构还是调参。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .tier2_dgcnn import EdgeConv, DROPOUT


# ---------------------------------------------------------------------------
# 柱状（pillar / BEV）分支
# ---------------------------------------------------------------------------


class PillarBranchV2(nn.Module):
    """柱状分支：柱状体素化 → 逐柱聚合 → BEV 伪图像 → 2D 卷积 → 回填到点。

    关键性质：**感受野由柱宽 × 卷积层数决定，与点数/密度无关**。
    因此在低密度域上不会像 kNN 那样"邻域被掐断" —— 这是本分支相对 points 分支
    的结构性差异，也是值得搜索的理由。

    ⚠⚠ 室内适配（这条是踩过才知道的，务必保留）
    ================================================
    初版把 z 完全丢掉、只做 XY 柱投影，结果 mIoU 只有 0.04。诊断（`diag_pillar.py`）：
      · BEV 栅格**占用率仅 4.6~6.5%**，柱内平均 **86 点（0.25m）/ 341 点（0.5m）**，
        柱内多数类投票的逐点 acc 上界只有 0.53~0.66；
      · BEV 图 **95% 是空像素**，batch=1 的 BN 被大量 0 污染。
    根因：**室内点云几乎全在墙面/天花板/地板上，在 XY 平面上是一圈薄壳**；
    柱状投影把三维结构压成极稀疏的环。柱状表示天生适合**户外激光雷达**
    （点铺在地面，BEV 占用率高），**不适合室内**。

    因此室内必须**保留高度信息**：不做单一 BEV 图，而是做 **z 分层栅格**——
    把 z 切成 `z_bins` 层，每层一张 XY 图，在 3D 张量上跑 (2+1)D 卷积。
    这样墙面/天花板/地板各自落在不同层里，占用率与可分性都恢复。
    """

    def __init__(self, in_c: int, out_c: int, xy_range: float = 8.0,
                 pillar_xy: float = 0.25, bev_layers: int = 3, z_bins: int = 8):
        super().__init__()
        self.xy_range = float(xy_range)
        self.pillar_xy = float(pillar_xy)
        self.z_bins = int(z_bins)
        self.side = int(np.ceil(2.0 * xy_range / pillar_xy))
        self.point_mlp = nn.Sequential(
            nn.Linear(in_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # (2+1)D 卷积：先在 XY 上做 2D 卷积（逐 z 层共享权重），再在 z 上做 1D 卷积。
        # 这样参数量不随 z_bins 增长，同时把高度结构利用起来。
        self.c = int(out_c) * 2
        # XY 卷积：**逐层串联**（第一层吃 2C，之后吃 C）。
        # （初版把单层 Conv2d 放进 for 循环重复调用 → 第二次通道就错了。）
        #
        # ⚠⚠ **这里绝不能用 BatchNorm**：室内体素图 **97.7% 是空体素**，
        #   而 BN 在 batch=1 时按 (B,C,H,W) 统计 → 均值/方差被大量填充的 0 主导。
        #   实测后果：跨点特征 std 只剩 0.0018（信号被稀释 3 个数量级），
        #   单 run 训练时间从 33s 掉到 1.2s（梯度近乎消失），mIoU 卡在 0.02~0.05。
        #   改用 **GroupNorm**：逐样本逐通道组归一化，**不受空体素比例影响**。
        n_grp = 8 if int(out_c) % 8 == 0 else 1
        xy: List[nn.Module] = []
        cin = self.c
        for _ in range(max(1, int(bev_layers))):
            xy += [nn.Conv2d(cin, out_c, 3, padding=1, bias=False),
                   nn.GroupNorm(n_grp, int(out_c)),
                   nn.LeakyReLU(0.2, inplace=True)]
            cin = int(out_c)
        self.xy_block = nn.Sequential(*xy)
        self.n_xy = max(1, int(bev_layers))
        self.z_conv = nn.Sequential(
            nn.Conv1d(int(out_c), out_c, 3, padding=1, bias=False),
            nn.GroupNorm(n_grp, int(out_c)), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_c, out_c, 1, bias=False),
        )

    def cell_ids(self, xyz: torch.Tensor):
        """返回 (线性 3D 栅格 id, gx, gy, gz)。以**房间中心**为原点。"""
        c = xyz.mean(dim=0, keepdim=True)
        p = xyz - c
        g = torch.floor((p[:, :2] + self.xy_range) / self.pillar_xy).long()
        g = g.clamp(0, self.side - 1)
        gx, gy = g[:, 0], g[:, 1]
        zmin = p[:, 2].min()
        zmax = p[:, 2].max()
        span = (zmax - zmin).clamp(min=1e-6)
        gz = (((p[:, 2] - zmin) / span) * self.z_bins).long().clamp(0, self.z_bins - 1)
        lin = (gz * self.side + gy) * self.side + gx
        return lin, gx, gy, gz

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                cell: Optional[Tuple[torch.Tensor, ...]] = None):
        if cell is None:
            cell = self.cell_ids(xyz)
        ids = cell[0]
        h = self.point_mlp(feat)                                   # (N, C)
        C = h.shape[1]

        uniq, inverse = torch.unique(ids, return_inverse=True)
        inverse = inverse.reshape(-1)
        P = uniq.numel()
        hmax = torch.zeros((P, C), dtype=h.dtype, device=h.device)
        hmax = hmax.index_reduce(0, inverse, h, "amax", include_self=True)
        cnt = torch.zeros((P, 1), dtype=h.dtype, device=h.device)
        cnt = cnt.index_add(0, inverse,
                            torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device))
        hsum = torch.zeros((P, C), dtype=h.dtype, device=h.device)
        hsum = hsum.index_add(0, inverse, h)
        hmean = hsum / cnt.clamp(min=1.0)

        # 唯一格 id → (gx, gy, gz)，按栅格坐标落像素（保持空间邻接）
        ux = uniq % self.side
        uy = (uniq // self.side) % self.side
        uz = uniq // (self.side * self.side)
        pix = (uz * self.side + uy) * self.side + ux

        vol = hmax.new_zeros((1, 2 * C, self.z_bins, self.side, self.side))
        flat = vol.view(1, 2 * C, -1)
        flat[0, :C, pix] = hmax.t()
        flat[0, C:, pix] = hmean.t()

        # XY 卷积（逐 z 层共享权重）：把 z 折进 batch 维
        B = self.z_bins
        v = vol.view(1, 2 * C, B, self.side * self.side).permute(0, 2, 1, 3)
        v = v.reshape(B, 2 * C, self.side, self.side)
        v = self.xy_block(v)                                       # (B, C, S, S)
        C2 = v.shape[1]
        v = v.reshape(1, B, C2, self.side * self.side).permute(0, 2, 1, 3)
        v = v.reshape(1, C2, B, self.side, self.side)

        # z 方向 1D 卷积：把 (S*S) 折进 batch，序列长度 = z_bins
        v = v.permute(0, 3, 4, 1, 2).reshape(-1, C2, B)            # (S*S, C, B)
        v = self.z_conv(v)                                         # 权重在 z 上共享
        v = v.reshape(1, self.side, self.side, C2, B).permute(0, 3, 4, 1, 2)
        out_vol = v.reshape(1, C2, -1).squeeze(0)                  # (C, B*S*S)
        out = out_vol[:, pix][:, inverse]                          # (C, N)
        return out.unsqueeze(0)                                    # (1, C, N)


# ---------------------------------------------------------------------------
# 多分支组合
# ---------------------------------------------------------------------------
class RangeViewBranch(nn.Module):
    """Range-view 分支：球面投影 → 距离图 → 2D 卷积 → 回填到点。

    为什么加第三个表示（NAS 的"表示搜索"叙事要完整）
    ================================================
    已测两个表示：`points`（笛卡尔 + kNN 动态图）与 `voxel`（规则 3D 栅格）。
    `range-view` 是点云文献里第三个主流家族，其**归纳偏置与前两者都不同**：
      · 它是**球面坐标**下的规则化 → 天然编码"方位角 / 仰角"，即**扫描仪视角**结构；
      · 分辨率**随距离衰减**（远处一个像素覆盖更大立体角）→ 与笛卡尔体素化的
        均匀分辨率完全不同；
      · 用标准 2D 卷积 → 计算量远低于 3D 卷积。

    ⚠⚠ **室内场景的一个关键修正**
    标准 range-view 是为**旋转式激光雷达**设计的：用 (方位角, 仰角) 建球面图。
    但 S3DIS 是**室内手持/固定扫描**，其"扫描中心"往往不是点云中心，强行套球面
    投影会把墙面投成极不均匀的条纹。
    因此这里用**以房间中心为原点的球面投影**：方位角 φ 绕竖直轴、仰角 θ 从水平面起算。
    这对室内是合理的（房间本身近似一个"从中心往外看"的封闭曲面），
    且仍保留了 range-view 的核心性质——**规则化 + 视角结构 + 距离分辨率衰减**。

    实现要点（与 pillar/voxel 同样的坑）
    ------------------------------------------------
    · 空像素占绝大多数 → **归一化必须用 GroupNorm**（BN 在 batch=1 下会被 0 主导，
      实测曾把跨点特征 std 从 0.326 压到 0.0018）。
    · 填图必须按 (φ, θ) 索引落像素，不能按点的出现顺序填（等价于随机置换，
      2D 卷积就失去空间意义）。
    · **每个像素可能收多个点**（室内近处点密）→ 取距离最小的那个（保留最近表面），
      这与 range-view 的物理语义一致（最近表面遮挡）。
    """

    def __init__(self, in_c: int, out_c: int, n_az: int = 128, n_el: int = 64,
                 rv_layers: int = 3):
        super().__init__()
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 距离作为额外通道输入图像（range 本身是 range-view 的核心信号）
        c_in = int(out_c) + 1
        n_grp = 8 if int(out_c) % 8 == 0 else 1
        layers: List[nn.Module] = []
        cin = c_in
        for _ in range(max(1, int(rv_layers))):
            layers += [nn.Conv2d(cin, out_c, 3, padding=1, bias=False),
                       nn.GroupNorm(n_grp, int(out_c)),
                       nn.LeakyReLU(0.2, inplace=True)]
            cin = int(out_c)
        self.conv = nn.Sequential(*layers)

    def angles(self, xyz: torch.Tensor):
        """(方位角索引, 仰角索引, 归一化距离)。以**房间中心**为原点。"""
        c = xyz.mean(dim=0, keepdim=True)
        p = xyz - c
        r = p.norm(dim=1).clamp(min=1e-6)
        az = torch.atan2(p[:, 1], p[:, 0])              # [-pi, pi]
        el = torch.asin((p[:, 2] / r).clamp(-1.0, 1.0))  # [-pi/2, pi/2]
        ia = ((az + np.pi) / (2 * np.pi) * self.n_az).long().clamp(0, self.n_az - 1)
        ie = ((el + np.pi / 2) / np.pi * self.n_el).long().clamp(0, self.n_el - 1)
        rmax = r.max().clamp(min=1e-6)
        return ia, ie, r / rmax

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        h = self.point_mlp(feat)                                  # (N, C)
        C = h.shape[1]
        ia, ie, rn = self.angles(xyz)
        pix = ie * self.n_az + ia                                  # 图像上的线性像素号
        P = self.n_el * self.n_az

        # 逐像素取**最近**的点（range-view 的物理语义：最近表面遮挡）。
        # 按距离升序稳定排序后，每个像素的**首次出现**即该像素的最近点。
        order = torch.argsort(rn, stable=True)
        pix_s, h_s, r_s = pix[order], h[order], rn[order]
        keep = torch.zeros_like(pix_s, dtype=torch.bool)
        keep[0] = True
        keep[1:] = pix_s[1:] != pix_s[:-1]          # 段首 = 该像素最近的那个点
        upix, ufeat, urange = pix_s[keep], h_s[keep], r_s[keep]

        # 按 (仰角, 方位角) 索引落像素 —— 不能按点序填，否则 2D 卷积失去空间意义
        img = ufeat.new_zeros((1, C, self.n_el, self.n_az))
        img.view(1, C, -1)[0, :, upix] = ufeat.t()
        ra = urange.new_zeros((1, 1, self.n_el * self.n_az))
        ra[0, 0, upix] = urange
        img = torch.cat([img, ra.view(1, 1, self.n_el, self.n_az)], dim=1)

        out_img = self.conv(img)                                   # (1, C, H, W)
        out_flat = out_img.view(C, -1)                             # (C, P)
        return out_flat[:, pix].unsqueeze(0)                       # (1, C, N)


class VoxelBranch(nn.Module):
    """体素分支：规则 3D 栅格 + **稀疏 3D 卷积（手动 scatter-gather 实现）**。

    与 `PillarBranchV2` 的关键区别
    ================================================
    · pillar：把 z 压进**通道维**，XY 上做 2D 卷积 → z 只作为特征而非空间轴
    · voxel ：z 是**第三个空间轴**，做真正的 3D 卷积（3×3×3 核）
    这是"表示"层面真正的差异，也正是值得测的地方。

    为什么手写而不是用稀疏卷积库
    ================================================
    本机没装 `torch_scatter`/`spconv`，且引入新依赖会破坏可复现性。
    手写版本用 `index_add`（逐体素聚合）+ `gather`（按偏移采样邻居）实现，
    对 16k 点、32³ 以下的栅格足够快（体素数 ~10³ 量级）。

    ⚠⚠ **归一化必须用 GroupNorm，不能用 BatchNorm**
    室内体素图实测 **97.7% 是空体素**；BN 在 batch=1 下按 (B,C,·) 统计会被
    大量填充的 0 主导 → 跨点特征 std 从 0.326 掉到 0.0018（稀释 180 倍），
    训练时间从 33s 掉到 1.2s（梯度近乎消失）。这条对任何稀疏图卷积都成立。
    """

    def __init__(self, in_c: int, out_c: int, xy_range: float = 8.0,
                 voxel_size: float = 0.25, layers: int = 2, z_bins: int = 16):
        super().__init__()
        self.xy_range = float(xy_range)
        self.voxel_size = float(voxel_size)
        self.z_bins = int(z_bins)
        self.side = int(np.ceil(2.0 * xy_range / voxel_size))
        self.point_mlp = nn.Sequential(
            nn.Linear(in_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        n_grp = 8 if int(out_c) % 8 == 0 else 1
        self.layers = int(layers)
        # 逐层：稀疏 3D 卷积（聚合自身 + 26 邻域）→ GroupNorm → ReLU
        self.convs = nn.ModuleList([
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
            for _ in range(self.layers)])
        self.norms = nn.ModuleList([
            nn.GroupNorm(n_grp, int(out_c)) for _ in range(self.layers)])
        # 26 邻域偏移（含自身用 0）
        offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        self.register_buffer("_offs", torch.tensor(offs, dtype=torch.long),
                             persistent=False)

    def voxel_ids(self, xyz: torch.Tensor):
        c = xyz.mean(dim=0, keepdim=True)
        p = xyz - c
        g = torch.floor((p[:, :2] + self.xy_range) / self.voxel_size).long()
        g = g.clamp(0, self.side - 1)
        zmin = p[:, 2].min()
        zmax = p[:, 2].max()
        span = (zmax - zmin).clamp(min=1e-6)
        gz = (((p[:, 2] - zmin) / span) * self.z_bins).long().clamp(0, self.z_bins - 1)
        return gz, g[:, 1], g[:, 0]          # (z, y, x)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        h = self.point_mlp(feat)                                # (N, C)
        C = h.shape[1]
        gz, gy, gx = self.voxel_ids(xyz)
        S, Z = self.side, self.z_bins

        uniq, inverse = torch.unique((gz * S + gy) * S + gx, return_inverse=True)
        inverse = inverse.reshape(-1)
        P = uniq.numel()

        # 体素级聚合（max）
        v = torch.zeros((P, C), dtype=h.dtype, device=h.device)
        v = v.index_reduce(0, inverse, h, "amax", include_self=True)

        # 稀疏 3D 卷积：用稠密稀疏张量 (1,C,Z,S,S) 跑 conv3d，但**只在非空体素上回填**
        # （空体素保持 0，GroupNorm 不受其比例影响）
        uz = uniq // (S * S)
        uy = (uniq // S) % S
        ux = uniq % S
        for li in range(self.layers):
            dense = v.new_zeros((1, C, Z, S, S))
            flat = dense.view(1, C, -1)
            lin = (uz * S + uy) * S + ux
            flat[0, :, lin] = v.t()
            dense = self.convs[li](dense)
            # 只在非空体素的**原位置**取回（不需要邻居的值回填到点）
            v = dense.view(1, C, -1)[0, :, lin].t()
            v = self.norms[li](v)
            v = torch.nn.functional.leaky_relu(v, 0.2)

        out = v[inverse]                                        # (N, C)
        return out.t().unsqueeze(0)                             # (1, C, N)


# ---------------------------------------------------------------------------
# 多分支组合
# ---------------------------------------------------------------------------
def _parse_voxel_spec(spec: str, default_z: int) -> Tuple[float, int]:
    """解析 voxel 分支名 → (体素边长 m, z_bins)。

    支持两种写法：
        `voxel25`        体素 0.25 m，z_bins 用外部默认值
        `voxel25_z64`    体素 0.25 m，**z_bins 强制 64**

    为什么允许在**分支名里指定 z_bins**：这是实现"**多尺度 z**"的关键 ——
    同一个网络里并行挂多个不同 z 分辨率的体素分支（如 `voxel25_z32,voxel25_z96`），
    让粗尺度提供大范围上下文、细尺度提供精确边界。

    为什么值得做：实测 **z_bins 是主导因素**（z8 0.3290 → z64 0.3885，跨度 0.0595，
    远大于体素尺寸的 0.0032），且呈**倒 U 形**（z64 峰值、z96 持平、z128 起下降）。
    倒 U 形意味着**不同 z 各有自己的盲区** —— 这正是多尺度融合能补上的地方，
    也是单靠调一个 z 拿不到的收益。
    """
    body = spec[len("voxel"):]
    z = int(default_z)
    if "_z" in body:
        body, zpart = body.split("_z", 1)
        try:
            z = int(zpart)
        except ValueError:
            z = int(default_z)
    try:
        vw = float(body) / 100.0
    except ValueError:
        vw = 0.25
    return vw, z


class MultiBranchSeg(nn.Module):
    """可组合的多分支分割网络。

    参数
    ----
    branches : 启用哪些分支，逗号分隔，如 "points,pillar25" 或 "points,voxel25"
        points       笛卡尔点集 + kNN 动态图（原 DGCNN）
        pillarXX     柱状分支，柱宽 = XX/100 米（z 压进通道，XY 上 2D 卷积）
        voxelXX      体素分支，体素边长 = XX/100 米（**z 是第三空间轴**，3D 卷积）
    fuse : 融合方式
        "late"   各分支各自出 logits 再相加（融合在最后一层）
        "early"  各分支特征拼接后共同进 head（融合在最早，参数量会上升）
    k : points 分支的邻域大小

    ⚠ 为了让"换分支"成为**唯一变量**，这里刻意不引入任何新的训练技巧；
      各分支的 out_c 固定，融合方式只有 late/early 两种。
    """

    def __init__(self, num_classes: int, branches: str = "points", k: int = 48,
                 feat: int = 64, layers: int = 3, fuse: str = "late",
                 xy_range: float = 8.0, bev_layers: int = 3, z_bins: int = 16,
                 voxel_layers: int = 2, rv_layers: int = 3):
        """`z_bins` 会**原样**传给 voxel/pillar 分支（不做任何倍乘）。

        ⚠ 曾经把 voxel 写成 `z_bins=max(8, z_bins*2)`，导致"配置里的 z_bins"
        与"实际生效的 z_bins"不一致，扫参时会得出错误的曲线。
        配置名里的 zNN 必须等于实际 z_bins，参数必须直白可信。
        """
        super().__init__()
        self.spec = [b.strip() for b in branches.split(",") if b.strip()]
        self.fuse = str(fuse)
        self.k = int(k)
        self.feat = int(feat)
        self.num_classes = int(num_classes)

        self.points_net = None
        self.pillars = nn.ModuleDict()
        self.voxels = nn.ModuleDict()
        self.ranges = nn.ModuleDict()
        # gate 融合的可学习权重（延迟创建：需要知道分支数）
        self.gate: Optional[nn.Parameter] = None
        for b in self.spec:
            if b == "points":
                convs = [EdgeConv(3, feat, k, agg="max")]
                for _ in range(int(layers) - 1):
                    convs.append(EdgeConv(feat, feat, k, agg="max"))
                self.points_net = nn.ModuleList(convs)
            elif b.startswith("pillar"):
                pw = float(b[len("pillar"):]) / 100.0
                self.pillars[b] = PillarBranchV2(3, feat, xy_range=xy_range,
                                                 pillar_xy=pw, bev_layers=bev_layers,
                                                 z_bins=z_bins)
            elif b.startswith("voxel"):
                vw, zb = _parse_voxel_spec(b, z_bins)
                self.voxels[b] = VoxelBranch(3, feat, xy_range=xy_range,
                                             voxel_size=vw, layers=voxel_layers,
                                             z_bins=zb)
            elif b.startswith("rangeview"):
                # `rangeview32` → n_az=n_el=32；不写数字则用默认 64
                body = b[len("rangeview"):]
                side = int(body) if body.isdigit() else 64
                self.ranges[b] = RangeViewBranch(3, feat, n_az=side, n_el=side,
                                                 rv_layers=rv_layers)
            else:
                raise ValueError(f"未知分支：{b}")

        n_br = len(self.spec)
        # 各分支输出通道：points 分支把每层输出拼接（DGCNN 原设计 → feat*layers），
        # 其余分支只出一份 feat。head 的输入维度必须按分支实际输出算，
        # 否则会静默地在 early 融合时通道不匹配、在 late 时矩阵乘错。
        self.br_out: List[int] = []
        for b in self.spec:
            self.br_out.append(feat * int(layers) if b == "points" else feat)

        if self.fuse == "concat":
            # concat 只用**一个吃全部通道的 head**；逐分支 head 会被白建、不参与前向
            # → 注册成死参数（实测 24 个参数无梯度），既虚增参数量又污染优化器。
            # 所以这里把 heads 留空，只建 concat_head。
            self.heads = nn.ModuleList()
            self.concat_head: Optional[nn.Sequential] = self._make_head(
                sum(self.br_out), num_classes)
        else:
            # late / gate 都用"每分支一个 head"（gate 需要逐分支 logits 才能加权）
            self.heads = nn.ModuleList(
                [self._make_head(c, num_classes) for c in self.br_out])
            self.concat_head = None

    @staticmethod
    def _make_head(in_c: int, num_classes: int) -> nn.Sequential:
        mid = max(in_c, 128)
        return nn.Sequential(
            nn.Conv1d(in_c, mid, 1, bias=False),
            nn.BatchNorm1d(mid), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(DROPOUT),
            nn.Conv1d(mid, mid, 1, bias=False),
            nn.BatchNorm1d(mid), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(mid, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz 形状 **(1, 3, N)**（与 DGCNNSeg 一致，保持同一套调用约定）。

        ⚠⚠ **必须把 xyz 降成 (N,3) 再交给分支**：
        分支内部的栅格索引按 (N,3) 语义写。若直接传 (1,3,N)，`xyz.mean(dim=0)`
        会得到 (3,N)，栅格索引计算虽**形状能广播通过、不报错**，但得到的索引全错，
        结果是分支输出 std 正常（0.62）而经 head 后 logits std 只剩 0.048
        → 模型对任何输入都预测同一类（实测预测类数 = 1）。
        这个 bug 曾让我把"分支表示不可用"误判了两轮（同房 mIoU 0.04 vs 修后 0.85）。
        """
        xyz_n3 = xyz[0].t().contiguous()            # (N, 3)
        feats: List[torch.Tensor] = []
        for b in self.spec:
            if b == "points":
                x = xyz
                outs = []
                for c in self.points_net:
                    x = c(x)
                    outs.append(x)
                feats.append(torch.cat(outs, dim=1))
            elif b.startswith("pillar"):
                feats.append(self.pillars[b](xyz_n3, xyz_n3))
            elif b.startswith("rangeview"):
                feats.append(self.ranges[b](xyz_n3, xyz_n3))
            else:
                feats.append(self.voxels[b](xyz_n3, xyz_n3))

        if self.fuse == "late":
            logits = None
            for f, h in zip(feats, self.heads):
                lg = h(f)
                logits = lg if logits is None else logits + lg
            return logits
        if self.fuse == "concat":
            # concat 需要"一个吃全部通道的 head"，与 late/gate 的逐分支 head 不同
            return self.concat_head(torch.cat(feats, dim=1))
        if self.fuse == "gate":
            # 可学习的**逐点软加权**融合：为每个分支学一个标量门（逐点共享），
            # 再 sigmoid 归一化相加。相比 late 的固定等权相加，
            # 这让网络自己决定"哪个尺度的分支更可信" —— 这正是多尺度融合的关键自由度。
            # 参数量几乎不增（每分支 1 个标量），所以是"结构变化而非容量变化"。
            if self.gate is None or self.gate.numel() != len(feats):
                self.gate = nn.Parameter(torch.zeros(len(feats), device=feats[0].device))
            w = torch.softmax(self.gate, dim=0)
            logits = None
            for k, (f, h) in enumerate(zip(feats, self.heads)):
                lg = h(f) * w[k]
                logits = lg if logits is None else logits + lg
            return logits
        raise ValueError(f"未知 fuse：{self.fuse}")


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
