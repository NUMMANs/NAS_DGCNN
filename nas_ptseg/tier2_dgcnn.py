"""Tier-2：真实 DGCNN 短训，回答"真实训练中是否存在 density x k 交互"。

为什么必须做这一步
==================
Tier-1（训练无关代理）已通过偏置检验、并量化了噪声，但分辨率不足：
sigma ≈ 0.0068，而目标效应只有 0.005~0.02。所以 scaling law 不能由代理产出，
只能由真实训练回答。

这个问题不是"k 越大越好"，而是**交互效应**：

    I(rho, k) = mIoU(k=64, rho) - mIoU(k=16, rho)

    I 与 rho 无关  -> 等度数律（density-invariant geometry），NAS 前提被削弱
    I 随 rho 单调变化 -> 存在 density x geometry 交互，NAS 前提成立

设计约束（用户裁定，不可改）
============================
  - 单一变量：只变 (density, k)。epoch 数 / optimizer / LR / seed / 增强 全固定。
  - 半径固定为"门控不生效"：本实现**不做半径门控**，但在开跑前实测每个
    (域, k) 的第 k 近邻距离并断言 < R_MAX_M，把该值写进结果 JSON 作为合规证据。
  - 不做任何 NAS。
  - GPU 1（由 run_tier2.ps1 设置 CUDA_VISIBLE_DEVICES=1）。

划分方式：空间块留出（关键设计）
================================
**不**用随机留点。随机留点会把训练集的点抽稀，等于在训练时人为改变局部密度 ——
而密度正是本实验的自变量。改用**空间块留出**（沿房间最长轴取上侧一段作为留出区）：
整片区域被留出，训练集里剩余区域的**局部密度不变**，因此"密度"这个变量保持干净。

同时这也给出真正的泛化测试：留出区在训练中完全不可见。

已知局限（必须写进论文）
========================
不同密度的训练点数天然不同（rho0≈11.5k vs rho4≈0.7k），而每 epoch 只做一次
全点前向 → 各 run 的"每步批大小"不同。这**不影响同一密度内 k 的对比**
（同密度下 N 相同），但会影响 Δ_k 的**跨密度**比较。
缓解：每 epoch 记录训练 loss，`--analyze` 同时给出
  (a) 末尾 epoch 的 Δ_k
  (b) 平均 mIoU 最优 epoch 的 Δ_k
  (c) 训练 loss 首次越过公共阈值那个 epoch 的 Δ_k（对齐"学到的程度"）
三者一致才认定结论稳健。

用法
====
    # 一个配置（先拿它确定"短训窗口"）
    python -m nas_ptseg.tier2_dgcnn --train-domain rho0 --k 16 --epochs 200

    # 窗口探针：同域三个 k，看曲线何时分离
    python -m nas_ptseg.tier2_dgcnn --window-probe --train-domain rho0 --epochs 200

    # 完整 9 次（3 域 x 3 k）
    python -m nas_ptseg.tier2_dgcnn --matrix --epochs 200

    # 汇总：算交互效应与 beta3
    python -m nas_ptseg.tier2_dgcnn --analyze
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import density as D
    from nas_ptseg import gpu as GPU

    _PKG_MODE = False
else:
    from . import data as DT
    from . import density as D
    from . import gpu as GPU

    _PKG_MODE = True

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 固定超参 —— 所有 run 必须完全一致，这是单一变量原则的底线
# ---------------------------------------------------------------------------
SEED = 12345
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
LR_STEP = 60          # StepLR 的步长（epoch）
LR_GAMMA = 0.5
FEAT_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.3
KNN_CHUNK = 4096

# 默认矩阵：3 域 x 3 k，半径全部不生效
MATRIX_DOMAINS = ["rho0", "rho2", "rho4"]
MATRIX_KS = [16, 32, 64]

# 密度条件化策略 k(ρ) = clip(round(k0 * (ρ/ρ_ref)^alpha)) 的合法范围
K_POLICY_MIN, K_POLICY_MAX = 4, 128

# 固定 N 对照的默认点数与目标 k 网格
CROP_N_DEFAULT = 1024
CROP_K_LIST = [8, 16, 32, 64]

# 多 seed 默认（给每个 (ρ,k) 一个误差棒）
SEEDS_DEFAULT = [12345, 23456, 34567]

# 密度条件化基线的 alpha 网格（负 = 稀疏域收缩 k；正 = 稀疏域扩大 k）
ALPHAS_DEFAULT = [-0.5, -0.3, 0.0, 0.3, 0.5]

# 半径门控"不生效"的断言上界（米）。与 proxy 的 R_MAX_M 一致。
RADIUS_NONBINDING_M = 2.0

RESULT_DIR = "tier2_results"


class _Log:
    """同时 print 与写 UTF-8 文件的日志器。

    为什么不用 PowerShell 捕获 stdout：Windows PowerShell 会按 ANSI 重新编码子进程
    输出，中文全变乱码（本机已多次踩到）。直接由 Python 写 UTF-8 最稳。
    """

    def __init__(self, path: Optional[str] = None):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, *args) -> None:
        s = " ".join(str(a) for a in args)
        try:
            print(s)
        except Exception:
            pass
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            try:
                self.f.close()
            except Exception:
                pass
            self.f = None


# ---------------------------------------------------------------------------
# numpy -> torch 的兼容转换
# ---------------------------------------------------------------------------
_NP2TORCH = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def to_torch(arr: np.ndarray, dev: torch.device) -> torch.Tensor:
    """numpy -> torch，优先走标准路径，失败才回退到 frombuffer。

    为什么需要它：曾遇到一个环境里 torch 与 numpy 的 C-API 不匹配
    （`numpy/_core/` 里残留了另一套编译好的 `_multiarray_umath*.pyd`，
    形成两套标量类型），于是：
        torch.from_numpy(...)  -> TypeError: expected np.ndarray (got numpy.ndarray)
        torch.tensor(ndarray)  -> RuntimeError: Could not infer dtype of numpy.float32
    `torch.frombuffer` 完全绕开 numpy 互操作层，可在这种环境下继续工作。

    现在先试标准路径（零拷贝、更快），只有抛异常才回退，因此：
      - 健康环境：走 `torch.from_numpy`，无额外开销；
      - 互操作损坏环境：自动回退，仍然能跑。
    回退路径用 .clone() 断开与临时 bytearray 的生命周期耦合，避免悬垂指针。
    """
    a = np.ascontiguousarray(arr)
    try:
        return torch.from_numpy(a).to(dev)
    except Exception:  # noqa: BLE001 —— 互操作层损坏，退回 frombuffer
        key = str(a.dtype)
        dt = _NP2TORCH.get(key)
        if dt is None:
            raise TypeError(f"to_torch 未支持 dtype: {a.dtype}")
        t = torch.frombuffer(bytearray(a.tobytes()), dtype=dt).reshape(tuple(a.shape))
        return t.clone().to(dev)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_domains() -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], dict]:
    return DT.load_cache()


def main_domain_names(meta: dict) -> List[str]:
    return [n for n, dm in (meta.get("domains") or {}).items()
            if dm.get("role", "main") == "main"]


def spatial_holdout_mask(
    coords_ref: np.ndarray,
    coords: np.ndarray,
    holdout_frac: float,
    holdout_m: Optional[float] = None,
) -> Tuple[np.ndarray, int, float]:
    """空间块留出：沿参考点云最长轴，把上侧一段作为留出区。

    为什么用**参考点云的绝对阈值**而不是每个域各算分位数：
    各域是同一房间的体素下采样，空间范围一致。用统一的绝对阈值，
    才能保证"留出的是同一片区域"，跨域结果才可比。

    两种模式（`holdout_m` 决定）
    ==========================
    - `holdout_m=None`（旧）：按**比例**取，`thr = quantile(coords_ref, 1-frac)`。
      ⚠ 已实测有致命混淆：补丁越小 → 留出板越薄 → 留出点离训练点越近 →
      **任务越容易**。上一轮 mIoU 随补丁变大单调下降（0.87 → 0.40），
      k 的效应被"任务难度"淹没，结论随裁剪方式翻转。
    - `holdout_m=T`（新，默认）：按**固定物理厚度**取，`thr = max - T`。
      各组留出板厚度恒为 T 米，留出点的"外推距离"大致恒定，任务难度可比。
      厚度由 `_probe_holdout.py` 实测选定：T=0.75 m 时五档裁剪的留出比例
      全落在 11%~33% 的健壮区间（T=0.3 会让三档留出 <5%，方差爆炸；
      T≥1.5 会让小补丁被整块留出，训练点不足）。

    注：无论哪种模式，都会在 run_once 里实测 `d_tr`（留出点到最近训练点的
    距离分布）并写进结果 JSON —— 混淆量的**直接测量**，而不是假定它已被消除。
    """
    ext = coords_ref.max(axis=0) - coords_ref.min(axis=0)
    axis = int(np.argmax(ext))
    lo = float(coords_ref[:, axis].min())
    hi = float(coords_ref[:, axis].max())
    if holdout_m is not None and float(holdout_m) > 0:
        thr = max(hi - float(holdout_m), lo + 1e-6)
    else:
        thr = float(np.quantile(coords_ref[:, axis], 1.0 - holdout_frac))
    return coords[:, axis] >= thr, axis, thr


def holdout_diag(
    ho_xyz: np.ndarray, tr_xyz: np.ndarray
) -> dict:
    """留出区诊断：有效留出比例 + 留出点到最近训练点的距离分布。

    `d_tr`（distance to nearest training point）是**任务难度混淆量的直接测量**。
    旧的比例留出下它在各配置组之间相差 2.6 倍；改成固定物理厚度后收到 2.3 倍，
    余下的部分作为协变量记录下来，供分析时控制或至少显式标注。
    """
    n_ho, n_tr = int(ho_xyz.shape[0]), int(tr_xyz.shape[0])
    out = {
        "n_holdout_points": n_ho,
        "n_train_points": n_tr,
        "holdout_eff_frac": float(n_ho / max(n_ho + n_tr, 1)),
        "holdout_to_train_med_m": None,
        "holdout_to_train_p90_m": None,
    }
    if n_ho < 1 or n_tr < 1:
        return out
    _i, dd = D.knn_query(ho_xyz, tr_xyz, 1)
    dd = np.asarray(dd, dtype=np.float64).reshape(-1)
    out["holdout_to_train_med_m"] = float(np.median(dd))
    out["holdout_to_train_p90_m"] = float(np.percentile(dd, 90))
    return out


# ---------------------------------------------------------------------------
# 模型：DGCNN (EdgeConv) 语义分割
# ---------------------------------------------------------------------------
def knn_idx(x: torch.Tensor, k: int, chunk: int = KNN_CHUNK) -> torch.Tensor:
    """在**特征空间**构建 kNN 动态图，返回 (N, k)。分块以控制显存。"""
    n = x.shape[0]
    kk = int(min(k + 1, n))
    out = torch.empty((n, k), dtype=torch.long, device=x.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = torch.cdist(x[s:e], x)
        _, idx = torch.topk(d, kk, dim=1, largest=False)
        sel = idx[:, 1:kk]                        # 去掉自身
        if sel.shape[1] < k:                      # n <= k 的极端边界，循环补齐
            reps = int(np.ceil(k / max(sel.shape[1], 1)))
            sel = sel.repeat(1, reps)[:, :k]
        out[s:e] = sel
    return out


class EdgeConv(nn.Module):
    """h_ij = MLP([x_i ; x_j - x_i])，然后对 j 做池化。

    `agg` 与 `idx` 是为**扩大搜索维度**加的可选开关（默认完全等价于原实现）：
      - `agg="max"`（默认）原 DGCNN 行为；
        `agg="mean"` 平均池化；`agg="maxmean"` 两者各半。
        池化方式是经典 NAS 轴，且**参数量不变**（只改聚合），可比性干净。
      - `idx`：外部传入固定的 kNN 索引 → 实现**坐标空间静态图**
        （原 DGCNN 是**特征空间动态图**，每层重算）。这是"图更新策略"这一维。
    """

    def __init__(self, in_c: int, out_c: int, k: int, agg: str = "max"):
        super().__init__()
        self.k = int(k)
        self.agg = str(agg)
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (1, C, N)
        xt = x[0].t().contiguous()                     # (N, C)
        if idx is None:
            idx = knn_idx(xt, self.k)                  # (N, k)
        nb = xt[idx]                                   # (N, k, C)
        edge = torch.cat([xt[:, None, :].expand_as(nb), nb - xt[:, None, :]], dim=-1)
        # (N, k, 2C) -> (1, 2C, N, k)：conv2d 要求 (batch, channel, H, W)
        e = edge.permute(2, 0, 1).unsqueeze(0)
        e = self.net(e)                                # (1, out_c, N, k)
        if self.agg == "mean":
            return e.mean(dim=-1)
        if self.agg == "maxmean":
            return 0.5 * (e.max(dim=-1)[0] + e.mean(dim=-1)[0])
        return e.max(dim=-1)[0]                        # (1, out_c, N)


class DGCNNSeg(nn.Module):
    """DGCNN 分割网络。k 只改变图度数，**参数量与 k 无关**（保证 k 轴对比公平）。

    为"扩大搜索维度"新增三个**默认等价于原实现**的开关：
        feat        每层特征宽度（原 64）
        layers      层数（原 3）
        agg         EdgeConv 池化方式（"max" 原实现 / "mean" / "maxmean"）
        graph_space "feat" 特征空间动态图（原实现）/ "coord" 坐标空间静态图

    ⚠ 只有改 `agg` / `graph_space` 时参数量不变，适合"同预算比结构"；
    改 `feat` / `layers` 会改参数量与计算量，作为**容量轴**单独报告（要配代价约束）。
    """

    def __init__(self, num_classes: int, k: int,
                 feat: int = FEAT_DIM, layers: int = NUM_LAYERS,
                 agg: str = "max", graph_space: str = "feat"):
        super().__init__()
        self.k = int(k)
        self.agg = str(agg)
        self.graph_space = str(graph_space)
        self.feat = int(feat)
        self.layers = int(layers)
        convs = [EdgeConv(3, feat, k, agg=agg)]
        for _ in range(int(layers) - 1):
            convs.append(EdgeConv(feat, feat, k, agg=agg))
        self.convs = nn.ModuleList(convs)
        self.head = nn.Sequential(
            nn.Conv1d(feat * int(layers), 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(DROPOUT),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz
        fixed_idx = None
        if self.graph_space == "coord":
            # 坐标空间静态图：只建一次，各层复用（与"逐层在特征空间重算"形成对照）
            fixed_idx = knn_idx(xyz[0].t().contiguous(), self.k)
        outs = []
        for c in self.convs:
            x = c(x, fixed_idx)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        return self.head(x)                            # (1, C, N)


# ---------------------------------------------------------------------------
# mIoU
# ---------------------------------------------------------------------------
def miou_np(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[float, float]:
    acc = float((pred == gt).mean()) if gt.size else 0.0
    ious = []
    for c in range(num_classes):
        g = gt == c
        p = pred == c
        if not g.any():
            continue
        inter = float((g & p).sum())
        union = float((g | p).sum())
        ious.append(inter / union if union > 0 else 0.0)
    return (float(np.mean(ious)) if ious else 0.0), acc


# ---------------------------------------------------------------------------
# 固定 N 对照：空间裁剪
# ---------------------------------------------------------------------------
def crop_points(
    xyz: np.ndarray,
    y: np.ndarray,
    n_target: int,
    center: Optional[np.ndarray] = None,
    stretch_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """取以锚点为球心的最近 n_target 个点，得到一个**空间连续**的点块。

    为什么必须用"球内最近点"而不是随机抽点
    ======================================
    随机抽点会把点云**均匀抽稀**，直接降低局部密度 —— 而密度是自变量，
    那样做等于同时改了两个变量。球内取点保持局部密度不变，只**缩小空间范围**。

    锚点必须落在**表面上**（关键修正）
    ==================================
    室内点云的几何中心在**空房间的空气里**。若以该点取最近邻，会同时命中
    地板与天花板（甚至两侧墙面），得到一个**多块不相连**的集合 —— 那就不是
    "一块小场景"，"缩小空间范围"这个操作也就名不副实了。
    因此默认锚点取**离几何中心最近的那个真实点**，再从它出发取最近邻，
    保证得到的是一块连续曲面片。

    各向异性拉伸 `stretch_s`（打破 k/N 与 r_k/L 的退化）
    ==================================================
    在**同质采样**的场景上，两个候选控制变量是**数学等价**的：
        面采样下 k ≈ σπr²（σ 为面密度），且 L ∝ (N/σ)^(1/2)
        ⇒  r_k/L ∝ (k/N)^(1/2)
    也就是说"k/N 大"与"r_k/L 大"只差一个单调（平方根）变换，**任何**
    单一房间上的实验都无法区分它们。唯一能打破退化的自由量是**形状**：

        固定 N、固定 ρ，把补丁**沿某一轴拉长**
            → k/N 不变，但 L 变大 → r_k/L 变小

    实现：把该轴的坐标差乘上 s<1 再算欧氏距离（各向异性度量），于是该方向
    可延伸到更远 → 补丁沿该轴伸长，而点数与局部密度都不变。
    s<1 越小，伸长越明显；s=None/1.0 即各向同性。
    """
    n = xyz.shape[0]
    if n_target <= 0 or n_target >= n:
        return xyz, y
    if center is None:
        centroid = xyz.mean(axis=0)
        anchor_idx = int(np.argmin(np.linalg.norm(xyz - centroid[None, :], axis=1)))
        anchor = xyz[anchor_idx]
    else:
        anchor = np.asarray(center, dtype=np.float32)
    dvec = (xyz - anchor[None, :]).astype(np.float32)
    if stretch_s is not None and float(stretch_s) > 0 and float(stretch_s) < 1.0:
        ext = xyz.max(axis=0) - xyz.min(axis=0)
        ax = int(np.argmax(ext))
        dvec[:, ax] = dvec[:, ax] * float(stretch_s)
    d = np.linalg.norm(dvec, axis=1)
    idx = np.sort(np.argsort(d)[: int(n_target)])
    return xyz[idx], y[idx]


# ---------------------------------------------------------------------------
# 单次训练
# ---------------------------------------------------------------------------
def resolve_k(spec: dict, rho_domain: float, rho_ref: float) -> int:
    """由 spec 决定本次训练用的 k（固定值，或密度条件化策略）。"""
    if spec.get("alpha") is None:
        return int(spec["k"])
    k0 = float(spec.get("k0", 32.0))
    a = float(spec["alpha"])
    return int(np.clip(round(k0 * (max(rho_domain, 1e-9) / max(rho_ref, 1e-9)) ** a),
                       K_POLICY_MIN, K_POLICY_MAX))


def run_once(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    meta: dict,
    spec: dict,
    epochs: int,
    holdout_frac: float,
    holdout_m: Optional[float] = None,
    verbose_every: int = 10,
    log=print,
) -> dict:
    """跑一个配置。spec 字段：
        tag, train_domain, seed, k | (alpha, k0, rho_ref), crop_n, stretch_s
    """
    tag = spec["tag"]
    train_domain = spec["train_domain"]
    seed = int(spec.get("seed", SEED))
    crop_n = int(spec.get("crop_n", 0) or 0)
    stretch_s = spec.get("stretch_s")
    stretch_s = float(stretch_s) if stretch_s is not None else None

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = GPU.setup_torch_device(verbose=False)

    mains = main_domain_names(meta)
    ref_xyz = domains[mains[0]][0]

    xyz_full, y_full = domains[train_domain]
    rho_domain = float(np.median(D.local_density(xyz_full)))
    rho_ref = float(spec.get("rho_ref") or rho_domain)
    k = resolve_k(spec, rho_domain, rho_ref)

    # ---- 固定 N 对照：空间裁剪（局部密度不变，只缩小空间范围）----
    xyz_c, y_c = crop_points(xyz_full, y_full, crop_n, stretch_s=stretch_s)
    cropped = bool(crop_n > 0 and xyz_c.shape[0] < xyz_full.shape[0])
    stretch_eff = stretch_s if (cropped and stretch_s is not None) else None

    # ---- 空间留出 ----
    eval_xyz: Dict[str, tuple] = {}
    if cropped:
        # 裁剪后必须用**裁剪集合自身**的范围定义留出区，否则阈值落在块外
        m_tr, axis, thr = spatial_holdout_mask(xyz_c, xyz_c, holdout_frac, holdout_m)
        if m_tr.any():
            eval_xyz[train_domain] = (xyz_c[m_tr], y_c[m_tr])
    else:
        # 阈值来自最密 main 域 → 各域留出的是**同一片区域**，跨域可比
        m_tr, axis, thr = spatial_holdout_mask(ref_xyz, xyz_c, holdout_frac, holdout_m)
        for n in mains:
            xyz, y = domains[n]
            m, _a, _t = spatial_holdout_mask(ref_xyz, xyz, holdout_frac, holdout_m)
            if not m.any():
                continue
            eval_xyz[n] = (xyz[m], y[m])

    train_mask = ~m_tr
    x_tr = xyz_c[train_mask]
    y_tr = y_c[train_mask]

    eval_sets: Dict[str, Tuple[torch.Tensor, np.ndarray]] = {
        n: (to_torch(np.ascontiguousarray(v[0].T[None, ...], dtype=np.float32), dev), v[1])
        for n, v in eval_xyz.items()
    }

    # ---- 留出区诊断：难度混淆量的直接测量 ----
    diag = holdout_diag(eval_xyz[train_domain][0], x_tr) if train_domain in eval_xyz else {}

    # ---- 局部性归一化量（本轮"相对局部性"课题的核心自变量）----
    # 两个候选控制变量，必须靠**两者分歧**的配置来区分：
    #   k_over_n        = k / N              （邻域占点云的比例，图规模）
    #   r_k_over_extent = r_k / L            （邻域物理半径占场景尺度的比例，几何）
    # 在**固定形状**的场景上二者退化相关（面采样下 r_k/L ∝ (k/N)^(1/2)），
    # 所以只有"稠密但很小"这种 k/N 大而 r_k/L 小的配置才能判别它们。
    bb = x_tr.max(axis=0) - x_tr.min(axis=0)
    extent_bbox = float(np.linalg.norm(bb))                       # 包围盒对角线
    extent_diam = float(2.0 * np.max(np.linalg.norm(x_tr - x_tr.mean(axis=0), axis=1)))
    n_train = int(x_tr.shape[0])
    k_over_n = float(k) / max(n_train, 1)

    # ---- 邻域半径记录（仅可比性提示，非正确性要求）----
    # ⚠ Tier-2 的 DGCNNSeg **没有半径门控**（Tier-1 代理才有），所以这里不是
    # "合规断言"，只是记录邻域物理尺度，便于与代理域的可比范围对照。
    # 判据用 p99 而非 max：max 会被少量孤立/边界点主导（实测 rho4/k=12 的 max
    # 达 2.26 m，而中位数只有 0.6 m 量级），拿 max 下断言会误报。
    rho_tr = D.local_density(x_tr)
    kk = int(min(k, max(x_tr.shape[0] - 1, 1)))
    _i, dk = D.knn_query(x_tr, x_tr, kk + 1)
    r_k_arr = dk[:, -1]
    r_k_med = float(np.median(r_k_arr))
    r_k_p99 = float(np.percentile(r_k_arr, 99))
    r_k_max = float(np.max(r_k_arr))
    radius_ok = bool(r_k_p99 <= RADIUS_NONBINDING_M)
    r_k_over_extent = r_k_med / max(extent_bbox, 1e-9)

    X = to_torch(np.ascontiguousarray(x_tr.T[None, ...], dtype=np.float32), dev)
    Y = to_torch(np.ascontiguousarray(y_tr, dtype=np.int64), dev)

    model = DGCNNSeg(int(DT.NUM_CLASSES), int(k)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)

    hist: List[dict] = []
    t0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)                                   # (1, C, Nt)
        loss = F.cross_entropy(logits[0].t(), Y)
        loss.backward()
        opt.step()
        sched.step()

        if ep % max(1, verbose_every) == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc_dom: Dict[str, float] = {}
                for n, (xe, ye) in eval_sets.items():
                    lg = model(xe)
                    # 走 .tolist() 而不是 .numpy()：后者依赖 numpy↔torch 互操作层，
                    # 在互操作损坏的环境里会直接崩（见 to_torch 的说明）。
                    pr = np.asarray(lg.argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
                    mi, ac = miou_np(pr, ye.astype(np.int64), int(DT.NUM_CLASSES))
                    acc_dom[n] = mi
            rec = {"epoch": ep, "train_loss": float(loss.item()),
                   "train_miou": acc_dom.get(train_domain, float("nan"))}
            rec.update({f"mIoU@{n}": v for n, v in acc_dom.items()})
            hist.append(rec)
            log(f"    ep {ep:>4d}  loss={rec['train_loss']:.4f}  "
                + "  ".join(f"{n}={v:.4f}" for n, v in acc_dom.items()))

    dur = time.perf_counter() - t0
    return {
        "tag": tag,
        "train_domain": train_domain,
        "k": int(k),
        "alpha": spec.get("alpha"),
        "k0": spec.get("k0"),
        "rho_ref": float(rho_ref),
        "crop_n": int(crop_n),
        "cropped": cropped,
        "stretch_s": stretch_eff,
        "epochs": int(epochs),
        "holdout_frac": float(holdout_frac),
        "holdout_m": holdout_m,
        "holdout_axis": int(axis) if axis is not None else None,
        "holdout_thr": thr,
        "holdout_to_train_med_m": diag.get("holdout_to_train_med_m"),
        "holdout_to_train_p90_m": diag.get("holdout_to_train_p90_m"),
        "n_eval_points": {n: int(v[1].shape[0]) for n, v in eval_sets.items()},
        "rho_domain_med": rho_domain,
        "rho_train_med": float(np.median(rho_tr)),
        "extent_bbox_diag_m": extent_bbox,
        "extent_diam_m": extent_diam,
        "k_over_n": k_over_n,
        "r_k_over_extent": r_k_over_extent,
        "n_train_points": n_train,
        "radius_nonbinding_ok": radius_ok,
        "r_k_med_m": r_k_med,
        "r_k_p99_m": r_k_p99,
        "r_k_max_m": r_k_max,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_params_theory_independent_of_k": True,
        "device": str(dev),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu"),
        "seed": seed, "lr": LR, "lr_step": LR_STEP, "lr_gamma": LR_GAMMA,
        "feat_dim": FEAT_DIM, "layers": NUM_LAYERS, "dropout": DROPOUT,
        "duration_sec": float(dur),
        "history": hist,
    }


# ---------------------------------------------------------------------------
# 汇总分析
# ---------------------------------------------------------------------------
def analyze(result_dir: str, log=print) -> None:
    files = []
    if os.path.isdir(result_dir):
        files = [os.path.join(result_dir, f) for f in sorted(os.listdir(result_dir))
                 if f.endswith(".json")]
    if not files:
        log(f"没有找到结果文件：{result_dir}")
        return
    runs = []
    for p in files:
        with open(p, "r", encoding="utf-8") as fh:
            runs.append(json.load(fh))

    all_runs = runs
    # 固定 k（未裁剪）的 run 才进 k 交互表；裁剪对照与 alpha 策略另有专表
    runs = [r for r in all_runs
            if r.get("alpha") is None and not r.get("cropped", False)]

    log("=" * 72)
    log("Tier-2 汇总：density x k 交互效应")
    log("=" * 72)
    log(f"  读入 {len(all_runs)} 个 run（其中固定 k 未裁剪 {len(runs)} 个）")

    # 邻域尺度检查（覆盖全部 run）。Tier-2 无门控，此项仅为可比性提示。
    bad = [r for r in all_runs if not r.get("radius_nonbinding_ok", False)]
    if bad:
        log(f"  ⚠ 有 {len(bad)} 个 run 的**邻域物理尺度偏大**（p99 r_k > "
            f"{RADIUS_NONBINDING_M}m）：")
        for r in bad:
            p99 = r.get("r_k_p99_m")
            p99s = f"{p99:.3f}m" if isinstance(p99, (int, float)) else "n/a(旧记录)"
            log(f"     {r['tag']}: k={r['k']} r_k_p99={p99s} "
                f"r_k_max={r['r_k_max_m']:.3f}m")
        log("     Tier-2 本身无半径门控，故这不是错误；但该配置的邻域已接近"
            "全局聚合，解读时应单独标注。")
    else:
        log(f"  ✅ 全部 run 的邻域物理尺度在可比范围内（p99 r_k <= "
            f"{RADIUS_NONBINDING_M}m）")

    def final_rec(r, key_mode):
        h = r.get("history") or []
        if not h:
            return None
        if key_mode == "last":
            return h[-1]
        # 按训练域留出区的 mIoU 选最优 epoch
        key = f"mIoU@{r['train_domain']}"
        return max(h, key=lambda d: d.get(key, -1.0))

    def val_of(r, mode):
        rec = final_rec(r, mode)
        if rec is None:
            return None
        v = rec.get(f"mIoU@{r['train_domain']}", float("nan"))
        return None if v is None or not np.isfinite(v) else float(v)

    for mode in ("last", "best"):
        log("")
        log(f"  ── 取点方式：{'末尾 epoch' if mode == 'last' else '训练域留出区 mIoU 最优 epoch'} ──")
        # 多 seed 时按 (域, k) 求均值与标准差
        acc: Dict[Tuple[str, int], List[float]] = {}
        for r in runs:
            v = val_of(r, mode)
            if v is None:
                continue
            acc.setdefault((r["train_domain"], r["k"]), []).append(v)
        vals: Dict[Tuple[str, int], float] = {kk: float(np.mean(vv))
                                             for kk, vv in acc.items()}
        stds: Dict[Tuple[str, int], float] = {
            kk: (float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0)
            for kk, vv in acc.items()}
        n_seed = max((len(vv) for vv in acc.values()), default=1)
        if n_seed > 1:
            log(f"    （每个 (域,k) 汇总 {n_seed} 个 seed，取均值；括号内为 seed 标准差）")

        doms = sorted({d for d, _k in vals})
        ks = sorted({k for _d, k in vals})
        log(f"    {'域':<8s} " + "".join(f"{'k='+str(k):>14s}" for k in ks)
            + f"{'I=Δ(k_max-k_min)':>18s}")
        deltas: Dict[str, float] = {}
        for d in doms:
            cells = []
            for k in ks:
                m = vals.get((d, k))
                if m is None:
                    cells.append(f"{'n/a':>14s}")
                elif n_seed > 1:
                    cells.append(f"{m:>9.4f}(±{stds.get((d, k), 0.0):.3f})")
                else:
                    cells.append(f"{m:>14.4f}")
            line = f"    {d:<8s} " + "".join(cells)
            if len(ks) >= 2 and (d, ks[-1]) in vals and (d, ks[0]) in vals:
                deltas[d] = float(vals[(d, ks[-1])] - vals[(d, ks[0])])
                line += f"{deltas[d]:>+18.4f}"
            log(line)

        if len(deltas) >= 2:
            ds = sorted(deltas.items(), key=lambda t: t[0])
            log("")
            log("    I(rho) = mIoU(k=%d) - mIoU(k=%d)：" % (ks[-1], ks[0]))
            for d, v in ds:
                log(f"      {d:<8s} {v:+.4f}")
            spread = max(deltas.values()) - min(deltas.values())
            log(f"      I 的跨密度极差 = {spread:+.4f}")
            # 判方向：I 是随"密度增大"上升还是下降。deltas 的 key 已按域名字典序，
            # 而 rho0/rho2/rho4 的域名字典序恰好与密度降序一致，所以这里显式用
            # 训练域的 ρ 排序，避免靠命名巧合。
            rho_of = {r["train_domain"]: r.get("rho_train_med", float("nan"))
                      for r in runs}
            ordered = sorted(deltas.items(), key=lambda t: rho_of.get(t[0], 0.0))
            if len(ordered) >= 2:
                i_low = ordered[0][1]      # 最稀疏
                i_high = ordered[-1][1]    # 最密
                trend = i_low - i_high
                log(f"      最稀疏域 I = {i_low:+.4f}，最密域 I = {i_high:+.4f}，"
                    f"差值 = {trend:+.4f}")
            else:
                trend = 0.0
            log("")
            if spread <= 0.02:
                log("      → 形态 A：I ≈ const（k 有影响，但密度不改变 k 的收益）")
                log("        支持 **等度数律 / density-invariant geometry**，")
                log("        而非 density-conditioned architecture → 不要强行做 NAS。")
            elif trend > 0.02:
                log("      → 形态 B：I 随密度**降低而增大**（稀疏域从大 k 获益更多）。")
                log("        方向与原假设一致：稀疏端需要更大的邻域。")
                log("        → 存在 density x geometry 交互，NAS 前提成立。")
            elif trend < -0.02:
                log("      → 形态 C：I 随密度**降低而减小**（稀疏域反而被大 k 伤害）。")
                log("        方向与原假设**相反**：稀疏端应**收缩** k，而不是扩大。")
                log("        物理解释：固定 k 时物理半径 r_k ∝ ρ^(-0.3)，稀疏域里 k 稍大")
                log("        就会逼近全局聚合 → 过平滑。所以真正该守恒的是**物理邻域上界**，")
                log("        而不是邻居数量。")
                log("        → 交互同样存在、同样值得做 NAS，但必须重新表述假设方向。")
            else:
                log("      → 形态介于 A 与其他之间，方向不明确，需要更多域/种子才能定论。")
            log("      （0.02 这个阈值取自代理噪声 sigma≈0.0068 的 ~3 倍；")
            log("        真实训练的噪声水平更小，故该阈值偏保守）")

    # ---- 线性交互模型（用 seed 平均后的点，避免多 seed 重复计数）----
    log("")
    log("=" * 72)
    log("交互模型  mIoU = b0 + b1*log(rho) + b2*k + b3*(log(rho) * k)")
    log("=" * 72)
    log("  核心看 b3：密度是否改变 k 的收益（b3≈0 → 无交互）")
    agg: Dict[Tuple[str, int], List[float]] = {}
    meta_of: Dict[Tuple[str, int], dict] = {}
    for r in runs:
        v = val_of(r, "best")
        if v is None:
            continue
        key = (r["train_domain"], r["k"])
        agg.setdefault(key, []).append(v)
        meta_of[key] = r
    y, X_ = [], []
    k_lo, k_hi = None, None
    for key, vv in sorted(agg.items()):
        r = meta_of[key]
        lg = float(np.log(max(r["rho_train_med"], 1e-9)))
        kk = float(r["k"])
        y.append(float(np.mean(vv)))
        X_.append([1.0, lg, kk, lg * kk])
        k_lo = kk if k_lo is None else min(k_lo, kk)
        k_hi = kk if k_hi is None else max(k_hi, kk)
    if len(y) >= 4:
        A = np.asarray(X_, dtype=np.float64)
        b = np.asarray(y, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coef
        ss_res = float(((b - pred) ** 2).sum())
        ss_tot = float(((b - b.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        names = ["b0", "b1(log rho)", "b2(k)", "b3(log rho x k)"]
        for nm, c in zip(names, coef):
            log(f"    {nm:<16s} = {c:+.6f}")
        log(f"    R^2 = {r2:.4f}   样本数 = {len(y)}（seed 平均后的 (域,k) 点）")
        b3 = float(coef[3])
        # b3 的量纲：log(rho) 跨 ln(60)≈4.1，k 跨 (k_hi-k_lo) → 交互项量级 ≈ 4.1*Δk*b3
        dk = (k_hi - k_lo) if (k_lo is not None and k_hi is not None) else 1.0
        scale = abs(np.log(60.0) * dk)
        log(f"    交互项的典型量级 ≈ {abs(b3) * scale:.4f} mIoU"
            f"（= |ln(60) x Δk({dk:g})| x |b3|）")
        if abs(b3) * scale < 0.02:
            log("    → 交互项可忽略：**principal 效应是 k，不是 density x k**")
        else:
            log("    → 交互项不可忽略：存在 density x geometry 交互")
    else:
        log("    样本不足（需要至少 4 个 (域,k) 点），跳过拟合。")
    log("")
    log("  注：本模型用训练域留出区的 mIoU（train 与 eval 空间不重叠）。")

    # ---- 补充实验专表（固定 N 对照 / alpha 策略 / 相对局部性）----
    report_crop_and_alpha(all_runs, log=log)
    report_locality(all_runs, log=log)


# ---------------------------------------------------------------------------
def report_crop_and_alpha(all_runs: List[dict], log=print) -> None:
    """两项补充实验的专表：固定 N 对照 + 密度条件化 alpha 策略。"""

    def best_val(r):
        h = r.get("history") or []
        if not h:
            return None
        rec = max(h, key=lambda d: d.get(f"mIoU@{r['train_domain']}", -1.0))
        v = rec.get(f"mIoU@{r['train_domain']}")
        return None if v is None else float(v)

    crop = [r for r in all_runs if r.get("cropped", False)]
    base = [r for r in all_runs if r.get("alpha") is None and not r.get("cropped", False)]

    # ---- 1) 固定 N 对照 ----
    if crop or base:
        log("")
        log("=" * 72)
        log("固定 N 对照：rho0 空间裁剪（局部密度不变，只缩小空间范围）")
        log("=" * 72)
        log("  问题：rho4 上 k=64 崩溃，是**密度**造成的，还是 **k/N 过大导致图接近全连接**？")
        log("  对照：把 rho0（稠密）裁到与 rho4 相近的点数 → 对齐 N、保留密度。")
        log("")

        agg: Dict[Tuple[str, int, int], List[float]] = {}
        info: Dict[Tuple[str, int, int], dict] = {}
        for r in crop + base:
            v = best_val(r)
            if v is None:
                continue
            key = (r["train_domain"], int(r.get("crop_n", 0)), r["k"])
            agg.setdefault(key, []).append(v)
            info[key] = r

        kset = sorted({k for (_d, _c, k) in agg})
        log(f"    {'配置':<26s} {'N训练':>7s} {'rho':>11s} "
            + "".join(f"{'k=' + str(k):>10s}" for k in kset))
        for key in sorted(set((d, cn) for (d, cn, _k) in agg)):
            d, cn = key
            r0 = info.get((d, cn, kset[0]))
            if r0 is None:
                continue
            label = f"{d} (未裁剪)" if cn == 0 else f"{d} 裁剪到 {cn}"
            cells = []
            for k in kset:
                vv = agg.get((d, cn, k))
                cells.append(f"{np.mean(vv):>10.4f}" if vv else f"{'n/a':>10s}")
            log(f"    {label:<26s} {r0['n_train_points']:>7d} "
                f"{r0['rho_train_med']:>11.3e} " + "".join(cells))

        # 关键对比：稠密小块 vs 稀疏全域，在同一 k 上
        if kset:
            k_big = kset[-1]
            dense_crop = None
            for (d, cn) in set((d, cn) for (d, cn, _k) in agg):
                if cn > 0 and d == "rho0":
                    dense_crop = (d, cn)
            sparse_full = ("rho4", 0)
            a = agg.get((dense_crop[0], dense_crop[1], k_big)) if dense_crop else None
            b = agg.get((sparse_full[0], 0, k_big))
            if a and b and dense_crop:
                ma, mb = float(np.mean(a)), float(np.mean(b))
                log("")
                log(f"    关键对比（k={k_big}）：")
                log(f"      稠密小块（{dense_crop[0]} 裁剪到 {dense_crop[1]}，"
                    f"ρ={info.get((dense_crop[0], dense_crop[1], k_big), {}).get('rho_train_med', float('nan')):.2e}）"
                    f"  mIoU = {ma:.4f}")
                log(f"      稀疏全域（rho4，ρ=1.2e1）"
                    f"                    mIoU = {mb:.4f}")
                gap = ma - mb
                log(f"      差值 = {gap:+.4f}")
                if gap > 0.05:
                    log("      → 稠密小块上 k 大**不崩** → 驱动因素是**密度**，")
                    log("        不是 k/N 图规模。『物理邻域上界』假说成立。")
                elif gap < -0.05:
                    log("      → 稠密小块上 k 大**也崩** → 驱动因素是**图规模（k/N）**，")
                    log("        与密度无关。必须重新解释 rho4 的崩溃（ρ 不是原因）。")
                else:
                    log("      → 两者接近，本对照无法区分，需要更多点数档位。")

    # ---- 2) 密度条件化 alpha 策略 ----
    alpha_runs = [r for r in all_runs if r.get("alpha") is not None]
    if alpha_runs:
        log("")
        log("=" * 72)
        log("人工 density-aware 基线：k(ρ) = clip(round(k0·(ρ/ρ_ref)^α))")
        log("=" * 72)
        log("  ⚠ 方向约定（易错，写清楚）：")
        log("     k ∝ ρ^α  ⇒  α<0 = **稀疏域扩大 k**（原假设方向）")
        log("                 α>0 = **稀疏域收缩 k**（反假设方向）")
        log("                 α=0 = 固定 k")
        log("")
        agg: Dict[Tuple[float, str], List[float]] = {}
        kused: Dict[Tuple[float, str], int] = {}
        for r in alpha_runs:
            v = best_val(r)
            if v is None:
                continue
            key = (float(r["alpha"]), r["train_domain"])
            agg.setdefault(key, []).append(v)
            kused[key] = int(r["k"])
        doms = sorted({d for (_a, d) in agg})
        alphas = sorted({a for (a, _d) in agg})
        k0 = next((r.get("k0") for r in alpha_runs if r.get("k0")), None)
        log(f"  k0 = {k0:g}，ρ_ref = {next((r.get('rho_ref') for r in alpha_runs), float('nan')):.4g}")
        log("")
        log(f"    {'α':>6s} " + "".join(f"{d:>16s}" for d in doms)
            + f"{'均值':>10s}{'最差':>10s}")
        scores: Dict[float, Tuple[float, float]] = {}
        for a in alphas:
            cells, vals_ = [], []
            for d in doms:
                vv = agg.get((a, d))
                if vv:
                    m = float(np.mean(vv))
                    vals_.append(m)
                    cells.append(f"{m:>10.4f}(k{kused[(a, d)]:>3d})")
                else:
                    cells.append(f"{'n/a':>16s}")
            if vals_:
                mean_, min_ = float(np.mean(vals_)), float(np.min(vals_))
                scores[a] = (mean_, min_)
                log(f"    {a:>+6.2f} " + "".join(cells)
                    + f"{mean_:>10.4f}{min_:>10.4f}")
        if scores:
            best_mean = max(scores.items(), key=lambda t: t[1][0])
            best_min = max(scores.items(), key=lambda t: t[1][1])
            log("")
            log(f"    → 均值最优 α = {best_mean[0]:+.2f}（mean={best_mean[1][0]:.4f}）")
            log(f"    → 最差域最优 α = {best_min[0]:+.2f}（min={best_min[1][1]:.4f}）")
            if best_mean[0] > 0.05:
                log("    → **α>0 胜出**：稀疏域应**收缩** k。方向与原假设**相反**，")
                log("      与 grid 实验的『形态 C』结论一致。C3 该报告的规律是 α_k > 0")
                log("      （即 k 随密度**增大**），这同时就是 NAS 必须打败的基线。")
            elif abs(best_mean[0]) <= 0.05:
                log("    → **α≈0 胜出**：固定 k 已是最优族，密度条件化 k 无收益。")
            else:
                log("    → **α<0 胜出**：稀疏域应**扩大** k，支持原假设方向。")


def report_locality(all_runs: List[dict], log=print) -> None:
    """判别哪个归一化量才是真正的控制变量：k/N（图规模）还是 r_k/L（几何局部性）。

    为什么必须判别
    ==============
    在**固定形状**的场景上两者退化相关：面采样下 k ≈ σπr²，故
        r_k/L ∝ (k/N)^(1/2)
    也就是说"k/N 大"与"r_k/L 大"在同一个房间里总是同步变化，无法区分。
    只有**两者分歧**的配置才有判别力：

        稠密但很小的一块（rho0 裁到 1024 点）：
            k/N 大（≈6%），但 r_k/L 小（邻域物理半径只占小块的一小部分）
        若它崩溃 → 支持 k/N 规律，否定 r_k/L 规律。
        若它不崩 → 支持 r_k/L 规律。

    做法：把同一"配置组"内的 mIoU 按最优 k 归一化成"退化量"
        degrad = mIoU(k) − max_k mIoU
    然后看 degrad 与 k/N、r_k/L 哪个相关性更强。
    """
    pts = [r for r in all_runs
           if r.get("k_over_n") is not None and r.get("r_k_over_extent") is not None]
    if len(pts) < 6:
        log("")
        log("（相对局部性判别：带 k_over_n / r_k_over_extent 的 run 不足 6 个，跳过。")
        log("  这类字段只在新跑的 run 里有；旧 run 无此字段。）")
        return

    def best_val(r):
        h = r.get("history") or []
        if not h:
            return None
        rec = max(h, key=lambda d: d.get(f"mIoU@{r['train_domain']}", -1.0))
        v = rec.get(f"mIoU@{r['train_domain']}")
        return None if v is None else float(v)

    # 分组键：域 + 裁剪点数 + 拉伸因子（同一组内比较不同 k 的退化）
    # ⚠ stretch_s 用 float（None 归为 1.0 表示各向同性），避免 None 与 float 混合
    # 导致 sorted() 在比较 tuple 第三元时抛 TypeError。
    groups: Dict[Tuple[str, int, float], List[dict]] = {}
    for r in pts:
        v = best_val(r)
        if v is None:
            continue
        r = dict(r)
        r["_miou"] = v
        st = float(r.get("stretch_s") or 1.0)
        groups.setdefault((r["train_domain"], int(r.get("crop_n", 0)), st), []).append(r)

    log("")
    log("=" * 72)
    log("相对局部性判别：k/N（图规模） vs r_k/L（几何局部性）")
    log("=" * 72)
    log(f"  {'配置组':<26s} {'N':>7s} {'L(m)':>7s} {'k':>4s} "
        f"{'k/N':>7s} {'r_k/L':>7s} {'mIoU':>8s} {'退化':>8s}")

    samples: List[dict] = []
    for key in sorted(groups):
        d, cn, st = key
        rs = groups[key]
        if len(rs) < 2:
            continue
        best = max(r["_miou"] for r in rs)
        label = f"{d}" + (f" crop{cn}" if cn else " full")
        if st != 1.0:
            label += f" s={st:.2f}"
        for r in sorted(rs, key=lambda x: x["k"]):
            deg = r["_miou"] - best
            log(f"  {label:<26s} {r['n_train_points']:>7d} "
                f"{r['extent_bbox_diag_m']:>7.2f} {r['k']:>4d} "
                f"{r['k_over_n']*100:>6.2f}% {r['r_k_over_extent']*100:>6.2f}% "
                f"{r['_miou']:>8.4f} {deg:>+8.4f}")
            samples.append({"deg": deg, "kn": r["k_over_n"],
                            "rL": r["r_k_over_extent"], "miou": r["_miou"],
                            "n": r["n_train_points"], "stretch": st,
                            "crop": cn, "dom": d})

    if len(samples) >= 6:
        deg = np.asarray([s["deg"] for s in samples])
        kn = np.asarray([s["kn"] for s in samples])
        rl = np.asarray([s["rL"] for s in samples])

        def corr2(a):
            if np.std(a) < 1e-12 or np.std(deg) < 1e-12:
                return float("nan")
            c = float(np.corrcoef(a, deg)[0, 1])
            return c * c

        r2_kn, r2_rl = corr2(kn), corr2(rl)
        log("")
        log(f"  退化量与 k/N 的相关 R^2 = {r2_kn:.4f}")
        log(f"  退化量与 r_k/L 的相关 R^2 = {r2_rl:.4f}")
        if np.isfinite(r2_kn) and np.isfinite(r2_rl):
            if r2_kn > r2_rl * 1.3:
                log("  → **k/N（图规模）解释力明显更强**。")
                log("    结论：约束是「邻域占点云的比例」，不是「邻域半径占场景比例」。")
            elif r2_rl > r2_kn * 1.3:
                log("  → **r_k/L（几何局部性）解释力明显更强**。")
                log("    结论：约束是「邻域物理半径相对场景尺度」，与点数无关。")
            else:
                log("  → 两者解释力接近（固定形状下二者退化相关，这是预期）。")
                log("    需要「两者分歧」的配置才能判别，见下面的配对判别。")

        # ---- 配对判别：固定 (crop, k)，对比 stretch=1.0 vs stretch<1.0 ----
        # 此时 N 与 k 都不变 → k/N 相同，唯一变的是形状（L 变大 → r_k/L 变小）。
        # 若 mIoU 显著变化 → r_k/L 是驱动；若几乎不变 → k/N 才是驱动。
        log("")
        log("  ── 配对判别（固定 crop/k，只改形状 stretch → k/N 不变、r_k/L 变）──")
        # 先按 (crop, k) 分组，再在组内找 stretch=1.0 基准与各 stretch<1.0 点
        # samples 里每个元素对应一个 (crop, k, stretch)（多 seed 已在上层平均，这里
        # 实际是单 seed 的行；直接用 mIoU 而非退化量，因为配对的是"同一 k 不同形状"）
        by_ck: Dict[Tuple[int, int], Dict[float, List[dict]]] = {}
        for s in samples:
            by_ck.setdefault((s["crop"], int(round(s["kn"] * s["n"]))), {}).setdefault(
                s["stretch"], []).append(s)
        found = False
        for (cn, _k), stmap in sorted(by_ck.items()):
            base = stmap.get(1.0)
            if not base:
                continue
            base_m = float(np.mean([b["miou"] for b in base]))
            for st in sorted(k for k in stmap if k != 1.0):
                others = stmap[st]
                o_m = float(np.mean([o["miou"] for o in others]))
                d_miou = o_m - base_m
                # 取各自 r_k/L 的中位
                b_rl = float(np.mean([b["rL"] for b in base]))
                o_rl = float(np.mean([o["rL"] for o in others]))
                if abs(o_rl - b_rl) < 1e-6:
                    continue
                found = True
                log(f"    crop{cn} k={_k}: stretch {st:.2f} 使 r_k/L "
                    f"{b_rl*100:.2f}%→{o_rl*100:.2f}%，mIoU {base_m:.4f}→{o_m:.4f} "
                    f"（Δ {d_miou:+.4f}）")
                if abs(d_miou) > 0.05:
                    log("      → 形状变化伴随显著 mIoU 变化 → **r_k/L 是驱动**。")
                else:
                    log("      → 形状变化而 mIoU 几乎不变 → **k/N 才是驱动**。")
        if not found:
            log("    （无 stretch 组，或 stretch 组内 r_k/L 无变化，跳过配对判别。）")

        # 逐点看：找出分歧最大的配置，单独报告
        log("")
        log("  分歧最大的配置（k/N 高但 r_k/L 低 → 最能判别）：")
        order = np.argsort(-(kn / np.maximum(rl, 1e-9)))
        for i in order[:3]:
            s = samples[int(i)]
            log(f"    k/N={s['kn']*100:.2f}%  r_k/L={s['rL']*100:.2f}%  "
                f"mIoU={s['miou']:.4f}  退化={s['deg']:+.4f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-2 真实 DGCNN 短训")
    ap.add_argument("--train-domain", default="rho0", help="训练所用密度域")
    ap.add_argument("--k", type=int, default=16, help="固定的邻居数")
    ap.add_argument("--epochs", type=int, default=200,
                    help="训练 epoch 数（1 epoch = 1 次全点前向）")
    ap.add_argument("--holdout-frac", type=float, default=0.3,
                    help="空间块留出比例（沿最长轴的上侧一段）")
    ap.add_argument("--holdout-m", type=float, default=0.75,
                    help="固定物理留出厚度（米）。>0 时忽略 --holdout-frac，"
                         "按 max-T 取留出区；默认 0.75 由 _probe_holdout.py 实测选定，"
                         "使五档裁剪的留出比例都落在健壮区间、任务难度可比。"
                         "传 0 或负值回退到比例模式。")
    ap.add_argument("--tag", default="", help="结果文件名（默认自动生成）")
    ap.add_argument("--out-dir", default=RESULT_DIR, help="结果目录")
    ap.add_argument("--verbose-every", type=int, default=10,
                    help="每多少 epoch 评估并打印一次")
    ap.add_argument("--window-probe", action="store_true",
                    help="对 --train-domain 跑 MATRIX_KS 全部 k，用于确定短训窗口")
    ap.add_argument("--matrix", action="store_true",
                    help="跑完整 3 域 x 3 k 矩阵")
    ap.add_argument("--grid", action="store_true",
                    help="跑 domains x --k-list x --seeds 的完整网格（k 网格加密 + 多 seed）")
    ap.add_argument("--crop-control", action="store_true",
                    help="固定 N 对照：把稠密域空间裁剪到 --crop-n 点后扫 k")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="人工 density-aware 基线：扫 k(ρ)=k0·(ρ/ρ_ref)^α")
    ap.add_argument("--k-list", type=int, nargs="+", default=None,
                    help="覆盖 k 网格（默认用 MATRIX_KS）")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="随机种子列表（>1 时结果按 seed 汇总，给出误差棒）")
    ap.add_argument("--crop-n", type=int, default=CROP_N_DEFAULT,
                    help="--crop-control 的裁剪目标点数（默认 1024，对齐 rho4）")
    ap.add_argument("--crop-ladder", type=int, nargs="+", default=None,
                    help="裁剪阶梯：对同一域按多个点数裁剪，用于分离 k/N 与 r_k/L")
    ap.add_argument("--stretch-s", type=float, nargs="+", default=None,
                    help="各向异性拉伸因子（0<s<1），打破 k/N 与 r_k/L 的退化相关。"
                         "配合 --crop-ladder/--crop-control 使用：把补丁沿最长轴"
                         "各向异性伸长，固定 N 与 ρ 时 r_k/L 变小、k/N 不变。"
                         "多个值则每个都各跑一组。")
    ap.add_argument("--crop-domains", type=str, nargs="+", default=None,
                    help="--crop-control 要裁剪的域（默认 --train-domain）")
    ap.add_argument("--k0", type=float, default=32.0,
                    help="--alpha-sweep 的基准 k")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="--alpha-sweep 的 α 网格")
    ap.add_argument("--analyze", action="store_true", help="只汇总已有结果")
    ap.add_argument("--log-file", default="", help="把日志同时写入该 UTF-8 文件")
    args = ap.parse_args()

    log = _Log(args.log_file or None)

    if args.analyze:
        analyze(args.out_dir, log=log)
        log.close()
        return

    os.makedirs(args.out_dir, exist_ok=True)

    k_list = [int(v) for v in (args.k_list or MATRIX_KS)]
    seeds = [int(v) for v in (args.seeds or [SEED])]
    holdout_m = float(args.holdout_m) if (args.holdout_m is not None
                                          and args.holdout_m > 0) else None
    stretches = [float(s) for s in (args.stretch_s or []) if 0.0 < s < 1.0]

    domains, meta = load_domains()
    mains = main_domain_names(meta)
    # ρ_ref：优先取标定期写入缓存的值（仅 main 域几何均值）
    rho_ref = float(meta.get("rho_ref") or 0.0)
    if not rho_ref:
        rho_ref = float(np.exp(np.mean([
            np.log(max(float(np.median(D.local_density(domains[n][0]))), 1e-9))
            for n in mains])))

    # ---- 构造任务规格 ----
    specs: List[dict] = []
    if args.grid:
        for d in MATRIX_DOMAINS:
            for k in k_list:
                for s in seeds:
                    tag = f"{d}_k{k}" + (f"_s{s}" if len(seeds) > 1 else "")
                    specs.append({"tag": tag, "train_domain": d, "k": int(k), "seed": s})
    elif args.crop_control or args.crop_ladder:
        cdoms = args.crop_domains or [args.train_domain]
        ladder = [int(v) for v in (args.crop_ladder or [args.crop_n])]
        ks_here = [int(v) for v in (args.k_list or CROP_K_LIST)]
        for d in cdoms:
            for cn in ladder:
                for k in ks_here:
                    for s in seeds:
                        tag = (f"crop{cn}_{d}_k{k}"
                               + (f"_s{s}" if len(seeds) > 1 else ""))
                        specs.append({"tag": tag, "train_domain": d, "k": k,
                                      "seed": s, "crop_n": cn})
            # 各向异性拉伸组（固定 N，改形状 → 打破 k/N 与 r_k/L 的退化）
            if stretches:
                for cn in ladder:
                    for st in stretches:
                        for k in ks_here:
                            for s in seeds:
                                tag = (f"stretch{st:.2f}_crop{cn}_{d}_k{k}"
                                       .replace(".", "p")
                                       + (f"_s{s}" if len(seeds) > 1 else ""))
                                specs.append({"tag": tag, "train_domain": d, "k": k,
                                              "seed": s, "crop_n": cn,
                                              "stretch_s": st})
            # 同域未裁剪参照组（同 k 网格，便于直接对照）
            for k in ks_here:
                for s in seeds:
                    tag = (f"full_{d}_k{k}"
                           + (f"_s{s}" if len(seeds) > 1 else ""))
                    specs.append({"tag": tag, "train_domain": d, "k": k, "seed": s})
    elif args.alpha_sweep:
        for a in (args.alphas or ALPHAS_DEFAULT):
            for d in MATRIX_DOMAINS:
                for s in seeds:
                    tag = (f"a{a:+.2f}_{d}".replace("+", "p").replace("-", "m")
                           + (f"_s{s}" if len(seeds) > 1 else ""))
                    specs.append({"tag": tag, "train_domain": d, "alpha": float(a),
                                  "k0": float(args.k0), "rho_ref": rho_ref, "seed": s})
    elif args.matrix:
        for d in MATRIX_DOMAINS:
            for k in k_list:
                for s in seeds:
                    tag = f"{d}_k{k}" + (f"_s{s}" if len(seeds) > 1 else "")
                    specs.append({"tag": tag, "train_domain": d, "k": int(k), "seed": s})
    elif args.window_probe:
        for k in k_list:
            specs.append({"tag": f"{args.train_domain}_k{k}",
                          "train_domain": args.train_domain, "k": int(k)})
    else:
        specs.append({"tag": args.tag or f"{args.train_domain}_k{int(args.k)}",
                      "train_domain": args.train_domain, "k": int(args.k)})

    log(f"[tier2] domains = {list(domains)}")
    log(f"[tier2] cuda = {torch.cuda.is_available()}  device = "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"[tier2] epochs = {args.epochs}  seeds = {seeds}  rho_ref = {rho_ref:.4g}")
    log(f"[tier2] {len(specs)} 个 run：{specs[0]['tag']} ... {specs[-1]['tag']}")

    for spec in specs:
        tag = spec["tag"]
        path = os.path.join(args.out_dir, f"{tag}.json")
        if os.path.exists(path):
            log(f"[tier2] skip (exists): {tag}")
            continue
        log("")
        log("=" * 72)
        log(f"[tier2] RUN {tag}  (spec={ {k: v for k, v in spec.items() if k != 'tag'} }, "
            f"epochs={args.epochs})")
        log("=" * 72)
        res = run_once(domains, meta, spec, args.epochs, args.holdout_frac,
                       holdout_m=holdout_m, verbose_every=args.verbose_every, log=log)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        log(f"[tier2] k={res['k']} 门控合规={res['radius_nonbinding_ok']} "
            f"(r_k_max={res['r_k_max_m']:.3f}m) 训练点数={res['n_train_points']} "
            f"耗时={res['duration_sec']:.1f}s")
        log(f"[tier2] 已写入 {path}")

    log("")
    log("[tier2] 全部 run 完成。汇总请加 --analyze")
    log.close()


if __name__ == "__main__":
    main()
