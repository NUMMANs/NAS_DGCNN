"""邻域几何的**实测**统计（不依赖任何"体填充"假设）。

为什么必须单独有这个模块
========================

`density.local_density` 用

    rho = k / ((4/3) * pi * r_k^3)

把 kNN 距离**反解**成一个"等效体密度"。这在 S3DIS 这种**表面采样**上是错的先验：
室内点云是墙面/地面/天花板这些二维流形上的采样，不是三维体填充。
更糟的是，报告里的"等效半径" `r = (3k/4pi*rho)^(1/3)` 又把这个假设代了回去，
于是得到一个**纯代数恒等式**：

    r = (3k / (4*pi*rho))^(1/3)   =>   alpha_r = (alpha_k - 1) / 3

实测 alpha_k = -0.0707 → alpha_r = -0.3569，与报告值逐位一致。
**所以 alpha_k 与 alpha_r 不是两条互相印证的证据，是同一条。**

本模块给出**不依赖假设**的几何量，回答一个具体问题：

    k 变大，究竟改变了"邻居数量"，还是改变了"物理覆盖范围"？

实测量
======

对每个点 i 与其前 k 个邻居 j：
    r_k(i)    : 第 k 近邻的实测距离（即邻域物理上界）
    dbar_k(i) : 前 k 个邻居的平均距离（邻域"紧凑程度"）
    C_k(i)    : 邻域体积 / k = (4/3)*pi*r_k^3 / k（物理覆盖率）

以及一个诊断内在维度的量（最有信息量的一条）：

    dbar_k ∝ k^gamma       =>   gamma ≈ 1/2 说明采样流形是**二维面**
                                gamma ≈ 1/3 说明是**三维体**

因为二维面上 k 个点铺满半径 r 的圆盘：k ∝ r^2 ⇒ r ∝ k^(1/2)；
三维体里 k ∝ r^3 ⇒ r ∝ k^(1/3)。这条**不需要知道密度**，只要 k，就能判断
"表面前提是否成立"。若 gamma ≈ 1/2 而论文却用三维体积公式解释 alpha，
那整个解释链条就是错的。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from . import density as D
except ImportError:  # 直接运行本文件时
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import density as D


def neighbor_stats(
    xyz: np.ndarray,
    ks: List[int],
    max_points: Optional[int] = None,
    seed: int = 20260923,
) -> Dict[int, Dict[str, float]]:
    """一次 kNN 查询算出所有 k 的邻域几何统计。

    只做一次 kNN（取 max(ks) 个邻居），再按 k 切片，避免重复查询。

    ⚠ 若 max_points 触发子采样，r_k 会被**系统性放大**（点变稀了），
    因此返回值里会带 `subsampled` 标记 —— 被标记的行**不可用于跨密度比较**。
    这一点踩过坑：diag0 有 131077 点、被 20000 上限子采样后 r_k 放大 ~2.5 倍，
    混进跨域拟合把 alpha_r 从 -0.31 拉到了 -0.20。

    返回 {k: {"r_k_mean","r_k_med","dbar_mean","dbar_med","C_k_med","ratio",
              "n_used","subsampled"}}
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n_total = xyz.shape[0]
    n = n_total
    ks = sorted({int(k) for k in ks if int(k) > 0})
    if n < 4 or not ks:
        return {}

    subsampled = False
    if max_points is not None and max_points > 0 and n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=int(max_points), replace=False)
        xyz = xyz[np.sort(sel)]
        n = xyz.shape[0]
        subsampled = True

    k_max = int(min(max(ks), n - 1))
    # 自身距离为 0，多取一列后丢掉第一列
    _idx, dist = D.knn_query(xyz, xyz, k_max + 1)
    dist = dist[:, 1:]                       # (n, k_max) 第 1..k_max 近邻距离
    if dist.shape[1] < k_max:
        k_max = dist.shape[1]

    out: Dict[int, Dict[str, float]] = {}
    for k in ks:
        if k > k_max:
            continue
        dk = dist[:, :k].astype(np.float64)          # (n, k)
        r_k = dk[:, -1]                              # 第 k 近邻距离
        dbar = dk.mean(axis=1)
        vol = (4.0 / 3.0) * np.pi * np.maximum(r_k, 1e-9) ** 3
        C_k = vol / float(k)
        out[k] = {
            "r_k_mean": float(r_k.mean()),
            "r_k_med": float(np.median(r_k)),
            "dbar_mean": float(dbar.mean()),
            "dbar_med": float(np.median(dbar)),
            "C_k_med": float(np.median(C_k)),
            # dbar/r_k → 1 表示邻居都贴在半径球面上（表面采样特征），
            # → 1/3 附近更像体填充
            "ratio": float(np.median(dbar) / max(np.median(r_k), 1e-9)),
            "n_used": int(n),
            "n_total": int(n_total),
            "subsampled": bool(subsampled),
        }
    return out


def fit_dimension_exponent(
    xyz: np.ndarray, ks: List[int], max_points: Optional[int] = None, seed: int = 20260923
) -> Optional[float]:
    """在**固定密度**下拟合 dbar_k ∝ k^gamma，用来判断采样流形的内在维度。

    gamma ≈ 1/2 → 二维面采样（室内扫描的真实情况）
    gamma ≈ 1/3 → 三维体填充

    这条测量**完全不用 rho**，所以不受"体密度定义"的循环污染，是干净的。
    """
    import math

    stats = neighbor_stats(xyz, ks, max_points=max_points, seed=seed)
    if len(stats) < 2:
        return None
    xs = [math.log(k) for k in stats]
    ys = [math.log(max(stats[k]["dbar_med"], 1e-12)) for k in stats]
    a, _b = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return float(a)


def coverage_table(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ks: List[int],
    max_points: int = 0,
    seed: int = 20260923,
    roles: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """跨域扫 k，产出"物理覆盖 vs 邻居数"的对照表。

    这是回答 "k 是数量还是范围" 的直接证据：
      - 若 k 增大时 dbar/r_k 基本不变、而 r_k 明显变大 → k 控制的是**物理覆盖范围**；
      - 若 r_k 几乎不动而邻居更密 → k 控制的是**采样密度内的邻居数量**。

    max_points=0（默认）表示**不子采样** —— 跨密度比较必须如此，否则大域的 r_k
    会被子采样放大，破坏密度关系（踩过坑，见 neighbor_stats 的说明）。
    """
    rows: List[dict] = []
    for name, (xyz, _y) in domains.items():
        rho_med = float(np.median(D.local_density(xyz)))
        stats = neighbor_stats(xyz, ks, max_points=max_points or None, seed=seed)
        gamma = fit_dimension_exponent(xyz, ks, max_points=max_points or None, seed=seed)
        role = (roles or {}).get(name, "main")
        for k, st in stats.items():
            rows.append({"domain": name, "role": role, "rho": rho_med,
                         "k": k, "gamma": gamma, **st})
    return rows
