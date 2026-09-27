"""局部密度估计与密度画像。

单位约定
========
坐标一律是**米**（S3DIS 原生公制）。因此：

    local_density(xyz) -> 每立方米点数（points / m^3）

这比之前 KITTI 版本的"归一化密度"更直观：数值本身就是物理量，
可以直接和"室内扫描 1e4~1e6 pts/m^3"这种常识对照。

RHO_REF 的作用
==============
策略（被进化的程序）不应该硬编码密度的绝对数值 —— 那样换一个房间就失效。
所以这里提供一个**跨域共享的参考密度** RHO_REF，策略只使用无量纲比值：

    ratio = rho / RHO_REF      # >1 稠密，<1 稀疏

RHO_REF 在数据准备阶段按各域实测密度的几何均值标定（见 prep.py），
并写入 meta.json。这样比值的量级始终是 O(1)，策略里的常数才有泛化性。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore
    _HAS_SCIPY = False


# kNN 估计密度的邻居数。太大→过度平滑掉局部起伏；太小→方差大。
DENSITY_K = 8

# 跨域共享的参考密度（points / m^3）。由 prep.build 标定后写入。
# 默认值取室内扫描的典型量级，仅作为未标定时的兜底。
RHO_REF_DEFAULT = 1.0e4

_rho_ref: float = RHO_REF_DEFAULT


def set_rho_ref(v: float) -> None:
    global _rho_ref
    _rho_ref = float(max(v, 1e-9))


def get_rho_ref() -> float:
    return float(_rho_ref)


# ---------------------------------------------------------------------------
# kNN 基础设施
# ---------------------------------------------------------------------------
def knn_query(
    query_pts: np.ndarray, ref_pts: np.ndarray, k: int
) -> tuple:
    """查询 query_pts 每个点在 ref_pts 中的 k 个最近邻。

    返回 (idx, dist)，形状均为 (n_query, k)。

    说明：优先用 scipy 的 cKDTree（C 实现，比暴力快 1~2 个量级）；
    没有 scipy 时退化为分块暴力计算，保证纯 numpy 环境也能跑。
    """
    q = np.ascontiguousarray(query_pts, dtype=np.float32)
    r = np.ascontiguousarray(ref_pts, dtype=np.float32)
    n_q = q.shape[0]
    k = int(max(1, min(k, r.shape[0])))

    if _HAS_SCIPY:
        tree = cKDTree(r)
        d, i = tree.query(q, k=k, workers=-1)
        # ⚠ scipy 的形状约定：k==1 时 d/i 是 1-D 的 (n_q,)，k>1 时是 (n_q, k)，
        # 而 n_q==1 且 k>1 时又是 (k,)。三种情况混在一起，用 atleast_2d 会出错：
        # atleast_2d((n_q,)) 得到 (1, n_q)，随后"列数 > k"的截断判断会成立，
        # 被切成 (1,1) —— 只返回第 0 个查询点的结果（k==1 时的静默数据丢失）。
        # 统一 reshape 成 (n_q, k) 对上述三种形状都正确。
        d = np.asarray(d, dtype=np.float32).reshape(n_q, k)
        i = np.asarray(i, dtype=np.int64).reshape(n_q, k)
        return i, d

    # ---- 纯 numpy 兜底：分块暴力 ----
    out_i = np.empty((n_q, k), dtype=np.int64)
    out_d = np.empty((n_q, k), dtype=np.float32)
    block = max(1, int(2_000_000 // max(1, r.shape[0])))
    for s in range(0, n_q, block):
        e = min(n_q, s + block)
        diff = q[s:e, None, :] - r[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        if k < r.shape[0]:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(r.shape[0]), (e - s, 1))
        dd = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(dd, axis=1)
        out_i[s:e] = np.take_along_axis(part, order, axis=1)
        out_d[s:e] = np.sqrt(np.maximum(np.take_along_axis(dd, order, axis=1), 0.0))
    return out_i, out_d


def local_density(xyz: np.ndarray, k: int = DENSITY_K) -> np.ndarray:
    """逐点局部密度（points / m^3）。

    定义：以到第 k 近邻的距离 r_k 为半径的球，其平均单位体积点数

        rho = k / (4/3 * pi * r_k^3)

    取 k 而非 k+1：球内除自身外恰有 k 个邻居，用它算体积更自洽。

    参数
    ----
    xyz : (N, 3) float32，单位**米**
    k   : 邻居数

    返回
    ----
    (N,) float64，每立方米点数。已加下限避免除零。
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if n < 4:
        return np.full((n,), float(_rho_ref), dtype=np.float64)

    kk = int(min(k, n - 1))
    # 自身距离为 0，所以要多取一个再丢掉第 1 列
    _idx, dist = knn_query(xyz, xyz, kk + 1)
    r_k = dist[:, -1].astype(np.float64)          # 到第 k 近邻的距离
    r_k = np.maximum(r_k, 1e-9)
    vol = (4.0 / 3.0) * np.pi * (r_k ** 3)
    return (kk / np.maximum(vol, 1e-18)).astype(np.float64)


def density_profile(
    rho: np.ndarray,
    layer_idx: int,
    num_layers: int,
    rho_ref: float | None = None,
) -> Dict[str, float]:
    """把逐点密度压缩成策略可用的画像（profile）。

    这是策略函数唯一的输入。设计原则：**只暴露无量纲量和层次信息**，
    不暴露房间 / 域的具体身份 —— 否则策略会退化成"按域查表"，
    失去泛化性，实验结果也不可信。

    返回字段
    --------
    rho      : 中位密度（points/m^3，原始物理量，供分析用）
    ratio    : rho / rho_ref，无量纲（策略应主要用这个）
    log_ratio: log(ratio)，取值更线性，便于策略做幂律映射
    spread   : 域内密度的对数跨度（p90/p10），反映局部不均匀程度
    layer_idx / num_layers / depth_frac : 层次位置
    scale_hint : rho^(-1/3)，"等效点间距"（米），物理先验锚点
    """
    ref = float(rho_ref) if rho_ref is not None else float(_rho_ref)
    ref = max(ref, 1e-9)

    rho = np.asarray(rho, dtype=np.float64)
    rho_med = max(float(np.median(rho)), 1e-9)
    rho_lo = max(float(np.percentile(rho, 10)), 1e-9)
    rho_hi = max(float(np.percentile(rho, 90)), 1e-9)

    depth_frac = 0.0 if num_layers <= 1 else layer_idx / float(num_layers - 1)

    return {
        # --- 核心状态量 ---
        "rho": rho_med,
        "rho_lo": rho_lo,
        "rho_hi": rho_hi,
        "rho_ref": ref,
        # --- 无量纲量：策略应该只用这些 ---
        "ratio": rho_med / ref,
        "log_ratio": float(np.log(rho_med / ref)),
        "spread": float(np.log(rho_hi / rho_lo)),
        # --- 层次信息 ---
        "layer_idx": float(layer_idx),
        "num_layers": float(num_layers),
        "depth_frac": float(depth_frac),
        # --- 物理先验锚点 ---
        # 等体积假设下"当前密度对应的特征尺度"：球内含 k 个点所需的半径量级
        "scale_hint": float(rho_med ** (-1.0 / 3.0)),
    }


def fit_exponent(xs: List[float], ys: List[float]) -> float | None:
    """在 log-log 空间拟合幂律指数 alpha，使 y ∝ x^alpha。

    用途：把"策略是否随密度自适应"量化成一个可报告的数。

    推导（x = rho，球内点数 k ≈ (4/3)*pi*r^3*rho，这是唯一的物理依据）：
      - 若 **k 恒定**（图度数不变）→ r ∝ rho^(-1/3)，即对 y=r 有 alpha = -1/3
      - 若 **r 恒定**（物理感受野不变）→ k ∝ rho^(+1)，即对 y=k 有 alpha = +1
      - alpha = 0 → y 完全不随密度变（等价于手工固定架构）

    因此 alpha 的含义**取决于 y 是 k 还是 r**，报告时必须写明。判别标准：
        alpha_k ≈ 1  ⇔ 感受野半径恒定（完全补偿密度漂移）
        alpha_r ≈ -1/3 ⇔ 图度数恒定（完全不补偿，只是重算图）
        0 < alpha_k < 1 ⇔ 部分补偿 —— 这本身就是有价值的发现：
                          网络可能只需部分补偿，而非完全补偿。

    返回 None 表示样本不足或 x 无变化（无法拟合）。
    """
    xs = np.asarray([x for x in xs if x and x > 0], dtype=np.float64)
    ys = np.asarray([y for y in ys if y and y > 0], dtype=np.float64)
    if xs.size < 2 or ys.size < 2 or xs.size != ys.size:
        return None
    if np.ptp(xs) <= 0:
        return None
    lx = np.log(xs)
    ly = np.log(ys)
    a, _b = np.polyfit(lx, ly, 1)
    return float(a)
