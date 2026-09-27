"""DGCNN (EdgeConv) 密度自适应代理评估器。

为什么是"代理"而不是真训练
==========================
大场景点云分割单次训练成本极高（这是本课题的出发点）。若评估器接真实训练，
进化算法会变成烧钱机器。所以这里评估一个**训练无关（training-free）的代理量**：

    DGCNN 结构的 EdgeConv 前向（随机固定权重）
        + 随机特征展开
        + 闭式 ridge 回归探针（有闭式解，无梯度迭代）

代理量衡量的是：**语义标签在"密度自适应特征"里有多容易被线性解码出来**。
这是随机特征 / 线性探针范式的标准做法，单次评估在毫秒到亚秒量级。

为什么从 PointNet++ 换成 DGCNN
==============================
用户要求先用 DGCNN 验证机制。从密度自适应的角度，DGCNN 其实更合适：

  - DGCNN 在**特征空间**逐层重算 kNN 图（dynamic graph）。固定 k 时，
    物理感受野满足 r_k ≈ (3k / 4*pi*rho)^(1/3)，会随密度漂移。
  - 因此"k 该不该随密度变、变多少"变成一个**可量化、可解释**的科学问题：
    拟合 k ∝ rho^alpha 的指数 alpha。
  - PointNet++ 的球查询把半径写死在配置里，密度适应被隐藏在采样层，
    不如 DGCNN 的 k 直接。

搜索空间
========
策略（被进化的程序）输入密度画像 profile，输出每层的：

    k            : 特征空间 kNN 图的度（邻居数）
    radius_m     : 边门控半径（米）。物理距离超过它的边不参与聚合
    sample_rate  : 图节点采样率，控制图规模 → 直接决定计算量

三者都允许是局部密度的函数，这正是"几何参数化搜索空间"的落地形式。

代价模型
========
解析式、与硬件无关，保证可复现、可比。分两部分：

    params : 投影矩阵参数量
    flops  : 图构建 + 边特征聚合的乘加次数

不做精确的硬件建模 —— 代理只需要**单调且近似成正比**，用来支撑 Pareto 分析。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 允许 `python nas_ptseg/proxy.py` 直接运行
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import density as D
else:
    from . import data as DT
    from . import density as D


# ---------------------------------------------------------------------------
# 搜索空间的数值边界
# ---------------------------------------------------------------------------
# 边界的作用是**防止退化解**。没有边界，LLM 很容易生成 k=1（没有邻域信息）
# 或 radius_m=1e6（把所有边都圈进来）这类无意义配置，而它们在代理里可能
# 恰好得分不低 → 污染整个进化轨迹。
#
# 为什么 k 上界取 48：最稀疏域的节点数才 1024，k 再大邻域就接近全图，
# "局部"这一概念本身失效。
#
# 为什么 radius 上限取 2.0 m：单房间尺度约 5~8 m，半径超过 2 m 已接近全局
# 聚合，超出部分没有物理意义。
K_MIN, K_MAX = 4, 48
R_MIN_M, R_MAX_M = 0.02, 2.0
SR_MIN, SR_MAX = 0.05, 1.0

# 探针默认重复次数（模块级，便于命令行统一覆盖）。
# 1 = 不重复，保持历史可比；正式做结论时应 ≥8（见 ProxyConfig.probe_repeats）。
DEFAULT_PROBE_REPEATS: int = 1
DEFAULT_PROBE_FOLDS: int = 1

# 均匀性下限：sample_rate 太低时节点太少，插值回全分辨率会严重失真。
# 这不是硬约束，只是给策略一个合理的活动范围。


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class ProxyConfig:
    """代理前向的结构超参。这些**不**参与进化，保持固定以保证可比性。"""

    num_layers: int = 3            # EdgeConv 层数（DGCNN 经典是 3~4）
    feat_dim: int = 32             # 每层边特征投影后的维度
    probe_dim: int = 512           # 随机特征展开维度（探针输入维度）
    # 探针重复次数：对**同一份特征**换种子跑多次闭式 ridge，取平均。
    #
    # 为什么这是性价比最高的一处改动：实测（prep.py --seed-noise）同一配置换种子
    # 的 mIoU 标准差达 0.007~0.028，而项目要分辨的效应量只有 0.02~0.05 ——
    # 信噪比 ≈ 1，k* 根本不可辨识，α 是在拟合噪声。
    # 而这套代理里**昂贵的是前向**（kNN 图 + EdgeConv），探针只是闭式解，
    # 单次几十毫秒。所以复用同一份 feats 反复跑探针，把方差降 √n 倍，
    # 代价几乎可以忽略 —— 这是先降噪再谈结论的正确顺序。
    probe_repeats: Optional[int] = None
    ridge_lambda: float = 1.0e-2   # ridge 正则强度
    train_frac: float = 0.6        # 单次划分时的训练集占比（probe_folds>1 时忽略）
    # 探针的 k 折交叉验证折数。1 = 沿用历史的一次性 60/40 划分。
    #
    # 为什么需要它：一次性划分有两个问题，在稀疏域尤其致命 ——
    #   (1) 测试集太小：rho4 只有 1024 点，40% 只有 410 点，逐类 IoU 方差极大；
    #   (2) 划分本身随机：换个种子，测试集换了样本，mIoU 就跳 —— 这是
    #       --seed-noise 实测 sigma 的主要来源之一。
    # 改成 k 折后，每个点都恰好被预测一次，评估覆盖全部点、且不再依赖某次划分，
    # 划分方差被彻底消除（只剩随机投影的方差，交给 probe_repeats 处理）。
    # 代价是求解次数 × folds，但闭式解极便宜，前向仍然只做一次。
    probe_folds: Optional[int] = None
    interp_k: int = 3              # 特征插值回全分辨率时的邻居数
    seed: int = 12345              # 固定随机权重与划分，保证评估可复现
    knn_cost_const: float = 6.0    # 图构建代价的经验常数（树查询的遍历开销）
    gate_floor: float = -1.0e9     # 被门控掉的边的填充值
    # EdgeConv 边特征投影的分块大小（节点数）。
    #
    # 为什么必须分块：边特征的中间张量是 (M, k, feat_dim)，M 为图节点数。
    # 密集端诊断域有 131072 个点，k=48、feat_dim=32 时单张量就是
    # 131072*48*32*4 ≈ 806 MB，而这一步同时存在 edge_feat / proj / gate 三张，
    # 峰值会到 2.5 GB 以上 —— 直接 OOM。
    # 分块后峰值只与块大小有关（32768 块约 200 MB/张），与全域点数解耦。
    edge_chunk_nodes: int = 32768
    # 诊断专用：把每节点的边数平铺到该值，**几何完全不变**。
    #
    # 为什么需要它：dense-diag 实测 mIoU 随 k 单调上升（三个域、密度跨 43 倍，
    # 增益都约 +0.04，最优都顶到 k=48）。这有两种解释，必须区分：
    #   (H1 几何解释) 更大邻域真的提供更多有用上下文 → 假设成立，k 该大；
    #   (H2 结构解释) 边数变多 → max-pool 覆盖更多随机投影 → 探针输入的特征
    #                 秩变高 → ridge 探针天然更容易解码。**与几何无关的偏置**。
    # 若 H2 成立，整个"mIoU 随几何参数变化"的代理信号都不可信。
    #
    # 判别方法：保持真实邻居集合不变（仍是 k 个真实邻居），把索引循环平铺到
    # pad_neighbors_to，使"边数"与"几何"解耦。
    #   若 8-邻居平铺到 48 边 ≈ 真 48-邻居 → H2 成立，代理须改造。
    #   若明显低于真 48-邻居 → H1 成立，信号是真几何信号。
    pad_neighbors_to: int = 0

    def __post_init__(self) -> None:
        if self.probe_repeats is None:
            self.probe_repeats = DEFAULT_PROBE_REPEATS
        self.probe_repeats = int(max(1, int(self.probe_repeats)))
        if self.probe_folds is None:
            self.probe_folds = DEFAULT_PROBE_FOLDS
        self.probe_folds = int(max(1, int(self.probe_folds)))


# ---------------------------------------------------------------------------
# 策略调用
# ---------------------------------------------------------------------------
def clamp_params(raw: dict) -> dict:
    """把策略返回的参数夹到合法范围。

    越界不报错（进化中 LLM 必然生成越界值），只夹取 —— 报错会让整轮评估失败、
    白白浪费一次 LLM 调用。但夹取次数记入 clamped，作为策略质量的诊断信号：
    一个经常越界的策略说明它没有理解搜索空间的尺度。
    """
    k = int(round(float(raw.get("k", 16))))
    radius_m = float(raw.get("radius_m", 0.3))
    sr = float(raw.get("sample_rate", 1.0))
    clamped = 0
    if k < K_MIN:
        k, clamped = K_MIN, clamped + 1
    elif k > K_MAX:
        k, clamped = K_MAX, clamped + 1
    if radius_m < R_MIN_M:
        radius_m, clamped = R_MIN_M, clamped + 1
    elif radius_m > R_MAX_M:
        radius_m, clamped = R_MAX_M, clamped + 1
    if sr < SR_MIN:
        sr, clamped = SR_MIN, clamped + 1
    elif sr > SR_MAX:
        sr, clamped = SR_MAX, clamped + 1
    if not np.isfinite(radius_m) or radius_m <= 0:
        radius_m, clamped = R_MIN_M, clamped + 1
    return {"k": k, "radius_m": radius_m, "sample_rate": sr, "clamped": clamped}


def safe_policy_call(
    policy: Callable[[Dict[str, float]], dict], profile: Dict[str, float]
) -> Tuple[dict, Optional[str]]:
    """调用策略并捕获异常，返回 (params, error)。

    策略是 LLM 生成的代码，必须假设它会抛异常。捕获后回退到基线参数，
    让这一轮评估仍能产出有意义的结果，而不是整轮失败。
    """
    try:
        raw = policy(profile)
        if not isinstance(raw, dict):
            return clamp_params({}), f"policy 返回了非 dict 类型：{type(raw).__name__}"
        return clamp_params(raw), None
    except Exception as e:  # noqa: BLE001
        return clamp_params({}), f"{type(e).__name__}: {e}"


def baseline_policy(profile: Dict[str, float]) -> dict:
    """手工基线：固定 k / 半径 / 采样率，**完全不随密度变化**。

    这是论文要对比的"手工架构"。它在密度漂移下必然在某些域偏优、某些域偏劣 ——
    这正是本项目要证伪的对象，也是搜索起点与代价归一化的参考点。

    radius_m 取 0.30 m 的理由：S3DIS 单房间在 16384 点量级时平均点间距约
    0.09 m，0.30 m 相当于覆盖约 3 圈邻居，是室内分割里常见的感受野量级。
    它同时也是"参照 D：固定架构扫描"的起点。
    """
    return {"k": 16, "radius_m": 0.30, "sample_rate": 1.0}


# ---------------------------------------------------------------------------
# 内部算子
# ---------------------------------------------------------------------------
def _random_proj(
    rng: np.random.Generator, in_dim: int, out_dim: int, scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """生成固定的随机投影矩阵与偏置（He 风格缩放）。"""
    s = float(scale) if scale is not None else float(np.sqrt(2.0 / max(1, in_dim)))
    W = (rng.standard_normal((in_dim, out_dim)) * s).astype(np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)
    return W, b


def _knn_graph(
    node_feat: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """在**特征空间**构建 kNN 图（DGCNN 的 dynamic graph）。

    返回 (nb_idx, ok_mask)：
      nb_idx  : (M, k) 邻居在 node 集合中的索引
      ok_mask : (M, k) bool，True 表示该边是"非自身"的有效边

    注意：查询后第一列是自身（距离 0），必须剔除 —— 否则 max-pool 会被
    自身特征主导，EdgeConv 就退化成了纯 MLP。
    """
    m = node_feat.shape[0]
    k_eff = int(max(1, min(k, m)))
    idx, _dist = D.knn_query(node_feat, node_feat, k_eff + 1)   # (M, k_eff+1)
    idx = idx[:, 1:]                                            # 去掉自身
    ok = np.ones(idx.shape, dtype=bool)
    if idx.shape[1] < k:
        pad = np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
        ok = np.concatenate([ok, np.zeros(pad.shape, dtype=bool)], axis=1)
    return idx.astype(np.int64), ok


def _interp_to(
    src_xyz: np.ndarray, src_feat: np.ndarray, dst_xyz: np.ndarray, k: int
) -> np.ndarray:
    """把 src 上的特征按反距离权重插值到 dst 坐标上。

    用途：sample_rate < 1 时图节点只是点云子集，但分割任务需要每个点都有输出。
    插值回全分辨率后，不同配置的 mIoU 都在**同一批点**上计算，比较才公平。
    """
    if src_feat.shape[0] == 0:
        return np.zeros((dst_xyz.shape[0], src_feat.shape[1]), dtype=np.float32)
    k_eff = int(max(1, min(k, src_feat.shape[0])))
    idx, dist = D.knn_query(dst_xyz, src_xyz, k_eff)             # (Nd, k_eff)
    w = 1.0 / np.maximum(dist, 1e-6)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("nk,nkd->nd", w, src_feat[idx]).astype(np.float32)


# ---------------------------------------------------------------------------
# DGCNN 前向
# ---------------------------------------------------------------------------
def forward(
    policy: Callable[[Dict[str, float]], dict],
    xyz: np.ndarray,
    cfg: ProxyConfig,
    rho_precomputed: Optional[np.ndarray] = None,
) -> dict:
    """执行一次密度自适应的 DGCNN 代理前向。

    参数
    ----
    policy : 密度画像 -> 参数字典。这是被进化的对象。
    xyz    : (N, 3) 米制点坐标（已居中）
    cfg    : 代理结构配置
    rho_precomputed : 可选，预先算好的逐点密度（跨域评估时复用，省时间）

    返回
    ----
    feats : (N, num_layers * feat_dim) 每层特征插值回全分辨率后拼接
    trace : 逐层参数与密度记录（用于诊断与 alpha 拟合）
    errors: 策略异常信息列表
    cost  : {'params': float, 'flops': float, 'n_nodes': [...]}
    """
    rng = np.random.default_rng(cfg.seed)
    n_total = xyz.shape[0]
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)

    rho = rho_precomputed if rho_precomputed is not None else D.local_density(xyz)
    rho_ref = float(np.median(rho)) if rho_precomputed is None else None

    trace: List[dict] = []
    errors: List[str] = []
    layer_feats_full: List[np.ndarray] = []

    cost_params = 0.0
    cost_flops = 0.0
    n_nodes_log: List[int] = []

    # 第 0 层的输入特征：坐标本身（DGCNN 原生做法），可选加颜色。
    # 这里不加 RGB —— 颜色会引入与密度无关的强语义线索（比如墙是白的、
    # 白板是白的），稀释我们对"几何密度"的关注。保持纯几何输入更干净。
    cur_feat = xyz.astype(np.float32)
    feat_dim_in = 3

    for l in range(cfg.num_layers):
        profile = D.density_profile(rho, l, cfg.num_layers)
        params, err = safe_policy_call(policy, profile)
        if err:
            errors.append(f"layer{l}: {err}")

        k = int(params["k"])
        radius_m = float(params["radius_m"])
        sr = float(params["sample_rate"])

        # ---- 图节点采样 ----
        n_node = int(max(1, min(n_total, round(sr * n_total))))
        if n_node >= n_total:
            node_idx = np.arange(n_total, dtype=np.int64)
        else:
            node_idx = np.sort(rng.choice(n_total, size=n_node, replace=False))
        node_xyz = xyz[node_idx]
        node_feat = cur_feat[node_idx]

        # ---- 特征空间 kNN 动态图 ----
        nb_idx, edge_ok = _knn_graph(node_feat, k)

        # ---- 诊断对照：把边数平铺放大，几何不变 ----
        # 见 ProxyConfig.pad_neighbors_to 的说明。邻居集合与物理半径门控都按
        # 原始 k 判定，只是把同一批真实邻居循环重复到目标边数 ——
        # 于是 "边数" 与 "邻域几何" 被解耦，可判定 mIoU-k 单调性来自哪一边。
        if cfg.pad_neighbors_to and cfg.pad_neighbors_to > nb_idx.shape[1]:
            target = int(cfg.pad_neighbors_to)
            reps = int(np.ceil(target / nb_idx.shape[1]))
            nb_idx = np.tile(nb_idx, (1, reps))[:, :target]
            edge_ok = np.tile(edge_ok, (1, reps))[:, :target]
            k = target          # 代价与 trace 记录"实际聚合的边数"

        nb_feat = node_feat[nb_idx]                       # (M, k, d_in)
        nb_xyz = node_xyz[nb_idx]                         # (M, k, 3)

        # ---- 物理半径门控 ----
        # 边虽然是在特征空间选的，但**物理距离**超过 radius_m 的边不参与聚合。
        # 这样 radius_m 才真正是一个"感受野上界"，而不是被特征空间的距离
        # 稀释掉的一个摆设。全部超界时回退为最近邻，避免整行被屏蔽导致输出恒定。
        phys_d = np.linalg.norm(nb_xyz - node_xyz[:, None, :], axis=2)     # (M, k)
        in_rad = phys_d <= radius_m
        none_in = ~in_rad.any(axis=1)
        if none_in.any():
            in_rad[none_in, 0] = True
        edge_ok = edge_ok & in_rad
        n_in = edge_ok.sum(axis=1).astype(np.float32)                       # (M,)

        # ---- EdgeConv：h_ij = MLP([f_i ; f_j - f_i]) 然后 max-pool over j ----
        # 边特征：自身特征拼接相对特征，这是 EdgeConv 的定义式。
        # 按节点分块计算，使峰值内存与全域点数解耦（见 ProxyConfig.edge_chunk_nodes）。
        in_dim = 2 * feat_dim_in
        W, b = _random_proj(rng, in_dim, cfg.feat_dim)
        k_used = int(nb_feat.shape[1])
        node_feat_out = np.empty((n_node, cfg.feat_dim), dtype=np.float32)
        chunk = int(max(1, cfg.edge_chunk_nodes))
        for s in range(0, n_node, chunk):
            e = min(n_node, s + chunk)
            nf = node_feat[s:e]                                        # (c, d_in)
            rel = nb_feat[s:e] - nf[:, None, :]                        # (c, k, d_in)
            edge = np.concatenate(
                [np.repeat(nf[:, None, :], k_used, axis=1), rel], axis=2
            ).astype(np.float32)                                       # (c, k, 2*d_in)
            proj = edge.reshape(-1, in_dim) @ W + b
            proj = np.maximum(proj, 0.0).reshape(e - s, k_used, cfg.feat_dim)
            gate = np.where(
                edge_ok[s:e, :, None], 0.0, cfg.gate_floor
            ).astype(np.float32)
            node_feat_out[s:e] = (proj + gate).max(axis=1)
            del rel, edge, proj, gate

        # ---- 代价（解析式）----
        # 图构建：树查询的代价近似为 M * log2(M) * d 次距离运算
        cost_flops += float(n_node) * float(np.log2(max(n_node, 2))) * float(feat_dim_in) * cfg.knn_cost_const
        # 边特征聚合：M * k * (2*d) * feat_dim 次乘加
        cost_flops += float(n_node) * float(k) * float(2 * feat_dim_in) * float(cfg.feat_dim)
        cost_params += float(in_dim * cfg.feat_dim) + float(cfg.feat_dim)

        # ---- 插值回全分辨率 ----
        # 必须插值回全分辨率再交给下一层，原因有两层，缺一不可：
        #   (1) 正确性：下一层会用 [0, n_total) 的索引去取样节点。若这里把
        #       M 行的 node_feat_out 直接当作 cur_feat，而本层 sample_rate < 1
        #       （M < n_total），下一层取样就会 IndexError。这是实测踩到的 bug。
        #   (2) 语义一致性：sample_rate 是搜索维度，图规模会逐层变化。只有把
        #       每层特征都对齐回同一批点，各层的 k / radius 才在**同一个点集**
        #       上可比，"逐层自适应"的 alpha 也才是同一物理量的跨层比较。
        full_l = _interp_to(node_xyz, node_feat_out, xyz, cfg.interp_k)
        layer_feats_full.append(full_l)
        cost_flops += float(n_total) * float(cfg.interp_k) * float(cfg.feat_dim)

        trace.append(
            {
                "layer": l,
                "k": k,
                "radius_m": radius_m,
                "sample_rate": sr,
                "n_nodes": int(n_node),
                "rho_med": float(profile["rho"]),
                "rho_ratio": float(profile["ratio"]),
                "scale_hint_m": float(profile["scale_hint"]),
                "n_in_med": float(np.median(n_in)),
                # 半径 / 物理锚点：宏观反映策略的"自适应强度"。
                # =1 表示刚好等于等效点间距；>1 表示感受野跨了更多圈邻居。
                "radius_over_spacing": float(radius_m / max(profile["scale_hint"], 1e-9)),
                "clamped": int(params["clamped"]),
            }
        )
        n_nodes_log.append(int(n_node))

        cur_feat = full_l
        feat_dim_in = cfg.feat_dim

    feats = np.concatenate(layer_feats_full, axis=1) if layer_feats_full else np.zeros((n_total, 1), dtype=np.float32)

    return {
        "feats": feats,
        "trace": trace,
        "errors": errors,
        "cost": {
            "params": float(cost_params),
            "flops": float(cost_flops),
            "n_nodes": n_nodes_log,
        },
        "rho": rho,
        "rho_ref_local": rho_ref,
    }


# ---------------------------------------------------------------------------
# 线性探针
# ---------------------------------------------------------------------------
def linear_probe_miou(
    feats: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cfg: ProxyConfig,
) -> Tuple[float, float, np.ndarray]:
    """随机特征 + 闭式 ridge 回归探针，返回 (mIoU, acc, 每类 IoU)。

    为什么用闭式解：ridge 回归有解析解
        W = (X^T X + lambda I)^-1 X^T Y
    不需要任何梯度迭代，因此评估是确定性的、毫秒级的。这正是"训练无关代理"
    能成立的技术基础。

    为什么先用随机特征展开：只有 L*feat_dim 维（本例 96 维）时线性可分性
    受限于特征维度。展开到 512 维并加 ReLU 能捕捉一部分非线性，提升代理
    与真实网络排序的相关性，代价仍然极低。

    为什么可重复多次（cfg.probe_repeats）：单次探针的随机投影与训练/评估
    划分都依赖种子，实测同配置换种子的标准差可达 0.03 —— 与要测的效应量同量级。
    前向（昂贵）只做一次，把 feats 复用给多次探针（便宜）取平均，方差降 √n 倍。
    """
    n, d = feats.shape
    if n < 8 or d == 0:
        return 0.0, 0.0, np.zeros((num_classes,), dtype=np.float32)

    reps = int(max(1, cfg.probe_repeats or 1))
    if reps == 1:
        return _linear_probe_once(feats, labels, num_classes, cfg, cfg.seed + 777)

    mious: List[float] = []
    accs: List[float] = []
    pcs: List[np.ndarray] = []
    for r in range(reps):
        m, a, pc = _linear_probe_once(feats, labels, num_classes, cfg,
                                      cfg.seed + 777 + r)
        mious.append(m)
        accs.append(a)
        pcs.append(pc)
    with np.errstate(invalid="ignore"):
        pc_mean = np.nanmean(np.stack(pcs, axis=0), axis=0).astype(np.float32)
    return float(np.mean(mious)), float(np.mean(accs)), pc_mean


def _linear_probe_once(
    feats: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cfg: ProxyConfig,
    seed: int,
) -> Tuple[float, float, np.ndarray]:
    """单次探针。seed 决定随机投影与划分。

    probe_folds <= 1：历史行为，一次性 train_frac 划分（默认 60/40）。
    probe_folds >  1：k 折交叉验证，每个点恰被预测一次 →
                      评估覆盖全部点、且不依赖某次具体划分。
    """
    n, d = feats.shape
    rng = np.random.default_rng(seed)
    W, b = _random_proj(rng, d, cfg.probe_dim)
    X = np.maximum(feats @ W + b, 0.0).astype(np.float32)      # (N, probe_dim)

    folds = int(max(1, cfg.probe_folds or 1))
    if folds > 1:
        return _probe_kfold(X, labels, num_classes, cfg, rng, folds)

    # 训练 / 评估划分（固定种子，保证同一配置重复评估结果一致）
    perm = rng.permutation(n)
    n_tr = int(max(1, min(n - 1, round(cfg.train_frac * n))))
    tr, te = perm[:n_tr], perm[n_tr:]

    Y = np.zeros((n_tr, num_classes), dtype=np.float32)
    Y[np.arange(n_tr), labels[tr]] = 1.0

    Xtr = X[tr]
    Xte = X[te]

    # 标准化：让 lambda 对不同尺度的特征都合理
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Wr = _ridge_solve(Xtr, Y, cfg.ridge_lambda)
    pred = np.argmax(Xte @ Wr, axis=1).astype(np.int64)
    gt = labels[te].astype(np.int64)

    acc = float((pred == gt).mean()) if gt.size else 0.0
    miou, per_class = _miou(pred, gt, num_classes)
    return miou, acc, per_class


def _ridge_solve(Xtr: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """闭式 ridge 解 W = (X^T X + lambda I)^-1 X^T Y。"""
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    B = Xtr.T @ Y
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def _probe_kfold(
    X: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cfg: ProxyConfig,
    rng: np.random.Generator,
    folds: int,
) -> Tuple[float, float, np.ndarray]:
    """k 折交叉验证探针：每个点恰被预测一次，mIoU 在全点集上算。

    这既消掉了"测试集太小"（稀疏域 40% 只有几百点）也消掉了"划分随机"。
    """
    n = X.shape[0]
    folds = int(max(2, min(folds, n)))
    perm = rng.permutation(n)
    parts = np.array_split(perm, folds)
    pred = np.empty(n, dtype=np.int64)

    for i in range(len(parts)):
        te = parts[i]
        if te.size == 0:
            continue
        tr = np.concatenate([parts[j] for j in range(len(parts)) if j != i])
        if tr.size == 0:
            pred[te] = int(np.bincount(labels, minlength=num_classes).argmax())
            continue
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

        Y = np.zeros((tr.size, num_classes), dtype=np.float32)
        Y[np.arange(tr.size), labels[tr]] = 1.0

        Wr = _ridge_solve(Xtr, Y, cfg.ridge_lambda)
        pred[te] = np.argmax(Xte @ Wr, axis=1).astype(np.int64)

    gt = labels.astype(np.int64)
    acc = float((pred == gt).mean()) if gt.size else 0.0
    miou, per_class = _miou(pred, gt, num_classes)
    return miou, acc, per_class


def _miou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[float, np.ndarray]:
    """mIoU，只对 ground-truth 中真实出现的类求平均。

    为什么跳过不存在的类：稀疏域 / 某些房间里个别类一个样本都没有，若把它
    算作 IoU=0 会严重低估，且不同域之间不可比。
    """
    ious: List[float] = []
    per_class = np.full((num_classes,), np.nan, dtype=np.float32)
    for c in range(num_classes):
        gt_c = gt == c
        if not gt_c.any():
            continue
        pd_c = pred == c
        inter = float((gt_c & pd_c).sum())
        union = float((gt_c | pd_c).sum())
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)
        per_class[c] = iou
    return (float(np.mean(ious)) if ious else 0.0), per_class


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 用合成点云做纯函数自测（不需要 S3DIS 数据）
    rng = np.random.default_rng(0)
    n = 6000
    # 构造一个"房间"：地面 + 墙面
    floor = np.stack(
        [rng.uniform(-3, 3, n // 2), rng.uniform(-2, 2, n // 2), np.zeros(n // 2)], axis=1
    )
    wall = np.stack(
        [np.full(n // 2, -3.0), rng.uniform(-2, 2, n // 2), rng.uniform(0, 2.5, n // 2)], axis=1
    )
    xyz = np.concatenate([floor, wall], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)])

    cfg = ProxyConfig()
    out = forward(baseline_policy, xyz, cfg)
    miou, acc, per = linear_probe_miou(out["feats"], y, 2, cfg)

    print("gating 自测")
    print(f"  N              = {xyz.shape[0]}")
    print(f"  feats shape    = {out['feats'].shape}")
    print(f"  mIoU / acc     = {miou:.4f} / {acc:.4f}")
    print(f"  params / flops = {out['cost']['params']:.0f} / {out['cost']['flops']:.3e}")
    print(f"  errors         = {out['errors']}")
    for t in out["trace"]:
        print(
            f"  layer{t['layer']}: k={t['k']} r={t['radius_m']:.3f}m "
            f"sr={t['sample_rate']:.2f} nodes={t['n_nodes']} "
            f"n_in_med={t['n_in_med']:.0f} r/spacing={t['radius_over_spacing']:.2f}"
        )

    # 密度自适应 vs 固定：在稀疏化后的点云上对比
    from nas_ptseg.data import voxel_downsample

    xyz_s, y_s = voxel_downsample(xyz, y, 0.08)
    m_s, a_s, _ = (lambda o: linear_probe_miou(o["feats"], y_s, 2, cfg))(
        forward(baseline_policy, xyz_s, cfg)
    )
    print(f"  稀疏域(N={xyz_s.shape[0]}) 固定策略 mIoU = {m_s:.4f}")

    def _adaptive(p):
        # 简单自适应：k 与半径随密度反向缩放（等体积先验）
        s = p["scale_hint"]
        return {
            "k": int(np.clip(round(12.0 + 26.0 * min(s / 0.3, 1.0)), K_MIN, K_MAX)),
            "radius_m": float(np.clip(2.0 * s, R_MIN_M, R_MAX_M)),
            "sample_rate": 1.0,
        }

    m_a, a_a, _ = (lambda o: linear_probe_miou(o["feats"], y_s, 2, cfg))(
        forward(_adaptive, xyz_s, cfg)
    )
    print(f"  稀疏域(N={xyz_s.shape[0]}) 自适应策略 mIoU = {m_a:.4f}")
