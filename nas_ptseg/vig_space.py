"""PointViG 半径集合的候选空间定义（供 NAS 搜索用）。

设计依据（都来自实测，不是猜）
================================================================
1. 单房间过拟合已过（同房 0.88）→ 骨干正确，可以搜
2. 首个对照实验（4+2 房 × 2 seed × 300ep）：
     A `[0.2]`          0.3495   seed 极差 0.0113
     B `[0.1,0.2,0.4]`  0.3615   seed 极差 0.0080   ← +0.0121 可信
     C `[0.15,0.3]`     0.3557   seed 极差 0.0060   ← +0.0063 噪声内
   → **多尺度确实赢，但不是"任意多尺度都赢"**（C 就在噪声内）
   → **尺度集合的具体选择是实质变量** —— 这正是值得搜的理由

3. 论文的消融只扫了**邻域数 k**（k=32 最优），**完全没扫半径**。
   所以"半径集合"是论文留下的空白，而它恰好是感受野的旋钮。

候选空间构造（保持可控 + 覆盖合理）
================================================================
· 半径取值：{0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6}（覆盖 indoor 物件~房间）
· 尺度个数：1 ~ 3（3 以上成本翻倍，先不做）
· 有序去重（半径集合与顺序无关 → 用 frozenset 去重）
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

RADII_POOL: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6)

# 已实测的（避免重复烧 GPU —— 这是本项目的一条硬规矩）
MEASURED: Dict[str, float] = {
    "0.2": 0.3495,                 # 论文的单尺度设计
    "0.1|0.2|0.4": 0.3615,         # 三尺度（首个可信增益）
    "0.15|0.3": 0.3557,            # 双尺度（噪声内）
}


def _key(radii: Sequence[float]) -> str:
    return "|".join("%g" % r for r in sorted(float(r) for r in radii))


def name_of(radii: Sequence[float]) -> str:
    return "vig_" + _key(radii).replace(".", "").replace("|", "_")


def build_candidates(max_scales: int = 3, include_measured: bool = False) -> List[List[float]]:
    """枚举候选半径集合（去重、按尺度个数升序）。

    ⚠ `include_measured=False` 时**跳过已实测的**，避免重复烧 GPU
    （本项目踩过"扫描范围过窄导致重复训练"的坑）。
    """
    from itertools import combinations

    out: List[List[float]] = []
    seen = set()
    for k in range(1, max_scales + 1):
        for combo in combinations(RADII_POOL, k):
            key = _key(combo)
            if key in seen:
                continue
            seen.add(key)
            if not include_measured and key in MEASURED:
                continue
            out.append([float(x) for x in combo])
    return out


def describe(radii: Sequence[float]) -> dict:
    """给一个集合，返回它的元信息（含成本粗估）。

    成本模型：ADGC 的代价 ∝ 尺度个数（每个半径各做一次 ball_query + 注意力），
    以实测 `[0.1,0.2,0.4]`（3 尺度）0.18 s/epoch 为锚点按比例外推。
    ⚠ 这是**成本**估计；**精度绝不估**（本项目硬原则：
      成本由结构决定可外推，精度由优化决定无法推）。
    """
    n = len(radii)
    lo = min(radii)
    hi = max(radii)
    return {
        "radii": [float(r) for r in radii],
        "name": name_of(radii),
        "n_scales": n,
        "r_min": lo,
        "r_max": hi,
        "span": hi - lo,
        # 抗噪：某半径太小（<0.05）在 16k 下采样后可能邻居不足
        "warn_small_r": lo < 0.05,
        "cost_s_per_epoch_est": round(0.06 * n, 3),
    }


if __name__ == "__main__":
    cands = build_candidates(max_scales=3)
    print("候选数（半径集合）= %d" % len(cands))
    print("已实测（跳过）= %s" % list(MEASURED))
    print()
    for c in cands[:20]:
        d = describe(c)
        print("  %-22s n=%d  span=%.2f  cost≈%.2fs/ep" % (
            d["name"], d["n_scales"], d["span"], d["cost_s_per_epoch_est"]))
    print("  … 共 %d 个" % len(cands))
