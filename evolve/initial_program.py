"""
跨房间几何策略 —— 这是被 OpenEvolve 进化的对象。

任务（第一轮：只搜 k）
====================
给出一个房间的**几何/密度统计量**，决定在这个房间上训练 DGCNN 时使用的
`k`（特征空间 kNN 图的度）。

评分方式（务必理解，否则会往错方向优化）
======================================
fitness = 用该房间的 k 训练后，在**若干未见过的验证房间**上的平均
          present-class mIoU（每个房间只对它真实出现的类别算 IoU）。

也就是说：**k 是"为这个房间挑的"，但分数来自别的房间**。所以策略必须表达
"什么样的房间该用多大的邻域"这一**可迁移的规律**，而不是记住某个房间。

为什么这是可搜的问题
====================
DGCNN 在特征空间逐层重算 kNN 图。固定 k 时第 k 近邻的物理距离满足
`r_k ∝ (k/ρ)^(1/2)`（室内是二维面采样，不是体采样）。于是：

    k 固定 ⇒ 物理感受野随密度漂移

但"该补偿多少"并不显然：等物理半径（k∝ρ）在小密度下会让 k 撞上界，
完全不补偿（k 固定）又会让稀疏房间的邻域跨越语义边界。真实最优可能是
**部分补偿**，而且可能依赖房间尺度、类别数、形状等。这正是要搜的东西。

输入：room_stats 字典（唯一的信息来源）
=====================================
    n_points            该房间下采样后的点数（本轮统一为 16384）
    extent_max_m        包围盒最长边长（米），房间尺度
    rho_med             中位局部密度（points/m^3，原始物理量）
    rho_p10 / rho_p90   密度分布的下/上十分位
    n_classes_present   该房间出现的语义类别数（任务复杂度）
    extent              三维包围盒尺寸，list[float]，米
    （area / room 字段**已在本轮被刻意移除**：用它们等于记住房间，跨房间必然失效）

输出：一个整数 k
==============
    允许值会被吸附到候选格 K_GRID = (8, 16, 24, 32, 48, 64)
    （返回任意数值都会被吸附到最近的一格，所以不必自己取整）

硬性约束
========
1. **必须是确定性函数**：同样的 room_stats 必须返回同样的 k。
2. **不要做 I/O**：不读文件、不联网、不打印。评估会被高频调用。
3. **只依赖 room_stats**：不要试图识别"这是哪个房间/哪栋楼"再查表。
4. 只用 Python 标准库和 numpy（不要 import torch / nas_ptseg）。
5. 越界/异常会被记为无效并给 0 分，所以做好防御性判断。

当前版本：固定 k（基线起点）
==========================
下面的初始实现**故意不随任何统计量变化**，就是论文里要对比的"手工固定架构"。
从它出发，是为了让"自适应是否有价值"这个结论由**搜索自己发现**。
"""

from typing import Dict

# 候选格（与 nas_ptseg/fitness.py 的 K_GRID 一致）
K_GRID = (8, 16, 24, 32, 48, 64)


# EVOLVE-BLOCK-START
def geometric_policy(room_stats: Dict[str, float]) -> int:
    """把房间的几何/密度统计映射为训练时使用的 k。

    当前实现：固定 k = 16，完全不随房间变化（手工架构基线）。
    """
    # 读取统计量（保留给未来的自适应版本使用；当前故意不用）
    rho_med = float(room_stats.get("rho_med", 1000.0))
    extent_max_m = float(room_stats.get("extent_max_m", 5.0))
    n_classes_present = float(room_stats.get("n_classes_present", 7.0))
    n_points = float(room_stats.get("n_points", 16384))

    k = 16

    # 防御性：夹到候选格范围
    k = int(max(K_GRID[0], min(K_GRID[-1], k)))
    _ = (rho_med, extent_max_m, n_classes_present, n_points)
    return k
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# 参考工具（不在 EVOLVE-BLOCK 内，可自由调用或忽略）
# ---------------------------------------------------------------------------
def _nearest_grid(x: float) -> int:
    """把任意数值吸附到最近的候选格。"""
    return int(min(K_GRID, key=lambda g: abs(g - float(x))))


def _reference_power_law(room_stats: Dict[str, float]) -> int:
    """参考实现：按密度幂律选 k。

    思路（仅供启发，不必照抄）：若想让"邻域里的物理范围"不随密度膨胀，
    就需要在稀疏房间用更大的 k。但补偿强度不是显然的 —— 完全补偿会撞上界，
    不补偿则感受野失控。指数取多少正是本任务要搜索的核心。

        k = k0 * (rho_med / rho_ref) ** alpha      # alpha < 0 → 稀疏房间用更大 k

    注意 rho_ref 这里用常数锚点即可（不要依赖全局状态）。
    """
    rho_med = max(float(room_stats.get("rho_med", 1000.0)), 1e-9)
    rho_ref = 1500.0          # 量级参考（验证房间的中位密度约在此附近）
    alpha = -0.3              # 负号：稀疏（rho 小）→ k 更大
    k0 = 24.0
    return _nearest_grid(k0 * (rho_med / rho_ref) ** alpha)
