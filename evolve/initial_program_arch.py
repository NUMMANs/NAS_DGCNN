"""
被进化的程序（**架构空间**）：把一个房间的统计量映射为一个局部几何/架构配置。

OpenEvolve 会反复改写 `arch_policy` 的函数体（也可以改辅助函数），
每次改写后用 `evolve/evaluator_arch.py` 打分。**函数签名不能改。**

搜索空间（离散，可被完整物化 → 评估是毫秒级查表，完全无偏）
========================================================
    k           邻域大小：8 / 16 / 24 / 32 / 48 / 64
    agg         EdgeConv 的池化方式：max（原 DGCNN）/ mean / maxmean  ← 参数量不变
    graph_space 图建在哪儿：feat（逐层在特征空间重算，原 DGCNN）/ coord（坐标空间静态图，只建一次）
                 ← 参数量不变
    （容量轴 feat / layers 在表里未启用，见 `nas_ptseg/arch_table.py`）

返回值
========================================================
返回 **配置字典**（推荐，可读、可组合）或 **cfg 下标 int**：
    return {"k": 48, "agg": "mean", "graph_space": "coord"}
    return 0
未知/缺失的轴会被自动补成最近的合法配置，所以不必保证字段齐全。

`room_stats` 里可用的量（**只有这些**，取不到别的）
========================================================
    n_points           该房间下采样后的点数（本项目固定 16384）
    extent_max_m       包围盒最长边（米）
    extent             三轴包围盒尺寸
    rho_med            中位局部点密度（pts/m^3）
    rho_p10 / rho_p90  密度的 10 / 90 分位
    n_classes_present  该房间出现的类别数（语义复杂度）

⚠ 不要试图识别具体房间（没有房间号/Area 可用，也不该用）。
⚠ `robust_objective` 必须返回一个 float，表示"除了平均精度之外你还想要什么"：
   例如最差房间的精度、或用 cfg 多样性换鲁棒性。评估器会把它一并记录，便于诊断。
"""

from typing import Dict


def default_config() -> Dict[str, object]:
    """
    当前这套配置的"常量基线"。

    说明：原 DGCNN 论文的默认设置就是 `k=48` 附近 + 特征空间动态图 + max 池化。
    初始程序**故意写成常量**（对所有房间给同一个配置），
    这样"是否值得按房间自适应"这件事必须由进化自己发现，而不是我们手工塞进去。
    """
    return {"k": 48, "agg": "max", "graph_space": "feat"}


def arch_policy(room_stats: Dict[str, float]) -> Dict[str, object]:
    """
    把一个房间的统计量映射为一个架构配置。

    这里目前是**不做任何自适应**的常量策略。
    """
    return default_config()


def robust_objective(room_stats: Dict[str, float]) -> float:
    """留给进化的"额外目标"钩子，返回 float。默认不表达任何偏好。"""
    return 0.0
