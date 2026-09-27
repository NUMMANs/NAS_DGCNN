"""被进化的程序（**精度 × 成本 双目标，且能提出新架构**）：组合几何分支与尺度。

与 `initial_program_pareto.py` 的关键区别
================================================================
`initial_program_pareto.py` 只能从**已测好的候选清单里挑一个**（返回架构名）→
搜索空间是离散的 63 个点，LLM 无论怎么写都无法创造新架构。
实测结果印证了这一点：60 迭代里 best 出现在 **iteration 1**，
之后 59 次迭代的 combined_score 只取到 4 个不同值（0.4216 出现 55 次）——
**这不是收敛，是"选择空间"被锁死了**。

本程序改为让策略**组合出新的架构描述**：
    build_architecture(...) -> dict(branches=[...], fuse=..., z_bins=..., ...)

评估器会把该描述**与所有已测配置一起纳入候选**：
  · 若描述的架构**已在结果表里**（名字匹配）→ 直接用实测的精度与成本（快、无偏）
  · 若是**新组合** → 用 `arch_cost_model` 预估成本，并**标记为待验证**
    （精度按"同分支数/同尺度数的最近邻实测配置"保守估算，绝不乐观外推）

这样进化的自由度从"63 选 1"扩展到"组合空间"，而实测表仍然兜住可信度。

⚠ 设计原则：**宁可保守，不可乐观**。新架构的精度估计取**同结构族里已知的最差**，
这样只有"显然更好"的组合才会被选中，避免幻想的架构骗过搜索。
"""

from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# 已实测的"构件"（这些名字都有真实结果表支撑，可安全复用）
# ---------------------------------------------------------------------------
KNOWN_BRANCHES: List[str] = [
    "points",          # 笛卡尔 + kNN 动态图（基线，最强单分支之一但慢）
    "pillar25",        # 柱状 BEV，z 压进通道（较弱）
    "voxel15", "voxel20", "voxel25", "voxel35", "voxel50",   # 体素分支（体素边长 m）
    "rangeview64",     # 球面投影（最弱，实测 0.2786 单独）
]

KNOWN_Z: List[int] = [8, 16, 32, 48, 64, 96, 128, 192]
KNOWN_FUSE: List[str] = ["late", "gate", "concat"]


def pareto_policy(room_stats: Dict[str, object]) -> Union[str, int, Dict, List]:
    """提出一个架构（**这是要进化的函数**）。

    ⚠⚠ **为什么必须返回配置 dict，而不是架构名字**
    ================================================
    返回名字只能从 63 个**已测好的固定点**里挑 → 搜索空间是离散且封闭的，
    LLM 无论怎么写都无法创造新架构。实测印证：该形态下 60 迭代里 best 出现在
    **iteration 1**，combined_score 只有 4 个不同值（0.4216 占 55 次）
    —— 这不是收敛，是**选择空间被锁死**。

    返回配置 dict（含 `branches` / `fuse` / `z_bins` / `feat`）则把自由度扩展到
    **组合空间**：评估器会把该描述与实测候选一起纳入；
    若组合已在表中 → 用实测；若是新组合 → 用**保守估计**（见评估器文档）。

    要进化的就是这个 dict 里的四个轴。
    """
    # ↓↓↓ 进化应主要改这一段 ↓↓↓
    return {
        "branches": ["voxel20_z32", "voxel50_z96"],
        "fuse": "gate",
        "z_bins": 16,
        "feat": 64,
    }
    # ↑↑↑ 可用的分支名见 KNOWN_BRANCHES；每项可带 `_zNN` 后缀指定该分支的 z 分层 ↑↑↑


def build_architecture(room_stats: Dict[str, object]) -> Dict[str, object]:
    """保留的兼容入口（等价于 pareto_policy，供外部调用与阅读）。"""
    out = pareto_policy(room_stats)
    if isinstance(out, dict):
        return out
    # 若 pareto_policy 被改回返回名字，这里退化为一个默认描述，保证不崩
    return {"branches": ["voxel20_z32", "voxel50_z96"], "fuse": "gate", "z_bins": 16}


def robust_objective(room_stats: Dict[str, object]) -> float:
    """留给进化的额外目标钩子（返回 float）。默认不表达任何偏好。"""
    return 0.0
