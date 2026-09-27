"""与框架无关的评测指标（**不得 import torch**）。

为什么单独一个模块
================================================================
OpenEvolve 跑在 `openevolve` conda 环境里（**没有 torch**），而训练/评估代码在
`lidar310`（有 torch）。为了让 OpenEvolve 的 evaluator 能直接复用 fitness 表逻辑，
被它依赖的模块必须能在无 torch 环境下 import。

因此凡是"纯 numpy 的指标计算"都放这里，`fitness.py` 与 `cross_room_exp.py` 共用，
避免两边实现漂移（同一指标两处实现是本项目踩过的坑：mIoU 口径不一致会让结果不可比）。
"""

from __future__ import annotations

from typing import List

import numpy as np


def present_class_miou(pred: np.ndarray, gt: np.ndarray, present: List[int]) -> float:
    """只对 `present` 里列出的类别算 mIoU。

    为什么不能直接用全部类别：不同房间的类别覆盖不同（有的没 door/table），
    若按全类别算，缺失类会被当作 IoU=0 计入平均，于是"类别少的房间"分数天然偏低 ——
    那是类别分布差异，不是分割质量差异。只对**该房间真实出现的类别**求平均，
    跨房间才可比。
    """
    ious: List[float] = []
    for c in present:
        g = gt == c
        p = pred == c
        if not g.any():
            continue
        inter = float((g & p).sum())
        union = float((g | p).sum())
        ious.append(inter / union if union > 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def miou_np(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> tuple:
    """全类别 mIoU（返回 (mIoU, acc)）。用于类别口径固定的场景（单房间受控密度）。"""
    acc = float((pred == gt).mean()) if gt.size else 0.0
    ious: List[float] = []
    for c in range(num_classes):
        g = gt == c
        p = pred == c
        if not g.any():
            continue
        inter = float((g & p).sum())
        union = float((g | p).sum())
        ious.append(inter / union if union > 0 else 0.0)
    return (float(np.mean(ious)) if ious else 0.0), acc
