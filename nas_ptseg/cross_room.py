"""跨房间跨域数据层（方向 1：真实跨域，取代单房间受控密度）。

背景与动机
================================================================
第七轮判定实验失败后，确认了"单房间 + 体素下采样造密度 + 空间留出"这套框架是
**结构性死胡同**：在受控密度下 `k/N` 与 `r_k/L` 数学退化（面采样下只差平方根变换），
唯一能打破退化的"改形状"操作又必然污染留出难度。连两轮用有缺陷对照产出不可信结论。

于是转向：**用 S3DIS 的真实多房间作为"域"**。不同房间的
  - N（点数）
  - L（空间尺度，包围盒）
  - rho（局部密度）
三个量**自然独立**（不再是同房间体素下采样那样只差平方根），从而第一次可以干净地
判别"最优 k 到底由哪个量决定"。

关键性质（与旧框架的根本区别）
================================================================
- 旧：域 = 同一房间的不同体素下采样（几何/语义一致，只有密度变）。
- 新：域 = **不同的真实房间**（几何、语义、密度、尺度、类别分布都不同）。
- 训练/测试划分 = **按房间分**（train on 一组房间，test on 完全不同的房间），
  而不是房间内切空间块 → "留出设计"这个死胡同直接消失。

要回答的问题变成
================================================================
对每个房间扫固定 k，最优 k* 如何随该房间的 (N, L, rho) 变化？
这是三个真正独立的自变量，跨房间回归即可判别。

⚠ 类别口径：不同房间类别覆盖不同。为避免"某房间缺 floor 类导致 mIoU 无定义"，
本模块对每个房间只用**该房间实际出现的类别**算 mIoU（present-class mIoU），
并在 meta 里记录每房间的类别覆盖，供分析时标注可比性。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import data as DT
from . import density as D

# 训练时每房间的读取点数上限。真实跨域里点数是**自变量**，不能像旧框架那样
# 用 read_max_pts 把点数统一 —— 所以这里取一个较大的上限，让各房间的真实
# 点数差异保留下来；只在房间过大（hallway 数百万点）时截断以控显存/时间。
ROOM_READ_MAX_PTS = 200_000


def load_room_raw(area: str, room: str, seed: int = 20260923) -> Tuple[np.ndarray, np.ndarray]:
    """读取一个房间的 (xyz, labels)，保留其**真实点数差异**（点数上限只兜底）。

    与 `data.load_room` 的区别：这里刻意不统一点数 —— 点数 N 本身就是自变量。
    """
    xyz, _rgb, labels = DT.load_room(area, room, max_pts=ROOM_READ_MAX_PTS, seed=seed)
    return xyz, labels


def room_stats(xyz: np.ndarray) -> dict:
    """一个房间的几何/密度统计（不含标签），用于跨房间回归的自变量。"""
    n = int(xyz.shape[0])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ext = hi - lo
    L = float(ext.max())                    # 最大包围盒边长（尺度）
    # 局部密度：用第 8 近邻反解（与旧 density.DENSITY_K 一致，保持可比）
    rho = D.local_density(xyz)              # (n,) pts/m^3
    rho_med = float(np.median(rho))
    return {
        "n_points": n,
        "extent_max_m": L,
        "extent": ext.astype(float).tolist(),
        "rho_med": rho_med,
        "rho_p10": float(np.percentile(rho, 10)),
        "rho_p90": float(np.percentile(rho, 90)),
    }


def load_rooms(
    rooms: List[Tuple[str, str]],
    seed: int = 20260923,
    target_n: Optional[int] = 16384,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], dict]:
    """把一组房间读成 {room_key: (xyz, labels)}，附每房间统计。

    room_key 形如 "Area_1/conferenceRoom_1"。

    关键设计：`target_n` 非 None 时，每个房间体素下采样到**统一目标点数**。
    为什么固定 N（而非保留真实点数）：
      - 训练成本可控（DGCNN 的逐层动态 kNN 在 18 万点上每 epoch 要 15s+，
        24 房间×5k 完全不可行；下采样到 16384 后单次训练 10s 内）；
      - **把 N 这个变量固定掉**，跨房间只剩尺度 L 和密度 rho 两个自变量，
        而 k/N 在每个房间都相同（= k/16384），于是若 k* 跨房间不同，
        只能归因于 L 或 rho —— 干净地排除了 k/N 的干扰；
      - 固定 N 时，大房间（L 大）→ 密度低（rho 小），二者负相关但不完全退化
        （房间形状不规则），仍可由跨房间回归区分。

    `target_n=None` 则保留真实点数（点数本身就是自变量，但训练慢）。
    """
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta: dict = {"rooms": {}, "classes": DT.COMPACT_CLASSES, "target_n": target_n}
    for area, room in rooms:
        key = f"{area}/{room}"
        xyz, labels = load_room_raw(area, room, seed=seed)
        if target_n is not None and target_n > 0 and xyz.shape[0] > target_n:
            _v, xyz, labels = DT.calibrate_voxel_for_target_n(xyz, labels, target_n)
        domains[key] = (xyz, labels)
        st = room_stats(xyz)
        present = sorted(int(c) for c in np.unique(labels))
        st["n_classes_present"] = len(present)
        st["present_classes"] = present
        meta["rooms"][key] = {"area": area, "room": room, **st}
    return domains, meta


def select_rooms(
    n_rooms: int = 24,
    areas: Optional[List[str]] = None,
    room_types: Optional[List[str]] = None,
    seed: int = 20260923,
) -> List[Tuple[str, str]]:
    """从 S3DIS 选一组代表性房间（跨 Area、跨房间类型），返回 [(area, room), ...]。

    选择标准（探路阶段，优先"信号清晰"而非"规模多样"）：
      - **类别覆盖全**：优先选含家具类（table/chair/clutter）的房间，
        否则跨房间迁移时类别分布差异会淹没 k 的效应（hallway 只有 3 类）。
      - **规模中等**：避免 auditorium（数百万点，读入慢）与 storage（太小）。
      - **跨 Area 多样**：尽量覆盖多个 Area。

    规模估计用"文件名数 × 平均点数"，但 hallway 类文件名少而每文件大，
    所以这里同时看文件数与文件名（hallway 文件前缀是 hallway_*，实际点更多）。
    为稳妥，用 Annotations 下文件总大小做代理（字节数 ≈ 点数）。
    """
    all_rooms = DT.list_rooms(area=None)
    if areas:
        all_rooms = [r for r in all_rooms if r[0] in areas]

    cands: List[dict] = []
    for area, room in all_rooms:
        ann = os.path.join(DT.S3DIS_ROOT, area, room, "Annotations")
        if not os.path.isdir(ann):
            continue
        files = [f for f in os.listdir(ann) if f.lower().endswith(".txt")]
        if not files:
            continue
        total_bytes = 0
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(ann, f))
            except OSError:
                continue
        # 类别覆盖：数不同类别前缀
        prefixes = set()
        for f in files:
            p = DT._parse_class_from_filename(f)
            if p:
                prefixes.add(p)
        n_classes = len(prefixes)
        # 房间类型
        rtype = "other"
        for t in ("conferenceRoom", "office", "hallway", "lounge", "lobby", "pantry"):
            if room.startswith(t):
                rtype = t
                break
        if room_types and rtype not in room_types:
            continue
        cands.append({
            "area": area, "room": room, "type": rtype,
            "n_files": len(files), "bytes": total_bytes,
            "n_classes": n_classes, "prefixes": prefixes,
        })

    if not cands:
        return []

    # 优先：类别 ≥5 且规模中等（bytes 在分位数 30%~70%）
    bytes_arr = np.asarray([c["bytes"] for c in cands], dtype=np.float64)
    lo, hi = np.percentile(bytes_arr, [25, 75])
    good = [c for c in cands if c["n_classes"] >= 5 and lo <= c["bytes"] <= hi]
    if len(good) < n_rooms:
        # 放宽：类别 ≥4 就够，规模放宽到 10%~90%
        lo2, hi2 = np.percentile(bytes_arr, [10, 90])
        good = [c for c in cands if c["n_classes"] >= 4 and lo2 <= c["bytes"] <= hi2]

    # 跨 Area 分层：每个 Area 尽量都取
    rng = np.random.default_rng(seed)
    by_area: Dict[str, List[dict]] = {}
    for c in good:
        by_area.setdefault(c["area"], []).append(c)
    selected: List[Tuple[str, str]] = []
    areas_list = sorted(by_area)
    # 轮询各 Area，每次各取一个（随机），直到凑够 n_rooms
    while len(selected) < n_rooms and any(by_area.values()):
        for a in areas_list:
            if not by_area[a]:
                continue
            c = by_area[a].pop(int(rng.integers(0, len(by_area[a]))))
            selected.append((c["area"], c["room"]))
            if len(selected) >= n_rooms:
                break
    return selected
