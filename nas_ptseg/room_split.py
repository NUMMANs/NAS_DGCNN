"""固定跨房间 train / val / test 划分（无泄漏），供搜索与可靠度验证共用。

为什么必须先做这一步
================================================================
在"搜索是否有意义"这个问题被回答之前，任何搜索都缺一个**固定的、可复现的**
评分底座。本模块只做一件事：把 S3DIS 的房间确定性地分成三份，写盘，之后所有
实验都读同一份划分。

无泄漏的三条规则
================================================================
1. **按房间分**，不是按点分。同一房间的点绝不跨 split —— 否则同一片几何会同时
   出现在训练与测试里，mIoU 会虚高。
2. **test 只在最终确认排名时碰一次**。日常搜索/校准只允许用 train + val。
   本模块因此在 API 上把 test 单独隔离（`load_split_rooms("test")` 需要显式调用）。
3. **分层**：每个 Area 都按 (train, val, test) 轮转分配，避免"test 全来自同一栋楼"
   这种伪泛化（那就是普通的场景泛化，不是域泛化）。

划分是**固定**的（种子写死），并落盘为 `data/room_split.json`：
换机器、换进程、换 seed 做训练，读到的都是同一份房间清单。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import data as DT
from . import cross_room as CR

# ---------------------------------------------------------------------------
# 划分参数（改动会改变划分 → 必须重建 room_split.json 并作废历史分数）
# ---------------------------------------------------------------------------
SPLIT_SEED = 20260923
N_TRAIN, N_VAL, N_TEST = 12, 12, 12

# 候选房间的准入条件：类别覆盖够、体量中等
MIN_CLASS_PREFIXES = 6
SIZE_PCTL_LO, SIZE_PCTL_HI = 15.0, 85.0

# 下采样目标点数：所有 split 统一，保证"点数"不是混淆变量
TARGET_N = 16384

SPLIT_PATH = os.path.join(DT.PROJECT_ROOT, "data", "room_split.json")
ROOM_CACHE_DIR = os.path.join(DT.PROJECT_ROOT, "data", "cache_rooms")

SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 候选收集
# ---------------------------------------------------------------------------
def _room_candidates() -> List[dict]:
    """收集所有满足准入条件的房间，附面积/类别/体量统计（不读点云）。"""
    out: List[dict] = []
    for area, room in DT.list_rooms(area=None):
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
        prefixes = set()
        for f in files:
            p = DT._parse_class_from_filename(f)
            if p:
                prefixes.add(p)
        if len(prefixes) < MIN_CLASS_PREFIXES:
            continue
        out.append({
            "area": area, "room": room, "key": f"{area}/{room}",
            "n_files": len(files), "bytes": total_bytes,
            "n_prefixes": len(prefixes),
        })
    return out


def build_split(
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    seed: int = SPLIT_SEED,
    write: bool = True,
) -> dict:
    """确定性构造划分并（默认）落盘。返回 {"train": [...], "val": [...], "test": [...]}。

    分层策略：按 Area 分组，组内按体量排序，再以 (train, val, test) 顺序轮转取，
    直到三份都装满。轮转保证每个 Area 都尽量进入每一份。
    """
    cands = _room_candidates()
    if not cands:
        raise RuntimeError("没有候选房间；确认 S3DIS 已解压且 Annotations 可读")

    sizes = np.asarray([c["bytes"] for c in cands], dtype=np.float64)
    lo, hi = np.percentile(sizes, [SIZE_PCTL_LO, SIZE_PCTL_HI])
    cands = [c for c in cands if lo <= c["bytes"] <= hi]
    if len(cands) < (n_train + n_val + n_test):
        raise RuntimeError(
            f"候选房间不足：{len(cands)} < {n_train + n_val + n_test}；"
            f"可放宽 MIN_CLASS_PREFIXES 或 SIZE_PCTL_*"
        )

    rng = np.random.default_rng(seed)
    by_area: Dict[str, List[dict]] = {}
    for c in cands:
        by_area.setdefault(c["area"], []).append(c)

    # 每个 Area 内随机打散一次（固定种子 → 可复现），再按体量排序，保证分层可控
    for a in by_area:
        lst = by_area[a]
        perm = rng.permutation(len(lst))
        by_area[a] = [lst[i] for i in perm]
        by_area[a].sort(key=lambda c: c["bytes"])

    targets = {"train": n_train, "val": n_val, "test": n_test}
    out: Dict[str, List[dict]] = {k: [] for k in SPLIT_NAMES}
    areas_sorted = sorted(by_area)
    # 轮转：多轮，每轮每个 Area 各贡献一个房间，依次填入 train/val/test
    order = list(SPLIT_NAMES)
    ptr = {a: 0 for a in areas_sorted}
    round_i = 0
    while any(len(out[k]) < targets[k] for k in SPLIT_NAMES):
        progressed = False
        for a in areas_sorted:
            if ptr[a] >= len(by_area[a]):
                continue
            # 找出这一轮还缺人的 split（按 order 轮转）
            want = order[round_i % 3]
            if len(out[want]) >= targets[want]:
                # 该 split 已满，换个还没满的
                for alt in order:
                    if len(out[alt]) < targets[alt]:
                        want = alt
                        break
                else:
                    break
            out[want].append(by_area[a][ptr[a]])
            ptr[a] += 1
            progressed = True
            if all(len(out[k]) >= targets[k] for k in SPLIT_NAMES):
                break
        if not progressed:
            break
        round_i += 1

    for k in SPLIT_NAMES:
        if len(out[k]) < targets[k]:
            raise RuntimeError(f"split '{k}' 只凑到 {len(out[k])}/{targets[k]} 个房间")

    split = {
        "seed": int(seed),
        "target_n": int(TARGET_N),
        "min_class_prefixes": int(MIN_CLASS_PREFIXES),
        "size_pctl": [SIZE_PCTL_LO, SIZE_PCTL_HI],
        "splits": {
            k: [{"area": c["area"], "room": c["room"], "key": c["key"],
                 "n_prefixes": c["n_prefixes"], "bytes": c["bytes"]}
                for c in out[k]]
            for k in SPLIT_NAMES
        },
    }
    if write:
        os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
        with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
            json.dump(split, fh, ensure_ascii=False, indent=2)
    return split


def load_split(path: Optional[str] = None) -> dict:
    p = path or SPLIT_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"划分文件不存在：{p}；先跑 `python -m nas_ptseg.room_split --build`")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_rooms(split: dict, which: str) -> List[Tuple[str, str]]:
    """取出某一份的房间清单 [(area, room), ...]。"""
    if which not in SPLIT_NAMES:
        raise ValueError(f"which 必须是 {SPLIT_NAMES} 之一，收到 {which!r}")
    return [(d["area"], d["room"]) for d in split["splits"][which]]


# ---------------------------------------------------------------------------
# 房间点云缓存（下采样后），避免每次评估都重读百万点文本
# ---------------------------------------------------------------------------
def _cache_path(area: str, room: str, target_n: int) -> str:
    return os.path.join(ROOM_CACHE_DIR, f"{area}__{room}__n{target_n}.npz")


def load_room_cached(
    area: str, room: str, target_n: int = TARGET_N, seed: int = 20260923
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读一个房间（体素下采样到 target_n），带磁盘缓存。返回 (xyz, labels, stats)。

    缓存的意义：百万点的文本解析约 0.7 s/房间，一次评估要读 24+ 房间。缓存后
    单房间加载降到 ~0.05 s，搜索阶段每轮评估的开销才可接受。
    """
    os.makedirs(ROOM_CACHE_DIR, exist_ok=True)
    p = _cache_path(area, room, target_n)
    if os.path.exists(p):
        with np.load(p) as z:
            xyz = z["xyz"]
            labels = z["y"]
            stats = json.loads(str(z["stats"]))
        return xyz, labels, stats

    xyz, labels = CR.load_room_raw(area, room, seed=seed)
    if target_n and target_n > 0 and xyz.shape[0] > target_n:
        _v, xyz, labels = DT.calibrate_voxel_for_target_n(xyz, labels, target_n)
    stats = CR.room_stats(xyz)
    stats["present_classes"] = sorted(int(c) for c in np.unique(labels))
    stats["n_classes_present"] = len(stats["present_classes"])
    stats["area"] = area
    stats["room"] = room
    np.savez_compressed(p, xyz=xyz, y=labels, stats=json.dumps(stats))
    return xyz, labels, stats


def load_split_rooms(
    split: dict, which: str, target_n: int = TARGET_N, verbose: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    """加载某一份 split 的全部房间 → {room_key: (xyz, labels, stats)}。"""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, dict]] = {}
    for area, room in split_rooms(split, which):
        key = f"{area}/{room}"
        out[key] = load_room_cached(area, room, target_n=target_n)
        if verbose:
            _x, _y, st = out[key]
            print(f"    [{which}] {key}: N={st['n_points']} "
                  f"L={st['extent_max_m']:.2f}m rho={st['rho_med']:.0f} "
                  f"cls={st['n_classes_present']}")
    return out


# ---------------------------------------------------------------------------
# 划分体检
# ---------------------------------------------------------------------------
def describe(split: dict, verbose: bool = True) -> dict:
    """打印 / 返回三份 split 的分布，用来确认它们**没有系统性差异**。

    若 train 与 val/test 的密度或尺度分布差很多，"训练→验证"的落差就会主要来自
    分布偏移而不是架构，结论会被污染。所以在开搜之前先看这张表。
    """
    rows: Dict[str, List[dict]] = {}
    summary: dict = {}
    for which in SPLIT_NAMES:
        rooms = load_split_rooms(split, which)
        recs = []
        for key, (_x, _y, st) in rooms.items():
            recs.append(st)
        rows[which] = recs
        if not recs:
            continue
        rho = np.asarray([r["rho_med"] for r in recs], float)
        L = np.asarray([r["extent_max_m"] for r in recs], float)
        cls = np.asarray([r["n_classes_present"] for r in recs], float)
        areas = sorted({r["area"] for r in recs})
        summary[which] = {
            "n_rooms": len(recs),
            "areas": areas,
            "n_areas": len(areas),
            "rho_med_min": float(np.min(rho)), "rho_med_max": float(np.max(rho)),
            "rho_med_geo": float(np.exp(np.mean(np.log(np.maximum(rho, 1e-9))))),
            "L_min": float(np.min(L)), "L_max": float(np.max(L)),
            "L_mean": float(np.mean(L)),
            "cls_min": int(np.min(cls)), "cls_max": int(np.max(cls)),
            "cls_mean": float(np.mean(cls)),
        }
    if verbose:
        print("=" * 84)
        print("房间划分体检（关键：三份的 rho / L / 类别覆盖不应有系统性差异）")
        print("=" * 84)
        hdr = f"{'split':<7s}{'房间数':>7s}{'Area数':>7s}{'rho几何均值':>13s}" \
              f"{'rho范围':>22s}{'L均值':>8s}{'L范围':>16s}{'类别均值':>9s}"
        print(hdr)
        for which in SPLIT_NAMES:
            s = summary.get(which)
            if not s:
                continue
            print(f"{which:<7s}{s['n_rooms']:>7d}{s['n_areas']:>7d}{s['rho_med_geo']:>13.1f}"
                  f"{s['rho_med_min']:>10.0f}~{s['rho_med_max']:<11.0f}"
                  f"{s['L_mean']:>8.2f}{s['L_min']:>7.2f}~{s['L_max']:<8.2f}"
                  f"{s['cls_mean']:>9.2f}")
        print("")
        for which in SPLIT_NAMES:
            s = summary.get(which)
            if s:
                print(f"  {which:<6s} areas: {', '.join(s['areas'])}")
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="固定跨房间 train/val/test 划分")
    ap.add_argument("--build", action="store_true", help="重新构造划分并写盘")
    ap.add_argument("--describe", action="store_true", help="只打印划分分布")
    ap.add_argument("--force", action="store_true", help="已有划分也重建")
    args = ap.parse_args()

    if args.build:
        if os.path.exists(SPLIT_PATH) and not args.force:
            print(f"划分已存在，未覆盖：{SPLIT_PATH}（要重建加 --force）")
        else:
            build_split(write=True)
            print(f"划分已写入：{SPLIT_PATH}")
    if args.describe or not args.build:
        describe(load_split())


if __name__ == "__main__":
    main()
