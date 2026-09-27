"""跨域协议：**按 Area 划分的留一域（Leave-One-Area-Out, LOAO）**。

为什么必须单独建这个协议
================================================================
此前所有数字（k 轴、架构轴、换表示、z 扫描）都是"训练 12 个房间 → 验证另 12 个房间"，
而这两个 split 里**每个 Area 都有房间**（`room_split.py` 的轮转分层）。
那是**场景泛化（scene generalization）**，不是**域泛化（domain generalization）**。

S3DIS 的 6 个 Area 是**不同的建筑/不同的采集批次**，是真实存在的分布位移。
只有"训练在若干 Area、测试在**完全没见过的 Area**"才能回答本课题的问题：

    **架构的选择，在跨域时会不会产生比同域更大的差异？**

三种划分，用途不同
================================================================
    `val`  / `test`  （现有）  同域内留出房间 → 测**场景泛化**
    `loao`          （本文件）  留一个 Area 完全不见 → 测**域泛化**（原域内）
    `loao_hard`     （本文件）  留两个 Area 完全不见 → 测**域泛化**（更远位移）

判据（这是本课题的立论点）
================================================================
设 `Δ_same(c)` = 同域下配置 c 相对基线的增益，`Δ_cross(c)` = 跨域下同一配置的增益。
    若 **跨域的分化 ≫ 同域的分化** → "架构对跨域更关键"，NAS 在跨域上有巨大作用
    若两者相当                  → 架构不是跨域的主因，NAS 的叙事不成立

⚠ 这条判据是**关于方差/分化的**，不是关于均值的。
   跨域均值整体下降是必然的（分布位移），我们问的是"配置之间的差距是否被放大"。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import room_split as RS

    _PKG_MODE = False
else:
    from . import data as DT
    from . import room_split as RS

    _PKG_MODE = True

LOAO_PATH = os.path.join(DT.PROJECT_ROOT, "data", "loao_split.json")

# 每个 Area 至少要有这么多房间才值得当"留出域"（否则测试集太小、噪声压倒效应）
MIN_ROOMS_PER_AREA = 6
# 训练域至少要有这么多房间，保证训练集不至于残缺
MIN_TRAIN_ROOMS = 24


def area_rooms() -> Dict[str, List[str]]:
    """列出每个 Area 的全部房间 key（按 key 排序，保证确定性）。"""
    out: Dict[str, List[str]] = {}
    for area, room in DT.list_rooms(area=None):
        ann = os.path.join(DT.S3DIS_ROOT, area, room, "Annotations")
        if not os.path.isdir(ann):
            continue
        out.setdefault(area, []).append(f"{area}/{room}")
    for a in out:
        out[a] = sorted(out[a])
    return out


def build(fold: str = "loao") -> dict:
    """构造留一域划分。

    `fold='loao'`      → 每个 Area 各做一次"留出域"，其余 Area 训练（6 折）
    `fold='loao_hard'` → 留出**最大**的两个 Area（位移更大，只 1 折）

    ⚠ 训练域内的房间**不再细分 val**：本协议每折只用一次评估，
      而且我们关心的是"域间"的差异，多一层 val 只会再抽稀训练集。
      若要早停，另建 `loao_val`（但那不属于当前要回答的问题）。
    """
    ar = area_rooms()
    usable = {a: v for a, v in ar.items() if len(v) >= MIN_ROOMS_PER_AREA}
    dropped = {a: len(v) for a, v in ar.items() if len(v) < MIN_ROOMS_PER_AREA}
    folds: List[dict] = []

    if fold == "loao":
        for held in sorted(usable):
            tr = [r for a in sorted(usable) if a != held for r in usable[a]]
            te = list(usable[held])
            if len(tr) < MIN_TRAIN_ROOMS:
                continue
            folds.append({"name": f"hold_{held}", "holdout_areas": [held],
                          "train_rooms": tr, "test_rooms": te})
    elif fold == "loao_hard":
        sizes = sorted(usable, key=lambda a: -len(usable[a]))
        held = sorted(sizes[:2])
        tr = [r for a in sorted(usable) if a not in held for r in usable[a]]
        te = [r for a in held for r in usable[a]]
        if len(tr) >= MIN_TRAIN_ROOMS:
            folds.append({"name": "hold_" + "_".join(held),
                          "holdout_areas": held, "train_rooms": tr, "test_rooms": te})
    else:
        raise ValueError(f"未知 fold：{fold!r}（只支持 loao / loao_hard）")

    return {"kind": fold, "created_from": "S3DIS room_split",
            "areas": {a: len(v) for a, v in sorted(ar.items())},
            "dropped_areas": dropped, "folds": folds}


def save(obj: dict, path: str = LOAO_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def load(path: str = LOAO_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def describe() -> dict:
    """打印划分体检：每个 fold 的域位移有多大（用密度/尺度/类别刻画）。"""
    rows: List[str] = []
    out: dict = {}
    for kind in ("loao", "loao_hard"):
        obj = build(kind)
        out[kind] = []
        rows.append("")
        rows.append("=" * 96)
        rows.append(f"跨域协议 {kind}：{len(obj['folds'])} 折")
        rows.append("=" * 96)
        rows.append(f"  {'折':<18s}{'留出域':<22s}{'训练房':>7s}{'测试房':>7s}"
                    f"{'ρ_med':>9s}{'ρ_test':>9s}{'L':>7s}{'L_test':>8s}{'类别':>6s}{'类别_t':>7s}")
        for fd in obj["folds"]:
            st_tr, st_te = [], []
            for key in fd["train_rooms"][:12]:
                a, r = key.split("/", 1)
                _x, _y, s = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
                st_tr.append(s)
            for key in fd["test_rooms"]:
                a, r = key.split("/", 1)
                _x, _y, s = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
                st_te.append(s)
            rho_tr = float(np.median([s["rho_med"] for s in st_tr])) if st_tr else float("nan")
            rho_te = float(np.median([s["rho_med"] for s in st_te])) if st_te else float("nan")
            l_tr = float(np.mean([s["extent_max_m"] for s in st_tr])) if st_tr else float("nan")
            l_te = float(np.mean([s["extent_max_m"] for s in st_te])) if st_te else float("nan")
            c_tr = float(np.mean([s["n_classes_present"] for s in st_tr])) if st_tr else float("nan")
            c_te = float(np.mean([s["n_classes_present"] for s in st_te])) if st_te else float("nan")
            rows.append(f"  {fd['name']:<18s}{','.join(fd['holdout_areas']):<22s}"
                        f"{len(fd['train_rooms']):>7d}{len(fd['test_rooms']):>7d}"
                        f"{rho_tr:>9.0f}{rho_te:>9.0f}{l_tr:>7.2f}{l_te:>8.2f}"
                        f"{c_tr:>6.1f}{c_te:>7.1f}")
            out[kind].append({"fold": fd["name"], "rho_train": rho_tr, "rho_test": rho_te,
                              "L_train": l_tr, "L_test": l_te,
                              "cls_train": c_tr, "cls_test": c_te})
    print("\n".join(rows), flush=True)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="跨域留一域协议（LOAO）")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--out", default=LOAO_PATH)
    args = ap.parse_args()

    if args.describe or not args.build:
        describe()
        print()
    if args.build:
        allf: Dict[str, list] = {}
        for kind in ("loao", "loao_hard"):
            obj = build(kind)
            allf[kind] = obj["folds"]
        p = save({"kinds": allf, "note": "按 Area 留一域；用于跨域泛化评测"}, args.out)
        print(f"[loao] 已写入 {p}")
        for k, v in allf.items():
            print(f"  {k}: {len(v)} 折  " + ", ".join(f["name"] for f in v))


if __name__ == "__main__":
    main()
