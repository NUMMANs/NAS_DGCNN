"""搜索空间上限探针：在动手搜索之前，先量清"最多能拿到多少"。

为什么必须做
================================================================
"再跑一轮进化"看起来总比"先算上限"更有进展感，但如果**可提升空间本身小于噪声**，
那轮进化只会产出一个漂亮的过拟合结果。本项目已多次在"指标定义/对照设计的偏差"
上把噪声读成发现，所以这次先算三件事：

    ① 最好固定架构        best const k
    ② 逐房间 oracle       每个房间各自挑最优 k（**6 值策略的理论上限**）
    ③ 最好阈值策略        在单一房间统计量上做 1~2 段阈值切分，穷举最优
       —— ⚠ 注意这是**在同样这 12 个验证房间上选择并评估**（in-sample），
       因此它是**乐观上界**而不是可报告的性能。用乐观上界来否证恰好最有力：
       连作弊都只能拿到这么多，那真去搜索只会更少。

同时给出 seed 噪声（σ）作参照：若 ②−① 或 ③−① 不到 3σ，则**这个空间不值得搜**，
应先扩大搜索维度（半径 / sample_rate / 图更新 / 池化），而不是继续在 k 上进化。

结论会直接决定下一步投算力的方向，所以它比多跑一轮进化更值钱。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import gpu as GPU  # noqa: F401
    from nas_ptseg import data as DT
    from nas_ptseg import room_split as RS
    from nas_ptseg import fitness as FZ

    _PKG_MODE = False
else:
    from . import gpu as GPU  # noqa: F401
    from . import data as DT
    from . import room_split as RS
    from . import fitness as FZ

    _PKG_MODE = True


STATS = ("rho_med", "extent_max_m", "n_points", "n_classes_present")


def load_matrix(table: dict) -> Tuple[List[str], Dict[str, Dict[int, float]]]:
    """{val_room: {k: mIoU}}（对 训练房间 × seed 平均）。"""
    ft = FZ.FitnessTable(table)
    mat: Dict[str, Dict[int, float]] = {v: {} for v in ft.val_rooms}
    for v in ft.val_rooms:
        for k in ft.ks:
            acc = []
            for s in ft.seeds:
                for tr in ft.train_rooms:
                    c = ft.cell(tr, k, s)
                    if c and v in c:
                        acc.append(float(c[v]))
            if acc:
                mat[v][k] = float(np.mean(acc))
    return list(ft.val_rooms), mat


def const_k_curve(mat: Dict[str, Dict[int, float]],
                  rooms: Sequence[str]) -> Dict[int, float]:
    ks = sorted({k for v in rooms for k in mat.get(v, {})})
    return {k: float(np.mean([mat[v][k] for v in rooms if k in mat[v]])) for k in ks}


def per_room_oracle(mat: Dict[str, Dict[int, float]], rooms: Sequence[str]) -> float:
    return float(np.mean([max(mat[v].values()) for v in rooms if mat.get(v)]))


def seed_sigma(table: dict, rooms: Sequence[str]) -> Optional[float]:
    """估计"同一配置换种子重训"带来的 σ。

    做法：对每个 (验证房间, k)，算两个 seed 下（对训练房间平均的）分数之差 d。
    两个 seed 是独立重训 → var(d) ≈ 2σ²，故 σ ≈ std(d)/√2。
    这样得到的 σ 与"一个策略的 fitness 在换 seed 时的抖动"同量级，可直接与
    headroom 比较。**不要**用"对全部 k 平均"的差值去估（那会把 k 之间的真实
    差异当成噪声，严重高估 σ）。
    """
    ft = FZ.FitnessTable(table)
    if len(ft.seeds) < 2:
        return None
    s0, s1 = ft.seeds[0], ft.seeds[1]
    diffs: List[float] = []
    for v in rooms:
        for k in ft.ks:
            a, b = [], []
            for tr in ft.train_rooms:
                c0 = ft.cell(tr, k, s0)
                c1 = ft.cell(tr, k, s1)
                if c0 and v in c0:
                    a.append(float(c0[v]))
                if c1 and v in c1:
                    b.append(float(c1[v]))
            if a and b:
                diffs.append(np.mean(a) - np.mean(b))
    if len(diffs) < 2:
        return None
    return float(np.std(diffs, ddof=1) / np.sqrt(2.0))


def _score_assign(mat, rooms, assign) -> float:
    vals = [mat[v][assign[v]] for v in rooms if assign[v] in mat.get(v, {})]
    return float(np.mean(vals)) if vals else float("nan")


def best_threshold_policy(mat: Dict[str, Dict[int, float]], val_stats: Dict[str, dict],
                          rooms: Sequence[str], ks: Sequence[int],
                          n_levels: int = 2) -> dict:
    """在**单一统计量**上穷举"阈值切分 → 分派 k"的最优策略。

    n_levels=2：一段阈值、两种 k。
    n_levels=3：两段阈值、三种 k。只有 12 个房间，3 段已接近插值，
    所以 2 段更可信、3 段只作上界参考。
    """
    best: Optional[dict] = None
    for stat in STATS:
        vals = np.asarray([val_stats[v][stat] for v in rooms], float)
        srt = np.sort(vals)
        cuts = [float(0.5 * (srt[i] + srt[i + 1])) for i in range(srt.size - 1)]
        if n_levels == 2:
            for cut in cuts:
                for k_lo in ks:
                    for k_hi in ks:
                        assign = {v: (k_lo if val_stats[v][stat] <= cut else k_hi) for v in rooms}
                        sc = _score_assign(mat, rooms, assign)
                        if not np.isfinite(sc):
                            continue
                        if best is None or sc > best["score"]:
                            best = {"stat": stat, "levels": 2, "score": sc,
                                    "params": {"k_low": k_lo, "cut": cut, "k_high": k_hi}}
        else:
            for i in range(len(cuts) - 1):
                lo, hi = cuts[i], cuts[i + 1]
                for k1 in ks:
                    for k2 in ks:
                        for k3 in ks:
                            assign = {}
                            for v in rooms:
                                x = val_stats[v][stat]
                                assign[v] = k1 if x <= lo else (k2 if x <= hi else k3)
                            sc = _score_assign(mat, rooms, assign)
                            if not np.isfinite(sc):
                                continue
                            if best is None or sc > best["score"]:
                                best = {"stat": stat, "levels": 3, "score": sc,
                                        "params": {"k_low": k1, "cuts": [lo, hi],
                                                   "k_mid": k2, "k_high": k3}}
    return best or {}


def run(table_path: str, log=print) -> dict:
    with open(table_path, "r", encoding="utf-8") as fh:
        table = json.load(fh)
    rooms, mat = load_matrix(table)
    tgt_n = int(table.get("target_n", RS.TARGET_N))

    val_stats: Dict[str, dict] = {}
    for key in rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=tgt_n)
        val_stats[key] = st

    ks = sorted({k for v in rooms for k in mat.get(v, {})})
    log("=" * 92)
    log("搜索空间上限探针：k 轴还能拿到多少？")
    log("=" * 92)
    log(f"  表          : {os.path.basename(table_path)}  (epochs={table.get('epochs')})")
    log(f"  验证房间    : {len(rooms)} 个     k 网格: {ks}")

    curve = const_k_curve(mat, rooms)
    log("")
    log("  固定 k 曲线（全部验证房间平均）：")
    for k, v in curve.items():
        log(f"    k={k:>3d}   {v:.4f}")
    best_k = max(curve, key=lambda k: curve[k])
    floor = curve[best_k]
    log(f"  → 最好固定架构   : k={best_k}  fitness={floor:.4f}   ← 基线①")

    orc = per_room_oracle(mat, rooms)
    log(f"  → 逐房间 oracle  : fitness={orc:.4f}   上限②")
    log(f"     headroom(②−①) = {orc - floor:+.4f}")

    log("")
    best2 = best_threshold_policy(mat, val_stats, rooms, ks, n_levels=2)
    best3 = best_threshold_policy(mat, val_stats, rooms, ks, n_levels=3)
    for nm, b in (("2 段阈值", best2), ("3 段阈值", best3)):
        if not b:
            continue
        p = b["params"]
        log(f"  → 最好{nm}策略 : stat={b['stat']}  {p}  fitness={b['score']:.4f}"
            f"   ← ③（headroom {b['score']-floor:+.4f}）")

    sigma = seed_sigma(table, rooms)
    sigma_fit = (sigma / np.sqrt(max(len(rooms), 1))) if sigma is not None else None
    log("")
    if sigma is None or sigma_fit is None:
        log("  seed 噪声 σ = 不可估（表只有 1 个 seed）")
    else:
        log(f"  seed 噪声（单房间，k 固定）  σ_room ≈ {sigma:.4f}")
        log(f"  seed 噪声（fitness 级，{len(rooms)} 房间平均） σ_fit ≈ {sigma_fit:.4f}"
            f"   ← headroom 要与它比")
        for nm, val in (("③2段", best2.get("score")), ("③3段", best3.get("score"))):
            if val is None:
                continue
            d = val - floor
            log(f"     {nm} − ① = {d:+.4f} = {d/max(sigma_fit,1e-9):.1f}σ_fit"
                f"   {'✅ 值得搜' if d > 3*sigma_fit else '⚠ 不足 3σ，不值得在这个空间上搜'}")

    log("")
    verdict = _verdict(floor, orc, best2.get("score"), best3.get("score"), sigma_fit)
    log(f"  判定：{verdict}")

    return {"table": os.path.basename(table_path), "ks": ks, "const_curve": curve,
            "best_const": best_k, "floor": floor, "oracle": orc,
            "best2": best2, "best3": best3, "sigma": sigma, "verdict": verdict}


def _verdict(floor: float, orc: float, s2: Optional[float], s3: Optional[float],
             sigma: Optional[float]) -> str:
    gains = [g for g in ((s2 - floor) if s2 is not None else None,
                         (s3 - floor) if s3 is not None else None,
                         orc - floor) if g is not None]
    if not gains:
        return "数据不足"
    top = max(gains)
    if sigma is None:
        return f"headroom 最大 {top:+.4f}，但噪声不可估 → 先加 seed 再判"
    if top < 1.0 * sigma:
        return (f"**k 轴已榨干**：全部 headroom {top:+.4f} 不到 1σ({sigma:.4f}) → "
                f"不要再在 k 上搜索，应扩大搜索维度。")
    if top < 3.0 * sigma:
        return (f"**k 轴空间可疑**：headroom {top:+.4f} 不到 3σ({sigma:.4f}) → "
                f"直接搜索很可能只得到噪声级收益；优先扩维度，或先降 σ。")
    return (f"k 轴仍有可搜空间：headroom {top:+.4f} > 3σ({sigma:.4f}) → "
            f"值得在这个空间上跑一轮进化。")


def main() -> None:
    ap = argparse.ArgumentParser(description="搜索空间上限探针")
    ap.add_argument("--table", default="", help="fitness 表路径（默认 full@300 val 表）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    p = args.table or FZ._table_path("full", args.epochs or FZ.EPOCHS_FULL,
                                     args.seeds, args.tag)
    res = run(p, log=log)
    if args.out:
        q = args.out if os.path.isabs(args.out) else os.path.join(DT.PROJECT_ROOT, args.out)
        with open(q, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"  结果已写入 {q}")


if __name__ == "__main__":
    main()
