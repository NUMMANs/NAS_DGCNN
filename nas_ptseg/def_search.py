"""fitness 定义搜索：哪种聚合口径的排序对**训练预算最不敏感**？

背景（为什么非做不可）
================================================================
实测 `ρ(cheap@25, full@300) = −0.825`：短预算的 k 排序与收敛后**反向**。
原因是预算改变时，各候选的**绝对水平**变化幅度不同（都在涨，但涨的斜率不同），
于是"按绝对 mIoU 排名"在不同预算下得出不同结论。

但注意：**这不一定是代理的错，可能是 fitness 定义选错了**。
如果我们换一个**对预算不敏感**的标量（例如"相对该预算下最优 k 的后悔值"），
短预算的排序有可能一下子就与长预算一致——这比多训几十分钟便宜得多。

被比较的口径（全部由同一张表离线算出，零额外训练成本）
================================================================
绝对分（对预算敏感）：
    mean          平均房间 mIoU（现用口径）
    median        中位房间
    worst         min 房间（现 MAP-Elites 特征之一）
    mean_std      mean − 1.0·std_across_rooms
    mean_hstd     mean − 0.5·std
    q25           25 分位

**相对分（对预算归一化，重点假设）**：
    neg_regret         − mean_v [ best(v) − score(v) ]        绝对后悔值取负
    neg_nregret        − mean_v [ (best(v) − score(v)) / best(v) ]   归一化后悔值取负
    其中 best(v) 是该**预算下**对房间 v 最好的那个 k 所得到的分数
    （即逐房间 oracle）。把"房间难度随预算的变化"整体扣除后，
    剩下的才是"这个策略挑 k 挑得好不好"。

判据
================================================================
对每个 (口径 g, 预算 E)：
    ρ( g@E , g@full )        ← 我们想最大化它
同时报告：
    ρ( g@full , mean@full )  ← **目标本身是否被口径改变**。
    若某个 g 改变了 full 的排序，那它不是一个"更好的度量"，而是一个**新目标**，
    必须另外论证它值得优化（例如它就是论文要的 worst-domain 鲁棒性）。
"""

from __future__ import annotations

import argparse
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
    from nas_ptseg import reliability as REL

    _PKG_MODE = False
else:
    from . import gpu as GPU  # noqa: F401
    from . import data as DT
    from . import room_split as RS
    from . import fitness as FZ
    from . import reliability as REL

    _PKG_MODE = True


AGG_ABSOLUTE = ("mean", "median", "worst", "mean_std", "mean_hstd", "q25")
AGG_RELATIVE = ("neg_regret", "neg_nregret")
ALL_AGG = AGG_ABSOLUTE + AGG_RELATIVE


# ---------------------------------------------------------------------------
# 建"每房间 × 每 k"的表（用于算 oracle / regret）
# ---------------------------------------------------------------------------
def per_room_k_matrix(ft: FZ.FitnessTable,
                      seeds: Optional[Sequence[int]] = None) -> Tuple[List[str], Dict[str, Dict[int, float]]]:
    """返回 {val_room: {k: score}}，score 已对 (训练房间 × seed) 平均。

    这是"逐房间 oracle"的基础：`best(v) = max_k matrix[v][k]`。
    """
    seeds = list(seeds or ft.seeds)
    mat: Dict[str, Dict[int, float]] = {v: {} for v in ft.val_rooms}
    for v in ft.val_rooms:
        for k in ft.ks:
            acc: List[float] = []
            for s in seeds:
                for tr in ft.train_rooms:
                    c = ft.cell(tr, k, s)
                    if c and v in c:
                        acc.append(float(c[v]))
            if acc:
                mat[v][k] = float(np.mean(acc))
    return list(ft.val_rooms), mat


def policy_per_room(policy, val_stats: Dict[str, dict],
                    mat: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """策略逐验证房间的得分（k 由该房间统计决定）。缺格则跳过该房间。"""
    out: Dict[str, float] = {}
    for v, st in val_stats.items():
        if v not in mat:
            continue
        k = FZ.clip_to_grid(policy(st))
        if k in mat[v]:
            out[v] = mat[v][k]
    return out


def aggregate(agg: str, per_room: Dict[str, float],
              mat: Dict[str, Dict[int, float]]) -> Optional[float]:
    """把一个策略的逐房间得分聚合成标量。"""
    if not per_room:
        return None
    vals = np.asarray(list(per_room.values()), float)
    if agg == "mean":
        return float(vals.mean())
    if agg == "median":
        return float(np.median(vals))
    if agg == "worst":
        return float(vals.min())
    if agg == "mean_std":
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return float(vals.mean() - sd)
    if agg == "mean_hstd":
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return float(vals.mean() - 0.5 * sd)
    if agg == "q25":
        return float(np.percentile(vals, 25))
    if agg in ("neg_regret", "neg_nregret"):
        reg, nreg = [], []
        for v, s in per_room.items():
            ks = mat.get(v) or {}
            if not ks:
                continue
            best = max(ks.values())
            reg.append(best - s)
            if best > 1e-9:
                nreg.append((best - s) / best)
        if not reg:
            return None
        return float(-np.mean(nreg if agg == "neg_nregret" else reg))
    raise ValueError(agg)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def detect_budgets(table_dir: str, seeds: Sequence[int], role: str = "cheap") -> List[int]:
    """扫出目录里建完的 cheap 预算（cells 数 = 房间 × k × seed）。"""
    out: List[int] = []
    s = "-".join(str(int(x)) for x in seeds)
    prefix = f"{role}_ep"
    for name in os.listdir(table_dir):
        if not (name.startswith(prefix) and name.endswith(f"_s{s}.json")):
            continue
        try:
            e = int(name[len(prefix):-len(f"_s{s}.json")])
        except ValueError:
            continue
        try:
            with open(os.path.join(table_dir, name), "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        need = len(d.get("train_rooms", [])) * len(d.get("ks", [])) * len(d.get("seeds", []))
        if need and len(d.get("cells", {})) >= need:
            out.append(e)
    return sorted(set(out))


def run(budgets: Sequence[int], target: int, seeds: Sequence[int],
        table_dir: str, out_path: str = "", log=print) -> dict:
    fp = os.path.join(table_dir, f"full_ep{int(target)}_s{'-'.join(str(int(s)) for s in seeds)}.json")
    if not os.path.exists(fp):
        log(f"[def] 目标表不存在：{fp}")
        return {}
    with open(fp, "r", encoding="utf-8") as fh:
        full_t = json.load(fh)
    ft_full = FZ.FitnessTable(full_t)

    bt: Dict[int, dict] = {}
    for e in budgets:
        p = os.path.join(table_dir, f"cheap_ep{int(e)}_s{'-'.join(str(int(s)) for s in seeds)}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                bt[int(e)] = json.load(fh)
    if not bt:
        log("[def] 没有可用的 cheap 表。")
        return {}

    # 验证房间统计（策略输入），用 target_n 一致
    base = bt[sorted(bt)[0]]
    tgt_n = int(base.get("target_n", RS.TARGET_N))
    val_stats: Dict[str, dict] = {}
    for key in ft_full.val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=tgt_n)
        val_stats[key] = st
    rho_arr = np.asarray([val_stats[k]["rho_med"] for k in val_stats], float)
    L_arr = np.asarray([val_stats[k]["extent_max_m"] for k in val_stats], float)
    rho_ref = float(np.exp(np.mean(np.log(np.maximum(rho_arr, 1e-9)))))
    L_ref = float(np.median(L_arr))

    cands = REL.dedupe_candidates(REL.build_candidates(rho_ref, L_ref), val_stats)
    names = [c[0] for c in cands]
    log("=" * 96)
    log("fitness 定义搜索：哪种聚合口径的排序对训练预算最不敏感？")
    log("=" * 96)
    log(f"  可用预算   : {sorted(bt)}   （目标 full@{target}）")
    log(f"  候选策略   : {len(names)} 个")

    # 逐预算建 per-room 矩阵
    mats: Dict[int, Dict[str, Dict[int, float]]] = {}
    for e, t in bt.items():
        _v, m = per_room_k_matrix(FZ.FitnessTable(t))
        mats[e] = m
    _v, mat_full = per_room_k_matrix(ft_full)

    # 逐 (预算, 口径) 得到候选得分向量
    def score_vector(mat, agg: str) -> np.ndarray:
        out = []
        for _n, fn, _sig in cands:
            pr = policy_per_room(fn, val_stats, mat)
            a = aggregate(agg, pr, mat)
            out.append(float("nan") if a is None else a)
        return np.asarray(out, float)

    target_vecs: Dict[str, np.ndarray] = {g: score_vector(mat_full, g) for g in ALL_AGG}
    base_target = target_vecs["mean"]

    log("")
    log("  ① 目标本身是否被口径改变？（ρ( g@full , mean@full )）")
    log(f"  {'口径':<14s}{'ρ vs mean@full':>16s}{'说明':>34s}")
    for g in ALL_AGG:
        rho = REL.spearman(target_vecs[g], base_target)
        note = ""
        if np.isfinite(rho) and rho < 0.85:
            note = "⚠ 改变了目标（是新度量而非更优度量）"
        log(f"  {g:<14s}{rho:>+16.3f}{note:>34s}")

    log("")
    log("  ② 各口径 × 各预算 与 full@{} 的排序一致性（越大越好）".format(target))
    hdr = f"  {'口径':<14s}" + "".join(f"{'E=' + str(e):>11s}" for e in sorted(bt))
    log(hdr)
    table: Dict[str, Dict[str, float]] = {}
    for g in ALL_AGG:
        row = {}
        line = f"  {g:<14s}"
        for e in sorted(bt):
            v = score_vector(mats[e], g)
            rho = REL.spearman(v, target_vecs[g])      # 同口径跨预算比
            row[str(e)] = rho
            line += f"{rho:>+11.3f}"
        table[g] = row
        log(line)

    log("")
    log("  ③ 附带：Top-3 召回（同口径跨预算，看能不能挑出最好的几个）")
    log(f"  {'口径':<14s}" + "".join(f"{'E=' + str(e):>11s}" for e in sorted(bt)))
    for g in ALL_AGG:
        line = f"  {g:<14s}"
        for e in sorted(bt):
            v = score_vector(mats[e], g)
            line += f"{REL.topk_recall(v, target_vecs[g], 3)['recall']*100:>10.0f}%"
        log(line)

    # 最优组合
    flat = [(g, int(e), r) for g, r in table.items() for e, r in r.items()
            if np.isfinite(r)]
    best = max(flat, key=lambda z: z[2]) if flat else None
    if best:
        log("")
        log(f"  → 最佳组合：口径 **{best[0]}** @ 预算 {best[1]}，ρ = {best[2]:+.3f}")
        cands_g = [(g, e, r) for (g, e, r) in flat if r >= 0.6]
        if cands_g:
            log(f"    达门槛（ρ>=0.6）的组合 {len(cands_g)} 个：")
            for g, e, r in sorted(cands_g, key=lambda z: -z[2])[:8]:
                log(f"      {g:<14s} E={e:<5d} ρ={r:+.3f}")
        else:
            log("     ⚠ 没有任何组合达到 ρ>=0.6。")

    res = {
        "budgets": sorted(bt),
        "target": int(target),
        "table": table,
        "best": {"agg": best[0], "budget": best[1], "rho": best[2]} if best else None,
        "target_self_consistency": {g: REL.spearman(target_vecs[g], base_target) for g in ALL_AGG},
    }
    if out_path:
        p = out_path if os.path.isabs(out_path) else os.path.join(DT.PROJECT_ROOT, out_path)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"  结果已写入 {p}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="fitness 定义搜索（预算鲁棒性）")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="要比较的 cheap 预算；默认自动探测目录里已建完的")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--table-dir", default=FZ.TABLE_DIR)
    ap.add_argument("--out", default="")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    budgets = args.budgets or detect_budgets(args.table_dir, args.seeds)
    run(budgets, args.target, args.seeds, args.table_dir, out_path=args.out, log=log)


if __name__ == "__main__":
    main()
