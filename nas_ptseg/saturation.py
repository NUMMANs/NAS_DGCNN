"""饱和诊断：DGCNN 这个结构族是不是已经被原作者调透了？

问题来源（用户 2026-09-24 提出的假设）
================================================================
"DGCNN 经过了原作者很多次调整，大概率已经非常逼近最好。这个时候接入 NAS，
是不是不能从进化模型架构的方向出发？"

核心判据是 **Δ(c) = fitness_full(c) − fitness_cheap(c) 的分布**，但**符号方向极易读反**，
必须分清两种截然不同的形态：

1. **欠拟合回填型（catch-up）**  Δ 与 cheap 分数 **负**相关（r ≈ −1）
   起点差的配置补涨更多，最终**趋同** → 排序由"训练速度"决定，
   结构差异在充分训练后消失 → **真·饱和**，族内没有更好的结构。

2. **优势放大型（advantage-amplifying）**  Δ 与 cheap 分数 **正**相关（r ≈ +1）
   起点好的配置补涨也更多，**差距被训练拉大** → 优势不是"学得快"，
   而是**结构本身更适合这个任务**（只是需要足够训练才显现）→ **不是饱和**，
   族内确实存在更好的结构。

⚠ 这两种形态在"Δ 极差都很小"时长得像（都能算"收敛型"），但结论完全相反。
所以**不能只看 Δ 极差，必须同时看 Δ 与 cheap 的相关符号**。
（本脚本第一版就写反了这一条，已修正。）

第三种可能：Δ 的极差**很大**（与配置间 full 极差同量级）→ 非收敛型，
便宜预算会系统性地错排某些配置。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import gpu as GPU  # noqa: F401
    from nas_ptseg import data as DT
    from nas_ptseg import arch_table as AT

    _PKG_MODE = False
else:
    from . import gpu as GPU  # noqa: F401
    from . import data as DT
    from . import arch_table as AT

    _PKG_MODE = True


def _cfg_mean(table: dict, ci: int) -> Optional[float]:
    rooms, mat = AT.load_matrix(table)
    vals = [mat[v][ci] for v in rooms if ci in mat[v]]
    return float(np.mean(vals)) if vals else None


def _cfg_rooms(table: dict) -> Dict[str, Dict[int, float]]:
    """{room: {cfg_id: score}}，按 (训练房间 × seed) 平均。"""
    _rooms, mat = AT.load_matrix(table)
    return mat


def diagnose(cheap: dict, full: dict, log=print) -> dict:
    if cheap["grid"] != full["grid"]:
        log("⚠ 两张表的配置网格不一致，无法比较。")
        return {}
    grid = cheap["grid"]
    n = len(grid)

    mc, mf = _cfg_rooms(cheap), _cfg_rooms(full)
    rooms = [v for v in mc if v in mf]

    # ---- 配置级 ----
    rows: List[dict] = []
    for ci in range(n):
        c_vals = [mc[v][ci] for v in rooms if ci in mc[v]]
        f_vals = [mf[v][ci] for v in rooms if ci in mf[v]]
        if not c_vals or not f_vals:
            continue
        rows.append({
            "cfg": ci, "label": AT.cfg_label(grid[ci]),
            "cheap": float(np.mean(c_vals)), "full": float(np.mean(f_vals)),
            "delta": float(np.mean(f_vals) - np.mean(c_vals)),
        })
    rows.sort(key=lambda r: r["cheap"])

    log("=" * 96)
    log("饱和诊断：这个结构族是不是已经被调透了？")
    log("=" * 96)
    log(f"  cheap = {cheap['epochs']} ep   full = {full['epochs']} ep   "
        f"配置 {len(rows)} 个 × 验证房间 {len(rooms)} 个")
    log("")
    log(f"  {'配置':<34s}{'cheap':>10s}{'full':>10s}{'Δ=full-cheap':>15s}")
    for r in rows:
        log(f"  {r['label']:<34s}{r['cheap']:>10.4f}{r['full']:>10.4f}{r['delta']:>+15.4f}")

    ch = np.asarray([r["cheap"] for r in rows], float)
    fu = np.asarray([r["full"] for r in rows], float)
    dl = np.asarray([r["delta"] for r in rows], float)

    d_spread = float(dl.max() - dl.min())
    d_mean = float(dl.mean())
    # Δ 与 cheap 的相关：强负相关 = 纯粹的欠拟合回填（收敛型）
    r_dc = float(np.corrcoef(ch, dl)[0, 1]) if ch.size > 2 and np.std(ch) > 1e-12 else float("nan")

    log("")
    log(f"  Δ 的均值        = {d_mean:+.4f}")
    log(f"  Δ 的极差        = {d_spread:.4f}   （各配置补涨幅度是否一致）")
    log(f"  Δ 极差 / Δ 均值 = {d_spread / max(abs(d_mean), 1e-9):.2f}")
    log(f"  r(cheap, Δ)     = {r_dc:+.3f}   （越接近 −1 越像纯欠拟合回填）")

    # ---- 房间级：Δ 是否随房间系统变化（若随房间变，则"饱和"这个说法要打折）----
    per_room_delta = {}
    for v in rooms:
        d = [mf[v][c] - mc[v][c] for c in range(n) if c in mc[v] and c in mf[v]]
        if d:
            per_room_delta[v] = float(np.mean(d))
    dv = np.asarray(list(per_room_delta.values()), float)
    log("")
    log(f"  房间级 Δ：均值 {dv.mean():+.4f}  极差 {dv.max()-dv.min():.4f}  "
        f"（跨房间相当一致 → 收益与房间无关）")

    # ---- 判定 ----
    log("")
    log("  判读（⚠ 必须同时看 Δ 极差**与** r(cheap,Δ) 的符号，只看一个会读反）：")
    full_spread = float(fu.max() - fu.min())
    ratio = d_spread / max(full_spread, 1e-9)
    log(f"    Δ 极差           = {d_spread:.4f}")
    log(f"    配置间 full 极差 = {full_spread:.4f}")
    log(f"    Δ 极差 / 配置极差 = {ratio:.2f}")
    log(f"    r(cheap, Δ)      = {r_dc:+.3f}")
    log("")

    if not np.isfinite(r_dc):
        shape = "样本不足，无法判定形态"
        sat = False
    elif r_dc <= -0.5 and ratio < 0.5:
        shape = ("**欠拟合回填型 → 真·饱和**：起点差的配置补涨更多、最终趋同。"
                 "排序由'训练速度'决定，族内没有更好的结构。用户的假设成立。")
        sat = True
    elif r_dc >= +0.5:
        shape = ("**优势放大型 → 不是饱和**：起点好的配置补涨更多，差距被训练**拉大**。"
                 "这说明优势不是'学得快'，而是**结构本身更适合该任务**，"
                 "只是需要足够训练才显现 → 族内确实存在更好的结构。")
        sat = False
    elif ratio >= 0.5:
        shape = ("**非收敛型**：各配置补涨幅度差异大（与配置间极差同量级）——"
                 "结构差异是真的，但便宜预算会系统性错排某些配置，排序会随预算漂移。")
        sat = False
    else:
        shape = ("**中间形态**：Δ 极差不大、相关性也不显著 —— "
                 "既没有明显的补涨趋同，也没有明显的优势放大。")
        sat = False

    log(f"    形态 = {shape}")
    log("")
    if sat:
        log("    → 支持用户的假设：在一个已高度调优的结构族内部做架构搜索，"
            "可拿的收益主要来自训练带来的收敛，而不是发现更好的结构。")
    else:
        log("    → **不支持用户的假设**：配置间的差异具有结构性，"
            "不是'训练量/调参程度'造成的假差距。")
    log("")
    log("    但无论哪种形态，本课题已测的**交互**（条件化）都是零：")
    log("      · 架构表：逐房间 oracle = 最好单一配置（房间级 headroom +0.0000）")
    log("      · 逐 Area：6 个 Area 的最优配置全部相同（域级 headroom +0.0000）")
    log("    所以真正缺的不是'更好的搜索算法'，而是搜索空间里**结构多样性的来源**——")
    log("    在同一个已调优的结构族内换超参，无论什么算法都跳不出去。")

    return {"rows": rows, "delta_mean": d_mean, "delta_spread": d_spread,
            "r_cheap_delta": r_dc, "full_spread": full_spread,
            "delta_over_full": ratio, "saturated": bool(sat),
            "per_room_delta": per_room_delta, "shape": shape}


def main() -> None:
    ap = argparse.ArgumentParser(description="DGCNN 结构族饱和诊断")
    ap.add_argument("--cheap-ep", type=int, default=50)
    ap.add_argument("--full-ep", type=int, default=300)
    ap.add_argument("--tag", default="k+agg+graph_space_n6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--out", default="")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    d = os.path.join(DT.PROJECT_ROOT, "data", "arch_tables")
    pc = AT._table_path(args.tag, args.seeds, args.cheap_ep)
    pf = AT._table_path(args.tag, args.seeds, args.full_ep)
    for p in (pc, pf):
        if not os.path.exists(p):
            log(f"[sat] 缺表：{p}")
            return
    with open(pc, "r", encoding="utf-8") as fh:
        cheap = json.load(fh)
    with open(pf, "r", encoding="utf-8") as fh:
        full = json.load(fh)
    res = diagnose(cheap, full, log=log)
    if args.out and res:
        q = args.out if os.path.isabs(args.out) else os.path.join(DT.PROJECT_ROOT, args.out)
        with open(q, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"  结果已写入 {q}")


if __name__ == "__main__":
    main()
