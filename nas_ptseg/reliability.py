"""排序可靠度验证：`fitness_cheap` 能否预测 `fitness_full`？

这是用户裁定的**唯一核心问题**，也是"搜索是否有意义"的判据：
    若 cheap 的排序预测不了 full 的排序 → 继续进化没有意义；
    若预测得了 → 才允许扩大搜索空间与预算。

四件事（按重要性排序）
================================================================
1. **Spearman ρ(cheap, full)** —— 我们要的答案。
2. **Spearman ρ(full_A, full_B)** —— **可靠性天花板**。
   full 用不同 init seed 独立重训两次，若它连自己都排不一致，那么"cheap 能否
   预测 full"这个问题本身没有意义（没有可预测的目标）。这一步必须先看。
3. **Top-k 召回** —— 搜索真正关心的是"能不能找出最好的那几个"。
   报告 full 的前 k 名有多少落在 cheap 的前 k 名里。
4. **分辨率** —— 两张表各自的 (k 效应极差 / seed 噪声)。信噪比不足时排序必不稳定，
   此时低 ρ 不能归咎于 cheap，而是数据本身分不开。

候选策略集
================================================================
用**确定性网格**生成（不依赖进化），先把"排序是否可靠"这件事单独量出来：
    - 固定 k：K_GRID 全部
    - 密度幂律 k = k0·(ρ/ρ_ref)^α：k0 × α 网格
    - 尺度幂律 k = k0·(L/L_ref)^β：k0 × β 网格
按"每个验证房间实际选到的 k"去重（同一 k 分配 = 同一个策略，fitness 必然相同）。
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
    from nas_ptseg import gpu as GPU  # noqa: F401  （导入即钉 GPU；本模块不训练但保持一致）
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


# ---------------------------------------------------------------------------
# 候选策略集
# ---------------------------------------------------------------------------
def build_candidates(rho_ref: float, L_ref: float) -> List[Tuple[str, FZ.PolicyFn]]:
    """确定性候选网格。返回 [(name, policy_fn), ...]，已按 k 分配签名去重。"""
    cands: List[Tuple[str, FZ.PolicyFn]] = []

    for k in FZ.K_GRID:
        cands.append((f"const_k{k}", FZ.policy_const(k)))

    for k0 in (16, 32, 48):
        for a in (-0.5, -0.25, 0.25, 0.5):
            cands.append((f"rho_k{k0}_a{a:+.2f}", FZ.policy_power(k0, a, rho_ref)))

    for k0 in (24, 40):
        for b in (-1.0, -0.5, 0.5, 1.0):
            cands.append((f"L_k{k0}_b{b:+.2f}", FZ.policy_scale(k0, b, L_ref)))

    # 按"在验证房间上的 k 分配"去重
    return cands


def dedupe_candidates(
    cands: Sequence[Tuple[str, FZ.PolicyFn]], val_stats: Dict[str, dict]
) -> List[Tuple[str, FZ.PolicyFn, Tuple[int, ...]]]:
    seen: Dict[Tuple[int, ...], str] = {}
    out: List[Tuple[str, FZ.PolicyFn, Tuple[int, ...]]] = []
    for name, fn in cands:
        sig = tuple(FZ.clip_to_grid(fn(val_stats[v])) for v in sorted(val_stats))
        if sig in seen:
            continue
        seen[sig] = name
        out.append((name, fn, sig))
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _rank(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），用于手算 Spearman（避免强依赖 scipy）。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # 处理并列
    s = a[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    sa, sb = np.std(ra), np.std(rb)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = a.size
    if n < 3:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = np.sign(da) * np.sign(db)
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return float((conc - disc) / tot) if tot else float("nan")


def topk_recall(cheap: Sequence[float], full: Sequence[float], k: int) -> dict:
    """full 的前 k 名有多少落在 cheap 的前 k 名里。"""
    n = len(cheap)
    k = min(k, n)
    if k <= 0:
        return {"k": 0, "hit": 0, "recall": float("nan")}
    cf = np.asarray(cheap, float)
    ff = np.asarray(full, float)
    top_full = set(np.argsort(-ff)[:k].tolist())
    top_cheap = set(np.argsort(-cf)[:k].tolist())
    hit = len(top_full & top_cheap)
    return {"k": k, "hit": hit, "recall": hit / k}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def analyse(
    cheap_table: dict,
    full_table: dict,
    log=print,
) -> dict:
    ft_c = FZ.FitnessTable(cheap_table)
    ft_f = FZ.FitnessTable(full_table)

    # 验证房间统计（策略输入）
    val_stats: Dict[str, dict] = {}
    for key in ft_c.val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=cheap_table.get("target_n", RS.TARGET_N))
        val_stats[key] = st

    # rho_ref / L_ref：用验证房间的中位（仅作策略的参考点）
    rho_arr = np.asarray([val_stats[k]["rho_med"] for k in val_stats], float)
    L_arr = np.asarray([val_stats[k]["extent_max_m"] for k in val_stats], float)
    rho_ref = float(np.exp(np.mean(np.log(np.maximum(rho_arr, 1e-9)))))
    L_ref = float(np.median(L_arr))

    log("=" * 88)
    log("排序可靠度验证：fitness_cheap 能否预测 fitness_full？")
    log("=" * 88)
    log(f"  cheap 表: {cheap_table['role']} epochs={cheap_table['epochs']} "
        f"seeds={cheap_table['seeds']} cells={len(cheap_table['cells'])}")
    log(f"  full  表: {full_table['role']} epochs={full_table['epochs']} "
        f"seeds={full_table['seeds']} cells={len(full_table['cells'])}")
    log(f"  验证房间 {len(val_stats)} 个；rho_ref={rho_ref:.1f}, L_ref={L_ref:.2f}m")

    cands = dedupe_candidates(build_candidates(rho_ref, L_ref), val_stats)
    log(f"  候选策略：{len(cands)} 个（已按 k 分配去重）")

    # ---- 逐候选打分 ----
    rows: List[dict] = []
    for name, fn, sig in cands:
        rc = ft_c.score(fn, val_stats)
        rf = ft_f.score(fn, val_stats)
        # 单 seed 版本（用于算"独立重训"的天花板）
        c_by_seed = {s: ft_c.score(fn, val_stats, seeds=[s])["fitness"]
                     for s in ft_c.seeds}
        f_by_seed = {s: ft_f.score(fn, val_stats, seeds=[s])["fitness"]
                     for s in ft_f.seeds}
        rows.append({
            "name": name,
            "sig": sig,
            "cheap": rc["fitness"],
            "full": rf["fitness"],
            "cheap_worst": rc["worst"],
            "full_worst": rf["worst"],
            "cheap_by_seed": c_by_seed,
            "full_by_seed": f_by_seed,
        })

    c = np.asarray([r["cheap"] for r in rows], float)
    f = np.asarray([r["full"] for r in rows], float)

    log("")
    log(f"  {'候选策略':<20s}{'cheap':>10s}{'full':>10s}{'Δ(full-cheap)':>15s}")
    for r in sorted(rows, key=lambda z: -z["full"]):
        log(f"  {r['name']:<20s}{r['cheap']:>10.4f}{r['full']:>10.4f}"
            f"{r['full']-r['cheap']:>+15.4f}")

    # ---- 1) 我们要的答案 ----
    rho_cf = spearman(c, f)
    tau_cf = kendall_tau(c, f)
    topk: Dict[int, float] = {}

    # ---- 2) 可靠性天花板：full 自己两次独立重训 ----
    ceil_str = "n/a"
    rho_ceiling = float("nan")
    seeds_f = list(full_table["seeds"])
    if len(seeds_f) >= 2:
        fa = np.asarray([r["full_by_seed"][seeds_f[0]] for r in rows], float)
        fb = np.asarray([r["full_by_seed"][seeds_f[1]] for r in rows], float)
        rho_ceiling = spearman(fa, fb)
        ceil_str = f"ρ(full_{seeds_f[0]}, full_{seeds_f[1]}) = {rho_ceiling:+.3f}"
    seeds_c = list(cheap_table["seeds"])
    rho_cheap_self = float("nan")
    if len(seeds_c) >= 2:
        ca = np.asarray([r["cheap_by_seed"][seeds_c[0]] for r in rows], float)
        cb = np.asarray([r["cheap_by_seed"][seeds_c[1]] for r in rows], float)
        rho_cheap_self = spearman(ca, cb)

    log("")
    log("=" * 88)
    log("结论")
    log("=" * 88)
    log(f"  ① Spearman ρ(cheap, full)        = {rho_cf:+.3f}   "
        f"(Kendall τ = {tau_cf:+.3f}, n={len(rows)})")
    log(f"  ② 可靠性天花板 {ceil_str}")
    if len(seeds_c) >= 2:
        log(f"     ρ(cheap_{seeds_c[0]}, cheap_{seeds_c[1]}) = {rho_cheap_self:+.3f}")

    for kk in (1, 3, 5):
        tk = topk_recall(c, f, kk)
        log(f"  ③ Top-{tk['k']} 召回 = {tk['hit']}/{tk['k']} "
            f"({tk['recall']*100:.0f}%)   ← full 前 {tk['k']} 名里被 cheap 命中的比例")
        topk[kk] = float(tk["recall"])

    # ---- 4) 分辨率 ----
    log("")
    res_c = ft_c.k_effect_resolution(val_stats)
    res_f = ft_f.k_effect_resolution(val_stats)
    for nm, res in (("cheap", res_c), ("full", res_f)):
        if res["seed_std"] is None:
            log(f"  {nm:<6s} k 效应极差={res['spread']:.4f}  σ=不可估（seed<2）")
        else:
            log(f"  {nm:<6s} k 效应极差={res['spread']:.4f}  σ={res['seed_std']:.4f}  "
                f"极差/3σ={res['spread_over_3sigma']:.2f}  "
                f"{'可辨识' if res['identifiable'] else '不可辨识'}")

    # ---- 判定 ----
    log("")
    THRESH = 0.6
    if not np.isfinite(rho_cf):
        verdict = "无法判定（样本或方差不足）"
    elif np.isfinite(rho_ceiling) and rho_ceiling < 0.5:
        verdict = (f"❌ **天花板太低**：full 自己两次独立重训只有 ρ={rho_ceiling:+.3f}，"
                   f"说明目标本身不稳定 —— 此时 cheap 排序没有改进空间，"
                   f"应先增大 seed 数或延长训练，而不是调 cheap。")
    elif rho_cf >= THRESH:
        verdict = (f"✅ **通过**（ρ={rho_cf:+.3f} ≥ {THRESH}）：cheap 可用于指导搜索。"
                   f"下一步：接进 OpenEvolve 跑小规模进化，再用 full 复核排名。")
    else:
        verdict = (f"❌ **不通过**（ρ={rho_cf:+.3f} < {THRESH}）：cheap 排序不能预测 full。"
                   f"候选修因：延长 cheap 的 epochs、增大 seed 数、"
                   f"或改 fitness 口径（如用 worst-room 而非 mean）。")
    log(f"  ④ 判定：{verdict}")

    return {
        "rho_cheap_full": rho_cf,
        "tau_cheap_full": tau_cf,
        "rho_ceiling_full_self": rho_ceiling,
        "rho_cheap_self": rho_cheap_self,
        "top3_recall": topk.get(3, float("nan")),
        "topk_recall": {str(k): v for k, v in topk.items()},
        "cheap_epochs": cheap_table.get("epochs"),
        "full_epochs": full_table.get("epochs"),
        "n_candidates": len(rows),
        "rows": [{k: v for k, v in r.items()} for r in rows],
        "verdict": verdict,
    }


def summarise(specs: List[Tuple[str, dict]], log=print) -> None:
    """跨多个 cheap 预算汇总：cheap 要训练多久，排序才可靠？

    `specs` 是 [(label, result_dict), ...]，label 里带 epochs 信息。
    这是"效果为先"最直接的一张表：**预算 → ρ**。
    """
    log("")
    log("=" * 88)
    log("跨预算汇总：cheap 训练预算 → 排序可靠度")
    log("=" * 88)
    log(f"  {'cheap 预算':<16s}{'ρ(cheap,full)':>15s}{'τ':>9s}"
        f"{'Top-3 召回':>12s}{'天花板 ρ(full,full)':>21s}")
    rows = []
    for label, res in specs:
        rho = res.get("rho_cheap_full", float("nan"))
        tau = res.get("tau_cheap_full", float("nan"))
        ceil = res.get("rho_ceiling_full_self", float("nan"))
        # Top-3 召回存在 rows 之外，这里从 verdict 里拿不到，改由调用方附带
        t3 = res.get("top3_recall", float("nan"))
        rows.append((label, rho, tau, t3, ceil))
    for label, rho, tau, t3, ceil in rows:
        t3s = "n/a" if not np.isfinite(t3) else f"{t3*100:.0f}%"
        ceils = "n/a" if not np.isfinite(ceil) else f"{ceil:+.3f}"
        log(f"  {label:<16s}{rho:>+15.3f}{tau:>+9.3f}{t3s:>12s}{ceils:>21s}")
    log("")
    best = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else -9))
    if np.isfinite(best[1]):
        log(f"  → 最佳预算 {best[0]}：ρ = {best[1]:+.3f}")
    ceiling = next((r[4] for r in rows if np.isfinite(r[4])), float("nan"))
    if np.isfinite(ceiling):
        log(f"  → 可靠性天花板 ρ(full_A, full_B) = {ceiling:+.3f}；"
            f"cheap 最多只能接近它。")
        if ceiling < 0.5:
            log("     ⚠ 天花板偏低：目标本身不稳定，应先加大 seed 数/延长 full 训练，")
            log("       而不是继续调 cheap（cheap 再好也超不过目标自身的稳定性）。")


def main() -> None:
    ap = argparse.ArgumentParser(description="cheap vs full 排序可靠度")
    ap.add_argument("--cheap", default="", help="cheap 表路径（给了就不用 --cheap-epochs）")
    ap.add_argument("--full", default="", help="full 表路径（给了就不用 --full-epochs）")
    ap.add_argument("--cheap-epochs", type=int, default=None,
                    help=f"cheap 表的 epochs（默认 {FZ.EPOCHS_CHEAP}）")
    ap.add_argument("--full-epochs", type=int, default=None,
                    help=f"full 表的 epochs（默认 {FZ.EPOCHS_FULL}）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456],
                    help="两张表共用的 seed 列表（用于定位文件名）")
    ap.add_argument("--out", default="", help="把结果 JSON 写到该路径")
    ap.add_argument("--summary-glob", default="",
                    help="汇总模式：glob 一批结果 JSON（如 _reliability_cheap*.json），"
                         "按 cheap epochs 升序打印 ρ vs 预算")
    ap.add_argument("--log-file", default="", help="UTF-8 日志")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    if args.summary_glob:
        import glob as _glob
        pat = args.summary_glob
        if not os.path.isabs(pat):
            pat = os.path.join(DT.PROJECT_ROOT, pat)
        specs: List[Tuple[str, dict]] = []
        for p in sorted(_glob.glob(pat)):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    res = json.load(fh)
            except Exception as e:  # noqa: BLE001
                log(f"[summary] 跳过无法解析的文件 {p}: {e}")
                continue
            ce = res.get("cheap_epochs")
            label = f"cheap@{ce}ep" if ce else os.path.basename(p)
            specs.append((label, res))
        specs.sort(key=lambda t: (t[1].get("cheap_epochs") or 0))
        if not specs:
            log(f"[summary] 没有匹配到文件：{pat}")
            return
        summarise(specs, log=log)
        return

    seeds = [int(s) for s in args.seeds]
    ce = args.cheap_epochs if args.cheap_epochs else FZ.EPOCHS_CHEAP
    fe = args.full_epochs if args.full_epochs else FZ.EPOCHS_FULL
    cp = args.cheap or FZ._table_path("cheap", ce, seeds)
    fp = args.full or FZ._table_path("full", fe, seeds)
    with open(cp, "r", encoding="utf-8") as fh:
        cheap = json.load(fh)
    with open(fp, "r", encoding="utf-8") as fh:
        full = json.load(fh)
    log(f"[reliability] cheap = {os.path.basename(cp)}")
    log(f"[reliability] full  = {os.path.basename(fp)}")

    res = analyse(cheap, full, log=log)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"[reliability] 结果已写入 {args.out}")


if __name__ == "__main__":
    main()
