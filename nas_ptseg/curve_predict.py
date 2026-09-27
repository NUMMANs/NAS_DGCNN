"""学习曲线外推预测器：用**多个短训练预算**的学习曲线预测长训练的排序。

为什么需要它
================================================================
实测（2026-09-23）：
    `ρ(cheap@25, full@300) = −0.825`  —— 单预算 cheap 不只不相关，而是**排序反向**。
机制是 k 效应的方向随训练预算翻转（欠拟合期 k 越大越差，收敛期 k 越大越好）。
所以"再用一个固定短预算当代理"是错的——**必须让代理知道曲线正在往哪走**。

做法：把学习曲线当作可观测量
================================================================
对每个候选策略，我们在若干预算 E ∈ {25, 50, 100, 150, ...} 上都有 fitness，
于是得到一条**部分观测的学习曲线** f(E)。用一小族单调饱和曲线拟合它、外推到目标预算，
再按外推值排序。这是 HPO 里的经典做法（Domhan et al. 2015, "Speeding up automatic
hyperparameter optimization by extrapolation of learning curves"）。

曲线族（都在给定 1 个非线性参数后**退化为 2 参数线性最小二乘** → 无需 scipy、
可用一维网格稳健求解，避免局部极小）：
    exp   : f(E) = a − b·exp(−cE)        渐近饱和
    power : f(E) = a − b·E^(−c)         幂律饱和（等价于 log-log 线性外推）
    log   : f(E) = a + b·ln(E)          对数增长（不饱和，保守）
    mm    : f(E) = a·E/(b+E)            Michaelis–Menten 饱和

约束：`a >= max(观测值)`（渐近线不应低于已观测到的分数）、`0 <= 参数` 合理范围。

方法选择不靠猜
================================================================
`--select-on` 指定一个**留出预算**（如 150）：只在 ≤ 该预算的数据上拟合、外推到它，
比较各曲线族的排序质量，选最好的那一族，再用它外推到真正的目标预算（如 300）。
这样"用哪个曲线族"这个决定本身也经过了验证，而不是我们拍脑袋。

用法
================================================================
    python -m nas_ptseg.curve_predict \
        --budgets 25 50 100 150 --target 300 --select-on 150
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import gpu as GPU  # noqa: F401  （导入即钉 GPU；本模块不训练）
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


FAMILIES = ("exp", "power", "log", "mm")


# ---------------------------------------------------------------------------
# 曲线拟合：非线性参数走一维网格，线性参数用最小二乘闭式解
# ---------------------------------------------------------------------------
def _design(family: str, E: np.ndarray, theta: float) -> Optional[np.ndarray]:
    """给定非线性参数 theta，返回线性设计矩阵 (n, 2)，列为 [基1, 基2]。"""
    E = np.asarray(E, dtype=np.float64)
    if family == "exp":
        return np.stack([np.ones_like(E), -np.exp(-theta * E)], axis=1)
    if family == "power":
        if theta <= 0:
            return None
        return np.stack([np.ones_like(E), -(E ** (-theta))], axis=1)
    if family == "log":
        return np.stack([np.ones_like(E), np.log(E)], axis=1)
    if family == "mm":
        if theta <= 0:
            return None
        return np.stack([E / (theta + E)], axis=1)
    return None


def _predict(family: str, theta: float, coef: np.ndarray, E) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if family == "exp":
        return coef[0] - coef[1] * np.exp(-theta * E)
    if family == "power":
        return coef[0] - coef[1] * (E ** (-theta))
    if family == "log":
        return coef[0] + coef[1] * np.log(E)
    if family == "mm":
        return coef[0] * E / (theta + E)
    raise ValueError(family)


def _theta_grid(family: str, Emax: float) -> np.ndarray:
    """非线性参数的搜索网格（覆盖从'几乎不饱和'到'早已饱和'）。"""
    if family == "exp":
        # c=1e-4 → 在 E<=300 上几乎线性；c=0.5 → E=10 就饱和
        return np.logspace(-4.0, np.log10(0.5), 200)
    if family == "power":
        return np.linspace(0.05, 3.0, 200)
    if family == "log":
        return np.zeros(1)
    if family == "mm":
        # b 相对 Emax：b>>Emax → 近似线性增长；b<<Emax → 早已饱和
        return np.logspace(np.log10(max(Emax, 1.0) * 0.01),
                           np.log10(max(Emax, 1.0) * 100.0), 200)
    raise ValueError(family)


def fit_curve(family: str, E: Sequence[float], y: Sequence[float],
              E_target: float, y_max_allowed: float = 1.05) -> Dict[str, float]:
    """拟合一条曲线并外推到 E_target。

    返回 {theta, params, sse, pred_target, ok}。ok=False 表示拟合无效（退化）。
    """
    E = np.asarray(E, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = E.size
    if n < 2:
        return {"ok": 0.0, "pred_target": float("nan")}

    Emax = float(E.max())
    best = None
    for th in _theta_grid(family, Emax):
        A = _design(family, E, float(th))
        if A is None:
            continue
        # 线性最小二乘（可能欠定/奇异 → 用 lstsq 兜住）
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A @ coef
        sse = float(((y - pred) ** 2).sum())
        # 约束 1：渐近线不低于已观测（避免外推向下跑）
        asym = _asymptote(family, float(th), coef, Emax)
        if asym is not None and asym < y.max() - 1e-9:
            continue
        # 约束 2：外推值落在物理区间内
        pt = float(_predict(family, float(th), coef, np.array([E_target]))[0])
        if not np.isfinite(pt) or pt < y.min() - 0.05 or pt > y_max_allowed:
            continue
        # 约束 3：外推不应超出观测趋势太远（防爆炸）
        if pt > y.max() + 0.5 * max(y.max() - y.min(), 0.02) + 0.05:
            continue
        if best is None or sse < best["sse"] - 1e-15:
            best = {"theta": float(th), "coef": coef.copy(), "sse": sse}
    if best is None:
        return {"ok": 0.0, "pred_target": float("nan")}

    pt = float(_predict(family, best["theta"], best["coef"], np.array([E_target]))[0])
    return {
        "ok": 1.0,
        "family": family,
        "theta": best["theta"],
        "coef": [float(v) for v in best["coef"]],
        "sse": best["sse"],
        "pred_target": pt,
    }


def _asymptote(family: str, theta: float, coef: np.ndarray, Emax: float) -> Optional[float]:
    if family in ("exp", "power"):
        return float(coef[0])
    if family == "mm":
        return float(coef[0])
    return None          # log 无渐近线


def predict_from_curves(curves: Dict[str, Dict[int, float]], fit_epochs: Sequence[int],
                        E_target: int) -> Dict[str, Dict[str, float]]:
    """对每个候选（策略名）在其可用预算上拟合曲线族，返回各族的预测值。

    `curves`: {cand_name: {E: fitness}}
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, by_e in curves.items():
        avail = [e for e in sorted(fit_epochs) if e in by_e and np.isfinite(by_e[e])]
        if len(avail) < 2:
            out[name] = {f: float("nan") for f in FAMILIES}
            continue
        E = [avail]
        y = [by_e[e] for e in avail]
        rec: Dict[str, float] = {}
        for fam in FAMILIES:
            r = fit_curve(fam, E[0], y[0], float(E_target))
            rec[fam] = r["pred_target"] if r["ok"] else float("nan")
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# 表格装载 / 候选打分
# ---------------------------------------------------------------------------
def load_budget_tables(budgets: Sequence[int], seeds: Sequence[int],
                       table_dir: Optional[str] = None) -> Dict[int, dict]:
    """装载若干 cheap 预算的表；缺哪个就跳过（返回实际存在的）。"""
    out: Dict[int, dict] = {}
    for e in budgets:
        p = (FZ._table_path("cheap", int(e), seeds) if table_dir is None
             else os.path.join(table_dir, _table_name("cheap", int(e), seeds)))
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if not d.get("cells"):
            continue
        # 只接受已建完的（cells 数 = 房间 × k × seed）
        need = len(d.get("train_rooms", [])) * len(d.get("ks", [])) * len(d.get("seeds", []))
        if len(d["cells"]) < need:
            continue                      # 未建完，跳过以免污染结论
        out[int(e)] = d
    return out


def _table_name(role: str, epochs: int, seeds: Sequence[int]) -> str:
    return f"{role}_ep{int(epochs)}_s{'-'.join(str(int(s)) for s in seeds)}.json"


def build_curves(tables: Dict[int, dict], log=print) -> Tuple[Dict[str, Dict[int, float]],
                                                             Dict[str, dict], List[str]]:
    """对确定性候选集，在每个预算上打分，得到学习曲线。

    返回 (curves, aux, val_keys)：
        curves  : {cand_name: {E: fitness}}
        aux     : {cand_name: {E: worst_room}}
        val_keys: 验证房间 keys
    """
    epochs = sorted(tables)
    base = tables[epochs[0]]
    val_rooms = list(base["val_rooms"])
    tgt_n = int(base.get("target_n", RS.TARGET_N))

    val_stats: Dict[str, dict] = {}
    for key in val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=tgt_n)
        val_stats[key] = st

    rho_arr = np.asarray([val_stats[k]["rho_med"] for k in val_stats], float)
    L_arr = np.asarray([val_stats[k]["extent_max_m"] for k in val_stats], float)
    rho_ref = float(np.exp(np.mean(np.log(np.maximum(rho_arr, 1e-9)))))
    L_ref = float(np.median(L_arr))

    cands = REL.dedupe_candidates(REL.build_candidates(rho_ref, L_ref), val_stats)
    log(f"  候选策略 {len(cands)} 个（按 k 分配去重后）")

    fts = {e: FZ.FitnessTable(t) for e, t in tables.items()}
    curves: Dict[str, Dict[int, float]] = {}
    aux: Dict[str, Dict[int, float]] = {}
    for name, fn, _sig in cands:
        curves[name] = {}
        aux[name] = {}
        for e, ft in fts.items():
            r = ft.score(fn, val_stats)
            curves[name][e] = float(r["fitness"])
            aux[name][e] = float(r["worst"])
    return curves, {"worst": aux, "val_stats": val_stats,
                    "rho_ref": rho_ref, "L_ref": L_ref}, val_rooms


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(budgets: Sequence[int], target_epochs: int, select_on: Optional[int],
        seeds: Sequence[int], out_path: str = "", log=print) -> dict:
    tables = load_budget_tables(budgets, seeds)
    have = sorted(tables)
    if not have:
        log("[curve] 没有任何可用的 cheap 表，退出。")
        return {}

    ft_path = FZ._table_path("full", int(target_epochs), seeds)
    if not os.path.exists(ft_path):
        log(f"[curve] 目标表不存在：{os.path.basename(ft_path)}")
        return {}
    with open(ft_path, "r", encoding="utf-8") as fh:
        full = json.load(fh)

    log("=" * 92)
    log("学习曲线外推：能否用多预算短训练预测长训练的排序？")
    log("=" * 92)
    log(f"  可用 cheap 预算 : {have}")
    log(f"  目标预算        : {target_epochs}")
    log(f"  留出选择预算    : {select_on}")

    curves, aux, val_rooms = build_curves(tables, log=log)
    ft_f = FZ.FitnessTable(full)

    # 目标真值（用与 cheap 相同的候选集）
    val_stats = aux["val_stats"]
    names = list(curves.keys())
    truth: Dict[str, float] = {}
    truth_worst: Dict[str, float] = {}
    for name in names:
        fn = _policy_lookup(name, curves, aux, val_stats)
        if fn is None:
            continue
        r = ft_f.score(fn, val_stats)
        truth[name] = float(r["fitness"])
        truth_worst[name] = float(r["worst"])

    names = [n for n in names if n in truth]
    log(f"  可比候选 {len(names)} 个")

    # ---- 基线：直接用各预算的原始值排序 ----
    log("")
    log("  ── 基线：单个预算的原始 fitness 排序 ──")
    log(f"  {'代理预算':<14s}{'ρ(代理, full)':>16s}{'Top-1':>9s}{'Top-3':>9s}")
    baselines: Dict[str, dict] = {}
    for e in have:
        pred = np.asarray([curves[n][e] for n in names], float)
        tru = np.asarray([truth[n] for n in names], float)
        rho = REL.spearman(pred, tru)
        t1 = REL.topk_recall(pred, tru, 1)["recall"]
        t3 = REL.topk_recall(pred, tru, 3)["recall"]
        baselines[str(e)] = {"rho": rho, "top1": t1, "top3": t3}
        log(f"  raw@{e:<10d}{rho:>+16.3f}{t1*100:>8.0f}%{t3*100:>8.0f}%")

    # ---- 方法选择：在留出预算上验证哪个曲线族/组合最好 ----
    chosen = None
    if select_on and select_on in have:
        fit_e = [e for e in have if e < select_on]
        log("")
        log(f"  ── 方法选择：只用预算 {fit_e} 拟合，外推到 {select_on} 验证 ──")
        if len(fit_e) >= 2:
            pred_sel = predict_from_curves(curves, fit_e, int(select_on))
            tru_sel = np.asarray([curves[n][select_on] for n in names], float)
            stats = {}
            for fam in FAMILIES + ("ens",):
                v = _stack_family(pred_sel, names, fam)
                rho = REL.spearman(v, tru_sel)
                stats[fam] = rho
                log(f"    外推族 {fam:<6s} ρ(外推@{select_on}, 实测@{select_on}) = {rho:+.3f}")
            ok = {k: v for k, v in stats.items() if np.isfinite(v)}
            if ok:
                chosen = max(ok, key=lambda k: ok[k])
                log(f"    → 选中曲线族：**{chosen}**（留出 ρ={ok[chosen]:+.3f}）")
        else:
            log(f"    可用拟合预算不足 2 个，跳过方法选择。")
    else:
        log("")
        log("  （未指定 --select-on 或该预算缺失，跳过方法选择）")

    # ---- 用全部可用预算外推到目标 ----
    log("")
    log(f"  ── 用预算 {have} 外推到 {target_epochs} ──")
    fit_all = have
    pred_all = predict_from_curves(curves, fit_all, int(target_epochs))
    tru = np.asarray([truth[n] for n in names], float)
    best = ("last", float("nan"))
    ext_stats: Dict[str, dict] = {}
    log(f"  {'外推族':<10s}{'ρ(外推, full)':>16s}{'Top-1':>9s}{'Top-3':>9s}")
    for fam in FAMILIES + ("ens",):
        v = _stack_family(pred_all, names, fam)
        rho = REL.spearman(v, tru)
        t1 = REL.topk_recall(v, tru, 1)["recall"]
        t3 = REL.topk_recall(v, tru, 3)["recall"]
        ext_stats[fam] = {"rho": rho, "top1": t1, "top3": t3}
        log(f"  {fam:<10s}{rho:>+16.3f}{t1*100:>8.0f}%{t3*100:>8.0f}%")
    # "last" 基线 = 用最大可用预算的原始值
    if have:
        v_last = np.asarray([curves[n][have[-1]] for n in names], float)
        rho = REL.spearman(v_last, tru)
        ext_stats["last"] = {"rho": rho,
                             "top1": REL.topk_recall(v_last, tru, 1)["recall"],
                             "top3": REL.topk_recall(v_last, tru, 3)["recall"]}
        log(f"  {'last':<10s}{rho:>+16.3f}"
            f"{ext_stats['last']['top1']*100:>8.0f}%{ext_stats['last']['top3']*100:>8.0f}%")

    # ---- 结论 ----
    ceiling = float("nan")
    if len(full.get("seeds", [])) >= 2:
        s = list(full["seeds"])
        fa, fb = [], []
        for n in names:
            fn = _policy_lookup(n, curves, aux, val_stats)
            if fn is None:
                fa.append(np.nan); fb.append(np.nan); continue
            fa.append(ft_f.score(fn, val_stats, seeds=[s[0]])["fitness"])
            fb.append(ft_f.score(fn, val_stats, seeds=[s[1]])["fitness"])
        ceiling = REL.spearman(fa, fb)

    log("")
    log("=" * 92)
    log("结论")
    log("=" * 92)
    log(f"  可靠性天花板 ρ(full_A, full_B) = {ceiling:+.3f}")
    ref = ext_stats.get(chosen) if chosen else None
    if ref and np.isfinite(ref["rho"]):
        log(f"  选中的外推族 {chosen}：ρ = {ref['rho']:+.3f}，Top-3 召回 {ref['top3']*100:.0f}%")
    gained = ""
    if have:
        raw = baselines.get(str(have[-1]), {}).get("rho", float("nan"))
        if ref and np.isfinite(ref["rho"]) and np.isfinite(raw):
            gained = (f"  相对 raw@{have[-1]}（ρ={raw:+.3f}）"
                      f"提升 {ref['rho'] - raw:+.3f}")
            log(gained)

    res = {
        "budgets_used": have,
        "target_epochs": int(target_epochs),
        "select_on": select_on,
        "chosen_family": chosen,
        "baselines": baselines,
        "extrapolation": ext_stats,
        "ceiling_full_self": ceiling,
        "n_candidates": len(names),
        "curves": {n: curves[n] for n in names},
        "truth_full": {n: truth[n] for n in names},
        "pred_target_exp": {n: pred_all[n].get(chosen or "exp") for n in names},
    }
    if out_path:
        p = out_path if os.path.isabs(out_path) else os.path.join(DT.PROJECT_ROOT, out_path)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"  结果已写入 {p}")
    return res


def _stack_family(pred: Dict[str, Dict[str, float]], names: Sequence[str],
                  fam: str) -> np.ndarray:
    """把每个候选的某族预测取出来；`ens` = 各可用族的均值（稳健化）。"""
    if fam != "ens":
        return np.asarray([pred[n].get(fam, float("nan")) for n in names], float)
    rows = []
    for n in names:
        vals = [pred[n].get(f, float("nan")) for f in FAMILIES]
        vals = [v for v in vals if np.isfinite(v)]
        rows.append(float(np.mean(vals)) if vals else float("nan"))
    return np.asarray(rows, float)


_POLICY_CACHE: Dict[str, object] = {}


def _policy_lookup(name: str, curves, aux, val_stats):
    """按名字重建策略函数（build_candidates 的 name 里编码了全部参数）。"""
    if name in _POLICY_CACHE:
        return _POLICY_CACHE[name]
    fn = None
    if name.startswith("const_k"):
        fn = FZ.policy_const(int(name[len("const_k"):]))
    elif name.startswith("rho_k"):
        body = name[len("rho_k"):]
        k0s, a_s = body.split("_a")
        fn = FZ.policy_power(int(k0s), float(a_s), float(aux["rho_ref"]))
    elif name.startswith("L_k"):
        body = name[len("L_k"):]
        k0s, b_s = body.split("_b")
        fn = FZ.policy_scale(int(k0s), float(b_s), float(aux["L_ref"]))
    _POLICY_CACHE[name] = fn
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description="学习曲线外推预测器")
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 150],
                    help="可用于拟合的 cheap 预算（只取实际存在且已建完的表）")
    ap.add_argument("--target", type=int, default=300, help="目标预算（full 表）")
    ap.add_argument("--select-on", type=int, default=150,
                    help="留出预算：在其上比较曲线族、选最优（0 表示跳过）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--out", default="", help="结果 JSON 路径")
    ap.add_argument("--log-file", default="", help="UTF-8 日志文件")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    run(args.budgets, args.target, (args.select_on or None), args.seeds,
        out_path=args.out, log=log)


if __name__ == "__main__":
    main()
