"""架构搜索空间的**物化表**：把 `(训练房间, 配置, seed) → {验证房间: mIoU}` 全建出来。

为什么物化，而不是继续做"廉价代理"
================================================================
k 轴的死因已经很清楚：**逐房间 oracle 的 headroom 只有 +0.0017**（`headroom.py`）。
换句话说 k 轴根本没有可搜的空间，不是"代理不够准"。

当搜索空间**足够小、可以被完整物化**时，"廉价代理"这件事就不需要了：
一旦表建好，**任何** room→config 策略的适应度都是毫秒级查表，而且**完全无偏、无预算敏感性**。
只有当场空间大到表建不出来（连续变量 / 组合爆炸）时，才需要廉价代理去筛。

所以这里的目标是：先在一个**可物化的中等规模空间**上，把"搜索能否真的赢"
这件事回答清楚。如果连物化空间里都搜不出东西，那做廉价代理只会更快地搜出噪声。

空间定义（`--grid` 选轴）
================================================================
    k           : 邻域大小（已知收益极小，留作对照）
    agg         : EdgeConv 池化 max / mean / maxmean   （**参数量不变**）
    graph_space : feat 动态特征图 / coord 静态坐标图     （**参数量不变**）
    feat        : 特征宽度 32 / 96 / 128                （改参数量 → 容量轴）
    layers      : 层数 2 / 4                            （改参数量 → 容量轴）

代价估算：`len(配置) × len(训练房间) × len(seed) × ~24s`。
18 个配置 × 12 房间 × 2 seed ≈ 2.9 h（GPU 1）。**必须先看 `arch_probe` 的极差/3σ 再决定建多大的表。**

表结构与 `fitness.py` 一致（每 cell 落盘、可断点续跑），只是把 `k=` 换成 `cfg=`。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import gpu as GPU
    from nas_ptseg import data as DT
    from nas_ptseg import room_split as RS
    from nas_ptseg import metrics as MT

    _PKG_MODE = False
else:
    from . import gpu as GPU
    from . import data as DT
    from . import room_split as RS
    from . import metrics as MT

    _PKG_MODE = True


TABLE_DIR = os.path.join(DT.PROJECT_ROOT, "data", "arch_tables")

# 轴 → 取值。k 固定 48（full@300 下最优固定 k），因为它自己的 headroom 已证明极小。
AXES: Dict[str, List] = {
    "k": [48],
    "agg": ["max", "mean", "maxmean"],
    "graph_space": ["feat", "coord"],
    "feat": [64],
    "layers": [3],
}
DEFAULT_AXES_ON = ("k", "agg", "graph_space")


def build_grid(axes_on: Sequence[str],
               overrides: Optional[Dict[str, Optional[Sequence]]] = None) -> List[dict]:
    """按启用的轴做笛卡尔积，返回配置列表（顺序稳定，cfg id = 下标）。

    `overrides` 可临时替换某个轴的取值（如 `{"feat": [32, 64, 128]}`），
    免得为了扫容量轴去改模块常量。
    """
    axes = {k: list(v) for k, v in AXES.items()}
    for key, vals in (overrides or {}).items():
        if vals:
            axes[key] = [int(v) if isinstance(v, (int, np.integer)) else v for v in vals]
    keys = [a for a in ("k", "agg", "graph_space", "feat", "layers") if a in axes_on]
    ranges = [axes[a] for a in keys]
    out: List[dict] = []
    for combo in itertools.product(*ranges):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def cfg_label(cfg: dict) -> str:
    return "|".join(
        f"{key}={cfg[key]}"
        for key in ("k", "agg", "graph_space", "feat", "layers") if key in cfg
    )


def _table_path(tag: str, seeds: Sequence[int], epochs: int) -> str:
    """⚠ 文件名**必须**含 epochs：同一个网格要在多个预算上各建一张表
    （用来检查"廉价预算下的架构排序能否预测充分训练的排序"）。
    若不含 epochs，两次建表会互相覆盖、并因"参数不一致"而被误判为需要重建。
    """
    s = "-".join(str(int(x)) for x in seeds)
    return os.path.join(TABLE_DIR, f"arch_{tag}_ep{int(epochs)}_s{s}.json")


def _torch_stack():
    import torch  # noqa: PLC0415
    from nas_ptseg import tier2_dgcnn as T2  # noqa: PLC0415
    return torch, T2


# ---------------------------------------------------------------------------
# 建表
# ---------------------------------------------------------------------------
def build(
    grid: List[dict], tag: str, seeds: Sequence[int], epochs: int,
    target_n: int, train_rooms: Sequence[str], val_rooms: Sequence[str],
    log=print,
) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = _table_path(tag, seeds, epochs)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
        if (table.get("grid") == grid and table.get("epochs") == epochs
                and list(table.get("seeds", [])) == [int(s) for s in seeds]):
            log(f"[arch] 续跑已有表 {os.path.basename(path)} "
                f"({len(table['cells'])}/{len(grid)*len(train_rooms)*len(seeds)})")
        else:
            log("[arch] 已有表参数不一致 → 重建")
            table = None
    else:
        table = None

    torch, T2 = _torch_stack()
    dev = GPU.setup_torch_device()

    if table is None:
        table = {
            "grid": grid, "epochs": int(epochs), "seeds": [int(s) for s in seeds],
            "target_n": int(target_n),
            "train_rooms": list(train_rooms), "val_rooms": list(val_rooms),
            "cells": {},
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "gpu_constraint": GPU.summary(),
        }
        log(f"[arch] 新建表：{len(grid)} 配置 × {len(train_rooms)} 训练房 × "
            f"{len(seeds)} seed = {len(grid)*len(train_rooms)*len(seeds)} cells, {epochs} ep")
        for i, c in enumerate(grid):
            log(f"    cfg[{i:>2d}] {cfg_label(c)}")

    # 预载数据（CRN：房间与下采样对所有 seed 一致）
    val_data = {}
    for v in val_rooms:
        a, r = v.split("/", 1)
        xyz, y, st = RS.load_room_cached(a, r, target_n=target_n)
        val_data[v] = (xyz, y, st["present_classes"])
    tr_data = {}
    for t in train_rooms:
        a, r = t.split("/", 1)
        xyz, y, _st = RS.load_room_cached(a, r, target_n=target_n)
        tr_data[t] = (xyz, y)
    log(f"[arch] 载入 {len(tr_data)} 训练 + {len(val_data)} 验证房间")

    total = len(train_rooms) * len(grid) * len(seeds)
    done = 0
    t0 = time.perf_counter()
    for tr in train_rooms:
        xyz_tr, y_tr = tr_data[tr]
        for ci, cfg in enumerate(grid):
            kk = int(min(int(cfg.get("k", 48)), max(xyz_tr.shape[0] - 1, 1)))
            arch = {kk2: vv for kk2, vv in cfg.items() if kk2 != "k"}
            for s in seeds:
                done += 1
                key = f"{tr}|cfg={ci}|seed={int(s)}"
                if key in table["cells"]:
                    continue
                torch.manual_seed(int(s))
                np.random.seed(int(s))
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                model = T2.DGCNNSeg(int(DT.NUM_CLASSES), kk, **arch).to(dev)
                opt = torch.optim.Adam(model.parameters(), lr=T2.LR,
                                       weight_decay=T2.WEIGHT_DECAY)
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP,
                                                        gamma=T2.LR_GAMMA)
                X = T2.to_torch(np.ascontiguousarray(xyz_tr.T[None, ...], dtype=np.float32), dev)
                Y = T2.to_torch(np.ascontiguousarray(y_tr, dtype=np.int64), dev)
                for _ in range(int(epochs)):
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss = T2.F.cross_entropy(logits[0].t(), Y)
                    loss.backward()
                    opt.step()
                    sched.step()
                model.eval()
                res: Dict[str, float] = {}
                with torch.no_grad():
                    for v, (xv, yv, pv) in val_data.items():
                        Xe = T2.to_torch(
                            np.ascontiguousarray(xv.T[None, ...], dtype=np.float32), dev)
                        pr = np.asarray(model(Xe).argmax(dim=1)[0].cpu().tolist(),
                                        dtype=np.int64)
                        res[v] = MT.present_class_miou(pr, yv.astype(np.int64), pv)
                table["cells"][key] = res
                table["n_params"] = int(sum(p.numel() for p in model.parameters()))
                del model
                torch.cuda.empty_cache()
                el = time.perf_counter() - t0
                if done % 3 == 0 or done == total:
                    log(f"[arch] {done}/{total}  ({el:.0f}s, {el/max(done,1):.1f}s/cell, "
                        f"ETA {el/max(done,1)*(total-done):.0f}s)  "
                        f"{tr} cfg[{ci}] seed={s} → {np.mean(list(res.values())):.4f}")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(table, fh, ensure_ascii=False, indent=2)
    log(f"[arch] 完成 → {path}")
    return table


# ---------------------------------------------------------------------------
# 分析：这个空间值不值得搜
# ---------------------------------------------------------------------------
def load_matrix(table: dict) -> Tuple[List[str], Dict[str, Dict[int, float]]]:
    """{val_room: {cfg_id: mIoU}}（对 训练房间 × seed 平均）。"""
    grid = table["grid"]
    seeds = [int(s) for s in table["seeds"]]
    rooms = list(table["val_rooms"])
    mat: Dict[str, Dict[int, float]] = {v: {} for v in rooms}
    for v in rooms:
        for ci in range(len(grid)):
            acc: List[float] = []
            for tr in table["train_rooms"]:
                for s in seeds:
                    c = table["cells"].get(f"{tr}|cfg={ci}|seed={s}")
                    if c and v in c:
                        acc.append(float(c[v]))
            if acc:
                mat[v][ci] = float(np.mean(acc))
    return rooms, mat


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
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


def spearman(x, y) -> float:
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_budgets(cheap: dict, full: dict, log=print) -> dict:
    """廉价预算下的**架构**排序，能否预测充分训练的架构排序？

    为什么必须单独做这一步：k 轴上 25→300 epoch 会**翻转排序**。
    架构轴完全可能有同类问题——更宽/更深的模型在短训练下更欠拟合，
    于是"短训练最优的架构"未必是"收敛后最优的架构"。
    所以在把 50 epoch 当廉价代理之前，必须先在这里验证。
    """
    if cheap["grid"] != full["grid"]:
        log("  ⚠ 两张表的配置网格不一致，无法比较。")
        return {}
    rc, mc = load_matrix(cheap)
    rf, mf = load_matrix(full)
    rooms = [v for v in rc if v in mf]
    grid = full["grid"]
    ncfg = len(grid)

    xs, ys = [], []
    for v in rooms:
        for ci in range(ncfg):
            if ci in mc[v] and ci in mf[v]:
                xs.append(mc[v][ci])
                ys.append(mf[v][ci])
    rho_cells = spearman(xs, ys)

    pc = [float(np.mean([mc[v][ci] for v in rooms if ci in mc[v]])) for ci in range(ncfg)]
    pf = [float(np.mean([mf[v][ci] for v in rooms if ci in mf[v]])) for ci in range(ncfg)]
    rho_cfg = spearman(pc, pf)

    top1_c = int(np.argmax(pc))
    top1_f = int(np.argmax(pf))
    orc_c = int(np.argmax([max(mc[v].values()) for v in rooms[:1]])) if rooms else -1
    # 逐房间"最优配置"的一致率
    agre = 0
    tot = 0
    for v in rooms:
        a = max(mc[v], key=lambda c: mc[v][c])
        b = max(mf[v], key=lambda c: mf[v][c])
        tot += 1
        agre += int(a == b)

    log("")
    log("=" * 96)
    log(f"架构排序的预算可迁移性：cheap@{cheap['epochs']}ep → full@{full['epochs']}ep")
    log("=" * 96)
    log(f"  {'cfg':>4s}  {'配置':<34s}{'cheap':>10s}{'full':>10s}{'Δ':>10s}")
    for ci in sorted(range(ncfg), key=lambda c: -pf[c]):
        log(f"  {ci:>4d}  {cfg_label(grid[ci]):<34s}{pc[ci]:>10.4f}{pf[ci]:>10.4f}"
            f"{pf[ci]-pc[ci]:>+10.4f}")
    log("")
    log(f"  ① 全部 (房间 × 配置) 单元格的 ρ = {rho_cells:+.3f}   ← 最敏感")
    log(f"  ② 配置均值的 ρ                = {rho_cfg:+.3f}   （只有 {ncfg} 个点）")
    log(f"  ③ 最优配置一致                : cheap→cfg[{top1_c}]  full→cfg[{top1_f}]  "
        f"{'✅ 一致' if top1_c == top1_f else '❌ 不一致'}")
    log(f"  ④ 逐房间最优配置一致率        = {agre}/{tot} ({agre/max(tot,1)*100:.0f}%)")
    ok = np.isfinite(rho_cfg) and rho_cfg >= 0.6
    log(f"  判定：{'✅ 50 epoch 的架构排序可用于搜索' if ok else '⚠ 架构排序在预算间不可迁移，需更长预算'}")
    return {"rho_cells": rho_cells, "rho_cfg": rho_cfg,
            "top1_cheap": top1_c, "top1_full": top1_f,
            "per_room_agree": agre / max(tot, 1), "ok": bool(ok)}


def sigma_fit(table: dict) -> Optional[float]:
    """配置内两 seed 之差的 σ（fitness 级）。"""
    grid = table["grid"]
    seeds = [int(s) for s in table["seeds"]]
    if len(seeds) < 2:
        return None
    s0, s1 = seeds[0], seeds[1]
    diffs: List[float] = []
    for ci in range(len(grid)):
        for v in table["val_rooms"]:
            a, b = [], []
            for tr in table["train_rooms"]:
                c0 = table["cells"].get(f"{tr}|cfg={ci}|seed={s0}")
                c1 = table["cells"].get(f"{tr}|cfg={ci}|seed={s1}")
                if c0 and v in c0:
                    a.append(float(c0[v]))
                if c1 and v in c1:
                    b.append(float(c1[v]))
            if a and b:
                diffs.append(np.mean(a) - np.mean(b))
    if len(diffs) < 2:
        return None
    return float(np.std(diffs, ddof=1) / np.sqrt(2.0) / np.sqrt(len(table["val_rooms"])))


def analyse(table: dict, log=print) -> dict:
    grid = table["grid"]
    rooms, mat = load_matrix(table)
    per_cfg = {ci: float(np.mean([mat[v][ci] for v in rooms if ci in mat[v]]))
               for ci in range(len(grid))}

    log("=" * 96)
    log("架构表分析：这个空间值不值得搜？")
    log("=" * 96)
    log(f"  配置数 {len(grid)}  验证房间 {len(rooms)}  epochs={table['epochs']}")
    log("")
    log(f"  {'cfg':>4s}  {'配置':<34s}{'mean':>10s}{'参数量':>10s}")
    best_ci = max(per_cfg, key=lambda c: per_cfg[c])
    for ci in sorted(per_cfg, key=lambda c: -per_cfg[c]):
        log(f"  {ci:>4d}  {cfg_label(grid[ci]):<34s}{per_cfg[ci]:>10.4f}"
            f"{table.get('n_params', 0):>10d}")

    oracle = float(np.mean([max(mat[v].values()) for v in rooms if mat.get(v)]))
    floor = per_cfg[best_ci]
    sig = sigma_fit(table)
    log("")
    log(f"  最好单一配置  ① = {floor:.4f}   (cfg[{best_ci}] {cfg_label(grid[best_ci])})")
    log(f"  逐房间 oracle ② = {oracle:.4f}   上限")
    log(f"  headroom(②−①)   = {oracle - floor:+.4f}")

    # ---- 按 Area 的**域级**交互 ----
    # 房间级 headroom ≈ 0 只说明"房间之间"没有交互。本课题真正要问的是**域**：
    # 不同 Area 是不同的建筑/扫描，构成真实的分布位移。若"每个 Area 的最优配置"
    # 与"全局最优配置"不同，则"按域条件化架构"有收益空间 —— 那才是 NAS 的立论基础。
    areas: Dict[str, List[str]] = {}
    for v in rooms:
        areas.setdefault(v.split("/", 1)[0], []).append(v)
    if len(areas) >= 2:
        log("")
        log("  按 Area 的域级交互（每个 Area = 一个域；room 级 headroom 为 0 不代表域级也为 0）")
        log(f"    {'Area':<10s}" + "".join(f"{'cfg[' + str(c) + ']':>10s}"
                                          for c in range(len(grid))) + f"{'最好':>8s}")
        area_best: Dict[str, int] = {}
        for a in sorted(areas):
            vals = [float(np.mean([mat[v][c] for v in areas[a] if c in mat[v]]))
                    for c in range(len(grid))]
            bi = int(np.argmax(vals))
            area_best[a] = bi
            log(f"    {a:<10s}" + "".join(f"{x:>10.4f}" for x in vals) + f"{bi:>8d}")
        area_oracle = float(np.mean([max(float(np.mean([mat[v][c] for v in areas[a]
                                                        if c in mat[v]]))
                                        for c in range(len(grid))) for a in areas]))
        # 全局最优配置在每个 Area 上的平均分
        glob = float(np.mean([float(np.mean([mat[v][best_ci] for v in areas[a]
                                             if best_ci in mat[v]])) for a in areas]))
        log("")
        log(f"    全局最优配置在此表上的域均值 = {glob:.4f}")
        log(f"    逐 Area oracle（每个 Area 各挑最好）= {area_oracle:.4f}")
        log(f"    域级 headroom = {area_oracle - glob:+.4f}")
        if sig is not None:
            r = (area_oracle - glob) / max(3.0 * sig, 1e-9)
            log(f"    域级 headroom / 3σ = {r:.2f}")
            if r > 1.0:
                log("    ✅ **域级存在交互** → 「按域条件化架构」有可搜空间。")
            else:
                log("    ⚠ 域级也没有交互（不足 3σ）→ 条件化架构在本数据上拿不到收益。")
        area_info = {"per_area_best": area_best, "area_oracle": area_oracle,
                     "global_on_areas": glob, "area_headroom": area_oracle - glob}
        # 注意：**逐 Area 的 oracle 与全局最优与 σ 无关**（σ 只用于判显著性）。
        # 若两者在所有 Area 上都指向同一个配置，那"按域条件化"就没有可拿的收益，
        # 这与 σ 大小无关 —— 不能把"σ 太小所以看不出差异"当成通过。
        area_info["all_areas_agree"] = bool(len(set(area_best.values())) == 1)
    else:
        area_info = {}

    if sig is None:
        log("  σ 不可估（seed < 2）")
        verdict = "需要至少 2 个 seed"
    else:
        ratio = (oracle - floor) / max(3.0 * sig, 1e-9)
        log(f"  σ_fit（fitness 级） = {sig:.4f}")
        log(f"  headroom / 3σ       = {ratio:.2f}")
        if ratio > 1.0:
            verdict = (f"✅ **空间值得搜**：headroom {oracle-floor:+.4f} > 3σ → "
                       f"下一步用这张表跑进化（策略 = 房间统计 → 配置）。")
        else:
            verdict = (f"⚠ **空间仍太小**：headroom {oracle-floor:+.4f} 未超 3σ={3*sig:.4f} → "
                       f"继续加轴（feat/layers 容量轴）、或加房间/seed 降噪。")
    log(f"  判定：{verdict}")
    return {"per_cfg": {str(k): v for k, v in per_cfg.items()},
            "best_cfg_id": best_ci, "floor": floor, "oracle": oracle,
            "headroom": oracle - floor, "sigma": sig, "verdict": verdict,
            "area": area_info}


def policy_const(ci: int):
    return lambda _stats: int(ci)


# ---------------------------------------------------------------------------
# 策略打分（供 OpenEvolve 的 evaluator 使用；torch-free）
# ---------------------------------------------------------------------------
_LEGAL: Dict[str, List] = {"agg": ["max", "mean", "maxmean"],
                          "graph_space": ["feat", "coord"]}


def normalise_cfg(raw, grid: List[dict]) -> int:
    """把策略返回的东西映射到网格里的一个 cfg id。

    容忍度设计：策略允许返回
      - int / 数字          → 直接当 cfg id（越界则截断）
      - dict                → 逐轴取网格里最接近的合法值（未知轴值退化为该轴首个取值）
      - 其它                → 退化为 cfg 0
    这样即使 LLM 写出畸形返回值，评估也不会崩、且行为确定（可复现）。
    """
    if isinstance(raw, (int, np.integer)):
        return int(min(max(int(raw), 0), len(grid) - 1))
    if isinstance(raw, float):
        if not np.isfinite(raw):
            return 0
        return int(min(max(round(raw), 0), len(grid) - 1))
    if not isinstance(raw, dict):
        return 0
    target: Dict[str, object] = {}
    for axis, vals in _LEGAL.items():
        if axis in raw:
            v = raw[axis]
            target[axis] = v if v in vals else vals[0]
    for axis in ("k", "feat", "layers"):
        if axis in raw:
            try:
                target[axis] = int(raw[axis])
            except (TypeError, ValueError):
                continue
    best_i, best_hit = 0, -1
    for i, cfg in enumerate(grid):
        hit = 0
        for axis, v in target.items():
            if axis in cfg:
                if axis in ("k", "feat", "layers"):
                    if int(cfg[axis]) == int(v):
                        hit += 2
                elif cfg[axis] == v:
                    hit += 2
        if hit > best_hit:
            best_i, best_hit = i, hit
    return best_i


class ArchTable:
    """封装架构表：给定"房间统计 → 配置"的策略，查表得到 fitness。

    fitness = mean_{v ∈ val} mean_{训练房间, seed} mIoU(train_room → v, cfg=policy(v))

    与 `fitness.FitnessTable` 同一口径，只是把"k"换成"完整配置"。
    """

    def __init__(self, table: dict):
        self.t = table
        self.grid: List[dict] = table["grid"]
        self.seeds = [int(s) for s in table["seeds"]]
        self.train_rooms = list(table["train_rooms"])
        self.val_rooms = list(table["val_rooms"])
        self._mat: Optional[Dict[str, Dict[int, float]]] = None

    def matrix(self) -> Dict[str, Dict[int, float]]:
        if self._mat is None:
            _rooms, self._mat = load_matrix(self.t)
        return self._mat

    def score(self, policy, val_stats: Dict[str, dict]) -> dict:
        mat = self.matrix()
        per_val: Dict[str, float] = {}
        cfg_used: Dict[str, int] = {}
        missing: List[str] = []
        for v in self.val_rooms:
            st = val_stats.get(v)
            if st is None:
                continue
            try:
                raw = policy(st)
            except Exception:  # noqa: BLE001  策略异常只记为该房间 0 分，不让整次评估崩
                cfg_used[v] = -1
                continue
            ci = normalise_cfg(raw, self.grid)
            cfg_used[v] = ci
            if v in mat and ci in mat[v]:
                per_val[v] = float(mat[v][ci])
            else:
                missing.append(v)
        if not per_val:
            return {"fitness": 0.0, "worst": 0.0, "per_val": {}, "cfg_used": cfg_used,
                    "missing": missing}
        vals = np.asarray(list(per_val.values()), float)
        return {"fitness": float(vals.mean()), "worst": float(vals.min()),
                "std_across_val": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "per_val": per_val, "cfg_used": cfg_used, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description="架构搜索空间物化表")
    ap.add_argument("--axes-on", nargs="+", default=list(DEFAULT_AXES_ON),
                    choices=["k", "agg", "graph_space", "feat", "layers"])
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="覆盖 k 的取值（只在 k 轴启用时生效）")
    ap.add_argument("--feat", type=int, nargs="+", default=None,
                    help="覆盖 feat 取值（容量轴，会改参数量）")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="覆盖 layers 取值（容量轴）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--val-split", choices=["val", "test"], default="val",
                    help="用哪个 split 当验证房间。**test 只用于最终验收**，"
                         "搜索期间一律用 val（test 会自动加 tag，表名不冲突）")
    ap.add_argument("--tag", default="", help="表文件名标识（默认按轴自动生成）")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--compare-with", type=int, default=0,
                    help="再加载该 epochs 的表并比较两者的排序可迁移性")
    # 规模守卫：本项目出过一次"参数被误绑成 50100150"的事故（PowerShell -File 把
    # 数组当字符串），结果启动了一个永不结束的任务。建表是长任务，必须显式确认规模。
    ap.add_argument("--max-cells", type=int, default=400,
                    help="cells 数上限（默认 400，约 2.7 GPU·h）。超出则拒绝启动")
    ap.add_argument("--allow-large", action="store_true",
                    help="显式允许超过 --max-cells（例如故意建大表）")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    grid = build_grid(args.axes_on, overrides={"k": args.k, "feat": args.feat,
                                               "layers": args.layers})
    tag = args.tag or "+".join(args.axes_on)
    if args.tag == "":
        bits = [f"n{len(grid)}"]
        if args.feat:
            bits.append("f" + "-".join(str(v) for v in args.feat))
        if args.layers:
            bits.append("l" + "-".join(str(v) for v in args.layers))
        tag = "+".join(args.axes_on) + "_" + "_".join(bits)
    if args.val_split == "test":
        tag += "_tval"
        log("[note] 验证房间取自 **test split**（tag 已加 _tval）："
            "该表只用于最终验收，不得用于搜索期间的选择。")
    path = _table_path(tag, args.seeds, args.epochs)

    if args.analyze_only:
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
    else:
        split = RS.load_split()
        tr_all = [d["key"] for d in split["splits"]["train"]][:args.n_train]
        va_all = [d["key"] for d in split["splits"][args.val_split]][:args.n_val]
        cells = len(grid) * len(tr_all) * len(args.seeds)
        est_h = cells * 24.0 / 3600.0        # ~24 s/cell 的实测经验值
        log(f"[arch] 计划：{len(grid)} 配置 × {len(tr_all)} 训练房 × {len(args.seeds)} seed "
            f"= {cells} cells, {args.epochs} ep → 预计 {est_h:.1f} GPU·h")
        if cells > args.max_cells and not args.allow_large:
            log(f"[arch] ✗ 拒绝启动：cells={cells} 超过 --max-cells={args.max_cells}。")
            log(f"[arch]   若确实要跑，请显式加 --allow-large（或先缩小网格/房间数）。")
            return
        table = build(grid, tag, args.seeds, args.epochs, RS.TARGET_N,
                      tr_all, va_all, log=log)
    res = analyse(table, log=log)
    if args.compare_with:
        p2 = _table_path(tag, args.seeds, args.compare_with)
        if os.path.exists(p2):
            with open(p2, "r", encoding="utf-8") as fh:
                other = json.load(fh)
            # 约定：epochs 小的当 cheap，大的当 full
            if int(table["epochs"]) <= int(other["epochs"]):
                cmpres = compare_budgets(table, other, log=log)
            else:
                cmpres = compare_budgets(other, table, log=log)
            res["budget_transfer"] = cmpres
        else:
            log(f"  ⚠ 找不到用于比较的表：{os.path.basename(p2)}（先建它）")
    q = os.path.join(TABLE_DIR, f"summary_{tag}_ep{args.epochs}.json")
    with open(q, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    log(f"  汇总已写入 {q}")


if __name__ == "__main__":
    main()
