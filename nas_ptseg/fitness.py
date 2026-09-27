"""跨房间 fitness：cheap（短训）与 full（长训）同源，用于验证排序可靠性。

核心问题（用户裁定）
================================================================
    搜索是否有意义，取决于：
        fitness_cheap 的排序  能否预测  fitness_full 的排序？
    若不能，继续进化没有意义。

设计：查表聚合，而不是每策略现训
================================================================
朴素做法是"每个候选策略现训一遍"，但一轮 60 次迭代 × 每次十几分钟完全不可行。
这里改成**两阶段查表**：

  阶段一（一次性、可断点续跑）：
      对每个 (训练房间, k, seed) 训一个模型（cheap 用短 epochs，full 用长 epochs），
      在**全部验证房间**上评估，把每个 (训练房间, 验证房间) 的 mIoU 存进表。

  阶段二（几乎免费）：
      给定策略 `k = policy(room_stats)`，对每个验证房间 v 取其策略 k_v，
      fitness = mean_{v} mean_{train} table[(train, k_v)][v]

这样 cheap 与 full 是**同一套数据、同一套评估、只差 epochs**，两者可直接比排序；
而且策略评估退化成查表，可以扫大量候选而不烧算力。

无泄漏
================================================================
- 训练房间取自 split['train']，验证房间取自 split['val']，两者**房间不相交**。
- test 房间在本模块里**完全不可达**（不传就不可能有泄漏）。
- 同一房间的点绝不跨 split。

噪声控制（CRN）
================================================================
- 房间、下采样、验证点集合全部固定（CRN），**只有模型初始化随 seed 变**。
- 每个 (房间, k) 跑多个 seed → 得到 seed 标准差，用于判断"效应是否大于噪声"。
  这是本项目反复踩坑的地方：任何结论前先看 σ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import room_split as RS
    from nas_ptseg import gpu as GPU
    from nas_ptseg.metrics import present_class_miou

    _PKG_MODE = False
else:
    from . import data as DT
    from . import room_split as RS
    from . import gpu as GPU
    from .metrics import present_class_miou

    _PKG_MODE = True

# ⚠ torch **不在此处导入**（延迟到真正训练的 build_table/train_model 内部）。
# 原因：OpenEvolve 的 evaluator 跑在无 torch 的 `openevolve` 环境里，却需要
# import 本模块来用 FitnessTable 查表打分。若模块级 import torch，evaluator 直接
# ImportError。凡新加依赖都要保持"模块级只依赖 numpy"。同样地，DGCNNSeg / to_torch
# 也只在训练路径内延迟导入。

# ---------------------------------------------------------------------------
# 训练超参（cheap 与 full 必须完全一致，只有 epochs 不同）
# ---------------------------------------------------------------------------
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
LR_STEP = 50
LR_GAMMA = 0.5

# k 的候选格：策略只能从这里选 k。离散化让"表"可建，也让 cheap/full 严格可比。
K_GRID: Tuple[int, ...] = (8, 16, 24, 32, 48, 64)

# 默认 epochs：cheap 取"曲线已分离"的窗口下沿（实测 epoch≈40~100 分离，取 25 更省）
EPOCHS_CHEAP = 25
EPOCHS_FULL = 300

TABLE_DIR = os.path.join(DT.PROJECT_ROOT, "data", "fitness_tables")

PolicyFn = Callable[[dict], int]


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------
def clip_to_grid(k: float, grid: Sequence[int] = K_GRID) -> int:
    """把任意 k 吸附到最近的候选格（策略可以返回连续值，落表时离散化）。"""
    arr = np.asarray(grid, dtype=np.float64)
    return int(arr[int(np.argmin(np.abs(arr - float(k))))])


def policy_const(k: int) -> PolicyFn:
    """固定 k（基线：完全不随房间变）。"""
    kk = clip_to_grid(k)
    return lambda stats: kk


def policy_power(k0: float, alpha: float, rho_ref: float) -> PolicyFn:
    """密度条件化：k = clip(round(k0 * (rho/rho_ref)^alpha))。

    方向约定：alpha < 0 → 稀疏域**扩大** k；alpha > 0 → 稀疏域**收缩** k。
    """
    def fn(stats: dict) -> int:
        rho = max(float(stats.get("rho_med", rho_ref)), 1e-9)
        return clip_to_grid(k0 * (rho / max(rho_ref, 1e-9)) ** alpha)
    return fn


def policy_scale(k0: float, beta: float, L_ref: float) -> PolicyFn:
    """尺度条件化：k = clip(round(k0 * (L/L_ref)^beta))。"""
    def fn(stats: dict) -> int:
        L = max(float(stats.get("extent_max_m", L_ref)), 1e-9)
        return clip_to_grid(k0 * (L / max(L_ref, 1e-9)) ** beta)
    return fn


# ---------------------------------------------------------------------------
# 延迟导入 torch 栈（保持本模块在无 torch 环境下可 import）
# ---------------------------------------------------------------------------
def _torch_stack():
    """返回 (torch, F, DGCNNSeg, to_torch)。只在真正训练时才调用。"""
    import torch
    import torch.nn.functional as F
    if _PKG_MODE:
        from .tier2_dgcnn import DGCNNSeg, to_torch
    else:
        from nas_ptseg.tier2_dgcnn import DGCNNSeg, to_torch
    return torch, F, DGCNNSeg, to_torch


# ---------------------------------------------------------------------------
# 训练 / 评估单个模型
# ---------------------------------------------------------------------------
def train_model(
    xyz_tr: np.ndarray,
    y_tr: np.ndarray,
    k: int,
    epochs: int,
    seed: int,
    dev,
):
    torch, F, DGCNNSeg, to_torch = _torch_stack()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X = to_torch(np.ascontiguousarray(xyz_tr.T[None, ...], dtype=np.float32), dev)
    Y = to_torch(np.ascontiguousarray(y_tr, dtype=np.int64), dev)
    model = DGCNNSeg(int(DT.NUM_CLASSES), int(k)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)

    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits[0].t(), Y)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, eval_rooms: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]],
               dev) -> Dict[str, float]:
    """在若干房间上评估，返回 {room_key: mIoU}（只用该房间实际出现的类别）。"""
    torch, _F, _M, to_torch = _torch_stack()
    out: Dict[str, float] = {}
    with torch.no_grad():
        for key, (xyz, labels, present) in eval_rooms.items():
            Xe = to_torch(np.ascontiguousarray(xyz.T[None, ...], dtype=np.float32), dev)
            pr = np.asarray(model(Xe).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
            out[key] = present_class_miou(pr, labels.astype(np.int64), present)
    return out


# ---------------------------------------------------------------------------
# 建表
# ---------------------------------------------------------------------------
def _table_path(role: str, epochs: int, seeds: Sequence[int], tag: str = "") -> str:
    s = "-".join(str(v) for v in seeds)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(TABLE_DIR, f"{role}_ep{epochs}_s{s}{suffix}.json")


def build_table(
    split: dict,
    role: str,
    epochs: int,
    seeds: Sequence[int],
    ks: Sequence[int] = K_GRID,
    target_n: int = RS.TARGET_N,
    train_rooms: Optional[Sequence[str]] = None,
    val_rooms: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    tag: str = "",
    log=print,
) -> dict:
    """建 `(训练房间, k, seed) → {验证房间: mIoU}` 的表，带断点续跑。

    表结构：
        {
          "role": "cheap"|"full", "epochs": int, "seeds": [...], "ks": [...],
          "train_rooms": [...], "val_rooms": [...],
          "cells": { "<train_room>|k=<k>|seed=<s>": {val_room: mIoU, ...}, ... }
        }
    每完成一个 cell 就落盘，长任务被中断后可原地续跑。

    ⚠ `tag` 用于区分"调试子集表"与"正式全量表"。若用 --max-train/--max-val 跑
    子集却不加 tag，子集表会写到正式表的路径上并把正式表覆盖掉。
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = out_path or _table_path(role, epochs, seeds, tag)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
        if (table.get("role") == role and table.get("epochs") == epochs
                and list(table.get("seeds", [])) == list(seeds)
                and list(table.get("ks", [])) == list(ks)):
            log(f"[table] 续跑已有表：{os.path.basename(path)} "
                f"（{len(table['cells'])}/{len(table['train_rooms'])*len(ks)*len(seeds)} cells）")
        else:
            log("[table] 已有表参数不一致，重建")
            table = None
    else:
        table = None

    dev = GPU.setup_torch_device()
    torch, _F, _M, _to = _torch_stack()

    if table is None:
        tr_all = list(train_rooms) if train_rooms else [d["key"] for d in split["splits"]["train"]]
        va_all = list(val_rooms) if val_rooms else [d["key"] for d in split["splits"]["val"]]
        table = {
            "role": role, "epochs": int(epochs),
            "seeds": [int(s) for s in seeds], "ks": [int(k) for k in ks],
            "target_n": int(target_n),
            "train_rooms": tr_all, "val_rooms": va_all,
            "cells": {},
            "device": str(dev),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "gpu_constraint": GPU.summary(),
        }
        log(f"[table] 新建 {role} 表: {len(tr_all)} 训练房间 × {len(ks)} k × "
            f"{len(seeds)} seed = {len(tr_all)*len(ks)*len(seeds)} cells, epochs={epochs}")

    # ---- 载入数据（CRN：房间与下采样对所有 seed 完全一致）----
    t_load = time.perf_counter()
    val_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for key in table["val_rooms"]:
        area, room = key.split("/", 1)
        xyz, labels, st = RS.load_room_cached(area, room, target_n=target_n)
        val_data[key] = (xyz, labels, st["present_classes"])
    tr_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key in table["train_rooms"]:
        area, room = key.split("/", 1)
        xyz, labels, _st = RS.load_room_cached(area, room, target_n=target_n)
        tr_cache[key] = (xyz, labels)
    log(f"[table] 载入 {len(tr_cache)} 训练 + {len(val_data)} 验证房间，"
        f"耗时 {time.perf_counter()-t_load:.1f}s")

    total = len(table["train_rooms"]) * len(table["ks"]) * len(table["seeds"])
    done = 0
    t0 = time.perf_counter()
    for tr_key in table["train_rooms"]:
        xyz_tr, y_tr = tr_cache[tr_key]
        for k in table["ks"]:
            kk = int(min(int(k), max(int(xyz_tr.shape[0]) - 1, 1)))
            for seed in table["seeds"]:
                cell = f"{tr_key}|k={int(k)}|seed={int(seed)}"
                done += 1
                if cell in table["cells"]:
                    continue
                t_one = time.perf_counter()
                model = train_model(xyz_tr, y_tr, kk, epochs, int(seed), dev)
                res = eval_model(model, val_data, dev)
                table["cells"][cell] = res
                del model
                torch.cuda.empty_cache()
                if done % 5 == 0 or done == total:
                    el = time.perf_counter() - t0
                    log(f"[table] {role} {done}/{total} cells  "
                        f"({el:.0f}s, {el/max(done,1):.1f}s/cell, "
                        f"ETA {el/max(done,1)*(total-done):.0f}s)  "
                        f"last {tr_key} k={int(k)} seed={seed} "
                        f"→ {np.mean(list(res.values())):.4f}")
                # 每个 cell 落盘：长任务被中断可原地续跑
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(table, fh, ensure_ascii=False)

    log(f"[table] 完成 {role}：{len(table['cells'])} cells，"
        f"总耗时 {time.perf_counter()-t0:.0f}s")
    return table


# ---------------------------------------------------------------------------
# 表的读取与策略打分
# ---------------------------------------------------------------------------
class FitnessTable:
    """封装一张表，提供"按策略查表聚合"的 fitness，并暴露噪声诊断。

    fitness 定义（务必与论文口径一致）
    ---------------------------------
        fitness(policy) = mean_{v ∈ val} mean_{train room} mIoU(train_room → v, k=policy(v))

    即：策略为每个房间选一个 k；对每个验证房间 v，取所有用 k_v 训练过的模型
    （每个训练房间一个）在 v 上的 mIoU，先在训练房间维平均，再在验证房间维平均。
    """

    def __init__(self, table: dict):
        self.t = table
        self.ks = [int(k) for k in table["ks"]]
        self.seeds = [int(s) for s in table["seeds"]]
        self.train_rooms = list(table["train_rooms"])
        self.val_rooms = list(table["val_rooms"])

    # -- 基础查询 --
    def cell(self, tr: str, k: int, seed: int) -> Optional[dict]:
        return self.t["cells"].get(f"{tr}|k={int(k)}|seed={int(seed)}")

    def val_room_mean(self, tr: str, k: int, seed: int) -> Optional[float]:
        c = self.cell(tr, k, seed)
        if not c:
            return None
        v = list(c.values())
        return float(np.mean(v)) if v else None

    def score(self, policy: PolicyFn, val_stats: Dict[str, dict],
              seeds: Optional[Sequence[int]] = None) -> dict:
        """按策略打分。`val_stats` 给出每个验证房间的统计（策略的输入）。"""
        seeds = list(seeds or self.seeds)
        per_val: Dict[str, float] = {}
        k_used: Dict[str, int] = {}
        missing: List[str] = []
        for v in self.val_rooms:
            st = val_stats.get(v)
            if st is None:
                continue
            k = clip_to_grid(policy(st))
            k_used[v] = k
            acc: List[float] = []
            for s in seeds:
                for tr in self.train_rooms:
                    m = self.val_room_mean(tr, k, s)
                    if m is None:
                        continue
                    # 该 cell 里房间 v 自己的那一格（同一 mIoU 口径）
                    c = self.cell(tr, k, s)
                    if v in c:
                        acc.append(float(c[v]))
            if not acc:
                missing.append(v)
                continue
            per_val[v] = float(np.mean(acc))
        if not per_val:
            return {"fitness": 0.0, "n_val": 0, "per_val": {}, "k_used": k_used,
                    "missing": missing}
        vals = np.asarray(list(per_val.values()), float)
        return {
            "fitness": float(np.mean(vals)),
            "worst": float(np.min(vals)),
            "std_across_val": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "n_val": int(vals.size),
            "per_val": per_val,
            "k_used": k_used,
            "missing": missing,
        }

    # -- 噪声诊断 --
    def seed_noise(self, val_stats: Dict[str, dict], policy: PolicyFn) -> dict:
        """同一策略、不同 seed 的 fitness 分布 → 衡量"分辨率"。

        这是最关键的前置检查：若 seed 噪声与要测的效应同量级，任何排序都不可信。
        ⚠ 只有 1 个 seed 时**噪声是未知的**（std 会算成 0），此时不能宣称"效应大于噪声"。
        所以额外返回 `n_seeds`，调用方必须据此判断可辨识性是否可估。
        """
        per_seed = {}
        for s in self.seeds:
            per_seed[s] = self.score(policy, val_stats, seeds=[s])["fitness"]
        v = np.asarray(list(per_seed.values()), float)
        n = int(v.size)
        return {
            "per_seed": per_seed,
            "n_seeds": n,
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if n > 1 else None,
        }

    def k_effect_resolution(self, val_stats: Dict[str, dict]) -> dict:
        """固定 k 策略下，fitness 随 k 的效应极差 —— 与 seed 噪声之比即分辨率。

        返回 {curve, spread, seed_std, spread_over_3sigma, identifiable, reason}
        比值 >> 1 才说明 k 轴可辨识。**seed 数 < 2 时噪声不可估，一律判为不可判定**，
        否则会把"σ=0"误读成"信噪比无穷大"（典型的假阳性）。
        """
        curve = {}
        for k in self.ks:
            r = self.score(policy_const(k), val_stats)
            curve[k] = r["fitness"]
        vals = np.asarray(list(curve.values()), float)
        noise = self.seed_noise(val_stats, policy_const(self.ks[len(self.ks)//2]))
        spread = float(vals.max() - vals.min())

        if noise["n_seeds"] < 2:
            return {
                "curve": curve, "spread": spread, "seed_std": None,
                "spread_over_3sigma": None, "identifiable": None,
                "reason": (f"只有 {noise['n_seeds']} 个 seed，seed 噪声不可估 —— "
                           f"无法判断 k 轴是否可辨识（请至少跑 2 个 seed）"),
            }

        sigma = max(float(noise["std"]), 1e-9)
        ratio = spread / (3.0 * sigma)
        if ratio < 1.0:
            reason = "k 效应被 seed 噪声淹没：此时任何 'k* 随什么变' 的结论都不可信。"
        else:
            reason = "k 效应大于噪声，可用于排序研究。"
        return {
            "curve": curve, "spread": spread, "seed_std": float(noise["std"]),
            "spread_over_3sigma": ratio,
            "identifiable": bool(ratio >= 1.0),
            "reason": reason,
        }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="跨房间 fitness 表（cheap / full）")
    ap.add_argument("--role", choices=["cheap", "full"], default="cheap")
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"训练 epoch（默认 cheap={EPOCHS_CHEAP} / full={EPOCHS_FULL}）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--ks", type=int, nargs="+", default=list(K_GRID))
    ap.add_argument("--max-train", type=int, default=0, help="只用前 N 个训练房间（调试）")
    ap.add_argument("--max-val", type=int, default=0, help="只用前 N 个验证房间（调试）")
    ap.add_argument("--val-split", choices=["val", "test"], default="val",
                    help="用哪个 split 当验证房间。**test 只用于最终验收**，"
                         "搜索期间一律用 val（自动加 tag，表名不会与 val 表冲突）")
    ap.add_argument("--tag", default="",
                    help="表文件名后缀。调试子集务必带 tag，否则会覆盖正式表")
    ap.add_argument("--analyze-only", action="store_true", help="只读表做诊断，不训练")
    ap.add_argument("--log-file", default="", help="UTF-8 日志文件")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    split = RS.load_split()
    epochs = args.epochs if args.epochs else (
        EPOCHS_CHEAP if args.role == "cheap" else EPOCHS_FULL)

    tag = args.tag
    if not tag and args.val_split == "test":
        tag = "tval"
        log("[note] 验证房间取自 **test split**（自动加 tag='tval'），"
            "该表只应用于最终验收，不得用于搜索期间的选择。")
    if not tag and (args.max_train or args.max_val):
        tag = f"sub{args.max_train or 0}m{args.max_val or 0}"
        log(f"[warn] 子集模式自动加 tag='{tag}'，避免覆盖正式表")

    if args.analyze_only:
        path = _table_path(args.role, epochs, args.seeds, tag)
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
    else:
        pool = args.val_split
        table = build_table(
            split, args.role, epochs, args.seeds, ks=args.ks,
            train_rooms=([d["key"] for d in split["splits"]["train"]][:args.max_train]
                         if args.max_train else None),
            val_rooms=([d["key"] for d in split["splits"][pool]][:args.max_val]
                       if args.max_val else [d["key"] for d in split["splits"][pool]]),
            tag=tag, log=log,
        )

    ft = FitnessTable(table)
    val_stats = {}
    for key in ft.val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=table["target_n"])
        val_stats[key] = st

    log("")
    log("=" * 78)
    log(f"表诊断  role={table['role']}  epochs={table['epochs']}  "
        f"seeds={table['seeds']}  k={table['ks']}")
    log("=" * 78)
    res = ft.k_effect_resolution(val_stats)
    log("  固定 k 的 fitness 曲线（val 房间平均）：")
    for k, v in res["curve"].items():
        log(f"    k={k:>3d}  fitness={v:.4f}")
    log("")
    log(f"  k 效应极差        = {res['spread']:.4f}")
    if res["seed_std"] is None:
        log("  seed 标准差 σ     = 不可估（seed 数 < 2）")
        log(f"  ⚠ {res['reason']}")
    else:
        log(f"  seed 标准差 σ     = {res['seed_std']:.4f}")
        log(f"  极差 / 3σ         = {res['spread_over_3sigma']:.2f}   "
            f"(>>1 才说明 k 轴可辨识)")
        if res["identifiable"]:
            log(f"  ✅ {res['reason']}")
        else:
            log(f"  ⚠ {res['reason']}")


if __name__ == "__main__":
    main()
