"""架构维度探针：k 轴已被证明榨干（headroom ≤ +0.0017 mIoU），
那么**换维度**有没有空间？动手前先量。

为什么先探针而不是直接建表
================================================================
建一张跨架构的表成本是 O(配置数 × 房间数 × seed)，而"这个维度值不值得搜"
只需要知道**配置之间的差异是否超过噪声**。所以先做单轴探针：
固定 k（默认 48 = full@300 下最好的固定 k）、固定 epoch，
逐个换一个架构开关，看 val 房间平均 mIoU 的**跨配置极差**相对 seed 噪声有多大。

被探的维度（全部与原 DGCNN 可比）
================================================================
容量轴（改参数量/计算量）：
    feat    : 32 / 64 / 96 / 128
    layers  : 2 / 3 / 4
结构轴（**参数量不变**，最干净的"同预算比结构"）：
    agg         : max（原实现）/ mean / maxmean
    graph_space : feat（原动态图）/ coord（坐标空间静态图）

判据
================================================================
    跨配置极差 / (3·σ_seed)  —— 与 k 轴同一把尺子。
    k 轴实测这个量只有 ~0.4（headroom 0.0017 vs σ_fit 0.0014）→ 不值得搜。
    若某个架构轴显著大于 1，它才值得进搜索空间。
"""

from __future__ import annotations

import argparse
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


PROBE_DIR = os.path.join(DT.PROJECT_ROOT, "data", "arch_probe")

# (名字, DGCNNSeg 关键字参数)
CONFIGS: List[Tuple[str, dict]] = [
    ("base_f64_l3_max_feat", {}),
    ("feat32",               {"feat": 32}),
    ("feat96",               {"feat": 96}),
    ("feat128",              {"feat": 128}),
    ("layers2",              {"layers": 2}),
    ("layers4",              {"layers": 4}),
    ("agg_mean",             {"agg": "mean"}),
    ("agg_maxmean",          {"agg": "maxmean"}),
    ("graph_coord_static",   {"graph_space": "coord"}),
]


def _torch_stack():
    import torch  # noqa: PLC0415
    from nas_ptseg import tier2_dgcnn as T2  # noqa: PLC0415
    return torch, T2


def run_config(
    name: str, arch: dict, k: int, epochs: int, seeds: Sequence[int],
    train_rooms: Sequence[str], val_rooms: Sequence[str], target_n: int,
    out_path: str, log=print,
) -> dict:
    """跑一个架构配置：每个 (训练房间, seed) 训一次，在所有验证房间上评估。"""
    torch, T2 = _torch_stack()
    dev = GPU.setup_torch_device()

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("name") == name and rec.get("arch_params") == arch \
                and rec.get("k") == k and rec.get("epochs") == epochs:
            log(f"  [skip] 已完成 {name}（{len(rec.get('cells', {}))} cells）")
            return rec
    rec = {
        "name": name, "arch_params": arch, "k": int(k), "epochs": int(epochs),
        "seeds": [int(s) for s in seeds],
        "train_rooms": list(train_rooms), "val_rooms": list(val_rooms),
        "target_n": int(target_n), "cells": {},
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "gpu_constraint": GPU.summary(),
    }

    t0 = time.perf_counter()
    total = len(train_rooms) * len(seeds)
    done = 0
    for tr in train_rooms:
        area, room = tr.split("/", 1)
        xyz_tr, y_tr, _st = RS.load_room_cached(area, room, target_n=target_n)
        for s in seeds:
            done += 1
            key = f"{tr}|seed={int(s)}"
            if key in rec["cells"]:
                continue
            torch.manual_seed(int(s))
            np.random.seed(int(s))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model = T2.DGCNNSeg(int(DT.NUM_CLASSES), int(k), **arch).to(dev)
            n_par = int(sum(p.numel() for p in model.parameters()))
            opt = torch.optim.Adam(model.parameters(), lr=T2.LR, weight_decay=T2.WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP, gamma=T2.LR_GAMMA)
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
            per_val: Dict[str, float] = {}
            with torch.no_grad():
                for v in val_rooms:
                    va, vr = v.split("/", 1)
                    xyz_v, y_v, st_v = RS.load_room_cached(va, vr, target_n=target_n)
                    Xe = T2.to_torch(
                        np.ascontiguousarray(xyz_v.T[None, ...], dtype=np.float32), dev)
                    pr = np.asarray(model(Xe).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
                    per_val[v] = MT.present_class_miou(pr, y_v.astype(np.int64),
                                                       st_v["present_classes"])
            rec["cells"][key] = per_val
            rec["n_params"] = n_par
            del model
            torch.cuda.empty_cache()
            el = time.perf_counter() - t0
            log(f"  [{done}/{total}] {name} {key} → mean {np.mean(list(per_val.values())):.4f}  "
                f"({el:.0f}s, {el/done:.1f}s/run)")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def interaction_report(recs: List[dict], log=print) -> dict:
    """**交互分析**：架构差异里，有多少能靠"按房间选配置"拿到？

    这是与 `summarise` 互补、且更关键的一问。
    `summarise` 量的是**主效应**（配置之间差多少）——它只说明"架构重要"，
    而"架构重要"是文献已知的常识，不构成本课题的发现。

    本课题要回答的是**交互**：最优配置是否随房间变？若最优配置对每个房间都一样，
    那么"按房间/按域条件化架构"这件事就没有任何可拿的收益，NAS 的立论基础不成立。

    判据（与 k 轴 `headroom.py` 同一把尺子）：
        ① 最好单一配置（对所有房间同一个配置）
        ② 逐房间 oracle（每个房间各挑自己最好的配置）—— 上界
        headroom = ② − ①；与 σ 比，不足 3σ 就是"没有交互"。
    """
    if not recs:
        return {}
    names = [r["name"] for r in recs]
    all_rooms = sorted({v for r in recs for v in r["val_rooms"]})

    # mat[room][cfg_index] = 对 (训练房间, seed) 平均
    mat: Dict[str, List[float]] = {v: [] for v in all_rooms}
    for r in recs:
        for v in all_rooms:
            acc: List[float] = []
            for key, per_val in r["cells"].items():
                if v in per_val:
                    acc.append(float(per_val[v]))
            mat[v].append(float(np.mean(acc)) if acc else float("nan"))

    per_cfg = []
    for i, r in enumerate(recs):
        vals = [mat[v][i] for v in all_rooms if np.isfinite(mat[v][i])]
        per_cfg.append(float(np.mean(vals)) if vals else float("nan"))
    best_i = int(np.nanargmax(per_cfg))
    floor = per_cfg[best_i]
    oracle = float(np.nanmean([np.nanmax(mat[v]) if np.isfinite(mat[v]).any() else np.nan
                               for v in all_rooms]))

    # σ：从每个配置的两 seed 均值之差估计（与 summarise 同一做法）
    sigmas = []
    for r in recs:
        by_seed: Dict[int, List[float]] = {}
        for key, per_val in r["cells"].items():
            s = int(key.split("seed=")[1])
            by_seed.setdefault(s, []).extend(list(per_val.values()))
        sm = [float(np.mean(v)) for _s, v in sorted(by_seed.items()) if v]
        if len(sm) >= 2:
            sigmas.append(abs(sm[0] - sm[1]) / np.sqrt(2.0))
    sigma = float(np.mean(sigmas)) if sigmas else float("nan")

    log("")
    log("=" * 92)
    log("交互分析：架构差异里有多少能靠「按房间选配置」拿到？")
    log("=" * 92)
    log(f"  配置 {len(recs)} 个 × 验证房间 {len(all_rooms)} 个")
    log(f"  {'配置':<26s}{'参数量':>10s}{'配置均值':>10s}")
    for i in sorted(range(len(recs)), key=lambda z: -(per_cfg[z] if np.isfinite(per_cfg[z]) else -9)):
        log(f"  {names[i]:<26s}{recs[i].get('n_params', 0):>10d}{per_cfg[i]:>10.4f}")
    log("")
    log(f"  ① 最好单一配置 = {floor:.4f}   ({names[best_i]})")
    log(f"  ② 逐房间 oracle（每房间各挑最好）= {oracle:.4f}")
    log(f"  交互 headroom(②−①) = {oracle - floor:+.4f}")
    if np.isfinite(sigma):
        d = oracle - floor
        log(f"  σ（配置内两 seed 之差）≈ {sigma:.4f}")
        log(f"  headroom / 3σ = {d / max(3.0 * sigma, 1e-9):.2f}")
        if d > 3.0 * sigma:
            verdict = (f"✅ **存在交互**：headroom {d:+.4f} > 3σ → "
                       f"「按房间条件化架构」有真实收益空间，值得做 NAS。")
        else:
            verdict = (f"⚠ **无交互**：headroom {d:+.4f} ≤ 3σ → 最优架构对每个房间都一样；"
                       f"「按房间/按域条件化架构」拿不到收益。"
                       f"架构的主效应（配置间差异）仍然显著，但那是**普通架构选择**，不是条件化 NAS。")
    else:
        verdict = "σ 不可估"
    log(f"  判定：{verdict}")
    return {"names": names, "per_cfg": per_cfg, "best": names[best_i], "floor": floor,
            "oracle": oracle, "headroom": oracle - floor, "sigma": sigma,
            "verdict": verdict}


def load_probe_dir(out_dir: str) -> List[dict]:
    """读取 out_dir 下所有配置结果（排除 summary.json），用于离线交互分析。"""
    recs: List[dict] = []
    if not os.path.isdir(out_dir):
        return recs
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        with open(os.path.join(out_dir, name), "r", encoding="utf-8") as fh:
            recs.append(json.load(fh))
    return recs


def summarise(recs: List[dict], log=print) -> dict:
    log("")
    log("=" * 92)
    log("架构维度探针汇总")
    log("=" * 92)
    log(f"  {'配置':<26s}{'参数量':>10s}{'2 seeds 的分数':>26s}{'均值':>10s}")
    stats: Dict[str, dict] = {}
    for rec in recs:
        by_seed: Dict[int, List[float]] = {}
        for key, pv in rec["cells"].items():
            s = int(key.split("seed=")[1])
            by_seed.setdefault(s, []).extend(list(pv.values()))
        seed_means = {s: float(np.mean(v)) for s, v in by_seed.items() if v}
        if not seed_means:
            continue
        vals = list(seed_means.values())
        stats[rec["name"]] = {
            "n_params": rec.get("n_params"),
            "seed_means": seed_means,
            "mean": float(np.mean(vals)),
            "arch": rec["arch_params"], "k": rec["k"],
        }
        ss = "  ".join(f"{s}:{v:.4f}" for s, v in sorted(seed_means.items()))
        log(f"  {rec['name']:<26s}{rec.get('n_params', 0):>10d}{ss:>26s}"
            f"{np.mean(vals):>10.4f}")

    if not stats:
        log("  没有可用结果。")
        return {}

    names = list(stats)
    means = np.asarray([stats[n]["mean"] for n in names], float)
    # 噪声：每个配置用其两 seed 之差估 σ（同配置内的重训抖动）
    sigmas = []
    for n in names:
        sm = list(stats[n]["seed_means"].values())
        if len(sm) >= 2:
            sigmas.append(abs(sm[0] - sm[1]) / np.sqrt(2.0))
    sigma = float(np.mean(sigmas)) if sigmas else float("nan")

    base = next((n for n in names if n.startswith("base_")), names[0])
    log("")
    log(f"  基线配置       : {base}  mean={stats[base]['mean']:.4f}")
    log(f"  seed 噪声 σ    : {sigma:.4f}")
    log("")
    log(f"  {'配置':<26s}{'Δ vs 基线':>12s}{'Δ/σ':>9s}{'Δ vs 最优':>12s}")
    best_i = int(np.argmax(means))
    for i, n in enumerate(names):
        d = means[i] - stats[base]["mean"]
        db = means[i] - means[best_i]
        log(f"  {n:<26s}{d:>+12.4f}{d/max(sigma,1e-9):>9.1f}{db:>+12.4f}")

    spread = float(means.max() - means.min())
    ratio = spread / max(3.0 * sigma, 1e-9)
    log("")
    log(f"  跨配置极差        = {spread:.4f}")
    log(f"  极差 / 3σ         = {ratio:.2f}   (k 轴实测约 0.4 → 不值得搜)")
    if ratio > 1.0:
        verdict = (f"✅ **存在值得搜的维度**：极差 {spread:.4f} > 3σ → "
                   f"把上面的维度纳入搜索空间。")
    else:
        verdict = (f"⚠ 跨配置极差仍在噪声内（{spread:.4f} vs 3σ={3*sigma:.4f}）→ "
                   f"这些开关也不够，需要换更本质的维度（采样率/池化/坐标编码）"
                   f"或先降低 σ（加 seed / 加房间）。")
    log(f"  判定：{verdict}")
    return {"stats": stats, "sigma": sigma, "spread": spread,
            "spread_over_3sigma": ratio, "verdict": verdict, "baseline": base}


def main() -> None:
    ap = argparse.ArgumentParser(description="架构维度探针")
    ap.add_argument("--k", type=int, default=48, help="固定 k（默认 48 = 最优固定 k）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--n-train", type=int, default=6, help="用前 N 个训练房间（探针够用）")
    ap.add_argument("--n-val", type=int, default=6, help="用前 N 个验证房间")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="只跑指定名字的配置（默认全部）")
    ap.add_argument("--out-dir", default=PROBE_DIR)
    ap.add_argument("--analyze-only", action="store_true",
                    help="不训练，只读 --out-dir 里已有结果做交互分析（零 GPU 成本）")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    split = RS.load_split()
    train_rooms = [d["key"] for d in split["splits"]["train"]][:args.n_train]
    val_rooms = [d["key"] for d in split["splits"]["val"]][:args.n_val]

    log("=" * 92)
    log("架构维度探针：k 轴已榨干，换维度有没有空间？")
    log("=" * 92)
    log(f"  k={args.k}  epochs={args.epochs}  seeds={args.seeds}")
    log(f"  训练房间 {len(train_rooms)} 个 / 验证房间 {len(val_rooms)} 个  "
        f"(target_n={RS.TARGET_N})")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.analyze_only:
        recs = load_probe_dir(args.out_dir)
        if not recs:
            log(f"[probe] {args.out_dir} 下没有可用结果。")
            return
        interaction_report(recs, log=log)
        return

    sel = set(args.configs) if args.configs else None
    recs: List[dict] = []
    for name, arch in CONFIGS:
        if sel and name not in sel:
            continue
        p = os.path.join(args.out_dir, f"{name}_k{args.k}_ep{args.epochs}.json")
        log("")
        log(f"── {name}  {arch} ──")
        recs.append(run_config(name, arch, args.k, args.epochs, args.seeds,
                               train_rooms, val_rooms, RS.TARGET_N, p, log=log))

    res = summarise(recs, log=log)
    if res:
        q = os.path.join(args.out_dir, "summary.json")
        with open(q, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        log(f"  汇总已写入 {q}")


if __name__ == "__main__":
    main()
