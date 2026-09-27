"""PointViG 半径集合的 NAS 搜索执行器。

复用本项目已有的三段闭环设计（提出 → 实测 → 选择），只是换了搜索对象：
  · DGCNN 那套搜的是"几何表示与体素尺度"
  · 这里搜的是 **ADGC 的半径集合**

⚠ 与 DGCNN 那套保持一致的三条硬原则
  1. **唯一变量是架构**（同样的 Adam/StepLR/epoch/数据）—— 否则赢了说不清
  2. **表每 cell 落盘**（原生崩溃最多损失 1 cell）+ 多轮重试
  3. **成本可估，精度绝不估**（合成候选只给成本，精度必须实测）

协议
================================================================
小规模快筛（与首个对照实验同协议，保证可比）：
  4 训练房 × 2 验证房 × 2 seed × 300 ep
→ 单配置约 0.14~0.20 s/ep × 300 × 4 房 × 2 seed ≈ 5~8 分钟
→ 89 个候选约 7~12 小时（可后台跑）

⚠ 先用**小规模**排掉明显差的，再对 Top-k 上全量 12+12 房确认
（这是本项目在 k 轴上验证过的两段式策略：粗筛 + 精验）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from nas_ptseg import vig_space as VS      # noqa: E402

TR = ["Area_1/pantry_1", "Area_2/office_8", "Area_3/office_10", "Area_5/office_25"]
VA = ["Area_1/office_4", "Area_6/office_20"]
SEEDS = [12345, 23456]
EPOCHS = 300
TARGET_N = 16384
NC = 13


def _torch_stack():
    """延迟导入 torch（与项目其余部分一致；本模块也可能被无 torch 环境 import）。"""
    import numpy as np
    import torch
    from nas_ptseg.pointvig import PointViGSeg
    from nas_ptseg import gpu as GPU
    return np, torch, PointViGSeg, GPU


def _miou(pred_np, y_np, present) -> float:
    import numpy as np
    ious = []
    for c in present:
        tp = int(((pred_np == c) & (y_np == c)).sum())
        fp = int(((pred_np == c) & (y_np != c)).sum())
        fn = int(((pred_np != c) & (y_np == c)).sum())
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else 0.0


def run_one(radii: Sequence[float], dev, log=print) -> dict:
    """跑一个半径集合：2 seed 的 val mIoU 均值 + seed 极差 + 成本。"""
    np, torch, PointViGSeg, GPU = _torch_stack()
    from nas_ptseg import room_split as RS

    val_data = {}
    for v in VA:
        a, r = v.split("/", 1)
        xyz, y, st = RS.load_room_cached(a, r, target_n=TARGET_N)
        val_data[v] = (torch.tensor(np.ascontiguousarray(xyz), dtype=torch.float32, device=dev),
                       torch.tensor(y, dtype=torch.long, device=dev),
                       st["present_classes"])
    tr_data = {}
    for t in TR:
        a, r = t.split("/", 1)
        xyz, y, _ = RS.load_room_cached(a, r, target_n=TARGET_N)
        tr_data[t] = (torch.tensor(np.ascontiguousarray(xyz), dtype=torch.float32, device=dev),
                      torch.tensor(y, dtype=torch.long, device=dev))

    per_seed, cost = [], None
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        m = PointViGSeg(in_ch=3, num_classes=NC, radii=list(radii), k=32).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
        t0 = time.perf_counter()
        for tr in TR:
            x, y = tr_data[tr]
            for _ in range(EPOCHS):
                m.train()
                lo = torch.nn.functional.cross_entropy(m(x), y)
                opt.zero_grad(set_to_none=True); lo.backward(); opt.step()
                sched.step()
            if cost is None:
                cost = (time.perf_counter() - t0) / EPOCHS
        m.eval()
        vals = []
        with torch.no_grad():
            for v in VA:
                xv, yv, pv = val_data[v]
                pr = m(xv).argmax(dim=1).cpu().numpy().astype("int64")
                vals.append(_miou(pr, yv.cpu().numpy().astype("int64"), pv))
        per_seed.append(float(np.mean(vals)))
        del m; torch.cuda.empty_cache()
    return {
        "radii": [float(r) for r in radii],
        "name": VS.name_of(radii),
        "n_scales": len(radii),
        "per_seed": per_seed,
        "mean": float(sum(per_seed) / len(per_seed)),
        "seed_range": float(max(per_seed) - min(per_seed)),
        "cost_s_per_epoch": cost,
    }


def measure_all(cands: List[Sequence[float]], out_path: str, log=print) -> dict:
    """**表每 cell 落盘 + 续跑**：已测过的跳过，崩溃最多损失 1 个候选。"""
    np, torch, _PV, GPU = _torch_stack()
    dev = GPU.setup_torch_device()

    rec: dict = {}
    if os.path.exists(out_path):
        try:
            rec = json.load(open(out_path, encoding="utf-8"))
            log("[vig] 续跑：已有 %d 个结果" % len(rec))
        except Exception:
            rec = {}

    for i, radii in enumerate(cands, 1):
        key = VS.name_of(radii)
        if key in rec:
            log("[%d/%d] skip %s（已完成 %.4f）" % (i, len(cands), key, rec[key]["mean"]))
            continue
        log("[%d/%d] %s  radii=%s" % (i, len(cands), key, list(radii)))
        t0 = time.perf_counter()
        try:
            r = run_one(radii, dev, log=log)
        except Exception as e:                     # noqa: BLE001
            log("    FAILED: %r" % (e,))
            continue
        r["wall_s"] = time.perf_counter() - t0
        rec[key] = r
        log("    mean=%.4f  seed极差=%.4f  %.2fs/ep  (%.0fs)"
            % (r["mean"], r["seed_range"], r["cost_s_per_epoch"], r["wall_s"]))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="PointViG 半径集合 NAS 实测")
    ap.add_argument("--max-scales", type=int, default=2,
                    help="最大尺度个数（2 = 快筛；3 = 更全但更贵）")
    ap.add_argument("--include-measured", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部）")
    ap.add_argument("--out", default=os.path.join(_PROJECT, "data", "pointvig", "search.json"))
    ap.add_argument("--name", default=None, help="可选：只跑指定的半径集合，如 0.1,0.2")
    args = ap.parse_args()

    if args.name:
        cands = [[float(x) for x in args.name.split(",")]]
        args.out = os.path.join(_PROJECT, "data", "pointvig", "single.json")
    else:
        cands = VS.build_candidates(max_scales=args.max_scales,
                                    include_measured=args.include_measured)
        if args.limit:
            cands = cands[:args.limit]

    print("[vig] 候选 %d 个 → %s" % (len(cands), args.out), flush=True)
    rec = measure_all(cands, args.out)

    # —— 选择：按 mean 排序，但**差异必须超过 seed 极差**才算可信 ——
    rows = sorted(rec.values(), key=lambda z: -z["mean"])
    print()
    print("=== 排序（前 15）===")
    print("%-24s %8s %10s %8s %8s" % ("radii", "mean", "seed极差", "s/ep", "n_scale"))
    for r in rows[:15]:
        print("%-24s %8.4f %10.4f %8.3f %8d" % (
            "|".join("%g" % q for q in r["radii"]), r["mean"],
            r["seed_range"], r["cost_s_per_epoch"] or -1, r["n_scales"]))

    base = None
    for r in rows:
        if r["radii"] == [0.2]:
            base = r
            break
    if base:
        print()
        print("=== 相对论文单尺度 [0.2]（%.4f）===" % base["mean"])
        for r in rows[:8]:
            d = r["mean"] - base["mean"]
            ok = "可信" if abs(d) > max(r["seed_range"], base["seed_range"]) else "噪声内"
            print("  %-24s %+.4f  (%s)" % ("|".join("%g" % q for q in r["radii"]), d, ok))


if __name__ == "__main__":
    main()
