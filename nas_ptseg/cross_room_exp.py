"""跨房间跨域实验（方向 1）：判别最优 k 到底由 N / L / rho 中的哪个决定。

与旧 Tier-2（单房间受控密度）的根本区别
================================================================
旧框架里"域"是同一房间的体素下采样 → N、L、rho 三者退化相关（只差平方根），
判别 `k/N` vs `r_k/L` 是死胡同。本模块改用**真实多房间**：
每个房间是一个独立的域，N、L、rho 自然独立。

实验设计
================================================================
对每个房间：
  1. 用该房间训练 DGCNN（固定 k）；
  2. 在**其余房间**上评估泛化 mIoU（present-class mIoU，见下）；
  3. 扫 k ∈ {8,16,32,48,64}，得到该房间的 k* = argmax_k 泛化 mIoU。
然后跨房间回归：k*（或 mIoU 的 k 依赖）如何随 (N, L, rho, k/N, r_k/L) 变化。

⚠ 训练/测试 = 跨房间，**不是**房间内切块 → 旧框架的"留出设计死胡同"彻底消失。

present-class mIoU
==================
不同房间类别覆盖不同（有的没 door，有的没 table）。为可比，每个 (训练房间 →
测试房间) 对只对**测试房间里实际出现的类别**算 mIoU。meta 记录类别覆盖供标注。

用法
====
    # 探路：24 房间 × 5 k，每房间训练后在其余房间评估
    python -m nas_ptseg.cross_room_exp --n-rooms 24 --ks 8 16 32 48 64 \
        --epochs 150 --out-dir cross_room_probe

    # 汇总：跨房间回归判别
    python -m nas_ptseg.cross_room_exp --analyze --out-dir cross_room_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import density as D
    from nas_ptseg import cross_room as CR
    from nas_ptseg import gpu as GPU
    from nas_ptseg.metrics import present_class_miou
    from nas_ptseg.tier2_dgcnn import DGCNNSeg, miou_np, to_torch, FEAT_DIM, NUM_LAYERS

    _PKG_MODE = False
else:
    from . import data as DT
    from . import density as D
    from . import cross_room as CR
    from . import gpu as GPU
    from .metrics import present_class_miou
    from .tier2_dgcnn import DGCNNSeg, miou_np, to_torch, FEAT_DIM, NUM_LAYERS

    _PKG_MODE = True

import torch
import torch.nn.functional as F

LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
LR_STEP = 50
LR_GAMMA = 0.5

DEFAULT_KS = [8, 16, 32, 48, 64]
RESULT_DIR = "cross_room_results"


class _Log:
    def __init__(self, path: Optional[str] = None):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, *args) -> None:
        s = " ".join(str(a) for a in args)
        try:
            print(s)
        except Exception:
            pass
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


# `present_class_miou` 已移到 `nas_ptseg/metrics.py`（torch-free），此处直接复用，
# 避免同一指标两处实现导致口径漂移。为向后兼容仍以本模块的名字导出。
__all__ = ["present_class_miou", "train_and_eval", "run_experiment", "analyze", "main"]


def train_and_eval(
    xyz_tr: np.ndarray,
    y_tr: np.ndarray,
    k: int,
    eval_rooms: List[Tuple[str, np.ndarray, np.ndarray, List[int]]],
    epochs: int,
    dev: torch.device,
    seed: int = 12345,
) -> Dict[str, float]:
    """在一个训练房间上训练固定 k 的 DGCNN，返回对每个测试房间的泛化 mIoU。"""
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
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, xe, ye, present in eval_rooms:
            Xe = to_torch(np.ascontiguousarray(xe.T[None, ...], dtype=np.float32), dev)
            lg = model(Xe)
            pr = np.asarray(lg.argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
            out[name] = present_class_miou(pr, ye.astype(np.int64), present)
    return out


def run_experiment(
    rooms: List[Tuple[str, str]],
    ks: List[int],
    epochs: int,
    seed: int,
    out_dir: str,
    target_n: Optional[int] = 16384,
    log=print,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    domains, meta = CR.load_rooms(rooms, seed=seed, target_n=target_n)

    dev = GPU.setup_torch_device()
    log(f"[cross_room] {len(domains)} 房间, ks={ks}, device={dev}")
    log(f"[cross_room] gpu = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # 预计算每个房间的 present classes（用其自身标签）
    present_of = {key: sorted(int(c) for c in np.unique(y)) for key, (_x, y) in domains.items()}

    # 对每个训练房间 × 每个 k，跑一次训练，评估到其余所有房间
    records = []
    n_room = len(domains)
    keys = list(domains.keys())
    t_start = time.perf_counter()
    for i, tr_key in enumerate(keys):
        xyz_tr, y_tr = domains[tr_key]
        # 评估房间 = 除自己外的所有房间
        eval_rooms = [
            (k, domains[k][0], domains[k][1], present_of[k])
            for k in keys if k != tr_key
        ]
        for k in ks:
            tag = f"{tr_key.replace('/', '__')}__k{k}"
            path = os.path.join(out_dir, tag + ".json")
            if os.path.exists(path):
                continue
            t0 = time.perf_counter()
            acc = train_and_eval(xyz_tr, y_tr, k, eval_rooms, epochs, dev, seed)
            # 汇总：跨房间平均泛化 mIoU + 最差房间
            vals = list(acc.values())
            mean = float(np.mean(vals)) if vals else 0.0
            worst = float(np.min(vals)) if vals else 0.0
            rec = {
                "tag": tag,
                "train_room": tr_key,
                "k": int(k),
                "epochs": int(epochs),
                "seed": seed,
                "n_train_points": int(xyz_tr.shape[0]),
                "extent_max_m": meta["rooms"][tr_key]["extent_max_m"],
                "rho_med": meta["rooms"][tr_key]["rho_med"],
                "n_classes_present": meta["rooms"][tr_key]["n_classes_present"],
                "cross_room_mean_miou": mean,
                "cross_room_worst_miou": worst,
                "per_eval_room": acc,
                "duration_sec": float(time.perf_counter() - t0),
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
            records.append(rec)
            log(f"  [{i+1}/{n_room}] {tr_key} k={k:>3d}  "
                f"mean={mean:.4f} worst={worst:.4f}  ({rec['duration_sec']:.1f}s)")

    log(f"[cross_room] 完成 {len(records)} 个 run, 总耗时 {time.perf_counter()-t_start:.0f}s")


def analyze(out_dir: str, log=print) -> None:
    files = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
             if f.endswith(".json")] if os.path.isdir(out_dir) else []
    if not files:
        log("无结果文件")
        return
    runs = [json.load(open(p, encoding="utf-8")) for p in files]

    log("=" * 72)
    log("跨房间实验汇总：最优 k 由 N / L / rho 中的哪个决定？")
    log("=" * 72)
    log(f"  读入 {len(runs)} 个 run")

    # 每房间的 k 扫描 → k*（按跨房间 mean）
    by_room: Dict[str, Dict[int, float]] = {}
    info: Dict[str, dict] = {}
    for r in runs:
        by_room.setdefault(r["train_room"], {})[r["k"]] = r["cross_room_mean_miou"]
        info[r["train_room"]] = r

    log("")
    log(f"  {'训练房间':<30s} {'N':>8s} {'L(m)':>6s} {'rho':>9s} "
        + "".join(f"{'k='+str(k):>9s}" for k in sorted(by_room[next(iter(by_room))])))
    rows = []
    for room in sorted(by_room):
        ks_here = sorted(by_room[room])
        mious = [by_room[room][k] for k in ks_here]
        kstar = ks_here[int(np.argmax(mious))]
        r = info[room]
        n, L, rho = r["n_train_points"], r["extent_max_m"], r["rho_med"]
        n_cls = r.get("n_classes_present", 0)
        koverN = kstar / max(n, 1)
        # r_k 估算：rho = k/(4/3 π r^3) → r = (3k/(4πρ))^(1/3)；但室内是面采样，
        # 这里只记录 k/N 与 L，r_k/L 用面采样近似 r ≈ (k/(πρ))^(1/2)
        rk_approx = float((kstar / (np.pi * max(rho, 1e-9))) ** 0.5)
        rk_over_L = rk_approx / max(L, 1e-9)
        rows.append({
            "room": room, "n": n, "L": L, "rho": rho, "kstar": kstar,
            "n_cls": n_cls,
            "koverN": koverN, "rk_over_L": rk_over_L,
        })
        cells = "".join(f"{by_room[room][k]:>9.4f}" for k in ks_here)
        mark = "".join(f"{'*' if k == kstar else ' ':>9s}" for k in ks_here)
        log(f"  {room:<30s} {n:>8d} {L:>6.2f} {rho:>9.3e} {cells}")
        log(f"  {'':<30s} {'':>8s} {'':>6s} {'':>9s} {mark}   <- k*={kstar}")

    # 跨房间回归：k* 与 log N / log L / log rho 的相关
    log("")
    log("=" * 72)
    log("跨房间回归：k* 随哪个量变化？")
    log("=" * 72)
    if len(rows) < 4:
        log("  房间数不足，跳过回归")
        return
    ks_arr = np.asarray([r["kstar"] for r in rows], dtype=np.float64)
    for name, get in [
        ("log N", lambda r: np.log(r["n"])),
        ("log L", lambda r: np.log(r["L"])),
        ("log rho", lambda r: np.log(r["rho"])),
        ("n_classes", lambda r: r["n_cls"]),
    ]:
        x = np.asarray([get(r) for r in rows], dtype=np.float64)
        if np.std(x) < 1e-12:
            log(f"  {name:<10s}: 无变化，跳过")
            continue
        c = float(np.corrcoef(x, ks_arr)[0, 1])
        log(f"  k* ~ {name:<10s}   Pearson r = {c:+.3f}  (R² = {c*c:.3f})")

    # 关键警示：N 已被下采样到统一 16384，"log N" 的微小方差是下采样残留噪声，
    # 其 r 值不可读。真正有信息的是 L / rho / n_classes 的跨房间方差。
    log("")
    log("  ⚠ 注意：N 已统一到 ~16384，『log N』的方差只是下采样残留噪声，")
    log("     那条 r 不可读。可读的是 L / rho / n_classes 的跨房间方差。")

    # 也看 mean mIoU 的 k 依赖（不只看 k*，因为 k* 有噪声）
    log("")
    log("  注：k* 本身有 seed 噪声；更稳的做法是看 mIoU(k) 的跨房间曲线形状。")
    log("  以上仅为探路，正式结论需多 seed + 更多房间。")


def main() -> None:
    ap = argparse.ArgumentParser(description="跨房间跨域实验")
    ap.add_argument("--n-rooms", type=int, default=24, help="房间数")
    ap.add_argument("--areas", type=str, nargs="+", default=None, help="限定 Area")
    ap.add_argument("--room-types", type=str, nargs="+", default=None,
                    help="限定房间类型（conferenceRoom/office/hallway/...）")
    ap.add_argument("--rooms-file", default="", help="显式指定房间清单文件（每行 Area/room）")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="k 网格")
    ap.add_argument("--epochs", type=int, default=150, help="训练 epoch")
    ap.add_argument("--target-n", type=int, default=16384,
                    help="每房间体素下采样目标点数（0=保留真实点数，训练慢）")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default=RESULT_DIR, help="结果目录")
    ap.add_argument("--analyze", action="store_true", help="只汇总")
    ap.add_argument("--log-file", default="", help="日志文件（UTF-8）")
    args = ap.parse_args()

    log = _Log(args.log_file or None)

    if args.analyze:
        analyze(args.out_dir, log=log)
        log.close()
        return

    if args.rooms_file:
        rooms = []
        with open(args.rooms_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, r = line.split("/", 1)
                rooms.append((a, r))
    else:
        rooms = CR.select_rooms(n_rooms=args.n_rooms, areas=args.areas,
                                room_types=args.room_types, seed=args.seed)
    log(f"[cross_room] 选定 {len(rooms)} 个房间：")
    for a, r in rooms:
        log(f"    {a}/{r}")

    run_experiment(rooms, [int(k) for k in args.ks], args.epochs, args.seed,
                   args.out_dir, target_n=args.target_n, log=log)
    analyze(args.out_dir, log=log)
    log.close()


if __name__ == "__main__":
    main()
