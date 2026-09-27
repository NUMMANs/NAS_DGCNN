"""PointNeXt 半径轴实测：单尺度扫描 + 固定公比对照。

为什么单独写一个（而不是复用 `multibranch_exp`）
--------------------------------------------------
`multibranch_exp` 的架构串语法（`nas_ptseg/arch_space.py`）是为 **DGCNN + 多分支几何表示**
设计的（`voxel*/pillar*/rangeview*/points*@fuse`）。PointNeXt 的搜索变量是**编码器逐层半径**
（一个 5 元组），语义完全不同 —— 硬塞进 arch_space 只会让两边都变脏。
但**闭环设计照搬**，因为那是本项目验证过的正确形态：

  ① 每个 cell（训练房 × seed）**立即落盘** → 崩溃最多损失 1 cell，天然可续跑
  ② 表名含 epochs 与 split 后缀 → 与验收表隔离
  ③ 启动前 GPU 守卫（导入 `nas_ptseg.gpu` 即钉 GPU 1 + 容量检查）
  ④ `_table_complete()` 复核产物，**不只信 returncode**

命名约定
--------
    s<r>              → 单尺度：5 层全用同一个半径 r
    a<r0>_<r1>_...    → 任意 5 元组（逐层独立半径）
    base              → 上游默认：标量 radius=0.1 被铺成固定公比 ×2

`--list` 只打印待测清单（不碰 GPU），`--dry-run` 打印命令。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ⚠ 必须在 import torch 之前让 GPU 守卫生效（nas_ptseg/__init__.py 顶部已钉卡）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import nas_ptseg.gpu as GPU  # noqa: E402

PN_ROOT = r"D:\vs-project\PointNeXt"
CFG = os.path.join(PN_ROOT, "cfgs", "s3dis", "pointnext-s.yaml")
OUT_DIR = os.path.join(_PROJECT, "data", "pointnext")

PY_EXE = sys.executable
SELF = os.path.abspath(__file__)

# 上游 S3DIS 基线的等价 5 元组（标量 0.1 经 _to_full_list 铺开）
BASELINE: Tuple[float, ...] = (0.1, 0.1, 0.2, 0.4, 0.8)

# 单尺度扫描用的半径值。范围覆盖基线端点：最细 0.02 → 最粗 1.2
SINGLE_RADII: Tuple[float, ...] = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2)


# ---------------------------------------------------------------- 命名 / 规格
def name_of(radii: Sequence[float]) -> str:
    """半径 5 元组 → 配置名。"""
    r = tuple(float(x) for x in radii)
    if len(set(r)) == 1:
        return "s%s" % (("%g" % r[0]).replace(".", "p"))
    return "a" + "_".join(("%g" % x).replace(".", "p") for x in r)


def spec_of(name: str) -> Optional[List[float]]:
    """配置名 → 半径列表。`base` 返回 None（表示用配置文件的标量默认值）。"""
    if name == "base":
        return None
    if name.startswith("s"):
        v = float(name[1:].replace("p", "."))
        return [v] * 5
    if name.startswith("a"):
        return [float(x.replace("p", ".")) for x in name[1:].split("_")]
    return None


def _table_path(name: str, out_dir: str, epochs: int, tag: str = "") -> str:
    """结果表路径。**必须与 `_run_one` 内部完全一致**，否则会无限重跑。"""
    return os.path.join(out_dir, "pn_%s_ep%d%s.json" % (name, int(epochs), tag))


def _table_complete(name: str, out_dir: str, epochs: int, expect: int, tag: str = "") -> bool:
    """表是否跑完（显式数 cell —— 只看文件存在不够，崩溃时文件已存在但内容不全）。"""
    p = _table_path(name, out_dir, epochs, tag)
    if not os.path.exists(p):
        return False
    try:
        rec = json.load(open(p, "r", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return len(rec.get("cells") or {}) >= int(expect)


# ---------------------------------------------------------------- 待测清单
def build_plan(kind: str, include_measured: bool, out_dir: str, epochs: int,
               expect: int) -> List[str]:
    if kind == "single":
        names = [name_of([r] * 5) for r in SINGLE_RADII]
    elif kind == "base":
        names = ["base"]
    else:
        raise SystemExit("未知 --kind: %s" % kind)
    if include_measured:
        return names
    return [n for n in names if not _table_complete(n, out_dir, epochs, expect)]


# ---------------------------------------------------------------- 单配置实测
def _run_one(name: str, radii: Optional[Sequence[float]], epochs: int, seeds: Sequence[int],
             n_train: int, n_val: int, out_dir: str, log_file: str,
             analyze_only: bool = False) -> dict:
    """实测一个半径配置（单进程；被 `main()` 作为子进程调用，也可直接调）。"""
    # ⚠ sys.path 必须在 import openpoints **之前**补好，否则 ModuleNotFoundError
    for _p in (PN_ROOT, _PROJECT):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import torch  # noqa: F401  (延迟导入：GPU 守卫已在 nas_ptseg/__init__ 生效)
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig

    import numpy as np
    from nas_ptseg import room_split as RS

    dev = GPU.setup_torch_device()
    NC = 13

    path = _table_path(name, out_dir, epochs)
    os.makedirs(out_dir, exist_ok=True)

    rec = {"name": name, "radii": None if radii is None else list(radii),
           "epochs": int(epochs), "seeds": [int(s) for s in seeds],
           "n_train": int(n_train), "n_val": int(n_val),
           "baseline_radii": list(BASELINE), "cells": {}}
    if os.path.exists(path):
        try:
            old = json.load(open(path, "r", encoding="utf-8"))
            if old.get("cells"):
                rec["cells"] = old["cells"]
                log("  [resume] 已有 %d cell，续跑" % len(rec["cells"]))
        except Exception:  # noqa: BLE001
            pass

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2, default=str)

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    log("── %s   radii=%s ──" % (name, "baseline(0.1 x2^n)" if radii is None else list(radii)))

    split = RS.load_split()
    tr = [d["key"] for d in split["splits"]["train"]][:n_train]
    va = [d["key"] for d in split["splits"]["val"]][:n_val]

    def build():
        cfg = EasyConfig()
        cfg.load(CFG, recursive=True)
        m = cfg["model"]
        # S3DIS 配置 in_channels=4；我们的协议只有 xyz → 改成 3
        m["encoder_args"]["in_channels"] = 3
        if radii is not None:
            m["encoder_args"]["radius"] = [float(x) for x in radii]
        return build_model_from_cfg(m)

    if not analyze_only:
        m0 = build()
        eff = getattr(m0.encoder, "radii", None)
        log("   params=%.4fM   encoder.radii=%s"
            % (sum(p.numel() for p in m0.parameters()) / 1e6, eff))
        rec["effective_radii"] = eff
        del m0
    save()

    for tr_key in tr:
        a, r = tr_key.split("/", 1)
        xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
        # ⚠ 实测确认：缓存返回的 xyz 已是 (N,3)，不要再转置（踩过一次）
        xyz = torch.tensor(np.ascontiguousarray(xyz_np), dtype=torch.float32, device=dev)
        y = torch.tensor(y_np, dtype=torch.long, device=dev)
        # ⚠ `target_n` 是**目标**点数，实际会浮动（实测有 16389）。
        #   别硬断言等于 16384 —— 用实际形状即可（PointNeXt 对 N 无硬要求）。
        assert xyz.shape[1] == 3 and xyz.shape[0] > 1000, xyz.shape
        assert y.shape[0] == xyz.shape[0], (xyz.shape, y.shape)

        pos = xyz.unsqueeze(0)
        denom = (xyz.max(0).values - xyz.min(0).values + 1e-6)
        x = ((xyz - xyz.min(0).values) / denom).permute(1, 0).unsqueeze(0)
        data = {"pos": pos, "x": x}
        yb = y.unsqueeze(0)

        for seed in seeds:
            ck = "%s|%d" % (tr_key, int(seed))
            if ck in rec["cells"] and rec["cells"][ck].get("miou") is not None:
                log("   [skip] %s（已落盘）" % ck)
                continue
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            model = build().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            model.train()
            for _ in range(3):                      # warmup 3 次（否则耗时虚高可达 40×）
                lo = torch.nn.functional.cross_entropy(model(data), yb)
                opt.zero_grad(set_to_none=True)
                lo.backward()
                opt.step()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for ep in range(int(epochs)):
                out = model(data)
                lo = torch.nn.functional.cross_entropy(out, yb)
                opt.zero_grad(set_to_none=True)
                lo.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / max(1, int(epochs))
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2

            model.eval()
            with torch.no_grad():
                o = model(data)
                mi = _miou(o, yb, st["present_classes"])
            log("   %-22s mIoU=%.4f  %.1f ms/ep  peak=%.0fMiB"
                % (ck, mi, dt * 1000, peak))
            rec["cells"][ck] = {"miou": mi, "ms_per_epoch": dt * 1000,
                                "peak_mib": peak, "train_room": tr_key, "seed": int(seed)}
            save()                                   # ★ 每 cell 落盘
            del model, opt
            torch.cuda.empty_cache()

    save()
    vals = [c["miou"] for c in rec["cells"].values() if c.get("miou") is not None]
    rec["mean"] = float(np.mean(vals)) if vals else None
    rec["seed_range"] = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    rec["cost_s_per_epoch"] = (float(np.mean([c["ms_per_epoch"] for c in rec["cells"].values()]))
                               / 1000.0) if rec["cells"] else None
    save()
    log("   ⇒ mean=%.4f  seed极差=%.4f" % (rec["mean"] or 0, rec["seed_range"] or 0))
    return rec


def _miou(logits, ytrue, present) -> float:
    import numpy as np
    pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    yt = ytrue.cpu().numpy().astype(np.int64)
    ious = []
    for c in present:
        tp = int(((pred == c) & (yt == c)).sum())
        fp = int(((pred == c) & (yt != c)).sum())
        fn = int(((pred != c) & (yt == c)).sum())
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------- 子进程入口
def main() -> None:
    ap = argparse.ArgumentParser(description="PointNeXt 半径轴实测")
    ap.add_argument("--kind", choices=["single", "base"], default="single")
    ap.add_argument("--name", default=None, help="直接指定配置名（子进程模式）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--include-measured", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--list", action="store_true", help="只打印待测清单（不碰 GPU）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single-proc", action="store_true",
                    help="当前进程内直接跑（供调试；默认走子进程以便重试）")
    args = ap.parse_args()

    expect = args.n_train * len(args.seeds)

    # ---- 子进程模式：只跑一个配置 ----
    if args.name:
        _run_one(args.name, spec_of(args.name), args.epochs, args.seeds,
                 args.n_train, args.n_val, args.out_dir, args.log_file)
        return

    plan = build_plan(args.kind, args.include_measured, args.out_dir, args.epochs, expect)
    print("[pn_radius] kind=%s  epochs=%d  seeds=%s  n_train=%d  → %d 个待测"
          % (args.kind, args.epochs, args.seeds, args.n_train, len(plan)), flush=True)
    for n in plan:
        print("    %-16s %s" % (n, spec_of(n) if n != "base" else "baseline(0.1 x2^n)"), flush=True)
    if args.list or not plan:
        return
    if args.dry_run:
        print("[dry-run] 不执行", flush=True)
        return

    if args.single_proc:
        for n in plan:
            _run_one(n, spec_of(n), args.epochs, args.seeds, args.n_train,
                     args.n_val, args.out_dir, args.log_file)
        return

    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "*"})
    for rnd in range(1, max(1, args.max_rounds) + 1):
        todo = [n for n in plan if not _table_complete(n, args.out_dir, args.epochs, expect)]
        if not todo:
            print("[pn_radius] 第 %d 轮：全部完成 ✅" % rnd, flush=True)
            return
        print("[pn_radius] 第 %d 轮：待跑 %d 个 %s" % (rnd, len(todo), todo), flush=True)
        for n in todo:
            cmd = [PY_EXE, "-W", "ignore", SELF, "--name", n,
                   "--epochs", str(args.epochs), "--seeds", *[str(s) for s in args.seeds],
                   "--n-train", str(args.n_train), "--n-val", str(args.n_val),
                   "--out-dir", args.out_dir]
            if args.log_file:
                cmd += ["--log-file", args.log_file]
            r = subprocess.run(cmd, cwd=_PROJECT, env=env)
            ok = _table_complete(n, args.out_dir, args.epochs, expect)
            # ⚠ 不只信 returncode：子进程可能 exit=0 但什么都没写
            print("    [%s] exit=%d  table_complete=%s" % (n, r.returncode, ok), flush=True)
    left = [n for n in plan if not _table_complete(n, args.out_dir, args.epochs, expect)]
    if left:
        print("[pn_radius] ⚠ %d 轮后仍缺：%s" % (args.max_rounds, left), flush=True)


if __name__ == "__main__":
    main()
