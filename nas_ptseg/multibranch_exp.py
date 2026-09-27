"""多分支架构对比实验：把"表示"作为唯一变量，在**同预算同 split** 下比效果。

目标（一切以出效果为主）
================================================================
前面几轮已确认：在 DGCNN 内部调超参（k / 池化 / 图位置 / 容量）**只有主效应、零交互**。
本实验直接回答一个更有用的问题：

    **换几何表示，能不能在同一个跨房间基准上拿到更高的 mIoU？**

设计（与 `fitness.py` 完全一致的口径，保证可比）
================================================================
· 同一份 `room_split.json`（12 训练房 → 12 验证房），**不碰 test**
· 同一套训练超参：Adam lr / StepLR / 300 epoch / 同 seed，**唯一变量是架构**
· 同一套指标：`present_class_miou`（只算该房间出现的类）
· 逐 (训练房, 配置, seed) 落盘，可断点续跑（本项目长任务被原生崩溃打断过）

被比较的配置
================================================================
    points            原 DGCNN（笛卡尔点集 + kNN 动态图 + max 池化）   ← 基线
    points_mean       DGCNN + mean 池化（已知：差 0.07）
    points_f128       DGCNN + 特征加宽到 128（已知：+0.019）
    pillar25          柱状 BEV，柱宽 0.25 m（**感受野与点数解耦**）
    pillar50          柱宽 0.50 m（更粗的感受野）
    points+pillar25   双分支 late 融合
    points+pillar50   双分支 late 融合（粗柱）

为什么这些配置的参数量**不齐**也没关系
================================================================
若只比"谁分高"，参数量不等就说不清是结构还是容量。所以本实验**同时报告参数量**，
并给出"**每万参数收益**"这一列；真正的结论看**同参数量档位**内部谁赢
（`points` vs `pillar25` 参数量接近；`points_f128` vs `points+pillar25` 也接近）。
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
    from nas_ptseg import arch_space as ARCH

    _PKG_MODE = False
else:
    from . import gpu as GPU
    from . import data as DT
    from . import room_split as RS
    from . import metrics as MT
    from . import arch_space as ARCH

    _PKG_MODE = True


OUT_DIR = os.path.join(DT.PROJECT_ROOT, "data", "multibranch")

# 名字 → (branches, fuse, DGCNNSeg 风格的关键字)
CONFIGS: List[Tuple[str, dict]] = [
    ("points",           {"branches": "points",           "fuse": "late"}),
    ("points_mean",      {"branches": "points",           "fuse": "late", "agg": "mean"}),
    ("points_f128",      {"branches": "points",           "fuse": "late", "feat": 128}),
    ("pillar25",         {"branches": "pillar25",         "fuse": "late"}),
    ("pillar50",         {"branches": "pillar50",         "fuse": "late"}),
    ("voxel25",          {"branches": "voxel25",          "fuse": "late"}),
    ("voxel50",          {"branches": "voxel50",          "fuse": "late"}),
    # 细调 voxel（第 2 步）：更细的体素 / 更多 z 层 / 更深 3D 卷积
    ("voxel15",          {"branches": "voxel15",          "fuse": "late"}),
    ("voxel20",          {"branches": "voxel20",          "fuse": "late"}),
    # ⚠ 名字里的 zNN 必须等于实际的 z_bins，否则会造成"同名的表内容不同"的混乱。
    #   （曾出现 `voxel25_z32` 实际 z_bins=16、新增的 `voxel25_z32` 又想指 32 的情况。）
    ("voxel25_z8",       {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 8}),
    ("voxel25_z16",      {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 16}),
    ("voxel25_z16_l3",   {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 16, "voxel_layers": 3}),
    ("voxel25_z32",      {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 32}),
    # ---- z_bins 边界扫描（z16→z32 已见 +0.0177，趋势仍在上升，必须扫到平台）----
    ("voxel25_z48",      {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 48}),
    ("voxel25_z64",      {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 64}),
    # 继续向上找平台（z48→z64 仍有 +0.0037，趋势未平）
    ("voxel25_z96",      {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 96}),
    ("voxel25_z128",     {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 128}),
    ("voxel25_z192",     {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 192}),
    ("points+pillar25",  {"branches": "points,pillar25",  "fuse": "late"}),
    ("points+pillar50",  {"branches": "points,pillar50",  "fuse": "late"}),
    ("points+voxel25",   {"branches": "points,voxel25",   "fuse": "late"}),
    ("points+voxel25_z64", {"branches": "points,voxel25", "fuse": "late",
                            "z_bins": 64}),
    ("voxel15_z64",      {"branches": "voxel15",          "fuse": "late",
                          "z_bins": 64}),
    ("voxel35_z64",      {"branches": "voxel35",          "fuse": "late",
                          "z_bins": 64}),
    ("voxel25_z64_l3",   {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 64, "voxel_layers": 3}),
    ("voxel25_z64_f128", {"branches": "voxel25",          "fuse": "late",
                          "z_bins": 64, "feat": 128}),
    # ---- 多尺度 z：并行挂不同 z 分辨率的分支（倒 U 形意味着各 z 有各自盲区）----
    ("ms_z32_z96",       {"branches": "voxel25_z32,voxel25_z96", "fuse": "late"}),
    ("ms_z32_z128",      {"branches": "voxel25_z32,voxel25_z128", "fuse": "late"}),
    ("ms_z64_z128",      {"branches": "voxel25_z64,voxel25_z128", "fuse": "late"}),
    ("ms_z16_z64",       {"branches": "voxel25_z16,voxel25_z64", "fuse": "late"}),
    # ---- 融合方式对比（同一组分支，只改融合结构；gate 只多 2 个参数）----
    ("ms_z32_z96_gate",  {"branches": "voxel25_z32,voxel25_z96", "fuse": "gate"}),
    ("ms_z32_z96_cat",   {"branches": "voxel25_z32,voxel25_z96", "fuse": "concat"}),
    ("pv64_gate",        {"branches": "points,voxel25_z64",      "fuse": "gate"}),
    # 三尺度：再加一个粗尺度提供大范围上下文
    ("ms3_z32_z64_z128", {"branches": "voxel25_z32,voxel25_z64,voxel25_z128",
                          "fuse": "gate"}),
    # ---- 多尺度加码：尺度数是 NAS 的直接搜索目标（3 尺度仍比 2 尺度净增益）----
    # 4 尺度：z32→z192 覆盖更宽的频率范围
    ("ms4_z16_z32_z64_z128", {"branches": "voxel25_z16,voxel25_z32,voxel25_z64,voxel25_z128",
                              "fuse": "gate"}),
    ("ms4_z32_z64_z96_z128", {"branches": "voxel25_z32,voxel25_z64,voxel25_z96,voxel25_z128",
                              "fuse": "gate"}),
    # 5 尺度：加最粗的 z192
    ("ms5_z16_z32_z64_z128_z192",
     {"branches": "voxel25_z16,voxel25_z32,voxel25_z64,voxel25_z128,voxel25_z192",
      "fuse": "gate"}),
    # 换体素尺寸的多尺度（横向也做一次尺度组合，检验"尺度"是否只在 z 上有用）
    ("ms_z32_z96_v35", {"branches": "voxel25_z32,voxel35_z64,voxel25_z128",
                        "fuse": "gate"}),
    # 融合方式在三尺度上的对比（此前只做过二尺度）
    ("ms3_cat", {"branches": "voxel25_z32,voxel25_z64,voxel25_z128",
                 "fuse": "concat"}),
    ("ms3_late", {"branches": "voxel25_z32,voxel25_z64,voxel25_z128",
                  "fuse": "late"}),
    # 最细 + 最粗的两端组合（检验"极值配对"是否比连续取值更好）
    ("ms_z16_z192", {"branches": "voxel25_z16,voxel25_z192", "fuse": "gate"}),
    ("ms_z16_z96", {"branches": "voxel25_z16,voxel25_z96", "fuse": "gate"}),
    ("ms_z48_z128", {"branches": "voxel25_z48,voxel25_z128", "fuse": "gate"}),
    # ---- 体素尺寸 × 尺度 联合搜索（此前几乎全用 0.25m，只试过一个 v35 变体）----
    # 单尺度基准线：同 z、不同体素尺寸（隔离"体素尺寸"这一维）
    ("v35_z64", {"branches": "voxel35_z64", "fuse": "late"}),
    ("v20_z64", {"branches": "voxel20_z64", "fuse": "late"}),
    ("v15_z64", {"branches": "voxel15_z64", "fuse": "late"}),
    ("v50_z64", {"branches": "voxel50_z64", "fuse": "late"}),
    # 多尺度 + 细体素（细体素给精确边界，多尺度给上下文）
    ("ms_v15_z16_z96", {"branches": "voxel15_z16,voxel15_z96", "fuse": "gate"}),
    ("ms_v20_z16_z96", {"branches": "voxel20_z16,voxel20_z96", "fuse": "gate"}),
    ("ms_v35_z16_z96", {"branches": "voxel35_z16,voxel35_z96", "fuse": "gate"}),
    # 多尺度 + 粗体素（粗体素更省，检验能否保持精度）
    ("ms_v50_z16_z96", {"branches": "voxel50_z16,voxel50_z96", "fuse": "gate"}),
    # 混合体素尺寸的多尺度（同一网络里既细又粗，检验"体素尺寸"是否也是有效尺度轴）
    ("ms_mix_v15_v35_z32_z96", {"branches": "voxel15_z32,voxel35_z96",
                                "fuse": "gate"}),
    ("ms_mix_v20_v50_z32_z96", {"branches": "voxel20_z32,voxel50_z96",
                                "fuse": "gate"}),
    ("ms_mix_v15_v25_v35_z32_z96", {"branches": "voxel15_z32,voxel25_z64,voxel35_z96",
                                    "fuse": "gate"}),
    # 三尺度 + 细体素（把上一轮的两条最优线合起来）
    ("ms3_v15_z32_z64_z96", {"branches": "voxel15_z32,voxel15_z64,voxel15_z96",
                             "fuse": "gate"}),
    ("ms3_v20_z32_z64_z96", {"branches": "voxel20_z32,voxel20_z64,voxel20_z96",
                             "fuse": "gate"}),
    # ---- 交叉加码轮：把"多尺度"与"混合体素尺寸"两条已验证轴交叉穷尽 ----
    # 已知：ms3_v20_z32_z64_z96 = 0.4062（最准）；ms_mix_v20_v50_z32_z96 = 0.4037 @15ms（最省）
    # 仍未测：3 尺度 + 混合体素尺寸；2 尺度混合 + 更多尺度；更多分支的组合
    ("ms3_mix_v20_v50_z16_z32_z96",
     {"branches": "voxel20_z16,voxel20_z32,voxel50_z96", "fuse": "gate"}),
    ("ms3_mix_v20_v50_z32_z64_z96",
     {"branches": "voxel20_z32,voxel20_z64,voxel50_z96", "fuse": "gate"}),
    ("ms4_mix_v20_v50_z16_z32_z64_z96",
     {"branches": "voxel20_z16,voxel20_z32,voxel20_z64,voxel50_z96", "fuse": "gate"}),
    # 混合体素尺寸的更多组合（细端用 v20，粗端扫 v35/v50/v96 体素）
    ("ms_mix_v20_v35_z32_z96", {"branches": "voxel20_z32,voxel35_z96",
                                "fuse": "gate"}),
    ("ms_mix_v15_v20_z16_z96", {"branches": "voxel15_z16,voxel20_z96",
                                "fuse": "gate"}),
    # 2 尺度混合的 z 配对扫描（细端固定 v20，粗端换 z）
    ("ms_mix_v20_v50_z16_z64", {"branches": "voxel20_z16,voxel50_z64",
                                "fuse": "gate"}),
    ("ms_mix_v20_v50_z16_z128", {"branches": "voxel20_z16,voxel50_z128",
                                 "fuse": "gate"}),
    ("ms_mix_v20_v50_z32_z128", {"branches": "voxel20_z32,voxel50_z128",
                                 "fuse": "gate"}),
    # 融合方式在大规模配置上的复测（gate 此前只在小配置上验证过）
    ("ms_mix_v20_v50_cat", {"branches": "voxel20_z32,voxel50_z96",
                            "fuse": "concat"}),
    ("ms_mix_v20_v50_late", {"branches": "voxel20_z32,voxel50_z96",
                             "fuse": "late"}),
    # 最省方向：粗体素 + 少尺度（探成本下界，看精度能保住多少）
    ("ms_mix_v35_v50_z32_z96", {"branches": "voxel35_z32,voxel50_z96",
                                "fuse": "gate"}),
    # ---- range-view：第三个表示家族（球面投影 + 距离分辨率衰减 + 视角结构）----
    # 与 points（笛卡尔 kNN）/ voxel（规则 3D 栅格）的归纳偏置都不同
    ("rv64", {"branches": "rangeview64", "fuse": "late"}),
    ("rv128", {"branches": "rangeview128", "fuse": "late"}),
    ("rv32", {"branches": "rangeview32", "fuse": "late"}),
    ("v20+rv64", {"branches": "voxel20_z32,rangeview64", "fuse": "gate"}),
    ("v20v50+rv64", {"branches": "voxel20_z32,voxel50_z96,rangeview64",
                     "fuse": "gate"}),
    ("pts+rv64", {"branches": "points,rangeview64", "fuse": "gate"}),
    ("v20_v50_rv64_late", {"branches": "voxel20_z32,voxel50_z96,rangeview64",
                           "fuse": "late"}),
    ("points+voxel50",   {"branches": "points,voxel50",   "fuse": "late"}),
]


def _torch_stack():
    import torch  # noqa: PLC0415
    from nas_ptseg.multibranch import MultiBranchSeg, count_params  # noqa: PLC0415
    from nas_ptseg import tier2_dgcnn as T2  # noqa: PLC0415
    return torch, T2, MultiBranchSeg, count_params


def measure_cost(model, k: int, dev, reps: int = 3, warmup: int = 3) -> Tuple[float, float]:
    """量单次前向+反传的 wall time，以及 peak 显存（MiB）。

    为什么必须量而不是估：voxel 与 points 的成本差异达 **20+ 倍**，
    这个量级差异是本项目新发现的搜索维度（精度–速度 Pareto）。
    估错会让双目标搜索退化，所以要在目标设备上实测同一份输入。

    ⚠ **必须先 warmup**：首次调用会触发 CUDA 上下文初始化、cuDNN 算法选择与
    算子编译。不 warmup 时 z_bins=8 这种"体素最少"的配置曾被量成 238 ms/step
    （真实值 1.0 ms），比最重的配置还慢 40 倍 —— 纯粹是测量伪影。
    伪影会直接污染 Pareto 前沿（把最快的配置判成最慢的）。
    """
    from nas_ptseg import tier2_dgcnn as T2  # noqa: PLC0415
    torch, _T2, _MBS, _cp = _torch_stack()
    sp = RS.load_split()
    a, r = sp["splits"]["val"][0]["key"].split("/", 1)
    xyz_np, y_np, _st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    X = T2.to_torch(np.ascontiguousarray(xyz_np.T[None, ...], dtype=np.float32), dev)
    Y = T2.to_torch(np.ascontiguousarray(y_np, dtype=np.int64), dev)
    model.train()

    def one_step():
        model.zero_grad(set_to_none=True)
        loss = T2.F.cross_entropy(model(X)[0].t(), Y)
        loss.backward()

    for _ in range(max(1, int(warmup))):
        one_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(int(reps)):
        one_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak = 0.0
    dt = (time.perf_counter() - t0) / max(int(reps), 1)
    return float(dt), float(peak)


def build_model(nc: int, cfg: dict, k: int):
    torch, T2, MBS, _cp = _torch_stack()
    if cfg["branches"] == "points" and cfg.get("agg") != "mean" and cfg.get("feat", 64) == 64:
        # 纯 points 且无改动 → 直接用原 DGCNNSeg，保证与历史数字**完全同构**
        return T2.DGCNNSeg(nc, k)
    if cfg["branches"] == "points" and cfg.get("agg") == "mean":
        return T2.DGCNNSeg(nc, k, agg="mean")
    if cfg["branches"] == "points" and cfg.get("feat", 64) != 64:
        return T2.DGCNNSeg(nc, k, feat=int(cfg["feat"]))
    return MBS(nc, branches=cfg["branches"], k=k, fuse=cfg.get("fuse", "late"),
               feat=int(cfg.get("feat", 64)),
               z_bins=int(cfg.get("z_bins", 16)),
               voxel_layers=int(cfg.get("voxel_layers", 2)),
               rv_layers=int(cfg.get("rv_layers", 3)))

def _known(name: str) -> bool:
    """`--only` 里写了不存在的名字会**静默跳过**（本项目踩过：voxel 从未被执行、
    结果表里却没有它，我还以为跑过了）。所以这里显式校验。

    ⚠ 额外支持 **架构串**（`nas_ptseg.arch_space` 的语法，如
    `voxel20_z32,voxel50_z96@gate`）：进化提出的新组合可以**不经注册**直接实测，
    这是"真正的进化"闭环的关键一步 —— 否则每次新组合都要我手工加 CONFIGS。
    """
    if any(n == name for n, _ in CONFIGS):
        return True
    return _arch_spec(name) is not None


def _arch_spec(name: str):
    """尝试把名字当架构串解析；成功则返回 MultiBranchSeg 的构造 kwargs。

    架构串里的 `|feat=N` 会转成 `feat` 参数，从而支持特征宽度轴。
    """
    spec = ARCH.parse(name)
    if spec is None:
        return None
    return {"branches": ",".join(str(b["name"]) for b in spec["branches"]),
            "fuse": spec["fuse"], "feat": int(spec.get("feat", 64))}


def run_config(name: str, cfg: dict, k: int, epochs: int, seeds: Sequence[int],
               tr_rooms: Sequence[str], va_rooms: Sequence[str], target_n: int,
               out_dir: str, log=print, tag_suffix: str = "") -> dict:
    torch, T2, MBS, count_params = _torch_stack()
    dev = GPU.setup_torch_device()
    # `tag_suffix` 用于区分不同验证 split（如 test 验收表加 "_tval"），
    # 否则验收表会覆盖搜索表 —— 而 test 数字绝不能泄漏回搜索。
    # ⚠ 文件名必须过 `safe_filename`：架构串里的 `|feat=N` 含 `|`，
    #   那是 Windows 非法字符，会让写盘抛 PermissionError →
    #   进程崩在第一格、表完全不存在、外层无限重跑、**整条 feat 轴静默不可测**。
    fname = ARCH.safe_filename(name.replace("+", "_plus_"))
    path = os.path.join(out_dir, f"{fname}_k{k}_ep{epochs}{tag_suffix}.json")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        need = len(tr_rooms) * len(seeds)
        if rec.get("config") == cfg and len(rec.get("cells", {})) >= need:
            log(f"  [skip] {name} 已完成（{len(rec['cells'])}/{need}）")
            return rec
        log(f"  [resume] {name} 续跑（{len(rec.get('cells', {}))}/{need}）")
    else:
        rec = {"name": name, "config": dict(cfg), "k": int(k), "epochs": int(epochs),
               "seeds": [int(s) for s in seeds], "train_rooms": list(tr_rooms),
               "val_rooms": list(va_rooms), "target_n": int(target_n), "cells": {},
               "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}

    m = build_model(int(DT.NUM_CLASSES), cfg, k).to(dev)
    rec["n_params"] = int(sum(p.numel() for p in m.parameters()))
    step_s, peak_mib = measure_cost(m, k, dev)
    rec["step_sec"] = round(step_s, 5)
    rec["peak_mib"] = round(peak_mib, 1)
    log(f"  [cost] {name}: {step_s*1000:.1f} ms/step, peak {peak_mib:.0f} MiB")
    del m
    torch.cuda.empty_cache()

    val_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for v in va_rooms:
        a, r = v.split("/", 1)
        xyz, y, st = RS.load_room_cached(a, r, target_n=target_n)
        val_data[v] = (xyz, y, st["present_classes"])

    t0 = time.perf_counter()
    total = len(tr_rooms) * len(seeds)
    done = 0
    for tr in tr_rooms:
        a, r = tr.split("/", 1)
        xyz_tr, y_tr, _st = RS.load_room_cached(a, r, target_n=target_n)
        X = T2.to_torch(np.ascontiguousarray(xyz_tr.T[None, ...], dtype=np.float32), dev)
        Y = T2.to_torch(np.ascontiguousarray(y_tr, dtype=np.int64), dev)
        for s in seeds:
            done += 1
            key = f"{tr}|seed={int(s)}"
            if key in rec["cells"]:
                continue
            torch.manual_seed(int(s))
            np.random.seed(int(s))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model = build_model(int(DT.NUM_CLASSES), cfg, k).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=T2.LR,
                                   weight_decay=T2.WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP,
                                                    gamma=T2.LR_GAMMA)
            for _ in range(int(epochs)):
                model.train()
                opt.zero_grad(set_to_none=True)
                lg = model(X)
                loss = T2.F.cross_entropy(lg[0].t(), Y)
                loss.backward()
                opt.step()
                sched.step()
            model.eval()
            per_val: Dict[str, float] = {}
            with torch.no_grad():
                for v, (xv, yv, pv) in val_data.items():
                    Xe = T2.to_torch(np.ascontiguousarray(xv.T[None, ...],
                                                          dtype=np.float32), dev)
                    pr = np.asarray(model(Xe).argmax(dim=1)[0].cpu().tolist(),
                                    dtype=np.int64)
                    per_val[v] = MT.present_class_miou(pr, yv.astype(np.int64), pv)
            rec["cells"][key] = per_val
            del model
            torch.cuda.empty_cache()
            el = time.perf_counter() - t0
            log(f"  [{done}/{total}] {name} {key} → {np.mean(list(per_val.values())):.4f}"
                f"  ({el:.0f}s, {el/done:.1f}s/run)")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def summarise(recs: List[dict], log=print) -> dict:
    log("")
    log("=" * 96)
    log("多分支对比：换几何表示能不能拿到更高的 mIoU？")
    log("=" * 96)
    base_name = "points"
    base = next((r for r in recs if r["name"] == base_name), None)

    rows = []
    for r in recs:
        vals = [v for cell in r["cells"].values() for v in cell.values()]
        if not vals:
            continue
        by_seed: Dict[int, List[float]] = {}
        for key, pv in r["cells"].items():
            by_seed.setdefault(int(key.split("seed=")[1]), []).extend(pv.values())
        rows.append({"name": r["name"], "mean": float(np.mean(vals)),
                     "n_params": r.get("n_params", 0),
                     "ms": (float(r["step_sec"]) * 1000.0) if r.get("step_sec") else None,
                     "peak_mib": r.get("peak_mib", 0.0),
                     "seed_means": {k: float(np.mean(v)) for k, v in by_seed.items()}})

    rows.sort(key=lambda z: -z["mean"])
    base_mean = next((x["mean"] for x in rows if x["name"] == base_name), float("nan"))

    log(f"  {'配置':<18s}{'mIoU':>9s}{'Δ vs DGCNN':>13s}{'参数量':>10s}"
        f"{'ms/step':>10s}{'峰值MiB':>10s}")
    for x in rows:
        d = x["mean"] - base_mean if np.isfinite(base_mean) else float("nan")
        log(f"  {x['name']:<18s}{x['mean']:>9.4f}{d:>+13.4f}{x['n_params']:>10,d}"
            f"{x.get('ms', float('nan')):>10.1f}{x.get('peak_mib', 0):>10.0f}")

    # ---- 精度–速度 Pareto 前沿（本项目新发现的搜索维度）----
    # 一个配置被支配 = 存在另一个配置精度 ≥ 它**且**步时 ≤ 它**且有一步严格更优**
    pareto: List[str] = []
    for x in rows:
        ms = x.get("ms")
        if ms is None:
            continue
        dominated = any((y["mean"] >= x["mean"] and y.get("ms", 1e9) <= ms
                         and (y["mean"] > x["mean"] or y.get("ms", 1e9) < ms))
                        for y in rows if y is not x and y.get("ms") is not None)
        if not dominated:
            pareto.append(x["name"])
    log("")
    log(f"  精度–速度 Pareto 前沿（不被任何其他配置同时'更准且更快'支配）：{pareto}")

    # 噪声：同一配置两 seed 均值之差 / √2
    sig = []
    for x in rows:
        sm = list(x["seed_means"].values())
        if len(sm) >= 2:
            sig.append(abs(sm[0] - sm[1]) / np.sqrt(2.0))
    sigma = float(np.mean(sig)) if sig else float("nan")

    log("")
    log(f"  基线 {base_name} = {base_mean:.4f}    σ（配置内 seed 差）≈ {sigma:.4f}")
    # ⚠ 判据必须覆盖**所有换表示的配置**，不能只查一个关键词。
    #   初版只查 "pillar"，于是把 voxel25（真换表示，+0.0279）误标成
    #   "同族容量调整"，并据此得出"换表示失败"的反结论。
    #   第二版只查 ("pillar","voxel")，又把多尺度配置（前缀 "ms"/"pv"）漏掉，
    #   于是宣布最优是 voxel25_z64 —— 而真正的冠军是 ms3_z32_z64_z128（+0.0643）。
    #   **教训：判据的关键词集合必须覆盖全部命名族；新增命名族时必须同步更新这里。**
    REP_KEYS = ("pillar", "voxel", "rangeview", "bev")
    REP_PREFIXES = ("ms", "pv", "ms3")
    rep_names = [x["name"] for x in rows
                 if any(k in x["name"] for k in REP_KEYS)
                 or x["name"].startswith(REP_PREFIXES)]
    best = rows[0]
    gain = best["mean"] - base_mean
    log(f"  全局最优 {best['name']} = {best['mean']:.4f}   Δ = {gain:+.4f}"
        f" = {gain/max(sigma,1e-9):.1f}σ")
    if rep_names:
        rep_best = max((x for x in rows if x["name"] in rep_names),
                       key=lambda z: z["mean"])
        rg = rep_best["mean"] - base_mean
        log(f"  换表示的最好配置 {rep_best['name']} = {rep_best['mean']:.4f}"
            f"   Δ = {rg:+.4f} = {rg/max(sigma,1e-9):.1f}σ")
        if rg > 3 * sigma:
            log("  ✅ **换表示拿到实质收益**（>3σ）")
        elif rg > 0:
            log("  ⚠ 换表示有提升但未超 3σ")
        else:
            log("  ❌ **换表示输给基线** —— 本表示在本数据上不成立")
    log("")
    log("  结论（按证据，不按期望）：")
    if rep_names:
        rep_best = max((x for x in rows if x["name"] in rep_names),
                       key=lambda z: z["mean"])
        rg = rep_best["mean"] - base_mean
        log(f"  换表示的最好配置 {rep_best['name']} = {rep_best['mean']:.4f}"
            f"   Δ = {rg:+.4f} = {rg/max(sigma,1e-9):.1f}σ")
        if rg > 3 * sigma:
            log(f"  ✅ **换表示拿到实质收益**（>3σ）：{rep_best['name']}")
        elif rg > 0:
            log("  ⚠ 换表示有提升但未超 3σ")
        else:
            log("  ❌ **换表示输给基线** —— 本表示在本数据上不成立")
    log("")
    log("  结论（按证据，不按期望）：")
    top_is_rep = bool(rep_names) and rows[0]["name"] in rep_names
    if top_is_rep and gain > 3 * sigma:
        log(f"    · 全局最优 **{rows[0]['name']}** 本身就是换表示配置 → 支持'换几何表示'路线。")
    else:
        log(f"    · 全局最优 {rows[0]['name']}；判'换表示'看上一行的"
            f"换表示最优（不要用全局最优冒充）。")
    log("    · ⚠ **速度必须一起看**：栅格分支单 run 远快于 points（points 的逐层 kNN 是主要开销）。")
    log("      若精度相当/更高而速度快数倍，那是与精度同等有价值的结论。")

    # 误差棒：每个配置的 seed 间极差
    log("")
    log("  逐配置 seed 波动（用于判断上面的 Δ 是否可信）：")
    for x in rows:
        sm = list(x["seed_means"].values())
        if len(sm) >= 2:
            log(f"    {x['name']:<18s} seed 均值 = "
                + ", ".join(f"{v:.4f}" for v in sm)
                + f"   极差 {max(sm)-min(sm):.4f}")
    return {"rows": rows, "sigma": sigma, "base": base_name, "base_mean": base_mean,
            "best": best["name"], "gain": gain}


def main() -> None:
    ap = argparse.ArgumentParser(description="多分支架构对比")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--only", nargs="+", default=None, help="只跑指定配置")
    ap.add_argument("--val-split", choices=["val", "test"], default="val",
                    help="用哪个 split 当验证房间。**test 只用于最终验收**（表名自动加 _tval），"
                         "搜索期间一律用 val")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.only:
        unknown = [n for n in args.only if not _known(n)]
        if unknown:
            log(f"[fail] --only 里有未定义的配置名：{unknown}")
            log(f"       已定义的：{[n for n, _ in CONFIGS]}")
            return
    recs: List[dict] = []
    if args.analyze_only:
        for name in sorted(os.listdir(args.out_dir)):
            if name.endswith(".json") and not name.startswith("summary"):
                with open(os.path.join(args.out_dir, name), "r", encoding="utf-8") as fh:
                    recs.append(json.load(fh))
    else:
        split = RS.load_split()
        tr = [d["key"] for d in split["splits"]["train"]][:args.n_train]
        va = [d["key"] for d in split["splits"][args.val_split]][:args.n_val]
        tsfx = "_tval" if args.val_split == "test" else ""
        if args.val_split == "test":
            log("[note] 验证房间取自 **test split**（表名加 _tval）：")
            log("       该表只用于最终验收，**不得**用于搜索期间的选择。")
        log("=" * 96)
        log(f"多分支对比实验：{len(CONFIGS)} 配置 × {len(tr)} 训练房 × {len(args.seeds)} seed "
            f"× {args.epochs} ep （k={args.k}，val_split={args.val_split}）")
        log("=" * 96)
        for name, cfg in CONFIGS:
            if args.only and name not in args.only:
                continue
            log("")
            log(f"── {name}  {cfg} ──")
            recs.append(run_config(name, cfg, args.k, args.epochs, args.seeds,
                                   tr, va, RS.TARGET_N, args.out_dir, log=log,
                                   tag_suffix=tsfx))

        # ---- 架构串（进化提出的新组合）：直接实测，无需注册到 CONFIGS ----
        # 这是"真正的进化"闭环的关键：进化提出任意合法架构串 →
        # 这里直接训练并落表 → 下一轮搜索即可读到实测值。
        # 旧行为要求每次新组合都手工加 CONFIGS，那等于把人变成搜索的瓶颈。
        arch_names = [n for n in (args.only or []) if not any(n == c for c, _ in CONFIGS)]
        for name in arch_names:
            spec = _arch_spec(name)
            if spec is None:
                log(f"[skip] 不是已知配置名、也不是合法架构串：{name}")
                continue
            log("")
            log(f"── [arch] {name}  {spec} ──")
            recs.append(run_config(name, spec, args.k, args.epochs, args.seeds,
                                   tr, va, RS.TARGET_N, args.out_dir, log=log,
                                   tag_suffix=tsfx))

    res = summarise(recs, log=log)
    q = os.path.join(args.out_dir,
                     "summary_tval.json" if args.val_split == "test" else "summary.json")
    with open(q, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    log(f"  汇总已写入 {q}")


if __name__ == "__main__":
    main()
