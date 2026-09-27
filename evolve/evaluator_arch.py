"""
OpenEvolve 评估器（**架构空间**）：把被进化程序（房间统计 → 架构配置）映射为适应度。

与 `evolve/evaluator.py`（k 空间）的关系
========================================
两个评估器共用同一套纪律，但查询的表不同：
    evaluator.py      → k 表      （`data/fitness_tables/`）
    evaluator_arch.py → 架构表    （`data/arch_tables/`）  ← 本文件
分开写是为了**不动已经跑通的 k 空间链路**。

为什么架构空间用"物化表"而不是廉价代理
========================================
k 轴已实测被榨干（`headroom.py`：逐房间 oracle 的 headroom 只有 +0.0017 < σ）；
真正的机会在架构轴。而架构表一旦建好（`arch_table.py`），任何
"房间 → 配置"策略的适应度都是**毫秒级查表**——无需廉价代理、无预算敏感性、无偏。
所以本评估器不做任何近似：它读的表就是最终口径。

适应度契约（同样最容易踩的坑）
========================================
返回字典**必须**含 `combined_score`；否则 OpenEvolve 会对所有数值指标求平均，
把 `runtime_sec` 之类混进适应度。候选异常早返回时**必须同时给 MAP-Elites 特征维度**，
否则那一轮迭代直接报错、白烧一次 LLM 调用。

⚠ 本文件绝不能 import torch（跑在无 torch 的 openevolve 环境）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from nas_ptseg import arch_table as AT          # noqa: E402  （torch-free）
from nas_ptseg import room_split as RS          # noqa: E402

try:  # 独立运行时允许退化为普通 dict
    from openevolve.evaluation_result import EvaluationResult  # type: ignore

    _HAS_OE = True
except Exception:  # noqa: BLE001
    EvaluationResult = None  # type: ignore
    _HAS_OE = False


# 只允许用这些字段做决策：**刻意剔除 area / room**，否则策略会"记住房间号"，
# 那是过拟合，跨房间必然失效。
ALLOWED_STATS = (
    "n_points", "extent_max_m", "rho_med", "rho_p10", "rho_p90",
    "n_classes_present", "extent",
)

DEFAULT_TABLE = os.path.join(_PROJECT, "data", "arch_tables",
                             "arch_k+agg+graph_space_n6_ep50_s12345-23456.json")

EMPTY_METRICS: Dict[str, float] = {
    "combined_score": 0.0,
    "mean_room_miou": 0.0,
    "worst_room_miou": 0.0,
    "cfg_switch_rate": 0.0,
    "n_val_rooms": 0.0,
    "n_missing": 0.0,
}

_STATE: dict = {}


def _table_path() -> str:
    p = os.environ.get("NAS_LIDAR_ARCH_TABLE", "").strip()
    if p:
        return p if os.path.isabs(p) else os.path.join(_PROJECT, p)
    return DEFAULT_TABLE


def _get_state() -> dict:
    if _STATE.get("ready"):
        return _STATE
    path = _table_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"架构表不存在：{path}\n"
            f"先建表（需 torch 环境，GPU 1）：\n"
            f"    python -m nas_ptseg.arch_table --axes-on k agg graph_space --epochs 50"
        )
    with open(path, "r", encoding="utf-8") as fh:
        table = json.load(fh)
    ft = AT.ArchTable(table)

    val_stats: Dict[str, dict] = {}
    for key in ft.val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room,
                                        target_n=table.get("target_n", RS.TARGET_N))
        val_stats[key] = {k: st[k] for k in ALLOWED_STATS if k in st}

    _STATE.update({"ready": True, "table": table, "ft": ft, "val_stats": val_stats,
                   "table_path": path})
    print(f"[evaluator_arch] 表 {os.path.basename(path)}  "
          f"configs={len(table['grid'])} epochs={table.get('epochs')} "
          f"val_rooms={len(ft.val_rooms)}", file=sys.stderr)
    return _STATE


def _load_policy(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_arch_policy", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载候选程序：{program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "arch_policy", None)
    if not callable(fn):
        raise AttributeError("候选程序必须定义可调用的 arch_policy(room_stats) -> dict|int")
    return fn


def _wrap(metrics: dict, artifacts: dict):
    clean = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            clean[k] = f
    clean["combined_score"] = float(clean.get("combined_score", 0.0))
    if _HAS_OE and EvaluationResult is not None:
        return EvaluationResult(metrics=clean, artifacts=artifacts)
    return clean


def evaluate(program_path: str):
    try:
        st = _get_state()
    except FileNotFoundError as e:
        print(f"[evaluator_arch] {e}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": str(e)})

    try:
        policy = _load_policy(program_path)
    except Exception as e:  # noqa: BLE001
        msg = f"候选程序载入失败：{type(e).__name__}: {e}"
        print(f"[evaluator_arch] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    ft: AT.ArchTable = st["ft"]
    out = ft.score(policy, st["val_stats"])
    per_val: Dict[str, float] = out["per_val"]
    used: Dict[str, int] = out["cfg_used"]
    missing: List[str] = out["missing"]
    if not per_val:
        msg = "所有验证房间都拿不到分数（策略全异常，或表缺该配置）"
        print(f"[evaluator_arch] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    vals = np.asarray(list(per_val.values()), float)
    cfgs = np.asarray([used[v] for v in per_val], float)
    n_distinct = float(len(set(used[v] for v in per_val)))

    metrics = {
        "combined_score": float(vals.mean()),
        "mean_room_miou": float(vals.mean()),
        "worst_room_miou": float(vals.min()),
        # 行为描述量：策略到底"分化"了没有（1 = 所有房间同一个配置）
        "n_distinct_cfgs": n_distinct,
        "cfg_switch_rate": float(n_distinct / max(vals.size, 1)),
        "n_val_rooms": float(vals.size),
        "n_missing": float(len(missing)),
    }
    return _wrap(metrics, {"report": _build_report(vals, per_val, used, missing, st)})


def _build_report(vals, per_val, used, missing, st) -> str:
    grid = st["table"]["grid"]
    lines: List[str] = []
    lines.append("### 跨房间架构泛化评估")
    lines.append("")
    lines.append(f"- fitness (mean room mIoU) = {vals.mean():.4f}   <- 适应度")
    lines.append(f"- worst room mIoU          = {vals.min():.4f}")
    lines.append(f"- 用到的不同配置数         = {len(set(used.values()))} / {len(vals)} 个房间"
                 f"（1 表示策略对所有房间给了同一个配置）")
    lines.append(f"- 可选配置共 {len(grid)} 个：")
    for i, c in enumerate(grid):
        lines.append(f"    - cfg[{i}] {AT.cfg_label(c)}")
    if missing:
        lines.append(f"- ⚠ {len(missing)} 个房间拿不到分数：{missing[:3]}")
    lines.append("")
    lines.append("### 逐验证房间明细")
    lines.append("")
    lines.append("| 验证房间 | 选中的配置 | mIoU |")
    lines.append("|---|---|---|")
    for key in sorted(per_val, key=lambda z: per_val[z]):
        lines.append(f"| {key} | cfg[{used[key]}] {AT.cfg_label(grid[used[key]])} "
                     f"| {per_val[key]:.4f} |")
    lines.append("")
    lines.append("### 改进方向")
    lines.append("")
    lines.append("- 返回值可以是 cfg 下标（int），也可以是 dict（如 "
                 "{\"agg\": \"mean\", \"graph_space\": \"coord\"}），自动匹配到最近配置。")
    lines.append("- 只看平均值容易被个别房间带偏：也盯 worst room。")
    lines.append("- 策略只能依赖 room_stats 里的统计量，**不要试图识别房间身份**。")
    return "\n".join(lines)


if __name__ == "__main__":
    prog = os.path.join(_HERE, "initial_program_arch.py")
    print(f"评估程序: {prog}")
    try:
        st = _get_state()
        print(f"表: {os.path.basename(st['table_path'])}")
        print(f"配置数: {len(st['table']['grid'])}   验证房间: {len(st['val_stats'])}")
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    out = evaluate(prog)
    m = out.metrics if _HAS_OE and EvaluationResult is not None else out
    print("\n--- metrics ---")
    print(json.dumps({k: round(v, 6) for k, v in sorted(m.items())},
                     ensure_ascii=False, indent=2))
    if _HAS_OE and EvaluationResult is not None:
        print(out.artifacts.get("report", ""))
