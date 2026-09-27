"""
OpenEvolve 评估器：把被进化程序（房间统计 → k 的策略）映射为适应度。

适应度契约（最容易踩的坑）
========================================
返回字典**必须**含 `combined_score`。OpenEvolve 的逻辑是：

    有 combined_score  → 用它
    否则               → 把所有数值型指标**求平均**

缺了它，`runtime_sec` / 各类诊断量会被一起平均，适应度完全失真。
所以 `combined_score` 永远是唯一适应度入口。

fitness 的定义（与论文口径一致）
========================================
    fitness = mean_{v ∈ val 房间}
                  mean_{训练房间 r}
                      mIoU(用 k=policy(v) 在 r 上训练的模型 → 在 v 上评估)

要点：
- **k 由验证房间自己的统计决定**（"部署这个策略"的语义），
  不是用训练房间的 k 去评估 —— 后者是 off-policy，语义不对。
- mIoU 只对**该房间真实出现的类别**算（`present_class_miou`），
  否则类别少的房间会因缺失类被记 0 而天然吃亏。
- 分数来自**未曾参与训练的验证房间**，所以测的是跨房间泛化。
- **test 房间在这里完全不可达**（不传就不可能有泄漏）。

为什么这么便宜
========================================
训练/评估被提前"物化"成一张表 `(训练房间, k, seed) → {验证房间: mIoU}`（见
`nas_ptseg/fitness.py`）。于是这里对每个候选策略只需**查表 + 平均**，
单次评估是毫秒级，且与 torch 无关（本文件跑在无 torch 的 openevolve 环境里）。

⚠ 因此本文件**绝不能 import torch**。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from nas_ptseg import fitness as FZ              # noqa: E402  （torch-free）
from nas_ptseg import room_split as RS           # noqa: E402  （torch-free）

try:  # 独立运行（不经 OpenEvolve）时允许退化为普通 dict
    from openevolve.evaluation_result import EvaluationResult  # type: ignore

    _HAS_OE = True
except Exception:  # noqa: BLE001
    EvaluationResult = None  # type: ignore
    _HAS_OE = False

# 只允许用这些字段做决策：**刻意剔除 area / room**，
# 否则策略可以"记住房间号"，那是过拟合，跨房间必然失效。
ALLOWED_STATS = (
    "n_points", "extent_max_m", "rho_med", "rho_p10", "rho_p90",
    "n_classes_present", "extent",
)

CHEAP_ROLE = "cheap"
CHEAP_SEEDS = [12345, 23456]


def _resolve_table_path() -> str:
    """定位要用的 fitness 表。

    可用环境变量（**不需要改代码就能换表**，实测需要频繁切换）：
        NAS_LIDAR_FITNESS_TABLE : 直接指定表文件路径（最高优先级）
        NAS_LIDAR_CHEAP_EPOCHS  : 指定 cheap 预算（默认 FZ.EPOCHS_CHEAP）

    ⚠ 表**必须**是 "训练房间 × 验证房间" 且验证房间取自 **val split**。
    test 表（tag=tval）只用于最终验收，绝不能进搜索回路，否则就是偷看测试集。
    """
    explicit = os.environ.get("NAS_LIDAR_FITNESS_TABLE", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(_PROJECT, explicit)
    ep = os.environ.get("NAS_LIDAR_CHEAP_EPOCHS", "").strip()
    epochs = int(ep) if ep else FZ.EPOCHS_CHEAP
    return FZ._table_path(CHEAP_ROLE, epochs, CHEAP_SEEDS)


# 适应度口径：mean（平均房间）| worst（最差房间）| mix（0.5·mean + 0.5·worst）
# 实测（_def_search.log）：口径**不能**修复短预算的方向反转，所以默认仍用 mean。
FITNESS_KEY = os.environ.get("NAS_LIDAR_FITNESS_KEY", "mean").strip().lower()

# MAP-Elites 的 feature_dimensions 取自 metrics，**任何返回路径都必须带上它们**。
# 否则 openevolve 会报 "Feature dimension ... not found in program metrics"
# （实测：候选程序异常 → 早返回只有 combined_score → 那一轮迭代直接报错、白烧一次 LLM）。
# feature_dimensions 现状：["worst_room_miou", "k_spread"]，见 config.yaml。
EMPTY_METRICS: Dict[str, float] = {
    "combined_score": 0.0,
    "mean_room_miou": 0.0,
    "worst_room_miou": 0.0,
    "k_spread": 0.0,
    "k_mean": 0.0,
    "n_val_rooms": 0.0,
    "n_missing": 0.0,
}


# ---------------------------------------------------------------------------
# 单例缓存
# ---------------------------------------------------------------------------
# 为什么缓存：评估器被高频调用。每次重新读表 + 读 12 个房间的 npz 会把单次评估
# 从毫秒拖到秒级，直接决定有限时间里能跑多少代。
_STATE: dict = {}


def _get_state() -> dict:
    if _STATE.get("ready"):
        return _STATE

    table_path = _resolve_table_path()
    if not os.path.exists(table_path):
        raise FileNotFoundError(
            f"fitness 表不存在：{table_path}\n"
            f"请先建表（需要 torch 环境，GPU 1）：\n"
            f"    python -m nas_ptseg.fitness --role {CHEAP_ROLE} "
            f"--epochs {FZ.EPOCHS_CHEAP} --seeds {' '.join(str(s) for s in CHEAP_SEEDS)}\n"
            f"或用 NAS_LIDAR_FITNESS_TABLE 指定已有表。"
        )
    with open(table_path, "r", encoding="utf-8") as fh:
        table = json.load(fh)

    split = RS.load_split()
    ft = FZ.FitnessTable(table)

    # 载入验证房间统计，并做字段白名单裁剪
    val_stats: Dict[str, dict] = {}
    for key in ft.val_rooms:
        area, room = key.split("/", 1)
        _x, _y, st = RS.load_room_cached(area, room, target_n=table.get("target_n", RS.TARGET_N))
        val_stats[key] = {k: st[k] for k in ALLOWED_STATS if k in st}

    _STATE.update({
        "ready": True,
        "table": table,
        "ft": ft,
        "val_stats": val_stats,
        "split_seed": split.get("seed"),
        "table_path": table_path,
    })
    print(f"[evaluator] 使用表 {os.path.basename(table_path)}  "
          f"(epochs={table.get('epochs')}, seeds={table.get('seeds')}, "
          f"val_rooms={len(ft.val_rooms)}, 口径={FITNESS_KEY})", file=sys.stderr)
    return _STATE


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ---------------------------------------------------------------------------
# 载入被进化程序
# ---------------------------------------------------------------------------
def _load_policy(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_policy", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载候选程序：{program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "geometric_policy", None)
    if not callable(fn):
        raise AttributeError("候选程序必须定义可调用的 geometric_policy(room_stats) -> int")
    return fn


# ---------------------------------------------------------------------------
# 评估入口
# ---------------------------------------------------------------------------
def evaluate(program_path: str):
    """OpenEvolve 调用的入口。program_path 是被进化程序的 .py 文件路径。"""
    try:
        st = _get_state()
    except FileNotFoundError as e:
        print(f"[evaluator] {e}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": str(e)})

    try:
        policy = _load_policy(program_path)
    except Exception as e:  # noqa: BLE001
        msg = f"候选程序载入失败：{type(e).__name__}: {e}"
        print(f"[evaluator] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    ft: FZ.FitnessTable = st["ft"]
    val_stats: Dict[str, dict] = st["val_stats"]

    # 逐验证房间跑策略（策略异常不得让整次评估崩掉，记为该房间 0 分）
    out = ft.score(policy, val_stats)
    per_val: Dict[str, float] = out["per_val"]
    k_used: Dict[str, int] = out["k_used"]
    missing: List[str] = out["missing"]

    if not per_val:
        msg = "所有验证房间都拿不到分数（策略全部异常，或表缺该 k 的 cell）"
        print(f"[evaluator] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    vals = np.asarray(list(per_val.values()), float)
    ks = np.asarray([k_used[v] for v in per_val], float)

    # 适应度口径（见 FITNESS_KEY 说明）
    mean_v = float(vals.mean())
    worst_v = float(vals.min())
    if FITNESS_KEY == "worst":
        fit_v = worst_v
    elif FITNESS_KEY == "mix":
        fit_v = 0.5 * mean_v + 0.5 * worst_v
    else:
        fit_v = mean_v

    # ---- metrics：数值型，供适应度与 MAP-Elites 分箱 ----
    metrics: dict = {
        # 唯一的适应度入口
        "combined_score": fit_v,
        "mean_room_miou": mean_v,
        "worst_room_miou": worst_v,
        # 行为描述量：策略是否真的"随房间变"（0 = 恒定策略）
        "k_spread": float(ks.max() - ks.min()),
        "k_mean": float(ks.mean()),
        "n_val_rooms": float(vals.size),
        "n_missing": float(len(missing)),
    }

    return _wrap(metrics, {"report": _build_report(vals, per_val, k_used, missing, st)})


def _build_report(vals, per_val: Dict[str, float], k_used: Dict[str, int],
                  missing: List[str], st: dict) -> str:
    """构造给 LLM 的反馈文本。

    原则：**指出可行动的方向**。只报一个总分，LLM 无从下手；
    要给"每个房间选了什么 k、得了多少分、哪个房间最差"。
    """
    lines: List[str] = []
    lines.append("### 跨房间泛化评估")
    lines.append("")
    lines.append(f"- fitness (mean room mIoU)  = {vals.mean():.4f}   <- 适应度")
    lines.append(f"- worst room mIoU           = {vals.min():.4f}   <- 最差房间")
    lines.append(f"- best  room mIoU           = {vals.max():.4f}")
    lines.append(f"- k 取值范围               = {min(k_used.values())} ~ {max(k_used.values())}"
                 f"（k_spread={max(k_used.values())-min(k_used.values())}，0 表示策略完全不随房间变）")
    lines.append(f"- 验证房间数               = {vals.size}")
    if missing:
        lines.append(f"- ⚠ 有 {len(missing)} 个房间拿不到分数：{missing[:3]}")
    lines.append("")
    lines.append("### 逐验证房间明细（策略为每个房间挑的 k 与得到的分）")
    lines.append("")
    lines.append("| 验证房间 | 选的 k | mIoU |")
    lines.append("|---|---|---|")
    for key in sorted(per_val, key=lambda z: per_val[z]):
        lines.append(f"| {key} | {k_used[key]} | {per_val[key]:.4f} |")
    lines.append("")
    lines.append("### 改进方向")
    lines.append("")
    lines.append("- 适应度是**所有验证房间的平均**，但也要看 worst：某个房间拖后腿时，")
    lines.append("  想想该房间的统计量（rho / 尺度 / 类别数）和别的房间差在哪、k 该怎么给。")
    lines.append(f"- 当前 k 只能在 {list(FZ.K_GRID)} 里选（返回值会自动吸附到最近的格）。")
    lines.append("- 策略必须只依赖 room_stats 里的统计量；**不要试图识别房间身份**。")
    return "\n".join(lines)


def _wrap(metrics: dict, artifacts: dict):
    # 过滤非有限值；combined_score 必须存在且有限
    clean = {k: v for k, v in metrics.items() if _finite(v) is not None}
    clean["combined_score"] = float(clean.get("combined_score", 0.0))
    if _HAS_OE and EvaluationResult is not None:
        return EvaluationResult(metrics=clean, artifacts=artifacts)
    return clean


# ---------------------------------------------------------------------------
# 独立自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prog = os.path.join(_HERE, "initial_program.py")
    print(f"评估程序: {prog}")
    try:
        st = _get_state()
        print(f"cheap 表: {os.path.basename(st['table_path'])}")
        print(f"验证房间: {len(st['val_stats'])} 个")
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    out = evaluate(prog)
    metrics = out.metrics if _HAS_OE and EvaluationResult is not None else out
    print("\n--- metrics ---")
    print(json.dumps({k: round(v, 6) for k, v in sorted(metrics.items())},
                     ensure_ascii=False, indent=2))
    if _HAS_OE and EvaluationResult is not None:
        print("\n--- artifacts.report ---")
        print(out.artifacts.get("report", ""))
