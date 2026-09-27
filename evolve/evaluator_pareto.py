"""双目标评估器（**精度 × 成本**）：把"房间统计 → 架构配置"的策略按 Pareto 打分。

为什么需要它（本项目的转向点）
================================================================
此前所有搜索都在"**固定预算下比精度**"：
  · k 轴：逐房间 oracle 的 headroom 仅 +0.0017（< σ）→ 无空间
  · 架构轴（池化 / 图位置 / 容量）：有主效应、**零交互** → 只能拿同族调整的收益
两者的共同问题：**候选之间成本几乎相同**，于是 MAP-Elites 的 `cost_rel` 维度
一直空转（所有点落在同一格），种群多样性只靠一个行为量撑着。

2026-09-24 实测发现：**换几何表示能同时改变精度和成本，且差异达 20 倍量级**：
    `voxel25_z64`  mIoU 0.3885（+0.0515）   15.9 ms/step    424 MiB
    `points`       mIoU 0.3370（基线）     104.2 ms/step   3524 MiB
    `voxel25_z8`   mIoU 0.3290             ~3 ms/step      ~113 MiB
于是"精度 × 成本"第一次成为**有真实差异可探索**的目标空间。

设计选择
================================================================
1. **成本用实测**（`step_sec` / `peak_mib`，由 `multibranch_exp.measure_cost` 产出，
   已修 warmup 伪影），不用参数量代理 —— 参数量与实际耗时在本项目里并不同向
   （voxel 与 points 参数量是 2.2× 差距，但耗时是 20× 差距）。
2. **适应度是标量化**的（`combined_score`，OpenEvolve 的契约），但**特征维度取
   「精度 × 成本」二维**，让 MAP-Elites 显式保留 Pareto 前沿上的多种形态：
   高精度高成本、中精度低成本、以及在两者之间取平衡的策略。
3. **成本归一化用对数刻度**：成本跨 3 个数量级（3ms ~ 200ms），线性分箱会把
   所有 cheap 配置挤进第一格。
4. **组合方案**：`combined_score` 用"相对基线的精度增益"按成本惩罚折算，
   使"用 1/10 成本拿到 90% 精度"这类策略也能得高分，而不是只看精度。

⚠ 本文件绝不能 import torch（跑在无 torch 的 openevolve 环境）。
"""

from __future__ import annotations

import glob
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

# `arch_space` 是纯 Python（不 import torch），可以在无 torch 的 openevolve 环境里用。
# 它负责：解析架构串（严格校验）、估算成本（只估成本，**不估精度**）。
from nas_ptseg import arch_space as ARCH        # noqa: E402

try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
    _HAS_OE = True
except Exception:  # noqa: BLE001
    EvaluationResult = None  # type: ignore
    _HAS_OE = False


# 只允许用这些统计量决策：**刻意剔除 area / room**，否则策略会"记住房间号"（过拟合）。
ALLOWED_STATS = (
    "n_points", "extent_max_m", "rho_med", "rho_p10", "rho_p90",
    "n_classes_present", "extent",
)

# 成本惩罚强度：combined = mean_miou - LAMBDA * log10(cost/cost_ref) 的线性化形式。
# 取 0.02 的含义：成本翻 10 倍 ≈ 扣 0.02 mIoU ≈ 相当于本项目实测的一个真实效应量级。
# 这样"精度高但慢 10 倍"与"精度低 0.02 但快 10 倍"被视为等价 —— 这正是双目标的意义。
LAMBDA_COST = 0.02

EMPTY_METRICS: Dict[str, float] = {
    "combined_score": 0.0,
    "mean_room_miou": 0.0,
    "worst_room_miou": 0.0,
    "log10_cost": 3.0,
    "n_candidates": 0.0,
    "n_distinct": 0.0,
    "is_synthetic": 0.0,
    "n_pending": 0.0,
    "proposed_arch_ms": 0.0,
}

_STATE: dict = {}


def _find_tables() -> List[str]:
    """收集所有多分支结果表（跨多个输出目录）。

    ⚠⚠ **两条过滤是防污染的关键，都不能删**：

    1. **排除 `_tval`（test split 验收表）**
       —— 否则会把 test 精度当成搜索/选择依据，即**测试集泄漏**。
       实测踩过：验收后冠军的值从 val 的 0.4126 变成 test 的 0.4184，
       若不堵死，候选池排序、Pareto 前沿都会基于 test 数字。
       （val 与 test 差距可能很小，但性质上是泄漏，必须从机制上排除。）

    2. **扫描 `data/*/` 全部目录**（不只 `data/multibranch*`）
       —— 否则自建目录（如 `_smoke_archstr`）会被漏掉，
       表现为"明明测过了却显示未测" → **重复训练**，白烧 GPU。
    """
    out: List[str] = []
    out.extend(sorted(glob.glob(os.path.join(_PROJECT, "data", "*", "*_k*_ep*.json"))))
    out.extend(sorted(glob.glob(os.path.join(_PROJECT, "data", "arch_tables", "*.json"))))
    keep: List[str] = []
    for p in out:
        b = os.path.basename(p)
        if b.startswith("summary"):
            continue
        if "_tval" in b:            # test 验收表 → 永不进入搜索候选池
            continue
        keep.append(p)
    return keep


def _load_candidates() -> List[dict]:
    """把各结果表展开成"候选架构"清单，每个候选带 (mean_miou, ms, peak)。

    ⚠ **只收有实测成本的表**（`step_sec` 存在且 > 0）。
    混合"实测成本"与"参数量估算成本"是危险的：本项目的参数量与耗时**并不同向**
    （voxel 与 points 参数量差 2.2×，耗时却差 20×），估算值会把 Pareto 前沿搞乱。
    宁可少几个候选，也不要污染成本轴。

    同名候选的处理：**保留成本数据最完整的那份**（有实测的优先），
    因为同名表在不同目录里可能是不同时期跑的（本项目出现过配置名与参数值不一致的旧表）。

    ⚠ **架构串表也会被收进来**（如 `voxel30_z48@gate_k48_ep300.json`）：
    这是"真正的进化"闭环的关键 —— 进化提出的新架构一经实测，
    下一轮搜索就能读到它的**真实精度**，从而摆脱"估计值天花板"。
    """
    cands: Dict[str, dict] = {}
    for p in _find_tables():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        name = rec.get("name")
        if not name or not rec.get("cells"):
            continue
        ms = float(rec.get("step_sec", 0.0) or 0.0) * 1000.0
        if ms <= 0:
            continue                      # 无实测成本 → 整条跳过（见上）
        vals = [v for cell in rec["cells"].values() for v in cell.values()]
        if not vals:
            continue
        # `@` 是架构串标志（`arch_space` 语法）→ 标记出来，便于统计"进化产物"占比
        cand = {"name": name, "mean_miou": float(np.mean(vals)),
                "n_params": int(rec.get("n_params", 0) or 0),
                "ms": ms, "peak_mib": float(rec.get("peak_mib", 0.0) or 0.0),
                "config": rec.get("config", {}), "source": os.path.basename(p),
                "n_cells": len(rec["cells"]),
                "from_archstr": ("@" in str(name))}
        old = cands.get(name)
        if old is None or cand["n_cells"] > old["n_cells"]:
            cands[name] = cand
    return list(cands.values())


def _get_state() -> dict:
    if _STATE.get("ready"):
        return _STATE
    cands = _load_candidates()
    if not cands:
        raise FileNotFoundError(
            "找不到任何多分支结果表。先跑：\n"
            "    python -m nas_ptseg.multibranch_exp --n-train 12 --n-val 12 ...")
    costs = np.asarray([c["ms"] for c in cands], float)
    mi = np.asarray([c["mean_miou"] for c in cands], float)
    # 基线取名为 points 的那些里精度最高的（没有就取中位数）
    base = [c for c in cands if c["name"] == "points"]
    base_miou = float(np.mean([c["mean_miou"] for c in base])) if base else float(np.median(mi))
    base_ms = float(np.mean([c["ms"] for c in base])) if base else float(np.median(costs))
    _STATE.update({
        "ready": True, "cands": cands, "base_miou": base_miou, "base_ms": base_ms,
        "cost_lo": float(max(costs.min(), 1e-3)), "cost_hi": float(costs.max()),
        "miou_lo": float(mi.min()), "miou_hi": float(mi.max()),
    })
    print(f"[evaluator_pareto] 候选 {len(cands)} 个；基线 points 精度 {base_miou:.4f} "
          f"/ 成本 {base_ms:.1f} ms", file=sys.stderr)
    return _STATE


def _load_policy(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_pareto_policy", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载候选程序：{program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "pareto_policy", None)
    if not callable(fn):
        raise AttributeError("候选程序必须定义 pareto_policy(room_stats) -> dict|int")
    return fn


def _nearest_cfg(raw, cands: List[dict]) -> int:
    """把策略返回值映射到候选清单里的一个下标。

    容忍 int（候选下标）、str（候选名）、dict（配置，按轴匹配）。
    """
    if isinstance(raw, (int, np.integer)):
        return int(min(max(int(raw), 0), len(cands) - 1))
    if isinstance(raw, str):
        for i, c in enumerate(cands):
            if c["name"] == raw:
                return i
        return 0
    if isinstance(raw, dict):
        best_i, best_hit = 0, -1
        for i, c in enumerate(cands):
            hit = 0
            cfg = c.get("config", {})
            for axis, v in raw.items():
                if axis in cfg and cfg[axis] == v:
                    hit += 2
                elif axis in cfg and str(cfg[axis]) == str(v):
                    hit += 1
            if hit > best_hit:
                best_i, best_hit = i, hit
        return best_i
    return 0


def _synth_from_spec(raw: dict, cands: List[dict]) -> Optional[dict]:
    """把"新架构描述"合成为一个候选 —— **成本用估、精度不用估**。

    ⚠⚠ 这是本项目第三次修正这里，前两次的设计都被实测证伪，务必理解为什么：

    **第一版（错）：保守估计精度**（取同分支数的最差实测值）。
    后果：估计值**结构上不可能超过**同分支数的最佳实测值 → 进化没有任何理由
    探索新架构。实测印证：29 迭代 `is_synthetic` **全为 0**。

    **第二版（半错）：加固定探索奖励**（`EXPLORE_BONUS=0.008`）。
    探索被激活了（`is_synthetic` 0 → 11），但仍是"给估计值加分"——
    它奖励的是**猜测**而不是**发现**，且新组合依然赢不过实测强项。

    **第三版（本版）：精度一律置 None，必须实测才能参与排序**。
    理由：成本由**结构**决定（分支数 × z 分层 × 体素尺寸），可稳定外推；
    精度由**优化**决定，**无法从结构推出**。强行估精度只会系统性地压制探索。
    所以这里只给成本，精度标 `pending`；`evaluate()` 对 pending 候选
    返回**该架构串本身作为"待实测建议"**，由 `arch_loop.py` 真正训练并写回表。
    这样架构一旦实测，就会以**真实精度**进入下一轮候选池 ——
    新架构**没有任何"估计值天花板"**，它是凭实测结果与已知配置竞争的。
    """
    br = raw.get("branches")
    if not br:
        return None
    if isinstance(br, (list, tuple)):
        br = ",".join(str(b) for b in br)
    br = str(br)
    fuse = str(raw.get("fuse", "late"))
    feat = int(raw.get("feat", 64))
    arch = f"{br}@{fuse}" + (f"|feat={feat}" if feat != 64 else "")

    # 1) 完全命中已有候选（名字或架构串）→ 直接用实测（最可信）
    for c in cands:
        if c["name"] == arch or str(c.get("config", {}).get("arch")) == arch:
            return dict(c, synthetic=False)
        cfg = c.get("config", {})
        if (str(cfg.get("branches", "")) == br and str(cfg.get("fuse", "")) == fuse):
            return dict(c, synthetic=False)

    # 2) 新架构 → 只估成本，精度 pending
    try:
        spec = ARCH.parse(arch)
    except Exception:  # noqa: BLE001
        spec = None
    if spec is None:
        return None                      # 非法架构串 → 直接拒绝（不猜）
    return {"name": arch, "mean_miou": None,               # ← 精度未知，必须实测
            "n_params": ARCH.estimate_params(spec["branches"], feat),
            "ms": float(spec["cost_ms"]), "peak_mib": 200.0,
            "config": {"branches": br, "fuse": fuse, "feat": feat, "arch": arch},
            "source": "(synthesized: cost-only)", "n_cells": 0,
            "synthetic": True, "pending": True,
            "from_archstr": True}


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
        print(f"[evaluator_pareto] {e}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": str(e)})

    cands: List[dict] = st["cands"]
    try:
        policy = _load_policy(program_path)
    except Exception as e:  # noqa: BLE001
        msg = f"候选程序载入失败：{type(e).__name__}: {e}"
        print(f"[evaluator_pareto] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    # ⚠ 这里**故意**不按房间分化：本项目已实测"房间/域条件化架构"零收益
    #   （逐房间 oracle = 最好单一配置）。所以策略只按**全局资源约束**选一个配置。
    #   保留策略接口是为了让 OpenEvolve 自己发现"不要条件化"，而不是我手工禁掉。
    miou_c, ms_c, n_c, name_used = [], [], 0, ""
    try:
        raw = policy({"n_rooms": len(cands), "note": "global selection"})
    except TypeError:
        try:
            raw = policy({})
        except Exception as e:  # noqa: BLE001
            msg = f"策略执行异常：{type(e).__name__}: {e}"
            print(f"[evaluator_pareto] {msg}", file=sys.stderr)
            return _wrap(dict(EMPTY_METRICS), {"report": msg})
    except Exception as e:  # noqa: BLE001
        msg = f"策略执行异常：{type(e).__name__}: {e}"
        print(f"[evaluator_pareto] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})

    # 允许一次返回多个候选（列表）→ 表示"这几档都可以"，取其中 combined 最高的
    picks = raw if isinstance(raw, (list, tuple)) else [raw]
    best = None
    synth_used: List[dict] = []
    pending: List[dict] = []
    for p in picks:
        if isinstance(p, dict) and "branches" in p:
            # 新架构描述 → 先查实测、查不到则只估成本（精度 pending）
            c = _synth_from_spec(p, cands)
            if c is None:
                continue
            if c.get("pending"):
                pending.append(c)
                synth_used.append(c)
                continue              # pending 候选**不参与打分**（精度未知）
            idx = -1
        else:
            idx = _nearest_cfg(p, cands)
            c = cands[idx]
        cost_pen = LAMBDA_COST * np.log10(max(c["ms"], 1e-3) / max(st["base_ms"], 1e-3))
        score = c["mean_miou"] - cost_pen
        if best is None or score > best[0]:
            best = (score, c, idx)

    # 若策略**只**提出了未实测的新架构（没有可打分的实测候选），
    # 不能凭空给分 —— 返回该架构串作为"待实测建议"，由 arch_loop.py 训练后写回表。
    if best is None and pending:
        prop = min(pending, key=lambda z: z["ms"])     # 建议里选成本最低的，先验证便宜货
        metrics = dict(EMPTY_METRICS)
        metrics.update({"n_candidates": float(len(cands)),
                        "proposed_arch_ms": float(prop["ms"]),
                        "n_pending": float(len(pending))})
        rep = ("### ⏳ 本轮只提出了**未实测**的新架构 —— 已记录为待验证建议\n\n"
               f"- 建议优先实测：`{prop['name']}`\n"
               f"- 估算成本 {prop['ms']:.1f} ms/step（**只有成本是估计**）\n"
               f"- ⚠ **精度未估计** —— 必须真实训练后才能参与排序\n"
               f"  （本项目已实测证明：给估计精度会系统性压制探索）\n\n"
               "实测方式：\n```\n"
               f"python -m nas_ptseg.multibranch_exp --n-train 12 --n-val 12 --only \"{prop['name']}\"\n"
               "```\n"
               "实测完成后，该架构会以**真实精度**出现在下一轮候选池中。")
        return _wrap(metrics, {"report": rep})

    if best is None:
        msg = "策略未返回任何可用候选（全部无法映射或合成）"
        print(f"[evaluator_pareto] {msg}", file=sys.stderr)
        return _wrap(dict(EMPTY_METRICS), {"report": msg})
    score, chosen, idx = best
    miou_c.append(chosen["mean_miou"])
    ms_c.append(chosen["ms"])
    name_used = chosen["name"]
    n_c += 1

    metrics = {
        "combined_score": float(score),
        # ⚠ 精度与成本在合成候选上是**估计值**（含探索奖励），不是实测值。
        #   单列 `is_synthetic` 让日志/数据库能一眼区分，
        #   避免"估计值被当成实测结论"（本项目最忌讳的错误类型）。
        "mean_room_miou": float(chosen["mean_miou"]),
        "delta_vs_base": float(chosen["mean_miou"] - st["base_miou"]),
        # MAP-Elites 特征维度：精度 × 成本（对数）。这两个维度必须有真实差异才有效，
        # 而实测成本跨 ~30×（4ms ~ 104ms），足以铺开。
        "cost_ms": float(chosen["ms"]),
        "log10_cost": float(np.log10(max(chosen["ms"], 1e-3))),
        "peak_mib": float(chosen["peak_mib"]),
        "n_candidates": float(len(cands)),
        "n_distinct": 1.0,
        "speedup_vs_base": float(st["base_ms"] / max(chosen["ms"], 1e-3)),
        # 0/1 标记：选中项是否为**合成的新架构**（其精度/成本均为估计，需实测验证）。
        # 这是本评估器最重要的可观测性字段 —— 没有它就无法判断进化
        # 是在"复述已测候选"还是在"探索新组合"。
        "is_synthetic": 1.0 if chosen.get("synthetic") else 0.0,
    }
    return _wrap(metrics, {"report": _build_report(chosen, st, cands, score, metrics,
                                                   synth_used=synth_used)})


def _build_report(chosen: dict, st: dict, cands: List[dict], score: float,
                  m: dict, synth_used: Optional[List[dict]] = None) -> str:
    lines: List[str] = []
    lines.append("### 双目标（精度 × 成本）评估")
    lines.append("")
    lines.append(f"- combined_score = **{score:.4f}**")
    lines.append(f"  （= 精度 {chosen['mean_miou']:.4f} − {LAMBDA_COST} × "
                 f"log10(成本/{st['base_ms']:.1f}ms)）")
    lines.append(f"- 选中架构：**{chosen['name']}**  {chosen['config']}")
    lines.append(f"- 精度 {chosen['mean_miou']:.4f}  （基线 points {st['base_miou']:.4f}，"
                 f"Δ {m['delta_vs_base']:+.4f}）")
    lines.append(f"- 成本 {chosen['ms']:.1f} ms/step  （基线 {st['base_ms']:.1f} ms，"
                 f"**快 {m['speedup_vs_base']:.1f}×**）")
    lines.append(f"- 峰值显存 {chosen['peak_mib']:.0f} MiB")
    lines.append("")
    lines.append("### 可选架构全表（按精度–成本 Pareto 排序）")
    lines.append("")
    lines.append("| 架构 | 精度 | Δ vs 基线 | ms/step | 加速比 | 峰值MiB | 配置 |")
    lines.append("|---|---|---|---|---|---|---|")
    base_ms = st["base_ms"]
    base_miou = st["base_miou"]
    for c in sorted(cands, key=lambda z: -z["mean_miou"]):
        lines.append(f"| {c['name']} | {c['mean_miou']:.4f} | "
                     f"{c['mean_miou']-base_miou:+.4f} | {c['ms']:.1f} | "
                     f"{base_ms/max(c['ms'],1e-3):.1f}× | {c['peak_mib']:.0f} | "
                     f"`{c['config']}` |")
    lines.append("")
    lines.append("### ⚠ 本轮的关键改进点（请重点看这里）")
    lines.append("")
    lines.append(f"- 本轮选中项：**{'合成的新架构（未实测）' if chosen.get('synthetic') else '已实测的候选'}**")
    lines.append(f"  （`is_synthetic={m.get('is_synthetic', 0.0):.0f}`）")
    lines.append("")
    lines.append("**要提升分数，必须让 `pareto_policy` 返回一个新的配置 dict**，例如：")
    lines.append("")
    lines.append("```python")
    lines.append("def pareto_policy(room_stats):")
    lines.append("    return {")
    lines.append("        \"branches\": [\"voxel20_z32\", \"voxel50_z96\"],   # 可增删/换体素尺寸/换 z 分层")
    lines.append("        \"fuse\": \"gate\",        # late | gate | concat")
    lines.append("        \"z_bins\": 16,          # 未带 _zNN 后缀分支的默认 z 分层数")
    lines.append("        \"feat\": 64,")
    lines.append("    }")
    lines.append("```")
    lines.append("")
    lines.append("**❌ 不要返回架构名字符串**（如 `\"voxel25_z64\"`）——")
    lines.append("那只能从已测好的 63 个固定点里挑，**搜索空间是封闭的**，")
    lines.append("实测已证明该形态下 best 出现在 iteration 1 且之后 59 次迭代毫无改进。")
    lines.append("")
    lines.append("### 已经在表里、但尚未被任何策略选中的高分组合（可直接复用其结构）")
    lines.append("")
    for c in sorted(cands, key=lambda z: -(z["mean_miou"] - LAMBDA_COST *
                                           np.log10(max(z["ms"], 1e-3) / max(st["base_ms"], 1e-3))))[:6]:
        sc = c["mean_miou"] - LAMBDA_COST * np.log10(max(c["ms"], 1e-3) / max(st["base_ms"], 1e-3))
        lines.append(f"- `{c['name']}` combined≈{sc:.4f}  精度 {c['mean_miou']:.4f}  "
                     f"{c['ms']:.1f} ms  `{c['config']}`")
    lines.append("")
    lines.append("### 权衡参考")
    lines.append("- 成本翻 10 倍 ≈ 精度要涨 0.02 才划算。")
    lines.append("- ⚠ 本项目已实测「按房间/域条件化架构」**零收益**"
                 "（逐房间 oracle = 最好单一配置）→ 最优策略是**全局选一个架构**。")
    lines.append("- ⚠ 实测同时表明：**弱分支参与融合会拖累整体**"
                 "（`points`+`rangeview64` = 0.3279 < 基线 0.3370）→ "
                 "加分支前先确认每个分支单独都不弱。")
    return "\n".join(lines)


if __name__ == "__main__":
    prog = os.path.join(_HERE, "initial_program_pareto.py")
    print(f"评估程序: {prog}")
    try:
        st = _get_state()
        print(f"候选 {len(st['cands'])} 个；基线精度 {st['base_miou']:.4f} "
              f"/ 成本 {st['base_ms']:.1f} ms")
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
