"""双目标（精度 × 成本）选择的**精确解**：直接枚举，不用 LLM。

为什么需要这个模块（一个重要的方法论修正）
================================================================
本项目此前用 OpenEvolve + LLM 来"选架构"，但这是一个**结构性错配**：

- **候选池是有限的**（~70 个已实测配置），**目标函数是确定的**
  （`combined = mIoU − λ·log10(cost/base_cost)`）。
- 在一个有限集上求确定目标的最优 → **枚举就是精确解**，且是确定性的、瞬时的。
- 让 LLM 反复生成策略来"选"一个有界集合里的元素，是绕远路：
  实测印证 —— 两轮各 60 迭代的搜索，best 都出现在**早期迭代**，
  且最终 best（0.4216）**还不如直接枚举出来的解（0.4244）**。

**正确的分工**：
| 环节 | 工具 | 理由 |
|---|---|---|
| **提出**新架构 | `arch_loop.py`（采样）/ LLM | 需要创造力，空间 3.2×10^8，枚举不了 |
| **选择**最优 | **本模块（枚举）** | 候选有限、目标确定 → 枚举即最优 |
| **验证** | 真实训练后写回表 | 摆脱"估计值天花板" |

**LLM 的价值在"提出"，不在"选择"。** 把这两件事分开，系统才既可靠又最优。

用法
================================================================
    python -m nas_ptseg.pareto_select                 # 打印精确最优与 Pareto 前沿
    python -m nas_ptseg.pareto_select --lambda-cost 0.05
    python -m nas_ptseg.pareto_select --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)


def _load_evaluator():
    """复用 `evolve/evaluator_pareto.py` 的候选加载逻辑，避免两处实现漂移。

    ⚠ 那个模块**必须保持 torch-free**（它跑在无 torch 的 openevolve 环境里），
    所以这里 import 它也安全。
    """
    p = os.path.join(_PROJECT, "evolve", "evaluator_pareto.py")
    spec = importlib.util.spec_from_file_location("_ev_sel", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combined_score(miou: float, ms: float, base_ms: float, lam: float) -> float:
    """双目标适应度：精度 − λ·log10(成本/基线成本)。

    成本取**对数**是必需的：本项目成本跨约 25×（4~104 ms），
    线性惩罚会把所有低精度-低成本配置一票否决，而它们正是 Pareto 前沿的左侧。
    """
    return float(miou) - float(lam) * math.log10(max(float(ms), 1e-3) / max(float(base_ms), 1e-3))


def pareto_front(cands: List[dict]) -> List[dict]:
    """真实 Pareto 前沿：按成本升序扫描，只保留精度严格创新高的。

    这样得到的每个点都"不被任何其他候选同时更准且更快支配"。
    """
    front: List[dict] = []
    for c in sorted(cands, key=lambda z: z["ms"]):
        if not front or c["mean_miou"] > front[-1]["mean_miou"]:
            front.append(c)
    return front


def select(lam: float = 0.02, verbose: bool = False) -> Dict[str, object]:
    """返回最优候选与完整排序/前沿。"""
    ev = _load_evaluator()
    st = ev._get_state()
    cands = st["cands"]
    base_miou, base_ms = st["base_miou"], st["base_ms"]

    rows = sorted(cands, key=lambda c: -combined_score(c["mean_miou"], c["ms"], base_ms, lam))
    best = rows[0]
    return {
        "best": best,
        "best_combined": combined_score(best["mean_miou"], best["ms"], base_ms, lam),
        "ranked": rows,
        "front": pareto_front(cands),
        "base_miou": base_miou,
        "base_ms": base_ms,
        "lambda": lam,
        "n_candidates": len(cands),
        "n_evolved": sum(1 for c in cands if c.get("from_archstr")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="双目标选择的精确解（枚举，不用 LLM）")
    ap.add_argument("--lambda-cost", type=float, default=0.02,
                    help="成本惩罚系数（默认 0.02：成本翻 10 倍需精度涨 0.02 才划算）")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    r = select(args.lambda_cost)
    b = r["best"]
    print(f"候选数 = {r['n_candidates']}（其中进化产出 {r['n_evolved']}）")
    print(f"基线 points: mIoU={r['base_miou']:.4f}  {r['base_ms']:.1f} ms"
          f"   λ={r['lambda']}")
    print()
    print(f"=== 适应度排序（combined = mIoU − {r['lambda']}·log10(cost/{r['base_ms']:.1f})）===")
    print(f"{'候选':<52s} {'mIoU':>8s} {'combined':>9s} {'ms':>8s} {'加速':>7s} {'来源':>6s}")
    for c in r["ranked"][:args.top]:
        print(f"{c['name'][:52]:<52s} {c['mean_miou']:8.4f} "
              f"{combined_score(c['mean_miou'], c['ms'], r['base_ms'], r['lambda']):9.4f} "
              f"{c['ms']:8.1f} {r['base_ms'] / max(c['ms'], 1e-3):6.1f}x "
              f"{'进化' if c.get('from_archstr') else '手工':>6s}")
    print()
    print("=== 真实 Pareto 前沿 ===")
    for c in r["front"]:
        print(f"{c['name'][:52]:<52s} mIoU={c['mean_miou']:.4f}  {c['ms']:7.1f} ms  "
              f"{r['base_ms'] / max(c['ms'], 1e-3):5.1f}x  "
              f"{'进化' if c.get('from_archstr') else '手工'}")
    print()
    print("=== 结论 ===")
    print(f"最优 = {b['name']}")
    print(f"  mIoU {b['mean_miou']:.4f}（基线 {r['base_miou']:.4f}，"
          f"Δ=+{b['mean_miou'] - r['base_miou']:.4f}）")
    print(f"  combined {r['best_combined']:.4f}")
    print(f"  成本 {b['ms']:.1f} ms（快 {r['base_ms'] / max(b['ms'], 1e-3):.1f}×）")
    print(f"  来源 = {'**进化产出**' if b.get('from_archstr') else '手工枚举'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"best": b, "best_combined": r["best_combined"],
                       "front": r["front"], "base_ms": r["base_ms"],
                       "base_miou": r["base_miou"], "lambda": r["lambda"]},
                      fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
