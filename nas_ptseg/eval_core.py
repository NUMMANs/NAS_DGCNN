"""多域评估编排：把 DGCNN 代理在各密度域上的结果合成为鲁棒分数。

评估目标（与用户需求对齐）
==========================
    搜索目标：多域鲁棒的 mIoU（最差域 + 平均域）

    最差域（min_domain_mIoU）是核心 —— 它正是"跨密度泛化差"这一痛点的
    直接量化。只优化平均域会让架构在稀疏域上崩掉却仍然得高分。

    代价作为次要约束：只在**超出基线代价**时施加惩罚，不奖励"变便宜"。
    这样第一轮验证机制时，"鲁棒性提升"这个信号不被效率项污染。
    效率单独用 Pareto 分析呈现（见 prep.py 的 --pareto）。

适应度契约（必须遵守）
======================
    返回字典**必须**含 combined_score。
    OpenEvolve 的 get_fitness_score() 优先取 combined_score；
    若缺失，则把所有数值型指标**求平均** —— 那会把 runtime_sec、clamp_rate
    这类量混进适应度，结果完全失真。这是本文件里最容易出错的一处。

α 指数（可解释性指标）
======================
    拟合 k ∝ rho^alpha（逐层），alpha 的含义：
        alpha ≈ 0   → 完全不随密度变（等价于手工固定架构）
        alpha ≈ 1   → k ∝ rho，即保持**物理感受野恒定**
        alpha ≈ -1/3→ 物理半径 ∝ rho^(-1/3)，即保持图度数恒定
        其他值      → 部分补偿。这本身就是一个有价值的发现：
                      "网络只需要部分补偿密度漂移，而不是完全补偿"
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import density as D
    from nas_ptseg import proxy as PX
else:
    from . import data as DT
    from . import density as D
    from . import proxy as PX


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------
# 最差域在鲁棒分数里的权重。0.5 表示"最差域与平均域同等重要"。
# 取 0.5 而不是更高，是为了避免只救最差域而牺牲其他域（那只是把短板挪个位置）。
W_MIN_DOMAIN = 0.5

# 代价惩罚：只在 cost_rel > 1 时线性惩罚。0.05 意味着"代价翻倍"要扣掉
# 0.05 的适应度 —— 相对 mIoU 的典型变化幅度（0.05~0.3），这是一个
# 温和但非零的约束。可在 config.yaml 里调整。
COST_PENALTY_ETA = 0.05

# 代价的加权方式：算力占主导，参数量次要
W_COST_FLOP = 0.7
W_COST_PARAM = 0.3


# ---------------------------------------------------------------------------
# 域角色（main / diag）
# ---------------------------------------------------------------------------
# role 由 data.build_domains 写入 meta.json。
#   main : 参与鲁棒目标、α 拟合、代价归一化 —— "部署/训练"域
#   diag : 密集端过平滑诊断探针 —— **不参与任何指标**
#
# 为什么必须分开：diag 域比主域更密，一旦混进指标会同时改写
# min_domain_mIoU、reference_cost 的 flops 基准、以及 α 拟合的 ρ 区间端点，
# 使历史标定结果不可比。旧缓存没有 role 字段 → 一律按 main 处理（向后兼容）。
def load_roles(cache_dir: Optional[str] = None, meta: Optional[dict] = None) -> Dict[str, str]:
    """读取各域角色。只读 meta.json，**不加载 npz**（避免为大域白付一次解压代价）。"""
    if meta is None:
        d = cache_dir or DT.CACHE_DIR
        p = os.path.join(d, "meta.json")
        meta = {}
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
    return {
        name: str(dm.get("role", "main"))
        for name, dm in (meta.get("domains") or {}).items()
    }


def main_domain_names(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cache_dir: Optional[str] = None,
    meta: Optional[dict] = None,
) -> List[str]:
    """按 meta 中声明的顺序返回 main 域名字。"""
    roles = load_roles(cache_dir, meta)
    return [n for n in domains if roles.get(n, "main") == "main"]


# ---------------------------------------------------------------------------
# 逐域密度缓存（避免每次评估都重算 kNN 密度）
# ---------------------------------------------------------------------------
_rho_cache: Dict[str, Dict[str, np.ndarray]] = {}


def get_rho_cache(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    key: str,
    names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """按需增量缓存逐点密度。

    带 names 过滤的意义：诊断域可达 131072 点，算一次 kNN 密度不便宜。
    标定/搜索只跑 main 域时不该为它们付这份代价。
    """
    want = list(names) if names is not None else list(domains)
    cache = _rho_cache.get(key)
    if cache is not None and all(d in cache for d in want):
        return cache
    cache = dict(cache or {})
    for name in want:
        if name not in cache and name in domains:
            cache[name] = D.local_density(domains[name][0])
    _rho_cache[key] = cache
    return cache


def set_rho_ref_from_domains(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    names: Optional[List[str]] = None,
) -> float:
    """把 RHO_REF 标定为各域中位密度的几何均值。

    为什么用几何均值：密度是乘性量级（跨数量级），算术均值会被最密的域
    拉高。几何均值让 ratio 在各域上大致围绕 1 对称分布。

    ⚠ names 必须只传 **main** 域。RHO_REF 是策略输入 `ratio = rho / RHO_REF`
    的归一化锚点，若把诊断域算进去，锚点会向高密度偏移，策略看到的
    ratio 整体变小 —— 这会静默改变搜索空间的语义，且与历史结果不可比。
    """
    items = [v for k, v in domains.items() if names is None or k in set(names)]
    meds = [float(np.median(D.local_density(xyz))) for xyz, _y in items]
    meds = [m for m in meds if m > 0]
    if not meds:
        return D.RHO_REF_DEFAULT
    ref = float(np.exp(np.mean(np.log(meds))))
    D.set_rho_ref(ref)
    return ref


# ---------------------------------------------------------------------------
# 单域评估
# ---------------------------------------------------------------------------
def eval_domain(
    policy: Callable[[Dict[str, float]], dict],
    xyz: np.ndarray,
    y: np.ndarray,
    cfg: PX.ProxyConfig,
    rho: Optional[np.ndarray] = None,
) -> dict:
    out = PX.forward(policy, xyz, cfg, rho_precomputed=rho)
    miou, acc, per_class = PX.linear_probe_miou(out["feats"], y, DT.NUM_CLASSES, cfg)

    # 该域实际出现的类（用于诊断：稀疏域可能丢类）
    present = [i for i in range(DT.NUM_CLASSES) if (y == i).any()]
    rho_med = float(np.median(out["rho"])) if out["rho"].size else 0.0
    rho_ratio = rho_med / max(D.get_rho_ref(), 1e-9)

    return {
        "mIoU": float(miou),
        "acc": float(acc),
        "per_class_iou": per_class,
        "n_points": int(xyz.shape[0]),
        "n_classes_present": len(present),
        "classes_present": present,
        "rho_med": rho_med,
        "rho_ratio": float(rho_ratio),
        "trace": out["trace"],
        "errors": out["errors"],
        "cost": out["cost"],
    }


# ---------------------------------------------------------------------------
# 多域编排
# ---------------------------------------------------------------------------
def evaluate_policy(
    policy: Callable[[Dict[str, float]], dict],
    domains: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    cache_dir: Optional[str] = None,
    cfg: Optional[PX.ProxyConfig] = None,
    ref_cost: Optional[dict] = None,
    cost_eta: float = COST_PENALTY_ETA,
    w_min_domain: float = W_MIN_DOMAIN,
    include_diag: bool = False,
) -> dict:
    """在密度域上评估一个策略，返回 report。

    参数
    ----
    policy      : 密度画像 -> {k, radius_m, sample_rate}。被进化的对象。
    domains     : {域名称: (xyz, labels)}。None 时从缓存加载。
    ref_cost    : 基线策略的代价，用于代价归一化。None 时本次不归一化（cost_rel=1）。
    include_diag: 是否也跑 diag 域。**默认 False** —— 诊断域只服务
                  `prep.py --dense-diag` 的定点扫描，不进搜索/标定。
                  即便置 True，标量指标也只由 main 域产生。
    """
    t0 = time.perf_counter()
    cfg = cfg or PX.ProxyConfig()

    meta = None
    if domains is None:
        domains, meta = DT.load_cache(cache_dir)

    main_set = set(main_domain_names(domains, cache_dir, meta))
    use_names = [n for n in domains if n in main_set or include_diag]

    cache_key = cache_dir or "default"
    rho_cache = get_rho_cache(domains, cache_key, names=use_names)

    per_domain: Dict[str, dict] = {}
    all_errors: List[str] = []
    for name in use_names:
        xyz, y = domains[name]
        r = eval_domain(policy, xyz, y, cfg, rho=rho_cache.get(name))
        r["role"] = "main" if name in main_set else "diag"
        per_domain[name] = r
        all_errors.extend(f"{name}: {e}" for e in r["errors"])

    # ---- 标量指标一律只在 main 域上算 ----
    metric_names = [n for n in use_names if n in main_set]
    mious = [per_domain[n]["mIoU"] for n in metric_names]
    if not mious:
        return _empty_report(all_errors, time.perf_counter() - t0)
    metric_reps = [per_domain[n] for n in metric_names]

    min_miou = float(np.min(mious))
    mean_miou = float(np.mean(mious))
    robust_raw = w_min_domain * min_miou + (1.0 - w_min_domain) * mean_miou

    # ---- 代价 ----
    params = float(np.mean([v["cost"]["params"] for v in metric_reps]))
    flops = float(np.mean([v["cost"]["flops"] for v in metric_reps]))
    if ref_cost:
        params_rel = params / max(ref_cost["params"], 1e-9)
        flops_rel = flops / max(ref_cost["flops"], 1e-9)
    else:
        params_rel = flops_rel = 1.0
    cost_rel = W_COST_PARAM * params_rel + W_COST_FLOP * flops_rel

    penalty = cost_eta * max(0.0, cost_rel - 1.0)
    combined = robust_raw - penalty

    # ---- 诊断量 ----
    clamps = [t["clamped"] for v in metric_reps for t in v["trace"]]
    clamp_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in clamps])) if clamps else 0.0

    return {
        # --- 适应度：必须存在且语义单一 ---
        "combined_score": float(combined),
        # --- 目标量 ---
        "robust_score": float(robust_raw),
        "min_domain_mIoU": min_miou,
        "mean_domain_mIoU": mean_miou,
        # --- 域角色（供报告与合规检查）---
        "main_domains": metric_names,
        "n_main_domains": len(metric_names),
        "diag_domains": [n for n in use_names if n not in main_set],
        # --- 代价 ---
        "cost_rel": float(cost_rel),
        "params_rel": float(params_rel),
        "flops_rel": float(flops_rel),
        "params_abs": float(params),
        "flops_abs": float(flops),
        "cost_penalty": float(penalty),
        # --- 逐域明细（含 diag，各自带 role）---
        "per_domain": {
            k: {
                "mIoU": v["mIoU"],
                "acc": v["acc"],
                "rho_med": v["rho_med"],
                "rho_ratio": v["rho_ratio"],
                "n_points": v["n_points"],
                "n_classes_present": v["n_classes_present"],
                "role": v["role"],
            }
            for k, v in per_domain.items()
        },
        # --- 可解释性：alpha 指数（只在 main 域上拟合）---
        "alphas": _fit_alphas({n: per_domain[n] for n in metric_names}),
        # --- 诊断 ---
        "clamp_rate": clamp_rate,
        "n_policy_errors": len(all_errors),
        "policy_errors": all_errors[:5],
        "runtime_sec": float(time.perf_counter() - t0),
        "rho_ref": float(D.get_rho_ref()),
        "trace": _collect_trace(per_domain),
    }


def _collect_trace(per_domain: Dict[str, dict]) -> List[dict]:
    out: List[dict] = []
    for dom, v in per_domain.items():
        for t in v["trace"]:
            tt = dict(t)
            tt["domain"] = dom
            out.append(tt)
    return out


def _fit_alphas(per_domain: Dict[str, dict]) -> Dict[str, Optional[float]]:
    """逐层拟合 k ∝ rho^alpha 与 radius ∝ rho^alpha。

    只保留 alpha 有明确物理含义的两条：
      alpha_k      : 图度数随密度的幂律指数（主线结论）
      alpha_r      : 感受野半径随密度的幂律指数（对照）
    """
    alphas: Dict[str, Optional[float]] = {}
    layers = sorted({t["layer"] for v in per_domain.values() for t in v["trace"]})
    for l in layers:
        rhos, ks, rs = [], [], []
        for v in per_domain.values():
            for t in v["trace"]:
                if t["layer"] != l:
                    continue
                rhos.append(t["rho_med"])
                ks.append(float(t["k"]))
                rs.append(t["radius_m"])
        if len(rhos) >= 2:
            alphas[f"alpha_k_layer{l}"] = D.fit_exponent(rhos, ks)
            alphas[f"alpha_r_layer{l}"] = D.fit_exponent(rhos, rs)
            # 自适应强度：k 与半径在不同域之间的实际跨度（比值形式，更直观）
            alphas[f"k_span_layer{l}"] = float(max(ks) / max(min(ks), 1e-9)) if ks else None
            alphas[f"r_span_layer{l}"] = float(max(rs) / max(min(rs), 1e-9)) if rs else None
    return alphas


def _empty_report(errors: List[str], runtime: float) -> dict:
    return {
        "combined_score": 0.0,
        "robust_score": 0.0,
        "min_domain_mIoU": 0.0,
        "mean_domain_mIoU": 0.0,
        "main_domains": [],
        "n_main_domains": 0,
        "diag_domains": [],
        "cost_rel": 1.0,
        "params_rel": 1.0,
        "flops_rel": 1.0,
        "params_abs": 0.0,
        "flops_abs": 0.0,
        "cost_penalty": 0.0,
        "per_domain": {},
        "alphas": {},
        "clamp_rate": 0.0,
        "n_policy_errors": len(errors),
        "policy_errors": errors[:5],
        "runtime_sec": runtime,
        "rho_ref": float(D.get_rho_ref()),
        "trace": [],
    }


def reference_cost(
    domains: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    cache_dir: Optional[str] = None,
    cfg: Optional[PX.ProxyConfig] = None,
) -> dict:
    """用基线策略算一次代价参考点（cost_rel == 1 的锚）。

    ⚠ 只统计 **main** 域。诊断域（可达 131072 点）若算进来会把 flops 基线
    整体抬高，cost_rel 的语义随之改变，与历史结果不可比。
    """
    cfg = cfg or PX.ProxyConfig()
    meta = None
    if domains is None:
        domains, meta = DT.load_cache(cache_dir)
    names = main_domain_names(domains, cache_dir, meta)
    rho_cache = get_rho_cache(domains, cache_dir or "default", names=names)

    params, flops = [], []
    for name in names:
        xyz, _y = domains[name]
        out = PX.forward(PX.baseline_policy, xyz, cfg, rho_precomputed=rho_cache.get(name))
        params.append(out["cost"]["params"])
        flops.append(out["cost"]["flops"])
    return {
        "params": float(np.mean(params)) if params else 1.0,
        "flops": float(np.mean(flops)) if flops else 1.0,
    }


def evaluate_policy_path(
    program_path: str,
    domains: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    cache_dir: Optional[str] = None,
    cfg: Optional[PX.ProxyConfig] = None,
    ref_cost: Optional[dict] = None,
) -> dict:
    """按文件路径加载策略并评估（供 evaluator 调用）。

    OpenEvolve 的 evaluator 收到的是程序文件路径，不是函数对象。
    这里用 importlib 从文件加载，并取其中的 density_adaptive_policy。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_evolved_program", program_path)
    if spec is None or spec.loader is None:
        return _empty_report([f"无法加载程序：{program_path}"], 0.0)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        return _empty_report([f"程序执行失败：{type(e).__name__}: {e}"], 0.0)

    policy = getattr(mod, "density_adaptive_policy", None)
    if policy is None:
        return _empty_report(
            ["程序里找不到 density_adaptive_policy 函数"], 0.0
        )
    return evaluate_policy(
        policy, domains=domains, cache_dir=cache_dir, cfg=cfg, ref_cost=ref_cost
    )


def format_report(rep: dict) -> str:
    """把 report 格式化成可读文本（用于日志与 LLM 反馈）。"""
    lines = [
        f"combined_score   = {rep['combined_score']:.4f}",
        f"robust_score     = {rep['robust_score']:.4f}   (min 权重 {W_MIN_DOMAIN})",
        f"min_domain_mIoU  = {rep['min_domain_mIoU']:.4f}   <- 最差域（核心指标）",
        f"mean_domain_mIoU = {rep['mean_domain_mIoU']:.4f}   <- 平均域",
        f"域角色           = main {rep.get('n_main_domains', len(rep['per_domain']))} 个"
        f"（标量指标只由 main 产生）",
        f"cost_rel         = {rep['cost_rel']:.4f}  "
        f"(params={rep['params_rel']:.3f}x, flops={rep['flops_rel']:.3f}x)",
        f"cost_penalty     = {rep['cost_penalty']:.4f}",
        f"clamp_rate       = {rep['clamp_rate']:.3f}",
        f"runtime          = {rep['runtime_sec']:.2f}s",
        f"rho_ref          = {rep['rho_ref']:.3e} pts/m^3",
        "逐域 mIoU:",
    ]
    for dom, m in rep["per_domain"].items():
        tag = "" if m.get("role", "main") == "main" else "  [诊断·不计入指标]"
        lines.append(
            f"  {dom:8s} mIoU={m['mIoU']:.4f}  acc={m['acc']:.4f}  "
            f"N={m['n_points']:6d}  rho={m['rho_med']:.3e} (x{m['rho_ratio']:.2f}){tag}"
        )
    if rep.get("alphas"):
        lines.append("alpha 指数（k ∝ rho^alpha，逐层）:")
        for k, v in rep["alphas"].items():
            lines.append(f"  {k:16s} = {v if v is None else round(v, 4)}")
    if rep.get("n_policy_errors"):
        lines.append(f"策略异常 {rep['n_policy_errors']} 次，首条：{rep['policy_errors'][:1]}")
    return "\n".join(lines)
