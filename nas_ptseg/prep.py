"""S3DIS 单房间数据准备、标定与对照实验。

用法
====
    python nas_ptseg/prep.py --list-rooms
    python nas_ptseg/prep.py --build --report _calib.txt
    python nas_ptseg/prep.py --calibrate --report _calib.txt
    python nas_ptseg/prep.py --pareto --report _pareto.txt

标定做了什么
============
为了让"密度自适应是否真的有用"这个结论站得住，标定阶段必须给出**四组参照**：

    A. 手工固定架构      —— 基线，完全不随密度变。这是要证伪的对象
    B. 解析式物理先验    —— k ∝ rho^alpha 的幂律先验（等体积/等半径两种锚定）
    C. 随机策略          —— 对照，确认代理不是凭空给分
    D. 固定架构扫描      —— 在固定架构族里找最优，给出"手工调参的上限"

如果 B 的 min_domain_mIoU 不能稳定超过 D 的最优固定架构，说明"密度自适应"
在这套代理上不成立 —— 那就**不该**往下投真实训练，应该先回来查代理。
这个判断顺序很重要，能省掉大量无效算力。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nas_ptseg import data as DT
    from nas_ptseg import density as D
    from nas_ptseg import proxy as PX
    from nas_ptseg import eval_core as EC
    from nas_ptseg import geom as GE
else:
    from . import data as DT
    from . import density as D
    from . import proxy as PX
    from . import eval_core as EC
    from . import geom as GE


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
class _Report:
    """同时写 stdout 与 UTF-8 文件的报告器。

    为什么不用 PowerShell 捕获输出：Windows PowerShell 会按 ANSI 重新编码
    子进程输出，中文全部变乱码，报告没法看。直接由 Python 写 UTF-8 最稳。
    """

    def __init__(self, path: Optional[str]):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, s: str = "") -> None:
        try:
            sys.__stdout__.write(s + "\n")
        except Exception:
            pass
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            try:
                self.f.close()
            except Exception:
                pass
            self.f = None


# ---------------------------------------------------------------------------
# 指令
# ---------------------------------------------------------------------------
def cmd_list_rooms(args, R: _Report) -> None:
    rooms = DT.list_rooms(args.area or None)
    R(f"S3DIS 根目录: {DT.S3DIS_ROOT}")
    R(f"存在 = {os.path.isdir(DT.S3DIS_ROOT)}")
    if not rooms:
        R("")
        R("没有找到任何房间。请确认：")
        R(f"  1. 数据集已解压到 {DT.S3DIS_ROOT}")
        R("  2. 该目录下含 Area_1 .. Area_6，每个 Area 下有若干 <room>/Annotations/")
        return
    R(f"共 {len(rooms)} 个房间：")
    for a, r in rooms:
        R(f"  {a}/{r}")


def _load_or_build(args, R: _Report):
    cache_dir = args.cache_dir or DT.CACHE_DIR
    npz = os.path.join(cache_dir, "s3dis_domains.npz")
    if os.path.exists(npz):
        domains, meta = DT.load_cache(cache_dir)
        R(f"已加载缓存: {cache_dir}")
    else:
        R(f"缓存不存在，开始构造: {cache_dir}")
        domains, meta = _do_build(args, R, cache_dir)
    return domains, meta, cache_dir


def _do_build(args, R: _Report, cache_dir: str) -> tuple:
    R("=" * 72)
    R("构造 S3DIS 受控密度域")
    R("=" * 72)
    R(f"房间            : {args.area}/{args.room}")
    R(f"主阶梯 main     : {args.ladder}   ← 参与鲁棒目标 / α 拟合 / 代价基准")
    R(f"诊断阶梯 diag   : {args.diag_ladder or '（未启用）'}   ← 仅用于密集端过平滑诊断")
    R(f"读取点数上限    : {args.read_max_pts}")
    R("")
    R("说明：所有域来自**同一房间、同一空间范围**，只有体素尺寸不同，")
    R("      因此几何与语义内容完全一致，唯一变量是点密度。")
    R("")
    R("      两档阶梯的角色必须分开：diag 域比 main 更密，一旦混进指标就会同时")
    R("      改写 min_domain_mIoU、代价基准 flops 与 α 拟合的密度区间端点，")
    R("      使与历史标定不可比 —— 且诊断域本身并不是部署场景。")
    R("")

    domains, meta = DT.build_domains(
        area=args.area,
        room=args.room,
        point_ladder=args.ladder,
        read_max_pts=args.read_max_pts,
        seed=args.seed,
        diag_ladder=args.diag_ladder or [],
    )

    R(f"原始读取点数    : {meta['raw_points_after_read']}")
    R("")
    R(f"{'域':<8s} {'角色':<6s} {'目标N':>7s} {'实际N':>7s} {'体素(cm)':>9s} "
      f"{'rho(pts/m^3)':>14s} {'类别数':>7s}")

    med_by_role: Dict[str, list] = {"main": [], "diag": []}
    for name, (xyz, y) in domains.items():
        rho = D.local_density(xyz)
        med = float(np.median(rho))
        dm = meta["domains"][name]
        med_by_role.setdefault(dm.get("role", "main"), []).append(med)
        n_cls = len([c for c in range(DT.NUM_CLASSES) if (y == c).any()])
        R(f"{name:<8s} {dm.get('role','main'):<6s} {dm['target_n']:>7d} "
          f"{xyz.shape[0]:>7d} {dm['voxel_m']*100:>9.2f} {med:>14.3e} {n_cls:>7d}")

    # ---- 密度参考锚点：⚠ 只用 main 域 ----
    main_names = [n for n, dm in meta["domains"].items()
                  if dm.get("role", "main") == "main"]
    ref = EC.set_rho_ref_from_domains(domains, main_names)
    main_meds = med_by_role.get("main", [])
    rho_ratio = max(main_meds) / max(min(main_meds), 1e-9) if main_meds else 1.0
    meta["rho_ref"] = float(ref)
    meta["rho_ref_scope"] = "main_only"
    meta["rho_ratio_max_over_min"] = float(rho_ratio)

    R("")
    R(f"RHO_REF（**仅 main 域**中位密度的几何均值） = {ref:.4e} pts/m^3")
    R(f"main 域密度跨度 max/min                  = {rho_ratio:.2f} 倍")
    R(f"→ 固定 k 时物理感受野需伸缩              = {rho_ratio ** (1/3):.2f} 倍")
    if med_by_role.get("diag"):
        dm_ratio = max(med_by_role["diag"]) / max(min(med_by_role["diag"]), 1e-9)
        R("")
        R(f"diag 域密度（{dm_ratio:.2f} 倍跨度的两个更密探针）:")
        for name in domains:
            dm = meta["domains"][name]
            if dm.get("role") == "diag":
                R(f"  {name:<8s} N={dm['n']:>7d}  体素 {dm['voxel_m']*100:>6.2f} cm")
        R("  用途：回答『密集区是否该收缩邻域』—— 见 prep.py --dense-diag")
    R("")

    DT.save_cache(domains, meta, cache_dir)
    R(f"缓存已写入: {cache_dir}")
    return domains, meta


def cmd_build(args, R: _Report) -> None:
    cache_dir = args.cache_dir or DT.CACHE_DIR
    _do_build(args, R, cache_dir)


def _gate_diag(R: _Report, rep: dict, title: str) -> None:
    """打印"半径门控"诊断：实际参与聚合的边数是否被 radius_m 掐死。

    为什么必须看这个 —— 这是本标定里最容易得出错误结论的地方：

        A 用固定 k=16 / r=0.30m。在稀疏域里，第 16 近邻的距离早已超过
        0.30 m，于是 in_rad 全 False，Proxy 的兜底逻辑只放行 1 条边 ——
        "16 邻域"实际退化成"1 近邻"。此时 A 的 mIoU 崩掉，**主要原因是
        半径门控把邻域掐断了，而不是密度本身**。

        于是 B2 相对 A 的增益里，混入了"解除门控"的收益。若不拆开，就会
        把"把半径放大"误读成"架构随密度自适应"，整篇立论建立在混淆之上。

    判别标准（n_in_med = 每节点落入半径内的邻居数中位数）：
        n_in_med ≈ k        → 门控不生效，radius 不是瓶颈
        n_in_med < 0.95k    → 部分受限
        n_in_med < 0.6k     → 门控掐断：该域结论不可直接归因于密度
    """
    R("")
    R(f"  [门控诊断] {title}")
    R(f"    {'域':<8s} {'层':>3s} {'k':>4s} {'radius(m)':>10s} "
      f"{'r/等效间距':>11s} {'n_in_med':>9s} {'状态':>10s}")
    by_dom: Dict[str, list] = {}
    for t in rep.get("trace", []):
        by_dom.setdefault(t["domain"], []).append(t)
    for dom, ts in by_dom.items():
        for t in sorted(ts, key=lambda x: x["layer"]):
            k = float(t["k"])
            nin = float(t["n_in_med"])
            if nin < 0.60 * k:
                state = "门控掐断"
            elif nin < 0.95 * k:
                state = "部分受限"
            else:
                state = "正常"
            R(f"    {dom:<8s} {t['layer']:>3d} {t['k']:>4d} {t['radius_m']:>10.3f} "
              f"{t['radius_over_spacing']:>11.2f} {nin:>9.1f} {state:>10s}")


def cmd_calibrate(args, R: _Report) -> None:
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)

    cfg = PX.ProxyConfig()

    R("")
    R("=" * 72)
    R("各域密度（仅 main 域参与标量指标）")
    R("=" * 72)
    rb = EC.get_rho_cache(domains, cache_dir, names=main_names)
    for name in main_names:
        xyz, _y = domains[name]
        rho = rb[name]
        R(f"  {name:<8s} N={xyz.shape[0]:>6d}  "
          f"rho: p10={np.percentile(rho,10):.3e} med={np.median(rho):.3e} "
          f"p90={np.percentile(rho,90):.3e}  ratio={np.median(rho)/ref:.3f}")

    # 代价参考点
    rc = EC.reference_cost(domains, cache_dir, cfg)
    R("")
    R(f"基线代价参考: params={rc['params']:.0f}  flops={rc['flops']:.3e}")

    # ---- 参照 A ----
    R("")
    R("=" * 72)
    R("参照 A：手工固定架构（完全不随密度变化）")
    R("=" * 72)
    ra = EC.evaluate_policy(PX.baseline_policy, domains, cfg=cfg, ref_cost=rc)
    R(EC.format_report(ra))
    _gate_diag(R, ra, "参照 A：固定 k=16, r=0.30m, sr=1.0")

    # ---- 参照 B ----
    R("")
    R("=" * 72)
    R("参照 B：解析式物理先验（k ∝ rho^alpha）")
    R("=" * 72)
    R("  B1: 等物理半径锚定 —— 想让感受野半径保持恒定，需 k ∝ rho")
    R("  B2: 等图度数锚定   —— 想让 k 恒定，半径需 ∝ rho^(-1/3)")
    R("  两组都试，看哪一族的先验更接近最优。")

    def _prior_const_radius(p):
        # 目标：物理半径不随密度变 → 球内点数 ∝ rho → k 随 rho 线性放大
        ratio = p["ratio"]
        k = int(np.clip(round(16.0 * ratio), PX.K_MIN, PX.K_MAX))
        r_m = float(np.clip(0.30, PX.R_MIN_M, PX.R_MAX_M))
        return {"k": k, "radius_m": r_m, "sample_rate": 1.0}

    def _prior_const_k(p):
        # 目标：k 恒定 → 半径按 rho^(-1/3) 缩放
        s = p["scale_hint"]
        k = 16
        r_m = float(np.clip(3.4 * s, PX.R_MIN_M, PX.R_MAX_M))
        return {"k": k, "radius_m": r_m, "sample_rate": 1.0}

    rb1 = EC.evaluate_policy(_prior_const_radius, domains, cfg=cfg, ref_cost=rc)
    R("")
    R("--- B1: k ∝ rho（等物理半径）---")
    R(EC.format_report(rb1))
    _gate_diag(R, rb1, "参照 B1：k ∝ rho, r=0.30m")

    rb2 = EC.evaluate_policy(_prior_const_k, domains, cfg=cfg, ref_cost=rc)
    R("")
    R("--- B2: radius ∝ rho^(-1/3)（等 k）---")
    R(EC.format_report(rb2))
    _gate_diag(R, rb2, "参照 B2：k=16, r=3.4*rho^(-1/3)")

    # ---- 参照 C ----
    R("")
    R("=" * 72)
    R("参照 C：随机策略（确认代理不是凭空给分）")
    R("=" * 72)
    R("  说明：若随机策略的分数与参照 A 接近，说明代理对该配置不敏感，")
    R("        后续一切对比都不可信 —— 这是必须先排除的情况。")

    def _rand_policy(p):
        rng = np.random.default_rng(int(abs(p["rho"]) * 1e3) % 2**31 + int(p["layer_idx"]))
        return {
            "k": int(rng.integers(PX.K_MIN, PX.K_MAX + 1)),
            "radius_m": float(rng.uniform(PX.R_MIN_M, PX.R_MAX_M)),
            "sample_rate": float(rng.uniform(PX.SR_MIN, PX.SR_MAX)),
        }

    rc_rep = EC.evaluate_policy(_rand_policy, domains, cfg=cfg, ref_cost=rc)
    R(EC.format_report(rc_rep))

    # ---- 参照 D ----
    R("")
    R("=" * 72)
    R("参照 D：固定架构扫描（手工调参能达到的上限）")
    R("=" * 72)
    R(f"  {'k':>4s} {'radius(m)':>10s} {'min_mIoU':>9s} {'mean_mIoU':>10s} "
      f"{'robust':>8s} {'cost_rel':>9s}")

    def _make_fixed(k: int, r_m: float):
        def _f(p):
            return {"k": k, "radius_m": r_m, "sample_rate": 1.0}

        return _f

    best = None
    grid: List[dict] = []
    for k in (8, 16, 24, 32):
        # 半径网格必须包含一个**不生效**的值（2.0m = R_MAX_M）。否则在稀疏域里
        # 所有候选都还卡着门控，扫出来的"上限"是被门控压过的假上限，会系统性
        # 高估"密度自适应"的净收益。
        for r_m in (0.10, 0.20, 0.30, 0.45, 0.70, PX.R_MAX_M):
            rr = EC.evaluate_policy(_make_fixed(k, r_m), domains, cfg=cfg, ref_cost=rc)
            grid.append({"k": k, "radius_m": r_m, **{kk: rr[kk] for kk in
                          ("min_domain_mIoU", "mean_domain_mIoU", "robust_score",
                           "cost_rel", "combined_score")}})
            R(f"  {k:>4d} {r_m:>10.2f} {rr['min_domain_mIoU']:>9.4f} "
              f"{rr['mean_domain_mIoU']:>10.4f} {rr['robust_score']:>8.4f} "
              f"{rr['cost_rel']:>9.3f}")
            if best is None or rr["robust_score"] > best["robust_score"]:
                best = {"k": k, "radius_m": r_m, **rr}

    R("")
    R(f"  → 固定架构最优: k={best['k']} radius={best['radius_m']:.2f}m  "
      f"robust={best['robust_score']:.4f} "
      f"(min={best['min_domain_mIoU']:.4f}, mean={best['mean_domain_mIoU']:.4f})")
    _gate_diag(R, best, f"参照 D 最优固定架构：k={best['k']}, "
                        f"r={best['radius_m']:.2f}m")

    # 关键对照：把 A 的半径放开到不卡门控，看它单独能拿到多少分。
    # 这一步是**必要**的 —— 它把"解除门控"的收益从"密度自适应"的收益里剥离出来。
    # 若放开半径的固定架构已经追平 B2，则"密度自适应"没有独立贡献。
    R("")
    R("--- 对照 A+：固定 k=16，但半径放开（r=2.0m，门控不再生效）---")
    R("  目的：把 A 的失分拆成【门控掐断】与【密度漂移】两部分。")

    def _fixed_k16_open_radius(p):
        return {"k": 16, "radius_m": PX.R_MAX_M, "sample_rate": 1.0}

    ra_plus = EC.evaluate_policy(_fixed_k16_open_radius, domains, cfg=cfg, ref_cost=rc)
    R(EC.format_report(ra_plus))
    _gate_diag(R, ra_plus, "对照 A+：固定 k=16, r=2.0m(不生效)")

    # ---- 结论 ----
    R("")
    R("=" * 72)
    R("结论指引")
    R("=" * 72)
    best_prior = rb1 if rb1["robust_score"] >= rb2["robust_score"] else rb2
    best_prior_name = "B1(等物理半径)" if rb1["robust_score"] >= rb2["robust_score"] else "B2(等k)"
    R(f"  基线 A           : robust={ra['robust_score']:.4f}  min={ra['min_domain_mIoU']:.4f}")
    R(f"  对照 A+ (放开半径): robust={ra_plus['robust_score']:.4f}  "
      f"min={ra_plus['min_domain_mIoU']:.4f}")
    R(f"  最优先验 {best_prior_name:<14s}: robust={best_prior['robust_score']:.4f}  "
      f"min={best_prior['min_domain_mIoU']:.4f}")
    R(f"  最优固定架构 D   : robust={best['robust_score']:.4f}  min={best['min_domain_mIoU']:.4f} "
      f"(k={best['k']}, r={best['radius_m']:.2f}m)")
    R(f"  随机 C           : robust={rc_rep['robust_score']:.4f}  min={rc_rep['min_domain_mIoU']:.4f}")
    R("")
    R("  —— 收益剥离：把'解除门控'与'密度自适应'分开 ——")
    gate_gain = ra_plus["min_domain_mIoU"] - ra["min_domain_mIoU"]
    R(f"  A → A+ 纯解除门控带来的最差域增益 = {gate_gain:+.4f}")
    R(f"      （这部分**不能**算作密度自适应的贡献，只是把半径设得太小的代价）")
    R(f"  A+ → D 最优固定架构的进一步增益   = "
      f"{best['min_domain_mIoU'] - ra_plus['min_domain_mIoU']:+.4f}")
    R("")
    delta_min = best_prior["min_domain_mIoU"] - best["min_domain_mIoU"]
    R(f"  先验 B 相对最优固定架构 D 的 min_domain_mIoU 增益 = {delta_min:+.4f}")
    if delta_min > 0.005:
        R("  → 密度自适应在**最差域**上有正增益，核心假设成立，"
          "值得投入后续的 DGCNN/PolarNet 真实训练验证。")
    elif delta_min > -0.005:
        R("  → 两者接近。手工调参已经能覆盖密度漂移，说明搜索空间的收益主要"
          "不在“随密度缩放”这一维。建议转向层次/宽度维度的搜索，再做真实验证。")
    else:
        R("  → 手写先验反而不如最优固定架构。先别投真实训练：")
        R("    优先怀疑代理（探针太弱 / 随机特征维度不足），而不是假设不成立。")
        R("    建议：提高 probe_dim、增加域数量，或改用真实 DGCNN 短训做 Tier-2 校验。")


def cmd_pareto(args, R: _Report) -> None:
    """扫出 (cost, min_domain_mIoU) 的 Pareto 前沿，用于效率-精度分析。"""
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)
    cfg = PX.ProxyConfig()
    rc = EC.reference_cost(domains, cache_dir, cfg)

    R("")
    R("=" * 72)
    R("Pareto 扫描：(cost, 鲁棒精度)")
    R("=" * 72)
    R(f"  {'k':>4s} {'r(m)':>6s} {'sr':>5s} {'min_mIoU':>9s} {'mean_mIoU':>10s} "
      f"{'cost_rel':>9s}")

    pts: List[dict] = []

    def _mk(k, r_m, sr):
        def _f(p):
            return {"k": k, "radius_m": r_m, "sample_rate": sr}

        return _f

    for k in (8, 12, 16, 24, 32):
        for r_m in (0.15, 0.30, 0.50):
            for sr in (0.25, 0.5, 1.0):
                rr = EC.evaluate_policy(_mk(k, r_m, sr), domains, cfg=cfg, ref_cost=rc)
                pts.append({
                    "k": k, "radius_m": r_m, "sample_rate": sr,
                    "min_domain_mIoU": rr["min_domain_mIoU"],
                    "mean_domain_mIoU": rr["mean_domain_mIoU"],
                    "cost_rel": rr["cost_rel"],
                })

    # Pareto 前沿：代价更低且精度更高的点
    pts.sort(key=lambda d: d["cost_rel"])
    front: List[dict] = []
    best_acc = -1.0
    for d in pts:
        if d["min_domain_mIoU"] > best_acc + 1e-9:
            front.append(d)
            best_acc = d["min_domain_mIoU"]

    for d in pts:
        mark = "  *" if d in front else "   "
        R(f"{mark}{d['k']:>4d} {d['radius_m']:>6.2f} {d['sample_rate']:>5.2f} "
          f"{d['min_domain_mIoU']:>9.4f} {d['mean_domain_mIoU']:>10.4f} "
          f"{d['cost_rel']:>9.3f}")
    R("")
    R("  * = Pareto 前沿点（在更低代价下达到该精度的最优解）")
    R(f"  前沿点数 = {len(front)} / {len(pts)}")


def cmd_dense_diag(args, R: _Report) -> None:
    """密集端过平滑诊断：固定架构在**更密的域**上，k 增大是否开始伤性能？

    为什么单独做这件事
    ==================
    "密度越低 → 感受野该越大"这个命题其实有两半：

        稀疏端：k 太小 → 邻域不够 → 欠拟合      （已被主阶梯验证）
        密集端：k 太大 → 邻域过平滑 → 丢细节    （主阶梯**看不到**）

    主阶梯最密的域只有 16384 点，mIoU 已经到 0.88 饱和 —— 过平滑根本没机会
    暴露。所以先用 DENSE_DIAG_LADDER 造两个更密的探针域，再固定架构扫 k
    （半径取到**不生效**，避免混入门控效应），看 mIoU-k 曲线有没有内部极大点。

    判读
    ====
      曲线有内部极大 k* → 存在过平滑拐点 → "密集区收缩邻域"有依据，
                            命题的两半都成立，密度自适应才有"两面性"可搜。
      曲线单调不降      → 本房间内 k 越大越好，命题的另一半在本基准内
                            **不成立**，应如实报告，不要硬编个先说辞。
                            （此时该换更密的场景 / 更大的空间范围，而不是调代理）
    """
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)
    cfg = PX.ProxyConfig()

    diag_names = [n for n in domains
                  if meta.get("domains", {}).get(n, {}).get("role") == "diag"]
    if not diag_names:
        R("")
        R("没有诊断域。请用 --diag-ladder 指定（默认 [131072, 65536]），")
        R("且必须 > 主阶梯最大点数，再重建缓存。")
        return

    # 目标域 = 诊断域（更密）+ 最密的主域（作为连续性锚点）
    targets = list(diag_names)
    if main_names:
        targets.append(main_names[0])

    ks = sorted({int(v) for v in args.dense_k}, reverse=False)

    R("")
    R("=" * 72)
    R("密集端过平滑诊断")
    R("=" * 72)
    R(f"  扫描 k        : {ks}")
    R(f"  半径          : {PX.R_MAX_M} m（取到不生效，隔离门控效应）")
    R(f"  采样率        : 1.0")
    R(f"  目标域        : {targets}")
    R("")
    R(f"  {'域':<8s} {'N':>7s} {'rho':>11s} {'k':>4s} {'mIoU':>8s} "
      f"{'acc':>8s} {'n_in_med':>9s} {'r_k(m)':>8s} {'k/N':>7s}")

    summary: List[dict] = []
    for name in targets:
        xyz, y = domains[name]
        rho = D.local_density(xyz)
        rho_med = float(np.median(rho))
        curve: List[Tuple[int, float, float]] = []
        for k in ks:
            pol = (lambda p, kk=k: {"k": kk, "radius_m": PX.R_MAX_M,
                                    "sample_rate": 1.0})
            rep = EC.eval_domain(pol, xyz, y, cfg, rho=rho)
            n_in = float(np.median([t["n_in_med"] for t in rep["trace"]]))
            r_k = float((3.0 * k / (4.0 * np.pi * max(rho_med, 1e-9))) ** (1.0 / 3.0))
            R(f"  {name:<8s} {xyz.shape[0]:>7d} {rho_med:>11.3e} {k:>4d} "
              f"{rep['mIoU']:>8.4f} {rep['acc']:>8.4f} {n_in:>9.1f} "
              f"{r_k:>8.3f} {k / xyz.shape[0]:>7.4f}")
            curve.append((k, rep["mIoU"], r_k))
        best_k, best_m, best_r = max(curve, key=lambda t: t[1])
        summary.append({"domain": name, "n": int(xyz.shape[0]), "rho": rho_med,
                        "best_k": best_k, "best_mIoU": best_m, "r_k": best_r,
                        "curve": curve})

    R("")
    R("=" * 72)
    R("判读")
    R("=" * 72)
    any_interior = False
    for s in summary:
        ks_here = [c[0] for c in s["curve"]]
        interior = s["best_k"] < max(ks_here)
        any_interior = any_interior or interior
        R(f"  {s['domain']:<8s} N={s['n']:>7d}  rho={s['rho']:.3e}  "
          f"最优 k={s['best_k']} (mIoU={s['best_mIoU']:.4f}, 等效半径 {s['r_k']:.3f} m)"
          f"  {'← 存在内部极大，过平滑拐点' if interior else '← 单调，拐点在扫描区间之外'}")
        if interior:
            tail = [c for c in s["curve"] if c[0] > s["best_k"]]
            if tail:
                R(f"            k 继续增大到 {tail[-1][0]} 时 mIoU 掉 "
                  f"{s['best_mIoU'] - tail[-1][1]:.4f}")

    R("")
    if any_interior:
        R("  → 至少一个密集域出现内部极大点：**过平滑拐点存在**。")
        R("    结论：密度自适应的两个方向都有依据（稀疏端要扩、密集端要收），")
        R("    搜索空间里的 k 才真正有两面性可搜。可以把 k* 随 ρ 的移动作为")
        R("    Level 3 规律分析的一部分（k* 与 ρ 的关系是否单调）。")
    else:
        R("  → 在扫描的 k 区间内 mIoU 单调不降，**没有观察到过平滑拐点**。")
        R("    含义：本房间 + 本代理下，「密集区该收缩邻域」这半个命题不成立，")
        R("    不能写进结论。下一步应改场景条件（更大房间 / 更高密度 / 更难任务）")
        R("    再试，而不是先怀疑代理 —— 先确认观测窗口是否存在。")

    R("")
    R("  注：本诊断只反映**训练无关代理**上的趋势。真实训练是否存在更强的")
    R("      过平滑，需要 Tier-2 短训才能确认（见 ROADMAP.md 的门禁设计）。")


def cmd_edge_null(args, R: _Report) -> None:
    """边数 vs 几何：把 max-pool 的结构性偏置从几何信号里剥离出来。

    动机
    ====
    `--dense-diag` 实测 mIoU 随 k **单调上升**（三个域、密度跨 43 倍，增益都是
    约 +0.04，最优全部顶到上界 k=48）。这有两种完全不同的解释：

      H1（几何）：更大的邻域真的带来更多有用上下文 → "k 越大越好"是真的。
      H2（结构）：边数变多 → max-pool 覆盖更多随机投影 → 探针输入的**特征秩**
                  变高 → 闭式 ridge 探针天然更容易解码。**与几何无关**。

    如果 H2 成立，那么"mIoU 随几何参数变化"这个代理信号本身就不可信 ——
    它测的是边数，不是几何。这是比"自适应不成立"更严重的问题，必须先排除。

    做法：保持**真实邻居集合不变**（仍是 k 个真实邻居、物理半径门控不变），
    把邻居索引循环平铺到目标边数。于是"边数"与"邻域几何"被解耦：
        mIoU(k=8 平铺到 48 边)  对比  mIoU(真实 k=48)  对比  mIoU(真实 k=8)

    判读
    ====
      平铺后逼近真 48 邻居（回收率 ≥ 0.8） → **H2 成立**，代理测的是边数，
                                             必须改造代理（换池化 / 正交化探针）
      平铺后几乎不动（回收率 ≤ 0.3）       → H1 成立，信号是几何信号，可用
      中间                                   → 混合，两者各占一部分
    回收率 = [mIoU(padded) − mIoU(k_small)] / [mIoU(真 k=48) − mIoU(k_small)]
    """
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)

    diag_names = [n for n in domains
                  if meta.get("domains", {}).get(n, {}).get("role") == "diag"]
    targets = list(diag_names) + ([main_names[0]] if main_names else [])
    if not targets:
        R("没有可用的目标域。")
        return

    smalls = [int(v) for v in args.null_small]
    pad_to = int(args.null_pad)

    def _pol(kk):
        return lambda p: {"k": kk, "radius_m": PX.R_MAX_M, "sample_rate": 1.0}

    R("")
    R("=" * 72)
    R("边数 vs 几何：对照实验（几何不变，只放大边数）")
    R("=" * 72)
    R(f"  真实邻域 k : {smalls} 与 {pad_to}")
    R(f"  平铺目标边数: {pad_to}（把 k 个真实邻居循环重复）")
    R(f"  半径        : {PX.R_MAX_M} m（不生效）")
    R(f"  目标域      : {targets}")
    R("")

    verdicts: List[dict] = []
    for name in targets:
        xyz, y = domains[name]
        rho = D.local_density(xyz)

        R(f"  ── 域 {name}（N={xyz.shape[0]}, rho={np.median(rho):.3e}）──")
        R(f"     {'配置':<26s} {'mIoU':>8s} {'acc':>8s} {'边数':>6s} {'几何邻居':>9s}")

        res: Dict[str, float] = {}
        # 真实几何参照
        for k in smalls:
            rep = EC.eval_domain(_pol(k), xyz, y, PX.ProxyConfig(), rho=rho)
            res[f"true{k}"] = rep["mIoU"]
            R(f"     {'真实 k=' + str(k):<26s} {rep['mIoU']:>8.4f} "
              f"{rep['acc']:>8.4f} {k:>6d} {k:>9d}")
        rep48 = EC.eval_domain(_pol(pad_to), xyz, y, PX.ProxyConfig(), rho=rho)
        res["true48"] = rep48["mIoU"]
        R(f"     {'真实 k=' + str(pad_to):<26s} {rep48['mIoU']:>8.4f} "
          f"{rep48['acc']:>8.4f} {pad_to:>6d} {pad_to:>9d}")

        # 几何不变、边数放大
        for k in smalls:
            cfg = PX.ProxyConfig(pad_neighbors_to=pad_to)
            rep = EC.eval_domain(_pol(k), xyz, y, cfg, rho=rho)
            res[f"pad{k}"] = rep["mIoU"]
            R(f"     {'k=' + str(k) + ' 平铺到 ' + str(pad_to) + ' 边':<26s} "
              f"{rep['mIoU']:>8.4f} {rep['acc']:>8.4f} {pad_to:>6d} {k:>9d}")

        for k in smalls:
            denom = res["true48"] - res[f"true{k}"]
            recov = (res[f"pad{k}"] - res[f"true{k}"]) / denom if abs(denom) > 1e-9 else 0.0
            verdicts.append({"domain": name, "k": k, "recovery": recov,
                             "gap": res["true48"] - res[f"pad{k}"]})
        R("")

    R("=" * 72)
    R("判读：回收率 = [mIoU(平铺) − mIoU(真 k)] / [mIoU(真 48) − mIoU(真 k)]")
    R("=" * 72)
    for v in verdicts:
        if v["recovery"] >= 0.8:
            tag = "→ H2 成立：纯边数效应，代理不可信"
        elif v["recovery"] <= 0.3:
            tag = "→ H1 成立：是真几何信号"
        else:
            tag = "→ 混合：两边各占一部分"
        R(f"  {v['domain']:<8s} k={v['k']:<3d} 回收率={v['recovery']:.3f}  "
          f"（平铺后仍比真 48 低 {v['gap']:+.4f}）  {tag}")

    worst = max(v["recovery"] for v in verdicts) if verdicts else 0.0
    R("")
    if worst >= 0.8:
        R("  结论：至少一个配置下，**只放大边数、几何完全不变**就能拿到绝大部分增益。")
        R("  → 代理对 k 的单调偏好是池化/特征秩的结构性偏置（H2），不是几何信号。")
        R("  → 在此之上做的任何'密度自适应'对比都无意义，必须先改造代理：")
        R("     · 把 max-pool 换成 mean-pool 或 top-1，破坏'边数→秩'的耦合；")
        R("     · 或对探针输入做正交化 / 白化，消除维度红利；")
        R("     · 或改用物理几何代理（如邻域物理覆盖率），完全绕开这一耦合。")
    elif worst <= 0.3:
        R("  结论：平铺边数几乎拿不到增益 → 增益来自**真实邻域几何**（H1）。")
        R("  → 代理测量的是几何，可用。此前的负面结论（B 不如 D）应归因于")
        R("    'k 最优值贴着上界、没有密度依赖'，而不是代理失真。")
    else:
        R("  结论：混合效应。需要先做代理改造（破坏边数→秩的耦合）再重跑标定，")
        R("        否则无法把几何贡献与结构贡献分开。")


def cmd_k_sweep(args, R: _Report) -> None:
    """逐域扫 k，直接测出 k*(ρ) —— 这是论文 C3（scaling law）的核心测量。

    为什么必须逐域扫
    ================
    参照 D 是**跨 5 个域取 min** 再比较，所以"D 的最优 k=24"只说明"k=24 时最差域
    最好"，无法回答"每个密度各自的 k 是多少"。而 C3 要的恰恰是 k*(ρ) 这条曲线。
    参照 D 的表格里已经露出线索：半径不生效时 k=24→0.7162、k=32→0.6724，
    **min 反而下降** —— 说明至少有一个域存在内部极大，但看不出是哪个。

    本命令把每个域单独扫一遍，直接给出 k*(ρ)，并拟合
        k* ∝ ρ^α_k      r*(ρ) = (3k*/4πρ)^(1/3) ∝ ρ^α_r
    这两个指数就是可写进论文的定量结论。

    关于"截断"（censoring）
    ======================
    若某个域的 argmax 落在扫描区间的边界上，说明真实最优点在区间之外，该点
    对幂律拟合是**右截断**的，会系统性压低 α。所以两种拟合都给：
        fit_all  : 含截断点（乐观）
        fit_free : 只含非截断点（保守，可信）
    报告时以 fit_free 为准，fit_all 仅作上界参照。

    关于扫描上界
    ============
    搜索空间里 k 的硬上界是 PX.K_MAX(=48)。但诊断需要看超过 48 的行为，
    所以这里允许用 --k-max 临时抬高上界（**仅本诊断有效，用完还原**），
    否则无法判断"密集域是真想要 48，还是只是被夹住了"。
    """
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)

    diag_names = [n for n in domains
                  if meta.get("domains", {}).get(n, {}).get("role") == "diag"]
    targets = list(main_names) + (list(diag_names) if args.sweep_with_diag else [])

    ks = sorted({int(v) for v in args.sweep_k})
    k_scan_max = max(ks)

    saved_kmax = PX.K_MAX
    if args.k_max is not None and int(args.k_max) > PX.K_MAX:
        PX.K_MAX = int(args.k_max)

    R("")
    R("=" * 72)
    R("逐域 k 扫描：测 k*(ρ)")
    R("=" * 72)
    R(f"  扫描 k    : {ks}")
    R(f"  半径      : {PX.R_MAX_M} m（不生效，隔离门控效应）")
    R(f"  采样率    : 1.0")
    R(f"  目标域    : {targets}")
    R(f"  K 上界    : {PX.K_MAX}（诊断用临时值），扫描上界 {k_scan_max}")
    R("")

    try:
        points: List[dict] = []
        for name in targets:
            xyz, y = domains[name]
            rho = D.local_density(xyz)
            rho_med = float(np.median(rho))
            role = meta.get("domains", {}).get(name, {}).get("role", "main")
            curve: List[Tuple[int, float]] = []
            R(f"  ── {name}（role={role}, N={xyz.shape[0]}, rho={rho_med:.3e}）──")
            for k in ks:
                pol = (lambda p, kk=k: {"k": kk, "radius_m": PX.R_MAX_M,
                                        "sample_rate": 1.0})
                rep = EC.eval_domain(pol, xyz, y, PX.ProxyConfig(), rho=rho)
                r_k = float((3.0 * k / (4.0 * np.pi * max(rho_med, 1e-9))) ** (1.0 / 3.0))
                R(f"     k={k:<4d} mIoU={rep['mIoU']:.4f}  acc={rep['acc']:.4f}  "
                  f"等效半径 r_k={r_k:.4f} m")
                curve.append((k, rep["mIoU"]))
            best_k, best_m = max(curve, key=lambda t: t[1])
            censored = int(best_k) == k_scan_max or int(best_k) == min(ks)
            r_star = float((3.0 * best_k / (4.0 * np.pi * max(rho_med, 1e-9))) ** (1.0 / 3.0))
            points.append({"domain": name, "role": role, "rho": rho_med,
                           "n": int(xyz.shape[0]), "k_star": int(best_k),
                           "mIoU": float(best_m), "r_star": r_star,
                           "censored": bool(censored), "curve": curve})
            R(f"     → k*={best_k}  r*={r_star:.4f} m  mIoU={best_m:.4f}"
              f"{'   [截断：最优点在扫描区间外]' if censored else ''}")
            R("")

        # ---- 幂律拟合（在 main 域上做，diag 只作旁证）----
        mpts = [p for p in points if p["role"] == "main"]
        free = [p for p in mpts if not p["censored"]]

        R("=" * 72)
        R("幂律拟合（这是 C3 的核心数字）")
        R("=" * 72)
        R(f"  {'域':<8s} {'主/诊':<6s} {'N':>7s} {'rho':>11s} {'k*':>5s} "
          f"{'r*(m)':>8s} {'mIoU':>8s}  截断")

        for p in points:
            R(f"  {p['domain']:<8s} {p['role']:<6s} {p['n']:>7d} {p['rho']:>11.3e} "
              f"{p['k_star']:>5d} {p['r_star']:>8.4f} {p['mIoU']:>8.4f}  "
              f"{'是' if p['censored'] else '否'}")

        R("")
        a_k_all = D.fit_exponent([p["rho"] for p in mpts], [p["k_star"] for p in mpts])
        a_k_free = D.fit_exponent([p["rho"] for p in free], [p["k_star"] for p in free])
        a_r_all = D.fit_exponent([p["rho"] for p in mpts], [p["r_star"] for p in mpts])
        a_r_free = D.fit_exponent([p["rho"] for p in free], [p["r_star"] for p in free])

        def _fmt(v):
            return "无法拟合" if v is None else f"{v:+.4f}"

        R(f"  alpha_k  (k* ∝ rho^alpha)  含截断 = {_fmt(a_k_all)}   "
          f"仅非截断 = {_fmt(a_k_free)}")
        R(f"  alpha_r  (r* ∝ rho^alpha)  含截断 = {_fmt(a_r_all)}   "
          f"仅非截断 = {_fmt(a_r_free)}")
        R("")
        R("  参照含义（推导：球内点数 k ≈ 4/3·pi·r^3·rho）:")
        R("    alpha_k ≈ 0    → k* 与密度无关（图度数守恒）")
        R("    alpha_k ≈ +1   → 物理感受野半径守恒（完全补偿密度漂移）")
        R("    alpha_r ≈ -1/3 → 等价于 alpha_k ≈ 0")
        R("    0 < alpha_k < 1 → 部分补偿")
        R("")

        R("=" * 72)
        R("判读")
        R("=" * 72)
        n_cens = len([p for p in mpts if p["censored"]])
        R(f"  main 域共 {len(mpts)} 个，其中 {n_cens} 个的最优 k 落在扫描边界（截断）")
        if len(free) < 2:
            R("  非截断点不足 2 个，**无法可信地拟合 alpha**。")
            R("  → 必须扩大扫描区间（--sweep-k 加到更大，并配 --k-max 抬高上界），")
            R("    否则 k*(ρ) 这条曲线根本没被测出来，C3 无从谈起。")
        else:
            ak = a_k_free
            if ak is None:
                R("  拟合失败（密度或 k* 无变化）。")
            elif ak > 0.5:
                R(f"  alpha_k = {ak:+.3f} → 接近 1：**物理感受野半径近似守恒**，")
                R("    稀疏域确实需要更大的 k。这是支持密度自适应的定量证据。")
            elif ak > 0.15:
                R(f"  alpha_k = {ak:+.3f} → **部分补偿**：k* 随密度增长，但不足以")
                R("    维持恒定物理半径。这本身就是有价值的发现（网络只需部分补偿），")
                R("    应作为 C3 的主结论写出来。")
            elif ak > -0.15:
                R(f"  alpha_k = {ak:+.3f} ≈ 0 → **k* 与密度无关**（图度数守恒）。")
                R("    含义：最优 k 是密度不变的常数，物理感受野自然按 rho^(-1/3) 收缩。")
                R("    → 固定 k 架构已经是最优族，密度条件化 k 没有可搜的空间。")
                R("      此时 C3 的结论应改写成『局部几何最优解是密度不变的』这一")
                R("      否定性规律 —— 也能写，但要如实定性，不能包装成自适应。")
            else:
                R(f"  alpha_k = {ak:+.3f} < 0 → **k* 随密度下降而下降**（反直觉）。")
                R("    含义：越稀疏的域反而要更小的 k（因为大 k 在稀疏域等于近全局聚合）。")
                R("    → 命题方向是反的，必须重新表述假设。")
        R("")
        R("  注：diag 域（更密）只作旁证，不参与拟合 —— 它们不是部署域。")
    finally:
        PX.K_MAX = saved_kmax


def cmd_seed_noise(args, R: _Report) -> None:
    """代理噪声底：同一配置换随机种子重复评估，量出 mIoU 的固有抖动。

    为什么这是必要条件
    ==================
    前面的 --k-sweep 得到 alpha_k = -0.071，但只用 3 个非截断点拟合，且曲线
    本身不光滑（rho1 在 k=48 掉到 0.8695，k=64 又回到 0.8780，相差 0.0085）。
    在下任何"k* 与密度无关"的结论之前必须先回答：

        **代理能不能分辨这个尺度的差异？**

    代理是确定性的（cfg.seed 固定），但固定种子只意味着"可复现"，不意味着
    "无噪声"：探针的训练/评估划分、随机投影矩阵、特征插值都依赖种子。
    换种子跑同一配置得到的分差，就是这套仪器的**分辨率下限**。

    判读
    ====
      sigma_seed << 同一域内 k 之间的差值  → 代理有分辨力，k* 可信
      sigma_seed 与差值同量级            → k* 不可辨识，**alpha 是在拟合噪声**，
                                            必须先降噪（多 seed 平均 / 加大
                                            探针 / 增加评估点）再谈结论

    这一步是"仪器校准"的最后一块：--edge-null 排除了**偏置**（边数效应），
    本命令量的是**噪声**。两者都过关，代理的读数才可用。
    """
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)
    ref = float(meta.get("rho_ref") or EC.set_rho_ref_from_domains(domains, main_names))
    D.set_rho_ref(ref)

    diag_names = [n for n in domains
                  if meta.get("domains", {}).get(n, {}).get("role") == "diag"]
    targets = list(main_names) + (list(diag_names) if args.sweep_with_diag else [])

    ks = sorted({int(v) for v in args.noise_k})
    seeds = [int(s) for s in args.noise_seeds]

    R("")
    R("=" * 72)
    R("代理噪声底：换种子重复评估同一配置")
    R("=" * 72)
    R(f"  k 取值    : {ks}")
    R(f"  种子      : {seeds}（共 {len(seeds)} 个）")
    R(f"  半径      : {PX.R_MAX_M} m（不生效）")
    R(f"  目标域    : {targets}")
    R("")
    R(f"  {'域':<8s} {'k':>4s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'极差':>8s}")

    rows: List[dict] = []
    by_domain: Dict[str, List[dict]] = {}
    for name in targets:
        xyz, y = domains[name]
        rho = D.local_density(xyz)
        for k in ks:
            vals = []
            for sd in seeds:
                cfg = PX.ProxyConfig(seed=sd)
                pol = (lambda p, kk=k: {"k": kk, "radius_m": PX.R_MAX_M,
                                        "sample_rate": 1.0})
                rep = EC.eval_domain(pol, xyz, y, cfg, rho=rho)
                vals.append(rep["mIoU"])
            arr = np.asarray(vals, dtype=np.float64)
            row = {"domain": name, "k": k, "mean": float(arr.mean()),
                   "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                   "min": float(arr.min()), "max": float(arr.max()),
                   "rng": float(arr.max() - arr.min())}
            rows.append(row)
            by_domain.setdefault(name, []).append(row)
            R(f"  {name:<8s} {k:>4d} {row['mean']:>8.4f} {row['std']:>8.4f} "
              f"{row['min']:>8.4f} {row['max']:>8.4f} {row['rng']:>8.4f}")

    R("")
    R("=" * 72)
    R("判读：把种子抖动与 k 造成的真实差异作比较")
    R("=" * 72)
    R(f"  {'域':<8s} {'种子std':>9s} {'k间极差':>9s} {'分辨率比':>10s}  结论")

    overall_ok = True
    for name, rs in by_domain.items():
        sig = float(np.mean([r["std"] for r in rs]))
        kspread = float(max(r["mean"] for r in rs) - min(r["mean"] for r in rs))
        ratio = sig / kspread if kspread > 1e-9 else float("inf")
        if ratio >= 0.5:
            verdict = "不可辨识：k* 落在噪声里"
            overall_ok = False
        elif ratio >= 0.25:
            verdict = "勉强可辨：结论需多 seed 平均"
        else:
            verdict = "可辨识"
        R(f"  {name:<8s} {sig:>9.4f} {kspread:>9.4f} {ratio:>10.3f}  {verdict}")

    R("")
    if overall_ok:
        R("  → 种子抖动远小于 k 造成的差异，代理具备分辨 k 的能力。")
        R("    alpha_k / alpha_r 的数值可以用，但仍建议报告时给出多种子均值。")
    else:
        R("  → 至少一个域的 k 间差异与种子抖动同量级：**该域上 k* 不可辨识**。")
        R("    → alpha_k 很可能是在拟合噪声，不能作为 C3 的结论。")
        R("    必须先降噪：多 seed 平均、加大 probe_dim、或在更多域/房间上重复。")
    R("")
    R("  注：sigma 用 ddof=1（样本标准差）；分辨率比 = 种子std / 域内 k 间极差。")


def cmd_geom_stats(args, R: _Report) -> None:
    """实测邻域几何：k 变大改变的到底是"邻居数量"还是"物理覆盖范围"？

    为什么必须有这一步
    ==================
    报告里的"等效半径" r = (3k/4πρ)^(1/3) 用了**三维体填充**假设，而 S3DIS 是
    **表面采样**。这导致两个后果：

      (1) 由该定义可推出恒等式 alpha_r = (alpha_k - 1)/3，所以 alpha_k 与
          alpha_r 是同一条结论的两种写法，**不能当作互相印证**。
      (2) "物理感受野随密度漂移"这类论述若依赖 3D 公式，前提本身就没被检验。

    本命令给出**不依赖假设**的实测量（r_k / dbar_k / C_k）与内在维度 gamma：
      dbar_k ∝ k^gamma，gamma≈1/2 → 二维面；gamma≈1/3 → 三维体。
    再跨域拟合实测的 alpha_r^emp / alpha_dbar，与代数恒等式对照。
    """
    domains, meta, cache_dir = _load_or_build(args, R)
    main_names = EC.main_domain_names(domains, cache_dir, meta)

    targets = list(main_names)
    if args.sweep_with_diag:
        targets += [n for n in domains
                    if meta.get("domains", {}).get(n, {}).get("role") == "diag"]

    ks = sorted({int(v) for v in args.geom_k})

    R("")
    R("=" * 72)
    R("实测邻域几何（无体密度假设）")
    R("=" * 72)
    R(f"  k 取值      : {ks}")
    R(f"  目标域      : {targets}")
    R(f"  子采样上限  : {args.geom_max_pts or '不子采样'}（子采样会放大 r_k，禁用跨域比较）")
    R("")
    R(f"  {'域':<8s} {'rho':>11s} {'k':>4s} {'r_k(中位)':>10s} {'dbar(中位)':>11s} "
      f"{'dbar/r_k':>9s} {'C_k(中位)':>11s}")

    roles = EC.load_roles(cache_dir, meta)
    sub = {n: domains[n] for n in targets}
    rows = GE.coverage_table(sub, ks, max_points=int(args.geom_max_pts),
                             seed=int(args.seed), roles=roles)
    gamma_by_dom: Dict[str, Optional[float]] = {}
    for r in rows:
        gamma_by_dom[r["domain"]] = r["gamma"]
        flag = "  [子采样·不可跨域比较]" if r.get("subsampled") else ""
        R(f"  {r['domain']:<8s} {r['rho']:>11.3e} {r['k']:>4d} {r['r_k_med']:>10.4f} "
          f"{r['dbar_med']:>11.4f} {r['ratio']:>9.3f} {r['C_k_med']:>11.3e}{flag}")

    R("")
    R("  ── 内在维度 gamma（同域内 dbar_k ∝ k^gamma）──")
    R(f"  {'域':<8s} {'gamma':>8s}   判读")
    for name in targets:
        g = gamma_by_dom.get(name)
        if g is None:
            R(f"  {name:<8s} {'n/a':>8s}   样本不足")
            continue
        if abs(g - 0.5) < 0.08:
            tag = "≈1/2 → **二维面采样**（3D 体假设不成立）"
        elif abs(g - 1.0 / 3.0) < 0.08:
            tag = "≈1/3 → 三维体填充"
        else:
            tag = f"介于两者之间（面/体混合或尺度相关）"
        R(f"  {name:<8s} {g:>8.4f}   {tag}")

    # ---- 跨域实测指数（固定 k，跨 ρ）----
    R("")
    R("  ── 跨域实测指数（固定 k，把 r_k / dbar / C_k 对 rho 拟合幂律）──")
    R("  ⚠ 只使用 **未子采样的 main 域**：子采样会放大 r_k，混进拟合会压低指数")
    R("    （踩过坑：把 diag0 混进来会使 alpha_r 从 ~-0.31 变成 -0.20）。")
    R(f"  {'k':>4s} {'n域':>4s} {'alpha_r^实测':>13s} {'alpha_dbar':>11s} {'alpha_C':>9s}")
    emp: Dict[int, Dict[str, Optional[float]]] = {}
    for k in ks:
        kr = [r for r in rows
              if r["k"] == k and r.get("role", "main") == "main"
              and not r.get("subsampled", False)]
        if len(kr) < 2:
            continue
        rhos = [r["rho"] for r in kr]
        a_r = D.fit_exponent(rhos, [r["r_k_med"] for r in kr])
        a_d = D.fit_exponent(rhos, [r["dbar_med"] for r in kr])
        a_c = D.fit_exponent(rhos, [r["C_k_med"] for r in kr])
        emp[k] = {"r": a_r, "dbar": a_d, "C": a_c}

        def _f(v):
            return "n/a" if v is None else f"{v:+.4f}"

        R(f"  {k:>4d} {len(kr):>4d} {_f(a_r):>13s} {_f(a_d):>11s} {_f(a_c):>9s}")

    R("")
    R("=" * 72)
    R("判读")
    R("=" * 72)
    R("  1) k 改变的是「数量」还是「范围」？看 dbar/r_k 是否随 k 变：")
    R("     该比值近似恒定 → 邻居是自相似的（k 只放大物理覆盖范围）；")
    R("     该比值随 k 明显下降 → k 增大时邻居更贴向半径球面（覆盖范围增长更快）。")
    R("")
    R("  2) ⚠ 报告里 alpha_r 与 alpha_k 的关系是**代数恒等式**，不是两条证据：")
    R("     r = (3k/4pi*rho)^(1/3)  ⇒  alpha_r = (alpha_k - 1)/3")
    R("     实证：alpha_k = -0.0707 → (alpha_k-1)/3 = -0.3569，与报告的 alpha_r 逐位一致。")
    R("     因此论文里**只能保留一个**，另一个作为换算结果说明，不能并列称「互相印证」。")
    R("")
    R("  3) 上表的 alpha_r^实测 才是**不依赖 3D 假设**的独立测量。若它与")
    R("     (alpha_k-1)/3 差别明显，说明「等效半径」这条定义在表面采样上有偏，")
    R("     论文里讨论「物理感受野」必须改用实测 r_k，而不是公式反解值。")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="S3DIS 单房间数据准备与标定")
    ap.add_argument("--build", action="store_true", help="构造受控密度域缓存")
    ap.add_argument("--calibrate", action="store_true", help="标定：四组参照对比")
    ap.add_argument("--pareto", action="store_true", help="扫 Pareto 前沿")
    ap.add_argument("--dense-diag", action="store_true",
                    help="密集端过平滑诊断（用 diag 域扫固定 k）")
    ap.add_argument("--edge-null", action="store_true",
                    help="边数 vs 几何对照（几何不变只放大边数，判定代理是否失真）")
    ap.add_argument("--null-small", type=int, nargs="+", default=[8, 16],
                    help="--edge-null 用的真实邻域 k")
    ap.add_argument("--null-pad", type=int, default=48,
                    help="--edge-null 的平铺目标边数（默认 48 = K_MAX）")
    ap.add_argument("--k-sweep", action="store_true",
                    help="逐域扫描 k，测 k*(rho)（论文 C3 的核心测量）")
    ap.add_argument("--sweep-k", type=int, nargs="+",
                    default=[8, 12, 16, 24, 32, 48, 64, 96],
                    help="--k-sweep 扫描的 k 取值")
    ap.add_argument("--sweep-with-diag", action="store_true",
                    help="--k-sweep 是否也扫 diag 域（仅旁证，不参与拟合）")
    ap.add_argument("--k-max", type=int, default=None,
                    help="诊断用：临时抬高 K 上界（仅 --k-sweep 生效，用完还原）")
    ap.add_argument("--seed-noise", action="store_true",
                    help="代理噪声底：换种子重复评估同一配置，量出可辨识分辨率")
    ap.add_argument("--noise-k", type=int, nargs="+", default=[16, 48],
                    help="--seed-noise 用的 k 取值")
    ap.add_argument("--noise-seeds", type=int, nargs="+",
                    default=[12345, 12346, 12347, 20260923],
                    help="--seed-noise 用的随机种子")
    ap.add_argument("--probe-repeats", type=int, default=None,
                    help="探针重复次数（对种子取平均，降噪 √n）。默认 1 = 历史可比；"
                         "做结论时建议 >=8")
    ap.add_argument("--probe-folds", type=int, default=None,
                    help="探针 k 折交叉验证折数（消除划分方差，全点参与评估）。"
                         "默认 1 = 历史的一次性 60/40 划分")
    ap.add_argument("--geom-stats", action="store_true",
                    help="实测邻域几何（无体密度假设）：k 改的是数量还是物理范围")
    ap.add_argument("--geom-k", type=int, nargs="+",
                    default=[8, 16, 32, 48, 64, 96, 128],
                    help="--geom-stats 统计的 k 取值")
    ap.add_argument("--geom-max-pts", type=int, default=0,
                    help="--geom-stats 每域统计用的子采样点数上限；0=不子采样（默认，"
                         "跨密度比较必须如此）")
    ap.add_argument("--list-rooms", action="store_true", help="列出可用房间")
    ap.add_argument("--area", default=DT.DEFAULT_AREA, help="S3DIS Area，如 Area_1")
    ap.add_argument("--room", default=DT.DEFAULT_ROOM, help="房间名，如 conferenceRoom_1")
    ap.add_argument(
        "--ladder", type=int, nargs="+", default=DT.POINT_LADDER,
        help="每域目标点数阶梯（必须全为正整数）",
    )
    ap.add_argument(
        "--diag-ladder", type=int, nargs="+", default=DT.DENSE_DIAG_LADDER,
        help="密集端诊断阶梯。只用于 --dense-diag，不计入任何标量指标。"
             "传 0 可关闭诊断域。",
    )
    ap.add_argument(
        "--dense-k", type=int, nargs="+", default=[8, 16, 24, 32, 48],
        help="密集端诊断扫描的 k 取值",
    )
    ap.add_argument("--read-max-pts", type=int, default=DT.ROOM_READ_MAX_PTS,
                    help="读取房间时的原始点数上限")
    ap.add_argument("--cache-dir", default="", help="缓存目录（默认 data/cache_s3dis）")
    ap.add_argument("--report", default="", help="把报告同时写入该 UTF-8 文件")
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    # 传 0 表示关闭诊断域
    if args.diag_ladder and all(v <= 0 for v in args.diag_ladder):
        args.diag_ladder = []

    # 探针重复次数：模块级默认，所有 ProxyConfig 统一继承
    if args.probe_repeats is not None:
        PX.DEFAULT_PROBE_REPEATS = int(max(1, args.probe_repeats))
    if args.probe_folds is not None:
        PX.DEFAULT_PROBE_FOLDS = int(max(1, args.probe_folds))

    R = _Report(args.report or None)
    try:
        if args.list_rooms:
            cmd_list_rooms(args, R)
            return
        if not (args.build or args.calibrate or args.pareto or args.dense_diag
                or args.edge_null or args.k_sweep or args.seed_noise
                or args.geom_stats):
            ap.print_help()
            return
        if args.build:
            cmd_build(args, R)
        if args.calibrate:
            cmd_calibrate(args, R)
        if args.pareto:
            cmd_pareto(args, R)
        if args.dense_diag:
            cmd_dense_diag(args, R)
        if args.edge_null:
            cmd_edge_null(args, R)
        if args.k_sweep:
            cmd_k_sweep(args, R)
        if args.seed_noise:
            cmd_seed_noise(args, R)
        if args.geom_stats:
            cmd_geom_stats(args, R)
    finally:
        R.close()


if __name__ == "__main__":
    main()
