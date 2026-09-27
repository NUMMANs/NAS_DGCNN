"""DGCNN 代理自测：不依赖 S3DIS 数据，用合成点云检查通路。

检查项
======
1. 前向能跑通、不抛异常、形状正确
2. 探针 mIoU 显著高于随机猜测（说明特征里确实有可解码的语义信息）
3. 密度自适应先验 vs 固定架构在稀疏域上的差异
4. 半径门控真的在起作用（改变 radius 会改变输出）
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas_ptseg import proxy as PX           # noqa: E402
from nas_ptseg import data as DT            # noqa: E402
from nas_ptseg import density as D          # noqa: E402

RNG = np.random.default_rng(7)


def make_room(n: int = 6000) -> tuple:
    """合成一个"房间"：地面 + 四面墙 + 一个桌子，共 3 类。"""
    per = n // 6
    parts = []
    ys = []
    # 地面 (z=0)
    parts.append(np.stack([RNG.uniform(-3, 3, per), RNG.uniform(-2.5, 2.5, per), np.zeros(per)], axis=1))
    ys.append(np.zeros(per, dtype=np.int64))
    # 四面墙
    for _ in range(4):
        t = RNG.uniform(0, 1, per)
        side = RNG.integers(0, 4, per)
        x = np.where(side < 2, np.where(side == 0, -3.0, 3.0), RNG.uniform(-3, 3, per))
        y = np.where(side < 2, RNG.uniform(-2.5, 2.5, per), np.where(side == 2, -2.5, 2.5))
        z = RNG.uniform(0, 2.6, per)
        parts.append(np.stack([x, y, z], axis=1))
        ys.append(np.ones(per, dtype=np.int64))
    # 桌子（长方体表面）
    for _ in range(2):
        x = RNG.uniform(-0.8, 0.8, per)
        y = RNG.uniform(-0.5, 0.5, per)
        z = np.where(RNG.random(per) < 0.7, np.full(per, 0.75), RNG.uniform(0, 0.75, per))
        parts.append(np.stack([x, y, z], axis=1))
        ys.append(np.full(per, 2, dtype=np.int64))

    xyz = np.concatenate(parts, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    return xyz, y


def main() -> None:
    out = []
    out.append("=" * 70)
    out.append("DGCNN 代理自测")
    out.append("=" * 70)

    xyz, y = make_room(9000)
    cfg = PX.ProxyConfig()
    out.append(f"合成点云: N={xyz.shape[0]}  类别数={DT.NUM_CLASSES}")

    rho = D.local_density(xyz)
    out.append(f"局部密度: med={np.median(rho):.3e} pts/m^3  "
               f"p10={np.percentile(rho,10):.3e}  p90={np.percentile(rho,90):.3e}")

    # ---- 1) 前向通路 ----
    t0 = time.perf_counter()
    res = PX.forward(PX.baseline_policy, xyz, cfg, rho_precomputed=rho)
    t_fwd = time.perf_counter() - t0
    out.append("")
    out.append(f"[1] 前向: feats={res['feats'].shape}  errors={res['errors']}  {t_fwd*1000:.0f} ms")
    for t in res["trace"]:
        out.append(f"    layer{t['layer']}: k={t['k']:>2d} r={t['radius_m']:.2f}m "
                   f"sr={t['sample_rate']:.2f} nodes={t['n_nodes']:>5d} "
                   f"n_in_med={t['n_in_med']:>6.0f} r/spacing={t['radius_over_spacing']:.2f}")
    out.append(f"    cost: params={res['cost']['params']:.0f} flops={res['cost']['flops']:.3e}")

    # ---- 2) 探针 ----
    t0 = time.perf_counter()
    miou, acc, per = PX.linear_probe_miou(res["feats"], y, DT.NUM_CLASSES, cfg)
    t_probe = time.perf_counter() - t0
    chance = 1.0 / 3.0
    out.append("")
    out.append(f"[2] 探针: mIoU={miou:.4f}  acc={acc:.4f}  (随机猜测 mIoU≈{chance/2:.3f})  {t_probe*1000:.0f} ms")
    out.append(f"    每类 IoU = {[None if np.isnan(v) else round(float(v),3) for v in per]}")
    out.append(f"    => {'PASS' if miou > chance * 0.9 else 'FAIL: 特征里几乎没有可解码信息'}")

    # ---- 3) 半径门控是否有效 ----
    out.append("")

    def _fixed(k, r):
        return lambda p: {"k": k, "radius_m": r, "sample_rate": 1.0}

    feats_small_r = PX.forward(_fixed(16, 0.05), xyz, cfg, rho_precomputed=rho)["feats"]
    diff = float(np.abs(feats_small_r - res["feats"]).max())
    out.append(f"[3] 半径门控: r=0.05 与 r=0.30 的输出最大差异={diff:.3e} "
               f"=> {'PASS（门控生效）' if diff > 1e-6 else 'FAIL（半径没起作用）'}")

    # ---- 4) 稀疏域：固定 vs 自适应 ----
    out.append("")
    out.append("[4] 稀疏域对比（体素下采样制造密度漂移）")
    n_max = xyz.shape[0]
    for target in (n_max, n_max // 3, n_max // 8):
        v, xd, yd = DT.calibrate_voxel_for_target_n(xyz, y, target)
        rd = D.local_density(xd)
        prof = D.density_profile(rd, 0, 3)
        m_fix, a_fix, _ = PX.linear_probe_miou(
            PX.forward(PX.baseline_policy, xd, cfg, rho_precomputed=rd)["feats"], yd, DT.NUM_CLASSES, cfg
        )

        def _adaptive(p):
            s = p["scale_hint"]
            return {
                "k": int(np.clip(round(14.0 + 30.0 * min(s / 0.30, 1.0)), PX.K_MIN, PX.K_MAX)),
                "radius_m": float(np.clip(2.2 * s, PX.R_MIN_M, PX.R_MAX_M)),
                "sample_rate": 1.0,
            }

        m_ada, a_ada, _ = PX.linear_probe_miou(
            PX.forward(_adaptive, xd, cfg, rho_precomputed=rd)["feats"], yd, DT.NUM_CLASSES, cfg
        )
        out.append(f"    N={xd.shape[0]:>5d} voxel={v*100:.2f}cm rho={np.median(rd):.3e} "
                   f"ratio={prof['ratio']:.3f} | 固定 mIoU={m_fix:.4f}  "
                   f"自适应 mIoU={m_ada:.4f}  delta={m_ada-m_fix:+.4f}")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_selftest.txt"),
              "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
