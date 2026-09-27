"""为什么 3D 栅格分支还是崩？（逐层定位，不猜）

现象：z 分层后 mIoU 仍 0.02~0.05，且**单 run 时间从 33s 掉到 1.2s**。
后者是关键信号 —— 训练几乎不耗时，说明**梯度极弱或特征被压成常数**。

可能的机制（按"会被误当成合理"的程度排序）
================================================================
M1. **BEV/体素图里 95% 是空像素**，而 BN 在 batch=1 时对 (B,C,H,W) 统计 →
    均值/方差被大量 0 主导，激活值被归一化到接近 0 → 回填到点的特征近乎常数
    → head 无论怎么学都输出同一类 → mIoU ≈ 该房间最大类占比的倒数。
    **这是最可疑的**：空像素占比 95% 且 batch=1。
M2. **回填后同格点特征相同**，逐点精度有硬上界（已知：柱宽 0.5m 时 acc 上界 0.53）。
    但这条解释不了"时间掉到 1.2s"。
M3. **梯度路径被 BN 截断**：BN 在 batch=1 下的 running stats 让训练/推理不一致。

本脚本逐层 dump 激活统计，定位是哪一层把信号压死。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nas_ptseg import gpu as GPU          # noqa: E402
from nas_ptseg import room_split as RS    # noqa: E402
from nas_ptseg.multibranch import PillarBranchV2  # noqa: E402


def stat(t: torch.Tensor, name: str) -> None:
    with torch.no_grad():
        print(f"    {name:<34s} shape={tuple(t.shape)}  "
              f"mean={float(t.mean()):+.4f} std={float(t.std()):+.4f} "
              f"nonzero_frac={float((t.abs() > 1e-8).float().mean()):.3f} "
              f"|max|={float(t.abs().max()):.3f}")


def main() -> int:
    dev = GPU.setup_torch_device()
    split = RS.load_split()
    a, r = split["splits"]["val"][0]["key"].split("/", 1)
    xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_np, dtype=np.float32)).to(dev)
    print(f"[diag] 房间 {a}/{r}  N={xyz.shape[0]}\n")

    pb = PillarBranchV2(3, 64, xy_range=8.0, pillar_xy=0.25, z_bins=8).to(dev)

    print("  A) 训练模式下的逐层统计（BN 用 batch 统计）")
    pb.train()
    with torch.no_grad():
        cell = pb.cell_ids(xyz)
        ids = cell[0]
        h = pb.point_mlp(xyz)
        stat(h, "point_mlp 输出")
        uniq, inverse = torch.unique(ids, return_inverse=True)
        inverse = inverse.reshape(-1)
        C = h.shape[1]
        hmax = torch.zeros((int(uniq.numel()), C), device=dev)
        hmax = hmax.index_reduce(0, inverse, h, "amax", include_self=True)
        stat(hmax, "柱内 max 聚合后（含空柱的 0）")
        # 关键：空柱占比
        uniq_ids = uniq
        px = uniq_ids % pb.side
        py = (uniq_ids // pb.side) % pb.side
        pz = uniq_ids // (pb.side * pb.side)
        pix = (pz * pb.side + py) * pb.side + px
        # 用"落在体素中间的比例"反推空占比
        total_vox = pb.z_bins * pb.side * pb.side
        print(f"    体素总数={total_vox}  非空={int(uniq.numel())}  "
              f"空占比={100*(1-int(uniq.numel())/total_vox):.1f}%")

        vol = hmax.new_zeros((1, 2 * C, pb.z_bins, pb.side, pb.side))
        flat = vol.view(1, 2 * C, -1)
        flat[0, :C, pix] = hmax.t()
        stat(vol, "体素 5D 张量（含 95% 空像素）")

        out = pb(xyz, xyz, cell=cell)
        stat(out, "分支输出（回填到点）")
        stat(pb.point_mlp[0].weight, "point_mlp 第0层权重")

    print("\n  B) BN 的 running stats（训练后偏置）")
    for name, m in pb.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            with torch.no_grad():
                print(f"    {name:<30s} rm={float(m.running_mean.mean()):+.4f} "
                      f"rv={float(m.running_var.mean()):.4f} "
                      f"w={float(m.weight.mean()):+.4f}")

    print("\n  C) 关键判据：分支输出在**不同点之间**是否几乎相同")
    pb.eval()
    with torch.no_grad():
        out = pb(xyz, xyz)
    o = out[0]                                     # (C, N)
    # 逐通道跨点的标准差；若全部趋 0 → 特征对点没有区分度
    sd = o.std(dim=1)
    print(f"    跨点 std：均值={float(sd.mean()):.6f}  最大={float(sd.max()):.6f}  "
          f"最小={float(sd.min()):.6f}")
    # 同一格 vs 不同格 的特征差
    cell = pb.cell_ids(xyz)
    ids = cell[0]
    d_within, d_between = [], []
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=min(2000, xyz.shape[0]), replace=False)
    for i in idx[:400]:
        same = (ids == ids[i]).nonzero().flatten()
        diff = (ids != ids[i]).nonzero().flatten()
        if same.numel() > 1:
            d_within.append(float((o[:, same] - o[:, i:i+1]).abs().mean()))
        if diff.numel():
            d_between.append(float((o[:, diff] - o[:, i:i+1]).abs().mean()))
    print(f"    同格内特征差 = {np.mean(d_within):.6f}（应为 0）")
    print(f"    异格间特征差 = {np.mean(d_between):.6f}（应显著 > 0，否则特征无区分度）")

    print("\n  D) 判读")
    if np.mean(d_between) < 1e-4:
        print("    → **特征对点完全没有区分度**（异格间差也≈0）：BN 被空像素压死。")
        print("      修法：BEV/体素卷积**不要用默认 BN**（batch=1 + 95% 空像素），")
        print("      改用 GroupNorm/LayerNorm，或只在非空体素上做归一化。")
    else:
        print("    → 特征本身有区分度，问题在 head 或优化；需进一步查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
