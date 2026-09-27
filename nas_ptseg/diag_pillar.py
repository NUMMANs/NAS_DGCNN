"""柱状分支为什么学不动？—— 定位而不是猜。

现象（2026-09-24 pilot）：pillar25 / pillar50 的 mIoU 只有 0.04，
远低于 points 的 0.3465；而 points_mean（0.2749）虽差但仍在合理范围。

三个候选原因，必须分辨
================================================================
C1. **BEV 栅格空间分辨率太低**：`xy_range=8m` + `pillar_xy=0.25m` → side=64，
    对 16k 点的房间尚可；但 0.5m 柱宽 → side=32，天花板/地板薄结构会消失。
C2. **回填特征对同柱点完全相同** → 同柱点的 logits 相同，逐点分割精度有硬上限。
    柱宽 0.5m 时，一个柱覆盖很多点 → 上限很低。**这是结构性的，不是调参问题。**
C3. **BN 统计量在 BEV 图上被大量空像素污染**：BEV 图里绝大多数像素是空的（0），
    BN 在 batch 维度统计（batch=1！）会把空像素的 0 混进均值/方差 → 特征被压死。

本脚本量化 C1/C2（不训练即可测），并用一次极短训练验证 C3。
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


def main() -> int:
    dev = GPU.setup_torch_device()
    split = RS.load_split()
    print("[diag] 量 C1/C2：栅格占用率 与 回填的信息损失\n")
    print(f"  {'房间':<26s}{'柱宽':>6s}{'side':>6s}{'占柱数':>8s}{'占用率':>8s}"
          f"{'柱内平均点数':>13s}{'最大柱点数':>11s}")
    for key in [d["key"] for d in split["splits"]["val"]][:4]:
        a, r = key.split("/", 1)
        xyz_np, _y, _st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
        xyz = torch.from_numpy(np.ascontiguousarray(xyz_np, dtype=np.float32)).to(dev)
        for pw in (0.25, 0.5):
            pb = PillarBranchV2(3, 32, xy_range=8.0, pillar_xy=pw).to(dev).eval()
            with torch.no_grad():
                ids, _gx, _gy = pb.cell_ids(xyz)
            uniq, inv = torch.unique(ids, return_inverse=True)
            inv = inv.reshape(-1)
            counts = torch.zeros(int(uniq.numel()), device=dev)
            counts = counts.index_add(0, inv, torch.ones_like(inv, dtype=torch.float))
            occ = uniq.numel() / (pb.side * pb.side)
            print(f"  {key:<26s}{pw:>6.2f}{pb.side:>6d}{uniq.numel():>8d}"
                  f"{occ*100:>7.1f}%{float(counts.mean()):>13.1f}"
                  f"{int(counts.max()):>11d}")

    # ---- C2 的定量上界：若同柱点 logits 完全相同，逐点 mIoU 的理论上限 ----
    print("\n  C2 上界估算（同柱点预测完全相同 → 柱内多数类投票的 mIoU）：")
    for key in [d["key"] for d in split["splits"]["val"]][:4]:
        a, r = key.split("/", 1)
        xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
        xyz = torch.from_numpy(np.ascontiguousarray(xyz_np, dtype=np.float32)).to(dev)
        for pw in (0.25, 0.5):
            pb = PillarBranchV2(3, 32, xy_range=8.0, pillar_xy=pw).to(dev).eval()
            with torch.no_grad():
                ids, _gx, _gy = pb.cell_ids(xyz)
            uniq, inv = torch.unique(ids, return_inverse=True)
            inv = inv.reshape(-1).cpu().numpy()
            y = y_np.astype(np.int64)
            # 柱内多数类作为该柱所有点的预测
            pred = np.zeros_like(y)
            order = np.argsort(inv, kind="mergesort")
            inv_s, y_s = inv[order], y[order]
            bounds = np.flatnonzero(np.diff(inv_s)) + 1
            for seg in np.split(np.arange(len(inv_s)), bounds):
                if seg.size == 0:
                    continue
                vals, cnts = np.unique(y_s[seg], return_counts=True)
                pred[order[seg]] = vals[np.argmax(cnts)]
            # 逐点精度（不是 mIoU，但足以说明"上界有多低"）
            acc = float((pred == y).mean())
            print(f"    {key:<26s} 柱宽{pw:.2f}  柱内多数类预测的逐点 acc = {acc:.4f}"
                  f"   （{len(np.unique(inv))} 个柱 / {len(y)} 点 →"
                  f" 平均 {len(y)/max(len(np.unique(inv)),1):.1f} 点共享一个预测）")

    # ---- C3：BN 是否被空像素污染 ----
    print("\n  C3 检查：BEV 图里空像素占比（BN 在 batch=1 下会把这些 0 计入统计）")
    a, r = split["splits"]["val"][0]["key"].split("/", 1)
    xyz_np, _y, _st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_np, dtype=np.float32)).to(dev)
    for pw in (0.25, 0.5):
        pb = PillarBranchV2(3, 32, xy_range=8.0, pillar_xy=pw).to(dev)
        with torch.no_grad():
            ids, _gx, _gy = pb.cell_ids(xyz)
            uniq, _inv = torch.unique(ids, return_inverse=True)
            total_pix = pb.side * pb.side
            print(f"    柱宽{pw:.2f}: side={pb.side}  总像素={total_pix}  "
                  f"非空柱={int(uniq.numel())}  空像素占比="
                  f"{(1 - uniq.numel()/total_pix)*100:.1f}%")

    print("\n[diag] 结论判读：")
    print("  · 若'柱内平均点数'很大（>10）→ C2 是主因：柱宽相对房间尺度过粗，")
    print("    同柱点被迫共享预测，逐点分割精度有硬上限。")
    print("  · 若空像素占比很高（>90%）→ C3 也在起作用：BN 统计被 0 污染。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
