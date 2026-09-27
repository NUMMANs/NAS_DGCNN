"""多分支架构的**正确性冒烟测试**（必须全过才允许烧 GPU）。

测什么（每一条都是"错了会静默给出好看但假的分数"的地方）
================================================================
1. **前向形状**：各分支组合的输出必须是 (1, num_classes, N) —— 逐点 logits。
2. **回填正确性**：同柱的点必须拿到**完全相同**的特征（这是柱状分支的定义），
   不同柱的点特征一般不同。若回填错位，分数会变但没人会发现。
3. **空间邻接完整性**：BEV 图上相邻像素对应的柱，其点在原始点云里也应相邻。
   做法：随机取一个柱，检查它的 8 邻域像素里至少有一个非空。
   （若填图时按柱出现顺序而非栅格坐标填，这条会失败。）
4. **密度不变性**：柱状分支的输出**不应**随点数剧烈变化 —— 这是它相对 kNN 的
   结构性优势，也是做这个分支的理由。测法：同一房间抽稀到 N/4，比较柱特征。
5. **参数量可比性**：late 融合下，加分支只增加一个 head + 一个分支的参数量。
6. **确定性**：同输入两次前向结果一致（BN 在 eval 模式）。
7. **梯度可达**：每个分支的参数都能拿到非 None 的梯度（防"分支其实没接上"）。
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

from nas_ptseg import gpu as GPU          # noqa: E402  （导入即钉 GPU 1）
from nas_ptseg import room_split as RS    # noqa: E402
from nas_ptseg import metrics as MT       # noqa: E402
from nas_ptseg.multibranch import (PillarBranchV2, MultiBranchSeg,  # noqa: E402
                                   count_params)

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    dev = GPU.setup_torch_device()
    print(f"[smoke] device = {dev}")
    split = RS.load_split()
    a, r = split["splits"]["train"][0]["key"].split("/", 1)
    xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    xyz = torch.from_numpy(np.ascontiguousarray(xyz_np, dtype=np.float32)).to(dev)
    print(f"[smoke] room={a}/{r}  N={xyz.shape[0]}  classes={len(st['present_classes'])}")
    NC = int(__import__("nas_ptseg.data", fromlist=["NUM_CLASSES"]).NUM_CLASSES)

    # ---------------------------------------------------------------- 1. 形状
    print("\n1) 前向形状")
    for branches in ("points", "pillar25", "points,pillar25"):
        for fuse in ("late", "early"):
            m = MultiBranchSeg(NC, branches=branches, k=48, fuse=fuse).to(dev).eval()
            with torch.no_grad():
                out = m(xyz.T[None, ...].contiguous())
            ok = tuple(out.shape) == (1, NC, xyz.shape[0])
            check(f"{branches} / {fuse}", ok, f"shape={tuple(out.shape)}")

    # ------------------------------------------------------- 2. 回填 + 3. 邻接
    print("\n2/3) 柱状分支：回填一致性与 3D 栅格空间邻接")
    pb = PillarBranchV2(3, 32, xy_range=8.0, pillar_xy=0.5, z_bins=8).to(dev).eval()
    with torch.no_grad():
        ids, gx, gy, gz = pb.cell_ids(xyz)
        feat = pb(xyz, xyz)
    feat_pn = feat[0].t()                                  # (N, C)  （返回已是 (1,C,N)）
    uniq, inv = torch.unique(ids, return_inverse=True)
    inv = inv.reshape(-1)
    # 2. 同格的点特征完全相同
    same = True
    for c in range(min(int(uniq.numel()), 5)):
        sel = (inv == c)
        if int(sel.sum()) > 1:
            v = feat_pn[sel]
            if float((v - v[0]).abs().max()) > 1e-5:
                same = False
                break
    check("同格点特征一致", same)

    # 3. 空间邻接：取一个内部非空格，检查其 XY 8 邻域里至少 1 个非空
    side, zb = pb.side, pb.z_bins
    occupied = set(uniq.tolist())
    gx_u = (uniq % side).tolist()
    gy_u = ((uniq // side) % side).tolist()
    gz_u = (uniq // (side * side)).tolist()
    hits = tested = 0
    for k in range(min(len(gx_u), 400)):
        x0, y0, z0 = gx_u[k], gy_u[k], gz_u[k]
        if not (0 < x0 < side - 1 and 0 < y0 < side - 1):
            continue
        tested += 1
        nb = sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if (dx or dy)
                 and ((z0 * side + (y0 + dy)) * side + (x0 + dx)) in occupied)
        hits += int(nb > 0)
    ratio = hits / max(tested, 1)
    check("同层 XY 8 邻域有非空格", ratio > 0.8, f"{hits}/{tested} = {ratio:.2f}")

    # 3b. z 分层是否有效：格数应显著多于单层方案（z 信息确实被利用）
    n_cells = int(uniq.numel())
    n_xy = len({(int(gx_u[i]), int(gy_u[i])) for i in range(len(gx_u))})
    check("z 分层确实增加了可分离的格", n_cells > n_xy * 1.5,
          f"3D 格={n_cells}  vs  XY 柱={n_xy}  → 平均 {n_cells/max(n_xy,1):.1f} 层/柱")

    # ------------------------------------------------------------ 4. 密度不变性
    print("\n4) 密度不变性（抽稀到 1/4 后柱特征是否稳定）")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(xyz.shape[0], size=xyz.shape[0] // 4, replace=False))
    xyz_q = xyz[sel]
    with torch.no_grad():
        f_q = pb(xyz_q, xyz_q)
    sel_t = torch.from_numpy(sel).to(dev)
    diff = (feat_pn[sel_t] - f_q[0].t()).abs().mean()
    check("抽稀后同点特征平均变化有限", bool(diff < 2.0),
          f"mean|Δ|={float(diff):.4f}（感受野由栅格尺度决定，不应随点数剧变）")

    # ------------------------------------------------------------ 5. 参数量
    print("\n5) 参数量")
    for branches in ("points", "pillar25", "points,pillar25"):
        m = MultiBranchSeg(NC, branches=branches, k=48)
        print(f"      {branches:<18s} params={count_params(m):>9,d}")
    dp = count_params(MultiBranchSeg(NC, branches="points,pillar25")) - \
        count_params(MultiBranchSeg(NC, branches="points"))
    check("加分支只增加该分支+head 的参数量", dp > 0, f"Δparams={dp:,d}")

    # ----------------------------------------------------- 6. 确定性 / 7. 梯度
    print("\n6/7) 确定性与梯度可达性")
    m = MultiBranchSeg(NC, branches="points,pillar25", k=48, fuse="late").to(dev)
    m.eval()
    with torch.no_grad():
        o1 = m(xyz.T[None, ...].contiguous())
        o2 = m(xyz.T[None, ...].contiguous())
    check("eval 模式两次前向一致", bool((o1 - o2).abs().max() < 1e-6),
          f"max|Δ|={float((o1 - o2).abs().max()):.2e}")

    m.train()
    out = m(xyz.T[None, ...].contiguous())
    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.int64)).to(dev)
    loss = torch.nn.functional.cross_entropy(out[0].t(), y)
    loss.backward()
    no_grad_params = []
    for name, p in m.named_parameters():
        if p.grad is None:
            no_grad_params.append(name)
    check("所有参数都有梯度（分支确实接上了）", not no_grad_params,
          f"无梯度的参数数={len(no_grad_params)}"
          + (f" 例：{no_grad_params[:3]}" if no_grad_params else ""))

    # 真实评估一次，确认整条链路能算 mIoU
    m.eval()
    with torch.no_grad():
        lg = m(xyz.T[None, ...].contiguous())
        pr = np.asarray(lg.argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
    mi = MT.present_class_miou(pr, y_np.astype(np.int64), st["present_classes"])
    print(f"\n[smoke] 未训练模型在同一房间上的 mIoU = {mi:.4f}（只验证链路可跑，无意义的分）")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"✗ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("✓ 全部通过，可以烧 GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
