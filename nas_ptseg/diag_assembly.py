"""定位 MultiBranchSeg 装配里把分支弄坏的环节。

已知（`diag_branch_ablate.py`，同房 300 ep）：
    [参考] DGCNNSeg 原版                同房 mIoU 1.0000
    C) point_mlp 直接出（不经栅格）     0.9700
    B) 聚合后直接回填（不做卷积）        0.8436
    A) 完整骨架（聚合+卷积+回填）        0.8467   ← **骨架本身完全可用！**

但 `MultiBranchSeg(branches="voxel25")` 的同房 mIoU 只有 0.0412。
同一套骨架，经过 MultiBranchSeg 装配就死 → **断点在装配层**，不在表示。

本脚本二分定位。嫌疑清单：
  S1. 输入特征给错：MultiBranchSeg 传的是 `xyz[0].t()`（(N,3)），AblationBranch 用 `xyz`（(N,3)）——
      看起来一致，但 MultiBranchSeg 的 `feat` 参数或 dtype/连续性可能不同。
  S2. **head 的 BN**：MultiBranchSeg 的 head 用 BatchNorm1d，而卷积段用 GroupNorm。
      分支输出跨点 std 只有 0.4 量级，若 head 的 BN 又把它压一次……
  S3. **late 融合的加法**：多分支时相加；单分支时其实只有一个 head，不该有问题。
  S4. 学习率/调度差异：MultiBranchSeg 经由 multibranch_exp 用 T2.LR 等同参数，应一致。
  S5. 参数未被优化：`br_out` 算错 → head 维度对但分支参数不在 optimizer 里？

做法：**把 AblationBranch 塞进 MultiBranchSeg 的装配路径**，以及反过来
用 diag_branch_ablate 的极简装配跑 VoxelBranch，看在哪一侧复现。
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nas_ptseg import gpu as GPU          # noqa: E402
from nas_ptseg import data as DT          # noqa: E402
from nas_ptseg import room_split as RS    # noqa: E402
from nas_ptseg import metrics as MT       # noqa: E402


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def run(model: nn.Module, xyz_np, y_np, present, epochs, dev, forward_feat: bool,
        use_mb_head: bool, tag: str, log=print, needs_xyz_pair: bool = False) -> dict:
    """`forward_feat=True` → 模型接受 (N,3)；False → 接受 (1,3,N)。
    `needs_xyz_pair=True` → 模型签名是 forward(xyz, feat)，两参都要传（VoxelBranch）。"""
    from nas_ptseg import tier2_dgcnn as T2
    NC = int(DT.NUM_CLASSES)
    model = model.to(dev)                      # ⚠ 必须显式搬设备（初版漏了 → device mismatch）
    head = None
    if use_mb_head:
        from nas_ptseg.multibranch import MultiBranchSeg
        # 用 MultiBranchSeg 的 _make_head，逐位复现其装配
        head = MultiBranchSeg._make_head(64, NC).to(dev)

    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=T2.LR, weight_decay=T2.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP, gamma=T2.LR_GAMMA)
    X3 = T2.to_torch(np.ascontiguousarray(xyz_np.T[None, ...], dtype=np.float32), dev)
    Xf = T2.to_torch(np.ascontiguousarray(xyz_np, dtype=np.float32), dev)
    Y = T2.to_torch(np.ascontiguousarray(y_np, dtype=np.int64), dev)

    def fwd():
        if needs_xyz_pair:
            return model(Xf, Xf)          # VoxelBranch(xyz, feat)
        return model(Xf) if forward_feat else model(X3)

    l0 = l1 = float("nan")
    for ep in range(int(epochs)):
        model.train()
        if head:
            head.train()
        opt.zero_grad(set_to_none=True)
        f = fwd()
        lg = head(f) if head else f
        loss = T2.F.cross_entropy(lg[0].t(), Y)
        loss.backward()
        opt.step(); sched.step()
        if ep == 0:
            l0 = float(loss.item())
        l1 = float(loss.item())
    model.eval()
    if head:
        head.eval()
    with torch.no_grad():
        f = fwd()
        pr = np.asarray((head(f) if head else f).argmax(dim=1)[0].cpu().tolist(),
                        dtype=np.int64)
    no_grad = [n for n, p in (list(model.named_parameters())
                              + (list(head.named_parameters()) if head else []))
               if p.grad is None]
    mi = MT.present_class_miou(pr, y_np.astype(np.int64), present)
    log(f"  {tag:<46s}{l0:>9.4f}{l1:>9.4f}{mi:>11.4f}"
        f"{int(len(np.unique(pr))):>9d}{len(no_grad):>9d}")
    return {"miou": mi, "loss_first": l0, "loss_last": l1,
            "n_no_grad": len(no_grad), "no_grad": no_grad[:5]}


def main() -> int:
    from nas_ptseg.multibranch import VoxelBranch, MultiBranchSeg
    from nas_ptseg.diag_branch_ablate import AblationBranch
    dev = GPU.setup_torch_device()
    split = RS.load_split()
    tr = split["splits"]["train"][0]["key"]
    a, r = tr.split("/", 1)
    xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    present = st["present_classes"]

    print("=" * 108)
    print("定位 MultiBranchSeg 装配层：同一套骨架，两种装配，看在哪一侧复现崩")
    print("=" * 108)
    print(f"  房间 {tr}   N={xyz_np.shape[0]}   300 ep   同房训练+评估\n")
    print(f"  {'变体':<46s}{'loss 首':>9s}{'loss 末':>9s}{'同房 mIoU':>11s}"
          f"{'类数':>6s}{'无梯度参数':>10s}")

    torch.manual_seed(12345); np.random.seed(12345)
    run(AblationBranch(3, 64, mode="full"), xyz_np, y_np, present, 300, dev,
        forward_feat=True, use_mb_head=False,
        tag="[对照] AblationBranch full + 极简 head")
    gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(12345); np.random.seed(12345)
    run(VoxelBranch(3, 64, voxel_size=0.25, layers=2, z_bins=16),
        xyz_np, y_np, present, 300, dev,
        forward_feat=True, use_mb_head=False, needs_xyz_pair=True,
        tag="A) VoxelBranch + 极简 head（新装配）")

    torch.manual_seed(12345); np.random.seed(12345)
    mbs = MultiBranchSeg(int(DT.NUM_CLASSES), branches="voxel25")
    run(mbs, xyz_np, y_np, present, 300, dev,
        forward_feat=False, use_mb_head=False,
        tag="B) MultiBranchSeg(voxel25) 自带头")

    torch.manual_seed(12345); np.random.seed(12345)
    run(AblationBranch(3, 64, mode="full"), xyz_np, y_np, present, 300, dev,
        forward_feat=True, use_mb_head=True,
        tag="C) AblationBranch full + MultiBranchSeg 的 head")

    torch.manual_seed(12345); np.random.seed(12345)
    run(VoxelBranch(3, 64, voxel_size=0.25, layers=2, z_bins=16),
        xyz_np, y_np, present, 300, dev,
        forward_feat=True, use_mb_head=True, needs_xyz_pair=True,
        tag="D) VoxelBranch + MultiBranchSeg 的 head")

    print("\n  判读：")
    print("    · A 高而 B 低  → 问题在 MultiBranchSeg 的装配（head / 输入 / 融合）")
    print("    · C 低而 A 高  → **BatchNorm head 是断点**（分组归一化 vs BN 的冲突）")
    print("    · 全高         → 之前的崩是别的配置差异（如 exp 里的 lr/参数）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
