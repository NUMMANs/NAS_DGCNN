"""最小可判定验证：栅格分支骨架本身有没有问题？

背景
================================================================
`pillar25` 与 `voxel25` 都**连单房间都过拟合不了**（同房 mIoU ≈ 0.04），
loss 几乎不降（1.94 → 1.68），且预测只用了 1 个类。而 `points` 分支同房能到 0.9997。
两者在骨架上的**共同点**是"点聚合到格子 → 格子上卷积 → 回填到点"，
差异只在 z 是通道还是空间轴。既然两者一起死，嫌疑指向骨架而非具体表示。

本脚本把该骨架拆成三段，逐段替换为"恒等直达"，定位是哪一段把梯度切断的：

    A) 完整分支（基线，应当崩）
    B) 跳过卷积：聚合 → **直接回填**（不做任何卷积）→ 看能否过拟合
    C) 跳过聚合与回填：point_mlp → **直接输出**（完全不经栅格）
        —— 这一段若都不能过拟合，说明问题不在分支而在 MultiBranchSeg 的
           head/融合装配（两条路都得重做）。

判据：同房 mIoU（train on room, eval on same room）
    > 0.8  → 这一段没问题，梯度能通
    < 0.25 → 这一段就是断点
"""

from __future__ import annotations

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


class AblationBranch(nn.Module):
    """栅格分支的消融版：`mode` 控制在哪一步截断。

    mode="full"  : point_mlp → 聚合 → densify+conv → 回填   （等价于 VoxelBranch）
    mode="noconv": point_mlp → 聚合 → 回填，**不做卷积**
    mode="nogrid": point_mlp → 直接返回，**完全不经栅格**
    """

    def __init__(self, in_c: int, out_c: int, mode: str = "full",
                 voxel_size: float = 0.25, xy_range: float = 8.0,
                 z_bins: int = 16, layers: int = 2):
        super().__init__()
        self.mode = mode
        self.xy_range = float(xy_range)
        self.voxel_size = float(voxel_size)
        self.z_bins = int(z_bins)
        self.side = int(np.ceil(2.0 * xy_range / voxel_size))
        self.point_mlp = nn.Sequential(
            nn.Linear(in_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_c, out_c, bias=False), nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        n_grp = 8 if int(out_c) % 8 == 0 else 1
        self.convs = nn.ModuleList([
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False) for _ in range(int(layers))])
        self.norms = nn.ModuleList([
            nn.GroupNorm(n_grp, int(out_c)) for _ in range(int(layers))])

    def voxel_ids(self, xyz):
        c = xyz.mean(dim=0, keepdim=True)
        p = xyz - c
        g = torch.floor((p[:, :2] + self.xy_range) / self.voxel_size).long()
        g = g.clamp(0, self.side - 1)
        zmin, zmax = p[:, 2].min(), p[:, 2].max()
        span = (zmax - zmin).clamp(min=1e-6)
        gz = (((p[:, 2] - zmin) / span) * self.z_bins).long().clamp(0, self.z_bins - 1)
        return gz, g[:, 1], g[:, 0]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        h = self.point_mlp(xyz)                                   # (N, C)
        if self.mode == "nogrid":
            return h.t().unsqueeze(0)

        C = h.shape[1]
        gz, gy, gx = self.voxel_ids(xyz)
        S, Z = self.side, self.z_bins
        uniq, inverse = torch.unique((gz * S + gy) * S + gx, return_inverse=True)
        inverse = inverse.reshape(-1)
        P = uniq.numel()

        v = torch.zeros((P, C), dtype=h.dtype, device=h.device)
        v = v.index_reduce(0, inverse, h, "amax", include_self=True)
        if self.mode == "noconv":
            return v[inverse].t().unsqueeze(0)

        uz = uniq // (S * S)
        uy = (uniq // S) % S
        ux = uniq % S
        lin = (uz * S + uy) * S + ux
        for li in range(len(self.convs)):
            dense = v.new_zeros((1, C, Z, S, S))
            dense.view(1, C, -1)[0, :, lin] = v.t()
            dense = self.convs[li](dense)
            v = dense.view(1, C, -1)[0, :, lin].t()
            v = self.norms[li](v)
            v = torch.nn.functional.leaky_relu(v, 0.2)
        return v[inverse].t().unsqueeze(0)


def train_and_score(branch: nn.Module, feat_mode: bool, xyz_np, y_np, present,
                    epochs: int, dev, log=print) -> dict:
    from nas_ptseg import tier2_dgcnn as T2
    NC = int(DT.NUM_CLASSES)
    torch.manual_seed(12345)
    np.random.seed(12345)
    model = nn.Sequential().to(dev)          # placeholder, replaced below
    # 组装：分支 + head
    head_in = 64
    head = nn.Sequential(
        nn.Conv1d(head_in, 128, 1, bias=False), nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(128, NC, 1),
    ).to(dev)
    br = branch.to(dev)
    params = list(br.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=T2.LR, weight_decay=T2.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP, gamma=T2.LR_GAMMA)

    X3 = T2.to_torch(np.ascontiguousarray(xyz_np.T[None, ...], dtype=np.float32), dev)
    Xf = T2.to_torch(np.ascontiguousarray(xyz_np, dtype=np.float32), dev)
    Y = T2.to_torch(np.ascontiguousarray(y_np, dtype=np.int64), dev)
    del model

    losses = []
    for ep in range(int(epochs)):
        br.train(); head.train()
        opt.zero_grad(set_to_none=True)
        f = br(Xf) if feat_mode else br(X3)
        lg = head(f)
        loss = T2.F.cross_entropy(lg[0].t(), Y)
        loss.backward()
        opt.step()
        sched.step()
        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
            losses.append(float(loss.item()))

    br.eval(); head.eval()
    with torch.no_grad():
        f = br(Xf) if feat_mode else br(X3)
        pr = np.asarray(head(f).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "miou": MT.present_class_miou(pr, y_np.astype(np.int64), present),
            "n_cls": int(len(np.unique(pr)))}


def main() -> int:
    from nas_ptseg import tier2_dgcnn as T2
    dev = GPU.setup_torch_device()
    split = RS.load_split()
    tr = split["splits"]["train"][0]["key"]
    a, r = tr.split("/", 1)
    xyz_np, y_np, st = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    present = st["present_classes"]
    print("=" * 88)
    print("最小可判定验证：栅格分支骨架在哪一步断了？")
    print("=" * 88)
    print(f"  房间 {tr}   N={xyz_np.shape[0]}   300 ep   同房训练+评估\n")
    print(f"  {'变体':<32s}{'loss 首':>9s}{'loss 末':>9s}{'同房 mIoU':>11s}{'预测类数':>9s}")

    # 参考：纯 DGCNN（它自己）
    torch.manual_seed(12345); np.random.seed(12345)
    dm = T2.DGCNNSeg(int(DT.NUM_CLASSES), 48).to(dev)
    opt = torch.optim.Adam(dm.parameters(), lr=T2.LR, weight_decay=T2.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP, gamma=T2.LR_GAMMA)
    X3 = T2.to_torch(np.ascontiguousarray(xyz_np.T[None, ...], dtype=np.float32), dev)
    Y = T2.to_torch(np.ascontiguousarray(y_np, dtype=np.int64), dev)
    l0 = l1 = float("nan")
    for ep in range(300):
        dm.train(); opt.zero_grad(set_to_none=True)
        lg = dm(X3)
        loss = T2.F.cross_entropy(lg[0].t(), Y)
        loss.backward(); opt.step(); sch.step()
        if ep == 0:
            l0 = float(loss.item())
        l1 = float(loss.item())
    dm.eval()
    with torch.no_grad():
        pr = np.asarray(dm(X3).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
    mi = MT.present_class_miou(pr, y_np.astype(np.int64), present)
    print(f"  {'[参考] DGCNNSeg 原版':<32s}{l0:>9.4f}{l1:>9.4f}{mi:>11.4f}"
          f"{int(len(np.unique(pr))):>9d}")

    results = {}
    for mode, feat_mode, label in (
        ("nogrid", True,  "C) point_mlp 直接出（不经栅格）"),
        ("noconv", True,  "B) 聚合后直接回填（不做卷积）"),
        ("full",   True,  "A) 完整骨架（聚合+卷积+回填）"),
    ):
        r_ = train_and_score(AblationBranch(3, 64, mode=mode), feat_mode,
                             xyz_np, y_np, present, 300, dev)
        results[mode] = r_
        print(f"  {label:<32s}{r_['loss_first']:>9.4f}{r_['loss_last']:>9.4f}"
              f"{r_['miou']:>11.4f}{r_['n_cls']:>9d}")

    print("\n  判读：")
    if results["nogrid"]["miou"] > 0.8:
        print("    · C 通过 → point_mlp + head 装配没问题")
    else:
        print("    · ❌ C 失败 → 问题在 head/装配，不在栅格分支本身")
    if results["noconv"]["miou"] > 0.8:
        print("    · B 通过 → 聚合/回填没问题，**断点在卷积段**")
    else:
        print("    · ❌ B 失败 → 聚合或回填本身就把特征压死了")
    if results["full"]["miou"] > 0.8:
        print("    · A 通过 → 骨架完全可用（那之前的崩是别的配置差异）")
    else:
        print("    · A 失败 → 与历史一致，骨架确实不可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
