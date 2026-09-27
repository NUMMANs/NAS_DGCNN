"""柱状分支最终诊断：它到底在学，还是仍然坏？

背景：GroupNorm 修好后（跨点 std 0.0018 → 0.326，训练时间恢复 33s），
但 mIoU 仍只有 0.04~0.05。这个分数**本身不能区分**两种情况：

    H1. 分支是好的，但 300 epoch 不足以学到东西（柱子太粗、信息太粗）
        → 对策：加宽特征 / 减柱宽 / 加 epoch / 改融合方式
    H2. 分支依然坏（仍有结构性阻断）
        → 对策：放弃柱状，换 voxel 或 range-view

判据（**不训练或极短训练即可**）
================================================================
A. **拟合能力下界**：用**同一个房间**训练 + 评估（train on room, eval on same points）。
   若连"过拟合单房间"都做不到 → 分支没有可用容量/梯度（H2）。
   若单房间能到高分却跨房间崩 → 是泛化问题，不是分支坏（偏 H1）。
B. **head 是否在学**：看训练 loss 是否显著下降。若 loss 卡住不动 → H2。
C. **输出分布**：若预测几乎全落在同一类 → H2（特征无区分度）。
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
from nas_ptseg import data as DT          # noqa: E402
from nas_ptseg import room_split as RS    # noqa: E402
from nas_ptseg import metrics as MT       # noqa: E402


def train_eval(branches: str, epochs: int, tr_key: str, ev_key: str,
               dev, log=print) -> dict:
    from nas_ptseg import tier2_dgcnn as T2
    from nas_ptseg.multibranch import MultiBranchSeg

    a, r = tr_key.split("/", 1)
    xtr, ytr, _ = RS.load_room_cached(a, r, target_n=RS.TARGET_N)
    b, r2 = ev_key.split("/", 1)
    xev, yev, stev = RS.load_room_cached(b, r2, target_n=RS.TARGET_N)

    torch.manual_seed(12345)
    np.random.seed(12345)
    model = MultiBranchSeg(int(DT.NUM_CLASSES), branches=branches, k=48,
                           fuse="late").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=T2.LR, weight_decay=T2.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=T2.LR_STEP, gamma=T2.LR_GAMMA)
    X = T2.to_torch(np.ascontiguousarray(xtr.T[None, ...], dtype=np.float32), dev)
    Y = T2.to_torch(np.ascontiguousarray(ytr, dtype=np.int64), dev)

    losses = []
    for ep in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        lg = model(X)
        loss = T2.F.cross_entropy(lg[0].t(), Y)
        loss.backward()
        opt.step()
        sched.step()
        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
            losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        # 同房间（过拟合能力）
        pr_tr = np.asarray(model(X).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
        mi_tr = MT.present_class_miou(pr_tr, ytr.astype(np.int64), _present(ytr))
        # 跨房间
        Xe = T2.to_torch(np.ascontiguousarray(xev.T[None, ...], dtype=np.float32), dev)
        pr_ev = np.asarray(model(Xe).argmax(dim=1)[0].cpu().tolist(), dtype=np.int64)
        mi_ev = MT.present_class_miou(pr_ev, yev.astype(np.int64), stev["present_classes"])
        n_pred_cls = int(len(np.unique(pr_tr)))
    return {"branches": branches, "epochs": epochs,
            "loss_first": losses[0] if losses else float("nan"),
            "loss_last": losses[-1] if losses else float("nan"),
            "miou_same": mi_tr, "miou_cross": mi_ev,
            "n_pred_classes": n_pred_cls}


def _present(y):
    return sorted(int(c) for c in np.unique(y))


def main() -> int:
    dev = GPU.setup_torch_device()
    split = RS.load_split()
    tr = split["splits"]["train"][0]["key"]
    ev = split["splits"]["val"][0]["key"]
    print("=" * 88)
    print("柱状分支最终诊断：在学，还是仍然坏？")
    print("=" * 88)
    print(f"  训练房 {tr}   评估房 {ev}   N={RS.TARGET_N}\n")
    print(f"  {'配置':<18s}{'ep':>5s}{'loss 首':>9s}{'loss 末':>9s}"
          f"{'同房 mIoU':>11s}{'跨房 mIoU':>11s}{'预测类数':>9s}")

    rows = []
    for branches, eps in (("points", 300),
                          ("pillar25", 300),
                          ("voxel25", 300),
                          ("points,voxel25", 300)):
        r = train_eval(branches, eps, tr, ev, dev)
        rows.append(r)
        print(f"  {r['branches']:<18s}{r['epochs']:>5d}{r['loss_first']:>9.4f}"
              f"{r['loss_last']:>9.4f}{r['miou_same']:>11.4f}"
              f"{r['miou_cross']:>11.4f}{r['n_pred_classes']:>9d}")

    print("\n  判读（**用同房 mIoU 判分支有没有容量**，不要只看跨房）：")
    for r in rows:
        if r["branches"] == "points":
            continue
        tag = f"{r['branches']}@{r['epochs']}ep"
        if r["miou_same"] > 0.5:
            print(f"    {tag}: ✅ 能过拟合单房间（{r['miou_same']:.3f}）→ "
                  f"分支有容量，问题在泛化/融合")
        elif r["miou_same"] > 0.25:
            print(f"    {tag}: ⚠ 同房 {r['miou_same']:.3f}，弱但非死 → 容量或步数不足")
        else:
            print(f"    {tag}: ❌ 连单房间都过拟合不了（{r['miou_same']:.3f}）→ "
                  f"分支结构性问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
