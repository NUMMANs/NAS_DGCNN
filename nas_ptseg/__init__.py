"""nas_ptseg —— NAS × 点云分割的实验代码包。

模块职责
========
    data       S3DIS 单房间读取 + 受控密度域构造（同几何、仅密度不同）
    density    局部密度估计（points/m^3）与密度画像
    proxy      DGCNN(EdgeConv) 密度自适应代理评估器（训练无关）
    eval_core  多域编排：鲁棒分数、代价归一化、alpha 指数
    prep       数据准备、四组参照标定、Pareto 扫描

技术路线
========
    用训练无关代理（随机权重 EdgeConv + 闭式 ridge 探针）替代真实训练，
    让 OpenEvolve 能在可接受的时间内搜索"局部密度 → 每层 k/半径/采样率"
    的参数化策略，优化目标为多域鲁棒 mIoU（最差域 + 平均域）。

基线演进
========
    PointNet++ (TF, 已弃用) → DGCNN (机制验证) → PolarNet (后续)
"""

# ⚠ 必须放在最前面：这个 import 有副作用——把 CUDA_VISIBLE_DEVICES 钉到 GPU 1。
# 用户硬约束：本机所有 GPU 操作只能在 GPU 1 上，且不许挤占其他程序资源。
# 它必须在任何 `import torch` 之前完成（torch 在首次 CUDA 调用时读取该变量），
# 而 `nas_ptseg/__init__.py` 先于包内任何子模块执行，所以放这里是可靠的。
from . import gpu  # noqa: F401  （导入即生效，勿删）

from . import data
from . import density
from . import proxy
from . import eval_core

__all__ = ["gpu", "data", "density", "proxy", "eval_core"]
