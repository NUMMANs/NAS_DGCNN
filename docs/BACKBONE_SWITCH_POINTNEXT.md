# 骨干选型变更：PointViG → **PointNeXt**（2026-09-26）

## 一、为什么换掉 PointViG

**PointViG 无官方代码**（CatalyzeX 标 "Request Code"；探测 3 个仓库名全 404）
→ 我只能自建 → **无法复现到论文精度（自建 0.36 vs 论文 71.7%）**
→ **所有结论都可能被审稿人一句"你实现错了"推翻。**

这是不可接受的：**复现未验证就堆结论 = 在沙地上盖楼。**

## 二、选定 **PointNeXt**（NeurIPS 2022，`guochengqian/PointNeXt`）

| 维度 | 情况 |
|---|---|
| **开源** | ✅ 主仓库 + 子模块 `openpoints`（**均已克隆**） |
| **数据集** | ✅ **原生 S3DIS 配置**（`cfgs/s3dis/pointnext-s.yaml`） |
| **精度** | ✅ **S3DIS 6-fold 74.9 mIoU**（SOTA 级，不是我自建的弱基线）|
| **硬件** | ✅ 官方："for S3DIS we only use **1 A100**" → 单卡可跑 |

### ⭐ 决定性发现：`radius` 是上游**一等参数**，且支持逐层列表

`openpoints/models/backbone/pointnext.py`：
```python
radius: float or List[float] = 0.1                 # L344：支持标量**或列表**
radius_scaling = kwargs.get('radius_scaling', 2)   # L366
self.radii = self._to_full_list(radius, radius_scaling)   # L369
```

**S3DIS 基线实际用的半径**（我按 `_to_full_list` 手算，blocks=[1]*5, strides=[1,4,4,4,4]）：
```
stage 0: [0.1]   stage 1: [0.1]   stage 2: [0.2]   stage 3: [0.4]   stage 4: [0.8]
```

**关键观察**：
1. 基线把 5 层的感受野压成**一条固定等比数列（公比 ×2）**
   —— **人类把一个"每层该多粗"的自由度，缩减成了一个单一公比**
2. 因为 `blocks` 全是 1，`radius_scaling` **实际上几乎不起作用**
   → **上游代码支持逐层独立半径，但配置里从未使用**

### 这给了比 PointViG 更好的命题

> **PointNeXt 的 S3DIS 配置把 5 层感受野压缩成固定公比（2.0）的等比数列，
> 而上游代码本就支持"逐层独立指定半径"——这个自由度从未被使用，也从未被消融。
> NAS 回答：逐层独立半径是否优于固定公比？**

**优势**：① **不需要复现**（开箱即用）→ 结论不会被"实现错误"推翻；
② 基线是 **SOTA 级（74.9）**→ 赢了有说服力；③ 搜的是**上游已支持的参数** → 可复现性极强。

---

## 三、⚠ 当前阻塞：CUDA 算子编译失败（需要你决策）

PointNeXt 依赖 3 个自编译 CUDA 扩展（`pointnet2_batch` / `pointops` / `subsampling`），
其中 `ball_query`、`gather_points`、`group_points`、`fps` **是训练必需**（无纯 PyTorch 回退）。

**实测环境与失败原因**：

| 项 | 状态 |
|---|---|
| CUDA toolkit | ✅ **12.1**（与 `lidar310` 的 torch 2.5.1+cu121 **完全匹配**） |
| Python / torch | ✅ 3.10 / 2.5.1+cu121，`cpp_extension` 可用 |
| **Windows 10 SDK** | ❌ **缺失**（`Windows Kits/10/Include` 不存在，只有 SDK 8.1） |
| MSVC | ⚠ 只有 **14.43**（对 CUDA 12.1 **过新**，官方要求 < 14.3x） |
| ninja | ❌ 未装（回退到慢速 distutils） |
| 报错 | `fatal error C1083: 无法打开包括文件: "stddef.h"` ← **SDK 缺失的直接症状** |

**根因**：`stddef.h` 属于 Windows 10 SDK。机器上只装了 SDK 8.1，
MSVC 找不到头文件 → 编译必然失败。

### 可行的四条路（各有代价）

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 装 Windows 10 SDK + 降 MSVC** | 装 SDK 10.0.20348 + VS 2019 BuildTools（v14.29，与 CUDA 12.1 匹配） | 下载数 GB，需你同意装系统组件 |
| **B. 用 WSL2 编译** | 在 WSL 里编译 `.so`（Linux 下 g++ 与 CUDA 12.1 兼容好得多） | 需 WSL + 再装一份 torch；产物跨平台可用性需验证 |
| **C. 换用**纯 PyTorch 实现的分支 | 只用不依赖自编译算子的骨干（如纯 `torch_grouping_operation` 路径） | 需改代码；可能失去部分配置 |
| **D. 换一个"零编译"的骨架** | 例如只用 `torch_scatter`/`spconv` 这种**有预编译 wheel** 的库 | 又要重新选型，但能立刻跑 |

### 我的建议：**A 或 D**

- **若你能装系统组件** → **A**（最直接，装完立刻可跑；我会给出精确的版本组合）
- **若不想动系统环境** → **D**：换一个依赖**纯预编译 wheel** 的开源骨架。
  例如：
  - `Pointcept` 系（PTv3）依赖 `spconv`（**有 cu121 预编译 wheel**）+ `torch_scatter`（同）→ **但仍需 `flash_attn`**（也可装预编译）
  - 或直接用本项目**已验证可跑**的 DGCNN 线（但它是弱基线）

⚠ **重要权衡**：PTv3 我们已实测 **6344 ms/step**（DGCNN 的 61×）→ 作为 NAS 搜索基底**成本不可接受**。
所以"零编译"路线里，能同时满足「开源 + 强基线 + 单卡可训 + 能跑 NAS」的选项并不多。

**在开工前我想先确认**：
1. 你能接受安装 **Windows 10 SDK + VS2019 BuildTools**（方案 A）吗？
2. 还是倾向我直接在 **WSL2** 里搭（方案 B）？
3. 或者你有装有完整 VS 工具链的机器/环境可用？

**这个问题不解决，PointNeXt 就跑不起来** —— 我不想在不告知的情况下反复试编译。
