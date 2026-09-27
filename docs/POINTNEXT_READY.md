# PointNeXt 就绪报告

> 2026-09-26 · 替换 PointViG 后的骨干落地结论
> 结论：**两道门禁全过，可以开始 NAS 搜索。**

---

## 一、为什么换掉 PointViG

PointViG **无官方代码**（CatalyzeX 标记 "Request Code"，探测 3 个仓库名全 404），只能自建。
自建版在 4+2 房小协议下绝对精度仅 **0.36**，远低于论文 **71.7%** ——
**无法证明复现正确**，那么"论文选的半径 r=0.2 不是最优"这类结论，
审稿人一句"你实现错了"就能推翻。

**复现未验证就堆结论 = 沙地上盖楼。** 换骨干是对的。

---

## 二、选定 PointNeXt 的理由

| 维度 | 情况 |
|---|---|
| 开源 | ✅ `guochengqian/PointNeXt`（主仓库 + `openpoints` 子模块均已克隆） |
| 数据集 | ✅ **原生 S3DIS 配置** |
| 精度 | ✅ 官方 S3DIS 6-fold **74.9 mIoU**（SOTA 级） |
| 硬件 | ✅ 官方："for S3DIS we only use 1 A100" → 单卡可跑 |

**最关键**：它暴露了一个**上游原生支持、但从未被使用**的搜索轴（见第四节）。

---

## 三、编译阻塞已解除（根因是三个坑叠加）

`pointnet2_batch` 是训练必需的自编译 CUDA 扩展（`ball_query`/`fps`/`gather_points` 等，
**无纯 PyTorch 回退**）。之前失败告终，现在**编译并安装成功**。

### 坑 1（最隐蔽，且是环境级问题）：环境变量 `CL` 被污染

```
CL = C:/Users/my/.workbuddy/binaries/PortableGit/versions/1.2.0/EHsc /W3 /D_DEBUG
```

`CL` 是 **MSVC 自动读取的"附加编译参数"变量**。这个值是 WorkBuddy 的 bash shim
把 `/EHsc` 做 MSYS 路径转换后的垃圾路径 —— 于是 `cl` 每次都去编译一个不存在的文件 `EHsc`，
**退出码 2**。而 nvcc 受影响时**静默退出、不打印任何错误**，极难定位。

> **修复：任何 MSVC / nvcc 编译前必须 `set CL=`（并清 `LINK` / `_CL_`）。**
> 附带：`CL` 里还含 `/D_DEBUG`，会污染 obj → 链接报 `LNK2038`
> （`_ITERATOR_DEBUG_LEVEL` 0 vs 2、`RuntimeLibrary` MD vs MDd）→ **必须先删 build 目录再重建**。

### 坑 2：Windows SDK 其实装了，只是没接进构建环境

**我此前判断"SDK 缺失"是错的** —— SDK 完整存在于 **`D:\Windows Kits\10\`**
（注册表 `KitsRoot10`），含 10.0.10240.0 / 10.0.18362.0 / **10.0.22621.0**，
`ucrt\stddef.h`、`corecrt.h` 齐全。我当时只搜了默认的 C 盘路径就下了结论，导致白跑一次安装。

> 真正的问题是：直接调 `setup.py` 时 torch 的 `_get_vc_env()` 没把 D 盘 SDK 注入 `INCLUDE`
> → `fatal error C1083: stddef.h`。
> **修复：在 `cmd` 里先 `call vcvars64.bat`（它按注册表补齐），再设 `DISTUTILS_USE_SDK=1`。**

### 坑 3：MSVC 14.43 + CUDA 12.1 需要官方旁路宏

```
yvals_core.h(921): error: static assertion failed with
  "error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer."
```

`yvals_core.h` L918 是 `#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` ——
**官方旁路宏就是它**。

> ⚠ 我一开始只读了 nvcc 自己的 `crt\host_config.h`（判据 `_MSC_VER ∈ [1910,2030)`，
> 14.43→1943 **通过**），就以为没问题 —— **漏了 STL 这一层更严的限制**。
> 教训：**编译器版本检查可能有多层，别只看第一层。**

---

## 四、⭐ 搜索轴：`radius` 是干净的 5 元组，且上游原生支持

`PointNextEncoder._to_full_list`（`openpoints/models/backbone/pointnext.py` L391）：

```python
if isinstance(param, List):
    for i, value in enumerate(param):
        param_list.append([value])        # list → 每个 stage 一个独立值
else:
    # 标量 → 按固定公比自动铺开
    for i, stride in enumerate(self.strides):
        if stride == 1: param_list.append([param] * self.blocks[i])
        else:
            param_list.append([param] + [param * param_scaling] * (self.blocks[i]-1))
            param *= param_scaling
```

S3DIS 基线用的是**标量** `radius: 0.1` → 实测 `radii = [[0.1],[0.1],[0.2],[0.4],[0.8]]`，
即**固定公比 ×2 的等比数列**。因为 `blocks` 全是 1，`radius_scaling` 实际几乎不起作用。

> ### 命题
> **PointNeXt 把 5 层感受野压缩成一个固定公比，而代码本就支持逐层独立半径 ——
> 这个自由度从未被使用、从未被消融。NAS 回答：逐层独立半径是否优于固定公比？**

**这个轴比 PointViG 那条好**：① 不用自建 → 结论不会被"实现错"推翻；
② 基线是 SOTA 级（74.9）；③ 搜的是**上游已支持的参数** → 可复现性极强。

---

## 五、验证结果：两道门禁全过

### 门禁 ① 算子编译与实跑
```
IMPORT OK -> site-packages\pointnet2_batch_cuda.cp310-win_amd64.pyd
exports: ball_query_wrapper, furthest_point_sampling_wrapper, gather_points_wrapper,
         group_points_wrapper, three_interpolate_wrapper, three_nn_wrapper,
         gather_points_grad_wrapper, group_points_grad_wrapper, three_interpolate_grad_wrapper
```

### 门禁 ② 单房间过拟合（判据：同房 mIoU > 0.8 = 有容量）
```
房间 Area_1/pantry_1  点数=16384  出现类别=6
baseline radii=[0.1, 0.1, 0.2, 0.4, 0.8]  → 编码器实测 [[0.1],[0.1],[0.2],[0.4],[0.8]]
  step 100  loss=0.4174  同房 mIoU=0.9694
  step 200  loss=0.1846  同房 mIoU=0.9852
  step 300  loss=0.1146  同房 mIoU=0.9903
  -> 同房 mIoU=0.9903   41.0 ms/step   峰值显存 214 MiB
  -> PASS
```

### 成本对比（NAS 可行性）
| 骨干 | ms/step | 峰值显存 | 参数 |
|---|---|---|---|
| **PointNeXt-s** | **41 ~ 55** | **214 ~ 411 MiB** | 0.792 M |
| DGCNN（原基线） | 104 | 3524 MiB | 0.35 M |
| PointViG（自建） | 123 | — | 3.50 M |

**PointNeXt 是三者中最便宜、显存最低的** → 搜索预算充足。

---

## 六、可复用的编译配方

见 `_sdk/build_pn2.bat`：

```bat
call "<VS>\VC\Auxiliary\Build\vcvars64.bat"     :: 补齐 MSVC + D 盘 SDK
set CL=                                          :: ★ 清掉被污染的 CL
set LINK=
set DISTUTILS_USE_SDK=1                          :: torch 用现有 env，不再自己推导
set TORCH_CUDA_ARCH_LIST=8.9                     :: RTX 4090
set NVCC_PREPEND_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
rmdir /s /q build                                :: ★ 防 debug obj 污染
python setup.py build_ext --inplace
python setup.py install
```

**必须在 `cmd` 里执行**（git-bash 会做 MSYS 路径转换，正是坑 1 的来源）。

---

## 七、下一步

1. **把 5 元组半径做成搜索表**（基线与候选），复用项目已有的三段闭环
   （提出 → 实测 → 写回；每 cell 落盘 + 续跑 + GPU 守卫）
2. 先做**单尺度半径扫描**：5 层全部用同一个 r，找最优单半径
   （这是"先把对照组自身搜到最优"的方法论要求 —— 项目已踩过 3 次同源坑）
3. 再做**逐层独立半径** vs **固定公比**的对照
4. 上全量协议（12+12 房）复核

---

## 八、环境/依赖变更记录

- **新增依赖**（装入 conda 环境 `lidar310`）：
  `multimethod easydict addict termcolor einops ninja timm shortuuid h5py gdown protobuf tensorboard`
- **无需编译**：`subsampling`（S3DIS 路径里 `grid_subsampling` 是注释掉的）、
  `pointops`（仅 PointTransformer 骨干）、`chamfer_dist`/`emd`（仅重建任务）
- **使用要点**：`cfg` 必须 `EasyConfig() + .load(path, recursive=True)`，
  否则报 `AttributeError: 'dict' object has no attribute 'radius'`
