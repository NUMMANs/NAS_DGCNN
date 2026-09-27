"""S3DIS 单房间数据层：读取 + 受控密度域构造。

设计要点（为什么这样做）
================================================================

1. 为什么改用 S3DIS 单房间
   - 用户约束：验证数据不使用 KITTI。
   - 单房间体量小（相对整个数据集），适合快速迭代机制验证。
   - S3DIS 是室内扫描，点密度比室外 LiDAR 高 2~3 个数量级，密度自适应的
     效应在室内更容易被观察到。

2. 密度域怎么构造 —— 这是整个实验的地基，必须说清楚
   受控实验的目标是：**几何完全相同，只有点密度不同**。这样任何性能差异
   都只能归因于密度，而不是场景内容。

   做法：对同一个房间、同一个空间范围，用不同体素尺寸下采样。
      - 体素越大 → 保留点数越少 → 局部密度越低
      - 几何支撑、语义标签、空间范围都不变

   这与之前 KITTI 方案的一个关键教训一致：密度 = N / V。在固定 N 和固定 V
   的框架下密度是常数，"隔线抽取"之类的操作根本改变不了密度。所以必须让
   N 随体素尺寸自然变化 —— 这里的目标点数阶梯是显式指定的，体素尺寸由二分
   搜索反解出来，从而**直接控制目标点数（≈ 直接控制密度）**，而不是反过来
   猜体素尺寸。

3. 坐标保持米制
   S3DIS 本身是公制真实尺度。不做全局缩放，只做居中：这样 radius_m 这个
   搜索参数始终有物理含义（"以米为单位的感受野半径"），可读性和可调试性
   都远好于归一化单位。

4. 标签压缩到 7 个紧凑类
   S3DIS 原始 13 类。单房间内 beam / column / board 这类样本极少，直接算
   mIoU 会被长尾噪声主导，且不同房间之间不可比。压成 7 个几何上有区分度的
   类，既保留任务难度又让 mIoU 稳定。beam / column 归入 wall（都是结构面）。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

# 解压后的 S3DIS 根目录（Stanford3dDataset_v1.2_Aligned_Version 的内容）
S3DIS_ROOT = os.path.join(
    PROJECT_ROOT, "data", "s3dis", "Stanford3dDataset_v1.2_Aligned_Version"
)
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache_s3dis")

# 默认房间。conferenceRoom_1 在多数 S3DIS 文献里被当作代表性房间：
# 类别覆盖较好（顶/地/墙/门/桌椅/杂物都有），体量适中。
DEFAULT_AREA = "Area_1"
DEFAULT_ROOM = "conferenceRoom_1"

# 读取房间时的原始点数上限。
# 为什么需要：S3DIS 单房间动辄上百万点，逐行解析会非常慢；而我们的目标点数
# 阶梯最高才 16384，读 30 万点已经绰绰有余。注意：这里的随机抽样**不会影响
# 密度阶梯**，因为阶梯是按"目标点数"反解体素尺寸的 —— 抽样只抬高了能达到的
# 最细体素，不改变各域最终的 (N, V)，因此 rho = N/V 不受影响。
ROOM_READ_MAX_PTS = 300_000

# ---------------------------------------------------------------------------
# 标签体系
# ---------------------------------------------------------------------------
# 原始 13 类 → 7 个紧凑类
COMPACT_CLASSES: List[str] = [
    "ceiling",   # 0
    "floor",     # 1
    "wall",      # 2  （含 beam / column）
    "opening",   # 3  （door / window）
    "table",     # 4  （table / bookcase / board）
    "seat",      # 5  （chair / sofa）
    "clutter",   # 6
]

_S3DIS13_TO_COMPACT: Dict[str, int] = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 2,
    "column": 2,
    "door": 3,
    "window": 3,
    "table": 4,
    "bookcase": 4,
    "board": 4,
    "chair": 5,
    "sofa": 5,
    "clutter": 6,
}

# Annotations 目录下的文件名前缀就是类别名（如 ceiling_1.txt → ceiling）
_KNOWN_PREFIXES = sorted(_S3DIS13_TO_COMPACT.keys(), key=len, reverse=True)

NUM_CLASSES = len(COMPACT_CLASSES)

# ---------------------------------------------------------------------------
# 目标点数阶梯 —— 直接决定密度阶梯
# ---------------------------------------------------------------------------
# 同一房间、同一空间范围内，rho = N / V，所以点数的比值就是密度的比值。
# 5 档等比阶梯 → 密度跨度 16 倍 → 若架构固定 k，物理感受野
# r_k ∝ rho^(-1/3) 就会漂移 16^(1/3) ≈ 2.52 倍。
#
# 为什么下限取 1024 而不是更低：k 的上界是 48，点数再低会让 k/N 大到退化
# （图几乎是全连接），"局部邻域"这一概念本身失效。
POINT_LADDER: List[int] = [16384, 8192, 4096, 2048, 1024]

# ---------------------------------------------------------------------------
# 密集端诊断阶梯 —— **只做诊断，不是训练/评估域**
# ---------------------------------------------------------------------------
# 存在的唯一理由：主阶梯的密集端（16384 点）mIoU 已到 0.88 饱和，感受野过大
# 导致的"过平滑"根本没机会暴露。于是"密度越低→感受野该越大"这个命题只被
# 验证了一半 —— 另一半（**密集区是否该收缩邻域**）在当前基准里没有观测窗口。
#
# 所以显式造两个更密的域作为**探针**，用来回答一个问题：
#     固定 k 扫描到更密的域上，mIoU 会不会开始随 k 增大而下降？
# 若会 → 存在过平滑拐点 → "密集区收缩邻域"有依据，"自适应"才有两面性；
# 若不会 → 该房间内 k 单调越大越好，命题的另一半在本基准内不成立，应如实报告。
#
# ⚠ 为什么**不能**直接把它们塞进主阶梯：一旦成为正式域，它们会同时改写三样东西 ——
#   (1) min_domain_mIoU（最差域权重最高，多两个密域会稀释）；
#   (2) 代价归一化基准 reference_cost（多两个大域会整体抬高 flops 基线）；
#   (3) α 拟合的 ρ 取值区间（端点被拉向高密度，α 数值直接变）。
#   → 与历史标定结果不再可比，且诊断域并非部署场景。
# 因此用 role 显式区分，鲁棒指标与 α 一律只在 main 域上算。
DENSE_DIAG_LADDER: List[int] = [131072, 65536]

# 域命名：从最密到最疏
def domain_names(point_ladder: List[int]) -> List[str]:
    n = len(point_ladder)
    if n == 1:
        return ["d0"]
    return [f"rho{i}" for i in range(n)]


def diag_domain_names(diag_ladder: List[int]) -> List[str]:
    """诊断域命名。与主域用不同前缀，避免缓存 npz 里键名撞车。"""
    return [f"diag{i}" for i in range(len(diag_ladder))]


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def _parse_class_from_filename(fname: str) -> Optional[str]:
    """从 Annotations 文件名解析类别名，如 chair_3.txt -> chair。"""
    stem = fname
    for ext in (".txt", ".TXT"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    for p in _KNOWN_PREFIXES:
        if stem == p or stem.startswith(p + "_"):
            return p
    return None


def _read_points_fast(path: str, max_pts: int, rng: np.random.Generator) -> np.ndarray:
    """快速读取 XYZRGB 文本，并在读取阶段限制点数。

    返回 (n, 6) float32：x y z r g b。

    实现说明
    --------
    逐行 np.loadtxt 对上百万行的文件太慢。这里按块读取、用 numpy 解析，
    并对每块做**等比例随机抽样**后再拼接 —— 保证整体是无偏的均匀抽样，
    而不是只读文件开头（那样会只拿到空间上局部的点，严重破坏几何分布）。

    解析容错：S3DIS 原版有个别行的列数异常（社区已知的越界坐标问题）。
    对列数不对的行直接跳过，而不是整体报错。
    """
    rows: List[np.ndarray] = []
    kept = 0
    # 先看文件行数，决定抽样比例
    try:
        with open(path, "rb") as fh:
            total_lines = sum(1 for _ in fh)
    except OSError:
        return np.zeros((0, 6), dtype=np.float32)
    if total_lines == 0:
        return np.zeros((0, 6), dtype=np.float32)

    keep_frac = min(1.0, max_pts / float(total_lines))

    with open(path, "rb") as fh:
        for batch in _iter_line_batches(fh, batch_lines=100_000):
            arr = _parse_batch(batch)
            if arr is None or arr.shape[0] == 0:
                continue
            if keep_frac < 1.0:
                m = arr.shape[0]
                take = int(round(m * keep_frac))
                if take <= 0:
                    continue
                sel = rng.choice(m, size=take, replace=False)
                arr = arr[sel]
            rows.append(arr)
            kept += arr.shape[0]
            if kept >= max_pts * 3:   # 已远超需求，提前停止
                break

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    out = np.concatenate(rows, axis=0)
    if out.shape[0] > max_pts:
        sel = rng.choice(out.shape[0], size=max_pts, replace=False)
        out = out[sel]
    return out


def _iter_line_batches(fh, batch_lines: int):
    """按行分批产出原始 bytes，每批 batch_lines 行。"""
    buf: List[bytes] = []
    for line in fh:
        buf.append(line)
        if len(buf) >= batch_lines:
            yield buf
            buf = []
    if buf:
        yield buf


def _parse_batch(lines: List[bytes]) -> Optional[np.ndarray]:
    """把一批文本行解析成 (n, 6) 数组，跳过列数异常的行。"""
    joined = b" ".join(lines)
    toks = joined.split()
    if not toks:
        return None
    try:
        vals = np.asarray(toks, dtype=np.float32)
    except (ValueError, TypeError):
        # 有非数值 token，退化为逐行解析（慢，但只在这种罕见情况下触发）
        good = []
        for ln in lines:
            t = ln.split()
            if len(t) != 6:
                continue
            try:
                good.append([float(x) for x in t])
            except ValueError:
                continue
        return np.asarray(good, dtype=np.float32) if good else None

    if vals.size % 6 != 0:
        # 列数不整齐 → 逐行过滤
        good = []
        for ln in lines:
            t = ln.split()
            if len(t) != 6:
                continue
            try:
                good.append([float(x) for x in t])
            except ValueError:
                continue
        return np.asarray(good, dtype=np.float32) if good else None
    return vals.reshape(-1, 6)


def list_rooms(area: Optional[str] = None) -> List[Tuple[str, str]]:
    """列出可用房间。返回 [(area, room), ...]。"""
    if not os.path.isdir(S3DIS_ROOT):
        return []
    areas = [area] if area else sorted(
        d for d in os.listdir(S3DIS_ROOT)
        if d.startswith("Area_") and os.path.isdir(os.path.join(S3DIS_ROOT, d))
    )
    out: List[Tuple[str, str]] = []
    for a in areas:
        ap = os.path.join(S3DIS_ROOT, a)
        for r in sorted(os.listdir(ap)):
            if os.path.isdir(os.path.join(ap, r)) and r != "Annotations":
                if os.path.isdir(os.path.join(ap, r, "Annotations")):
                    out.append((a, r))
    return out


def load_room(
    area: str = DEFAULT_AREA,
    room: str = DEFAULT_ROOM,
    max_pts: int = ROOM_READ_MAX_PTS,
    seed: int = 20260923,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取一个房间的点云与标签。

    返回
    ----
    xyz    : (n, 3) float32，**米制真实坐标**（仅做了居中）
    rgb    : (n, 3) float32，0-1 归一化
    labels : (n,)   int64，0..NUM_CLASSES-1（7 个紧凑类）

    实现说明
    --------
    从 room/Annotations/*.txt 逐文件读取并打标签。每个文件是一个语义实例
    （如 chair_3.txt），文件名前缀给出类别。相比读 `<room>.txt`（全部点但无
    标签）再和实例做点匹配，直接读 Annotations 更直接、无匹配误差。
    """
    ann_dir = os.path.join(S3DIS_ROOT, area, room, "Annotations")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(
            f"找不到房间标注目录：{ann_dir}\n"
            f"请确认 S3DIS 已解压到 {S3DIS_ROOT}"
        )

    files = sorted(f for f in os.listdir(ann_dir) if f.lower().endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"{ann_dir} 下没有 .txt 文件")

    rng = np.random.default_rng(seed)
    # 每文件配额：按文件数均分，保证小类不会被大类挤掉
    per_file = max(1000, max_pts // max(1, len(files)))

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for fname in files:
        cls_name = _parse_class_from_filename(fname)
        if cls_name is None:
            continue
        arr = _read_points_fast(os.path.join(ann_dir, fname), per_file, rng)
        if arr.shape[0] == 0:
            continue
        xs.append(arr)
        ys.append(np.full((arr.shape[0],), _S3DIS13_TO_COMPACT[cls_name], dtype=np.int64))

    if not xs:
        raise ValueError(f"{ann_dir} 中未能解析出任何有效点")

    pts = np.concatenate(xs, axis=0)
    labels = np.concatenate(ys, axis=0)

    # 清洗异常坐标：S3DIS 原版已知有越界值，会导致后续 bbox / 体素索引爆炸
    xyz = pts[:, :3].astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    in_range = (np.abs(xyz) < 100.0).all(axis=1)      # 房间尺度远小于 100 m
    ok = finite & in_range
    n_drop = int((~ok).sum())
    if n_drop:
        xyz = xyz[ok]
        labels = labels[ok]
        rgb_raw = pts[ok, 3:6]
    else:
        rgb_raw = pts[:, 3:6]

    # 居中（保持米制，不做缩放 —— radius_m 必须始终有物理含义）
    xyz = xyz - xyz.mean(axis=0, keepdims=True)

    # RGB 归一化：原始值可能是 0-255 或 0-1，统一到 0-1
    rgb_max = float(rgb_raw.max()) if rgb_raw.size else 1.0
    rgb = rgb_raw.astype(np.float32) / (255.0 if rgb_max > 1.5 else 1.0)

    return xyz, rgb, labels


# ---------------------------------------------------------------------------
# 体素下采样
# ---------------------------------------------------------------------------
def _voxel_keys(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """把点映射成体素线性索引（用于去重）。"""
    keys = np.floor(xyz / float(voxel)).astype(np.int64)
    keys -= keys.min(axis=0, keepdims=True)
    dims = (keys.max(axis=0) + 1).astype(np.int64)
    return (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]


def voxel_count(xyz: np.ndarray, voxel: float) -> int:
    """该体素尺寸下会保留多少点（不实际构造，只计数）。"""
    if xyz.shape[0] == 0:
        return 0
    return int(np.unique(_voxel_keys(xyz, voxel)).shape[0])


def voxel_downsample(
    xyz: np.ndarray, labels: np.ndarray, voxel: float
) -> Tuple[np.ndarray, np.ndarray]:
    """体素下采样：每个体素保留一个点（取索引最小的那个）。

    取"索引最小"而非"质心"，是为了保持每个点都是**真实观测点**，
    坐标不被人工平均改变 —— 密度分析里这一点很重要（质心会轻微改变
    局部间距分布）。代价是存在微弱的索引顺序偏置，可忽略。
    """
    if xyz.shape[0] == 0:
        return xyz, labels
    flat = _voxel_keys(xyz, voxel)
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    first = np.ones(flat_s.shape[0], dtype=bool)
    first[1:] = flat_s[1:] != flat_s[:-1]
    idx = order[first]
    return xyz[idx], labels[idx]


def calibrate_voxel_for_target_n(
    xyz: np.ndarray,
    labels: np.ndarray,
    target_n: int,
    v_lo: float = 1e-4,
    v_hi: float = 0.5,
    iters: int = 26,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """二分搜索体素尺寸，使下采样后的点数接近 target_n。

    为什么用二分反解而不是拍一个体素值：S3DIS 是**表面采样**（不是体填充），
    点数随体素的变化并不是简单的 1/v^3，没有解析式。直接按目标点数反解，
    才能让密度阶梯真正可控。

    返回 (voxel, xyz_down, labels_down)
    """
    n = xyz.shape[0]
    if target_n >= n:
        return 0.0, xyz, labels

    lo, hi = v_lo, v_hi
    # 先确保区间有效
    while voxel_count(xyz, hi) > target_n and hi < 10.0:
        hi *= 2.0
    while voxel_count(xyz, lo) < target_n and lo > 1e-6:
        lo /= 2.0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = voxel_count(xyz, mid)
        if c > target_n:
            lo = mid
        else:
            hi = mid
    v = 0.5 * (lo + hi)
    xd, yd = voxel_downsample(xyz, labels, v)
    return v, xd, yd


# ---------------------------------------------------------------------------
# 密度域构造
# ---------------------------------------------------------------------------
def build_domains(
    area: str = DEFAULT_AREA,
    room: str = DEFAULT_ROOM,
    point_ladder: Optional[List[int]] = None,
    read_max_pts: int = ROOM_READ_MAX_PTS,
    seed: int = 20260923,
    diag_ladder: Optional[List[int]] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], dict]:
    """构造受控密度域。

    返回
    ----
    domains : {domain_name: (xyz, labels)}
    meta    : 元信息（房间、体素、实际点数、实际密度、**role** 等）

    关键性质：所有域来自**同一个房间、同一个空间范围**，只是体素尺寸不同。
    因此几何与语义内容完全一致，唯一变量是点密度。

    两种 role
    ---------
    main : 参与鲁棒目标、α 拟合、代价归一化。这是"部署/训练"域。
    diag : 仅用于密集端过平滑诊断（见 DENSE_DIAG_LADDER）。**不参与任何指标**。
    """
    ladder = list(point_ladder or POINT_LADDER)
    ladder = sorted(set(int(v) for v in ladder), reverse=True)
    diag = sorted(set(int(v) for v in (diag_ladder or [])), reverse=True)
    # 诊断域必须比主域更密，否则"密集端探针"这个身份不成立
    diag = [v for v in diag if not ladder or v > max(ladder)]

    xyz, _rgb, labels = load_room(area, room, max_pts=read_max_pts, seed=seed)
    names = domain_names(ladder)

    domains: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta: dict = {
        "area": area,
        "room": room,
        "point_ladder": ladder,
        "diag_ladder": diag,
        "read_max_pts": int(read_max_pts),
        "raw_points_after_read": int(xyz.shape[0]),
        "classes": COMPACT_CLASSES,
        "domains": {},
    }

    def _add(name: str, target: int, role: str) -> None:
        v, xd, yd = calibrate_voxel_for_target_n(xyz, labels, target)
        domains[name] = (xd.astype(np.float32), yd.astype(np.int64))
        meta["domains"][name] = {
            "target_n": int(target),
            "voxel_m": float(v),
            "n": int(xd.shape[0]),
            "role": role,
            "desc": f"体素 {v*100:.2f} cm",
        }

    for name, target in zip(names, ladder):
        _add(name, target, "main")
    for name, target in zip(diag_domain_names(diag), diag):
        _add(name, target, "diag")

    return domains, meta


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
_CACHE_NPZ = "s3dis_domains.npz"
_CACHE_META = "meta.json"


def save_cache(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]], meta: dict, cache_dir: Optional[str] = None
) -> str:
    d = cache_dir or CACHE_DIR
    os.makedirs(d, exist_ok=True)
    arrays = {}
    for name, (xyz, y) in domains.items():
        arrays[f"{name}__xyz"] = xyz
        arrays[f"{name}__y"] = y
    np.savez_compressed(os.path.join(d, _CACHE_NPZ), **arrays)
    with open(os.path.join(d, _CACHE_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return d


def load_cache(cache_dir: Optional[str] = None) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], dict]:
    d = cache_dir or CACHE_DIR
    npz_path = os.path.join(d, _CACHE_NPZ)
    meta_path = os.path.join(d, _CACHE_META)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"缓存不存在：{npz_path}，请先运行 prep.py --build")

    domains: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with np.load(npz_path) as z:
        for key in z.files:
            if key.endswith("__xyz"):
                name = key[: -len("__xyz")]
                domains[name] = (z[key], z[f"{name}__y"])

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    # 保持域顺序与 meta 一致
    if meta.get("domains"):
        ordered = {k: domains[k] for k in meta["domains"].keys() if k in domains}
        for k, v in domains.items():
            ordered.setdefault(k, v)
        domains = ordered
    return domains, meta
