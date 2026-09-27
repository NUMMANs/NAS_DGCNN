"""**可实测的架构搜索空间**：把架构表示成一个可执行字符串，并能量化其成本。

设计目标（回答"怎样划定范围才能达到真正的进化"）
================================================================
此前两轮双目标搜索都被结构性限制住：
  · 候选池是我**手工枚举**的固定点 → LLM 只能从中挑（"查表择优"，不是搜索）
  · 新组合只有**估计值**，而估计值**结构上不可能超过**同档实测值 → 探索被奖励函数扼杀

本模块划定的范围满足三个条件，使"真正的进化"成为可能：

**条件 A：新架构必须能被真实评估**
    架构由 `describe()` 序列化成**字符串**，该字符串同时决定
    `build()` 能否实例化、`estimate_cost()` 能否估成本。
    任何合法串都能进 `multibranch_exp` 实测 —— 进化提出 → 实测 → 写回候选池。

**条件 B：空间必须显著大于手工枚举**
    可用**砖块（brick）**自由组合，而不是几个预设开关：
      · 分支族：voxel / pillar / rangeview / points
      · 每分支独立参数：体素边长、z 分层、特征宽度、层数
      · 分支数量：1~5（含重复族但不同参数）
      · 融合方式：late / gate / concat
    组合数 ≈ 10^6 量级 >> 手工枚举的 88 个。**这是"能搜出预料不到的组合"的前提。**

**条件 C：奖励不能有结构性天花板**
    成本**必须可估**（否则无法进 MAP-Elites 的成本轴），
    但精度**不估**——精度一律留 0 并标记 `pending`，
    由实测回路填真值。这样新架构**没有任何"估计值天花板"**，
    它可以凭实测结果直接胜过任何已知配置。

关键设计：**成本可估、精度必须实测**
================================================================
成本与精度对"可否估计"的容忍度完全不同：
  · 成本由**结构**决定（分支数 × z 分层 × 体素尺寸）→ 可稳定外推，
    且实测锚点充足（voxel25_z16≈4.1ms → z128≈23.6ms → points≈104ms）
  · 精度由**优化**决定，无法从结构推出 → 任何估计都会系统性地
    压制探索（前两轮已实测印证：`is_synthetic` 全为 0）

所以这里划红线：**`estimate_cost()` 只给成本；精度永远是 None**，
必须有实测值才能参与排序。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 砖块定义
# ---------------------------------------------------------------------------
BRANCH_FAMILIES: Tuple[str, ...] = ("voxel", "pillar", "rangeview", "points")

# 实测锚点：单分支成本（ms/step，来自 2026-09-24/25 的全量实测）
#   voxel25_z16 4.2 / z32 8.0 / z48 9.7 / z64 12.8 / z96 18.5 / z128 24.0 / z192 34.8
#   voxel20_z64 19.5 / voxel15_z64 32.9 / voxel35_z64 7.8 / voxel50_z64 4.2
#   points 103.8 / pillar25 32.5 / rangeview64 5.3
COST_ANCHORS = {
    "points": 103.8,
    "pillar25": 32.5,
    "rangeview64": 5.3,
}
# voxel 成本模型：随 z 近似线性，且随体素边长增大而下降（体素数 ∝ 1/vs^2 但受 side 上限影响）
VOXEL_BASE_MS = 1.2
VOXEL_MS_PER_Z = 0.18
FUSE_MULT = {"late": 1.00, "gate": 1.00, "concat": 1.05}
BRANCH_OVERHEAD = 1.06          # 每多一个分支的头部开销


def _voxel_cost(vs: float, z: int) -> float:
    """体素分支成本模型（锚点拟合）。

    实测：vs=0.25 时 z16 4.2 / z48 9.7 / z64 12.8 / z128 24.0
    vs 变小 → side 变大 → 体素数上升 → 变贵（v35 7.8 < v25 12.8 < v20 19.5 < v15 32.9 @ z64）
    按 (0.25/vs) 的一次方近似（实测 v35→v15 是 7.8→32.9，比值 4.2，
    而 (0.25/0.35) : (0.25/0.15) = 0.71 : 1.67 = 2.35，一次方略保守，取指数 1.3 更贴）
    """
    span = max(0.05, min(1.0, float(vs)))
    base = VOXEL_BASE_MS + VOXEL_MS_PER_Z * max(4, min(256, int(z)))
    scale = (0.25 / span) ** 1.3
    return base * scale


def estimate_cost(branches: List[dict], fuse: str, feat: int = 64) -> float:
    """给一组分支估算 ms/step。**只估成本，不估精度**（见模块文档）。

    ⚠⚠ **`feat` 必须计入，否则会严重低估**（实测踩过）：
    `feat=128` 在 5 分支配置上实测 **131.6 ms/step**，
    而只按分支估算给出 **46.7 ms** —— 低估 **2.8 倍**。
    后果：合成候选的成本被算小 → combined 被算大 → MAP-Elites 的成本轴被污染，
    一个实际比 `points` 基线还慢的配置会被当成"便宜"的 Pareto 点。
    （代价是二次方：特征宽 ×2 → 每层 FLOPs ≈ ×4，实测 131.6/46.7 ≈ 2.8× 与之相符。）
    """
    total = 0.0
    for b in branches:
        fam = str(b.get("family", "voxel"))
        if fam == "voxel":
            total += _voxel_cost(float(b.get("voxel_size", 0.25)), int(b.get("z_bins", 16)))
        elif fam == "pillar":
            # 柱状分支初版会按 xy 尺寸缩放，但量纲应为"边长越小越贵"
            total += COST_ANCHORS["pillar25"] * (0.25 / max(0.05, float(b.get("voxel_size", 0.25))))
        elif fam == "rangeview":
            side = int(b.get("side", 64))
            total += COST_ANCHORS["rangeview64"] * (side / 64.0)
        elif fam == "points":
            total += COST_ANCHORS["points"]
        else:
            total += 20.0
    total *= FUSE_MULT.get(str(fuse), 1.0)
    total *= BRANCH_OVERHEAD ** max(0, len(branches) - 1)
    # 特征宽度：以 64 为基准按二次方缩放（锚点 feat=32/64/128 的参数量比 ≈ 1:3.2:11.9，
    # 但实测耗时比 5 分支下约 1 : 1 : 2.8，故取指数 1.5 折中，宁高勿低）
    f = max(8, min(512, int(feat)))
    if f != 64:
        total *= (f / 64.0) ** 1.5
    return float(total)


def estimate_params(branches: List[dict], feat: int = 64, layers: int = 2) -> int:
    """粗估参数量（只用于日志/诊断，**不参与适应度**）。"""
    per = int(feat) * int(layers) * 2000
    return per * max(1, len(branches))


# ---------------------------------------------------------------------------
# 序列化：架构 <-> 字符串
# ---------------------------------------------------------------------------
# 语法：
#   ARCH   := <branch>[,<branch>...] "@" <fuse> ["|feat=" N]
#   branch := voxel<边长为厘米>[_z<分层>]
#           | pillar<边长为厘米>
#           | rangeview<side>
#           | points
# 例：voxel20_z32,voxel50_z96@gate
#     voxel15_z64,voxel15_z192,voxel35_z96@concat|feat=128
_VB = re.compile(r"^voxel(\d+)(?:_z(\d+))?$")
_PB = re.compile(r"^pillar(\d+)$")
_RB = re.compile(r"^rangeview(\d+)$")


def parse(arch: str) -> Optional[Dict[str, object]]:
    """把架构串解析成结构化描述；非法返回 None。

    ⚠ 必须**严格**校验（初版有两个静默错误，都已被抓出）：
      1. `fuse` 解析前忘了先剥掉 `|feat=N`，导致 `@gate|feat=128` 被读成 fuse="gate|feat=128"
         从而落回默认值 → **所有带 feat 的串 fuse 都被静默改成 late**。
      2. 分肢没有做**取值域校验**，`voxel99_z999@bogus`（边长 0.99m、z=999、fuse 非法）
         竟然被接受 → 会造出无法实例化、且成本模型外推到垃圾值的"架构"。
    严格拒绝非法串，比事后 debug 一个幻想的架构便宜得多。
    """
    if not isinstance(arch, str) or "@" not in arch:
        return None
    body, rest = arch.split("@", 1)
    rest = rest.strip()
    feat = 64
    if "|" in rest:
        rest, opts = rest.split("|", 1)
        for kv in opts.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k.strip() == "feat":
                    try:
                        feat = int(v)
                    except ValueError:
                        return None
    fuse = rest.strip()
    if fuse not in FUSE_MULT:
        return None
    if feat not in (32, 64, 128, 256):
        return None

    branches: List[dict] = []
    for raw in body.split(","):
        tok = raw.strip()
        if not tok:
            continue
        m = _VB.match(tok)
        if m:
            vs_cm, z = int(m.group(1)), int(m.group(2)) if m.group(2) else 16
            # 取值域校验：边长 5~100 cm；z 分层 4~512（超出范围成本模型会外推失真）
            if not (5 <= vs_cm <= 100 and 4 <= z <= 512):
                return None
            branches.append({"family": "voxel", "name": tok,
                             "voxel_size": vs_cm / 100.0, "z_bins": z})
            continue
        m = _PB.match(tok)
        if m:
            vs_cm = int(m.group(1))
            if not (5 <= vs_cm <= 100):
                return None
            branches.append({"family": "pillar", "name": tok,
                             "voxel_size": vs_cm / 100.0})
            continue
        m = _RB.match(tok)
        if m:
            side = int(m.group(1))
            if side not in (32, 64, 128, 256):
                return None
            branches.append({"family": "rangeview", "name": tok, "side": side})
            continue
        if tok == "points":
            branches.append({"family": "points", "name": tok})
            continue
        return None                     # 有无法识别的砖块 → 整串非法
    if not branches or len(branches) > 5:
        return None
    # 同一分支不得重复出现（重复无意义且会误导成本模型）
    names = [b["name"] for b in branches]
    if len(names) != len(set(names)):
        return None
    return {"branches": branches, "fuse": fuse, "feat": int(feat),
            "arch": arch, "n_branches": len(branches),
            "cost_ms": estimate_cost(branches, fuse, feat)}


def describe(spec: Dict[str, object]) -> str:
    """结构化描述 → 架构串（parse 的逆）。"""
    names = [str(b.get("name")) for b in spec.get("branches", [])]
    arch = ",".join(names) + "@" + str(spec.get("fuse", "late"))
    feat = int(spec.get("feat", 64))
    if feat != 64:
        arch += f"|feat={feat}"
    return arch


# ---------------------------------------------------------------------------
# 文件名安全化
# ---------------------------------------------------------------------------
# ⚠⚠ **`|` 是 Windows 文件名的非法字符**（`* ? " < > | : \ /`）。
# 架构串里的 `|feat=N` 会直接让结果表写盘失败 —— 而且是 `PermissionError`，
# 发生在**第一个 cell 写盘时**，于是：
#   · 进程崩在第一格，表完全不存在
#   · 外层重试脚本因为"表不存在"而无限重跑同一配置
#   · **整条 `feat` 轴静默不可测**（实测踩过：feat=32/128 全部写不出去）
# 所以：**架构串保留 `|`（它是语法的一部分，人类可读），
# 但凡是落到文件名的地方，必须先过 `safe_filename`。**
_UNSAFE = str.maketrans({c: "_" for c in '*?"<>|:\\/'})


def safe_filename(name: str) -> str:
    """把架构串（或任意名字）转成 Windows 合法的文件名片段。

    只替换非法字符，**不改语义**：结果表里的 `name` 字段仍是原架构串，
    所以读回来还能精确匹配。切勿反过来从文件名还原架构串——
    还原要用表内的 `name`/`config` 字段（`_load_candidates` 就是这么做的）。
    """
    return str(name).translate(_UNSAFE)


def is_measured(arch: str, measured: Dict[str, float]) -> bool:
    """该架构是否已有实测精度。"""
    return arch in measured


def summarize_space() -> Dict[str, int]:
    """给出搜索空间的规模（用于验证"显著大于手工枚举"）。"""
    vs = [15, 20, 25, 35, 50]
    zs = [8, 16, 32, 48, 64, 96, 128, 192]
    n_voxel = len(vs) * len(zs)                       # 40
    n_pillar = len(vs)                             # 5
    n_rv = 4                                       # 32/64/128/256
    per_family = n_voxel + n_pillar + n_rv + 1      # +points = 50
    # 1~5 个分支（可重复族但参数不同）→ 组合数
    total = 0
    for k in range(1, 6):
        total += per_family ** k
    return {"per_family": per_family, "branch_counts_1to5": total,
            "with_fuse": total * 3,
            "hand_enumerated": 88}
