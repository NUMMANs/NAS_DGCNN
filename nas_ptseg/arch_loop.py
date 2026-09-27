"""**进化闭环**：提出 → 实测 → 写回 → 再提出。

回答"怎样划定范围才能达到真正的进化"
================================================================
前两轮双目标搜索都不是真正的进化，原因是结构性的：

| 轮次 | 搜索空间 | 新架构能否被评估 | 结果 |
|---|---|---|---|
| `pareto`（第一轮） | 63 个**预先测好**的固定点 | 不能（只能挑名字） | best 在 iteration 1，60 迭代只 4 个不同分数 |
| `pareto-gen`（第二轮） | 组合空间（能提新架构） | 只能拿到**估计精度** | `is_synthetic` 0→11，但估计值赢不过实测强项 |
| **本轮 `arch_loop`** | **架构串（≈10^8 组合）** | **能（真训练 + 写回表）** | ← 本文件 |

**真正的进化需要三个条件，本文件补齐最后一个：**

- **条件 A（搜索空间够大）**：`nas_ptseg/arch_space.py` 用"砖块"组合描述架构，
  组合数 ≈ 3.2×10^8（1~5 个分支 × 50 种分支砖块 × 3 种融合），
  远超我手工枚举的 88 个配置。**这是"找到预料不到的组合"的前提。**
- **条件 B（新架构可被评估）**：架构串可直接被 `multibranch_exp --only "<arch>"` 训练，
  无需注册到 CONFIGS（已打通并冒烟验证）。
- **条件 C（奖励无结构性天花板）**：**精度绝不估计** —— 只估成本，精度留 None，
  必须实测才有值。`evaluator_pareto._synth_from_spec` 的第三版即按此改写。

本文件的职责：把评估器标记为 `pending` 的建议架构**真正训练出来**并写回结果表，
于是下一轮搜索就能读到**真实精度**。新架构因此可以凭实测结果胜过任何已知配置 ——
**没有"估计值天花板"**。

实测成本控制
================================================================
每轮只测少量（默认 3 个），且**优先测成本低的**：
- 单次 300ep × 12 房 × 2 seed ≈ 4~15 分钟（取决于架构成本）
- 候选池越大，单轮搜索越贵，所以用小步快跑：每轮测几个 → 重跑搜索 → 再测

用法
================================================================
    python -m nas_ptseg.arch_loop --suggest-file <path>     # 从评估器建议里取
    python -m nas_ptseg.arch_loop --archs "a@gate" "b,c@late"  # 直接指定
    python -m nas_ptseg.arch_loop --from-evolve              # 从进化库自动提取 pending
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from nas_ptseg import arch_space as ARCH          # noqa: E402

PY_EXE = r"D:\anaconda\envs\lidar310\python.exe"


def measured_archs() -> Dict[str, float]:
    """扫描所有结果表，返回 {架构串: 实测平均 mIoU}。

    只认**架构串形式**的表名（含 `@`），因为只有那类是进化产物。

    ⚠⚠ **必须排除 `_tval` 表（test split 验收表）** —— 否则会把
    test 精度当成搜索依据 → **测试集泄漏**。
    实测踩过：验收后 `measured_archs()` 里冠军的值从 val 的 0.4126
    变成了 test 的 0.4184，若不修，后续采样/排序都会基于 test 数字。
    val 与 test 的差距虽小（此处 0.006），但性质上是泄漏，必须从机制上堵死。

    ⚠ 扫描范围必须覆盖**所有**可能的输出目录（`data/*/`），不能只扫
    `data/multibranch*` —— 否则自建目录（如 `_smoke_archstr`）里的表会被漏掉，
    表现为"明明测过了却显示未测"，进而**重复训练**（白烧 GPU）。
    """
    out: Dict[str, float] = {}
    for p in glob.glob(os.path.join(_PROJECT, "data", "*", "*_k*_ep*.json")):
        base = os.path.basename(p)
        if base.startswith("summary"):
            continue
        if "_tval" in base:                    # ← test 验收表，永不参与搜索
            continue
        m = re.match(r"^(.*?)_k\d+_ep\d+", base)
        if not m:
            continue
        name = m.group(1)
        if "@" not in name:
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        cells = rec.get("cells") or {}
        vals = [v for c in cells.values() for v in c.values()]
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


def extract_from_evolve(out_dir: str, max_n: int = 40) -> List[str]:
    """从进化输出目录里提取**策略提出过的架构串**（去重、按出现顺序）。

    进化把每个候选程序写在 checkpoint/programs 里；这里用正则抓架构串字面量，
    因为策略里写的就是这些字符串，抓出来最稳（不依赖 openevolve 的内部格式）。

    ⚠⚠ **正则必须避开两个陷阱**（初版在这里直接把进程打崩，
    症状是 exit code 1 且没有任何输出/回溯 —— 容易误判成"环境坏了"）：
      1. **不要在长行上用 `.*?` 这种带回溯的结构**。候选程序里有很长的
         docstring/字符串，`(?:,?...)*` 这类嵌套量词会在其上组合爆炸，
         在 Windows 上表现为原生栈溢出（`0xC0000409`），**连 Python 回溯都来不及打**。
      2. **逐行匹配 + 显式字符类**（不含 `.`）把回溯压到常数级。
    架构串的字符集很小（字母/数字/逗号/下划线/@/|/=），完全可以写死。
    """
    # 只允许架构串里真实出现的字符：字母、数字、下划线、逗号、@、|、=
    tok = r"[A-Za-z0-9_,]+@(?:late|gate|concat)(?:\|feat=[0-9]+)?"
    pat = re.compile(r"""["'](""" + tok + r""")["']""")
    found: List[str] = []
    seen = set()
    roots = [out_dir, os.path.join(out_dir, "best")]
    files: List[str] = []
    for r in roots:
        files.extend(glob.glob(os.path.join(r, "**", "*.py"), recursive=True))
        # 也扫 checkpoint 里的程序（若开了 checkpointing）
        files.extend(glob.glob(os.path.join(out_dir, "checkpoints", "**", "*"),
                               recursive=True))
    for f in files[:400]:                # 上限，防止目录异常膨胀时卡死
        try:
            if os.path.getsize(f) > 2_000_000:      # 单文件 >2MB 跳过（不是候选程序）
                continue
            txt = open(f, "r", encoding="utf-8", errors="ignore").read()
        except Exception:  # noqa: BLE001
            continue
        for line in txt.splitlines():               # ← 逐行，避免长行回溯
            for m in pat.finditer(line):
                a = m.group(1)
                if a in seen:
                    continue
                if ARCH.parse(a) is None:
                    continue
                seen.add(a)
                found.append(a)
                if len(found) >= max_n:
                    return found
    return found


def propose_by_sampling(measured: Dict[str, float], budget: int = 3,
                        seed: int = 0, strategy: str = "greedy_local") -> List[str]:
    """**自主提出**待测架构 —— 不依赖 LLM，直接从空间里采样。

    为什么需要这条路：LLM 提出的候选会把进程打崩前的那一轮最终收敛到
    "复述已测候选"（best 程序返回的是名字字符串，`extract_from_evolve` 抓到 0 个）。
    为了不让进化闭环依赖 LLM 的行为，这里提供一个**确定性的、纯结构驱动**的提出器，
    它利用本项目已实测的两条规律来定向搜索：

      · **z_bins 倒 U，峰值≈64**（z8 0.3290 → z64 0.3885 → z192 0.3806）
      · **voxel_size 倒 U，峰值≈0.20m**（v50 0.3685 → v20 0.3902 → v15 0.3879）
      · **混合体素尺寸**（细+粗在同一网络内）比纯一档更好
      · **强分支才值得融合**（弱分支参与会拖累，实测 `points+rangeview` 低于基线）

    `strategy="greedy_local"`：以当前**精度最高**的实测配置为锚点，
    在它的邻域里做**单轴扰动**（改一个分支的 z / 体素尺寸 / 换融合），
    优先低成本。
    这样每次提出的都是"看起来有希望但从未测过"的点，
    而不是随机噪声 —— 在预算很小（每轮 3 个）时这一点很重要。
    """
    import random
    rng = random.Random(int(seed))

    # 从已有实测（含手工配置名）里挑锚点：取精度最高的几个
    anchor_archs = _anchor_archs(measured)
    if not anchor_archs:
        anchor_archs = ["voxel20_z32,voxel50_z96@gate"]

    # 已测过的手工配置名也要排除（避免重复测同名架构）
    done = set(measured.keys())
    done |= _hand_config_archs()

    out: List[str] = []
    seen = set()
    ZS = [16, 32, 48, 64, 96, 128]
    VS = [15, 20, 25, 35, 50]

    def consider(a: str) -> bool:
        if a in done or a in seen:
            return False
        if ARCH.parse(a) is None:
            return False
        seen.add(a)
        out.append(a)
        return len(out) >= budget

    # 1) 锚点邻域：逐分支改 z / 体素尺寸
    for anc in anchor_archs[:3]:
        spec = ARCH.parse(anc)
        if spec is None:
            continue
        brs = list(spec["branches"])
        for i, b in enumerate(brs):
            if b["family"] == "voxel":
                for z in ZS:
                    if z == b.get("z_bins"):
                        continue
                    nb = dict(b, z_bins=z, name=f"voxel{int(b['voxel_size']*100)}_z{z}")
                    cand = brs[:i] + [nb] + brs[i + 1:]
                    a = ARCH.describe({"branches": cand, "fuse": spec["fuse"]})
                    if consider(a):
                        return out
                for vs in VS:
                    if vs == int(b["voxel_size"] * 100):
                        continue
                    nb = dict(b, voxel_size=vs / 100.0,
                              name=f"voxel{vs}_z{b.get('z_bins', 16)}")
                    cand = brs[:i] + [nb] + brs[i + 1:]
                    a = ARCH.describe({"branches": cand, "fuse": spec["fuse"]})
                    if consider(a):
                        return out
        # 融合方式
        for fu in ("late", "gate", "concat"):
            if fu == spec["fuse"]:
                continue
            a = ARCH.describe({"branches": brs, "fuse": fu})
            if consider(a):
                return out

    # 2) 若邻域已被测完 → 随机采样补足（保证闭环永远能推进）
    tries = 0
    while len(out) < budget and tries < 5000:
        tries += 1
        k = rng.choice([2, 2, 2, 3, 4])           # 偏爱 2 个分支（实测甜点）
        brs = []
        for _ in range(k):
            vs = rng.choice(VS)
            z = rng.choice(ZS)
            brs.append({"family": "voxel", "voxel_size": vs / 100.0, "z_bins": z,
                        "name": f"voxel{vs}_z{z}"})
        # 去重分支
        names = [b["name"] for b in brs]
        if len(set(names)) != len(names):
            continue
        a = ARCH.describe({"branches": brs, "fuse": rng.choice(["late", "gate"])})
        if consider(a):
            break
    return out


def _anchor_archs(measured: Dict[str, float]) -> List[str]:
    """把"精度最高的已测配置"翻译成架构串，作为扰动锚点。

    ⚠ **只认"可信"的实测**（`n_cells >= 20`，即至少 ~10 房 × 2 seed）。
    否则一次 1 房间 × 2 epoch 的冒烟测试（如 `voxel30_z48@gate` 得 0.3729）
    会被当成高精度锚点，把整个采样邻域**带偏到它附近** ——
    表现为"提出来的全是同一个架构的单轴变体"，浪费实测预算。
    """
    anchors: List[str] = []
    reliable = _reliable_archs()
    for k, v in sorted(measured.items(), key=lambda z: -z[1]):
        if k in reliable and ARCH.parse(k) is not None:
            anchors.append(k)
    # 手工配置里的冠军（转成架构串）—— 这些是全量实测过的，可信
    hand = {
        "ms3_mix_v20_v50_z16_z32_z96": "voxel20_z16,voxel20_z32,voxel50_z96@gate",
        "ms4_mix_v20_v50_z16_z32_z64_z96":
            "voxel20_z16,voxel20_z32,voxel20_z64,voxel50_z96@gate",
        "ms_mix_v20_v50_z32_z96": "voxel20_z32,voxel50_z96@gate",
        "ms3_v20_z32_z64_z96": "voxel20_z32,voxel20_z64,voxel20_z96@gate",
    }
    for _k, v in hand.items():
        if v not in anchors and ARCH.parse(v) is not None:
            anchors.append(v)
    return anchors[:4]


def _reliable_archs(min_cells: int = 20) -> set:
    """返回**评估充分**的架构串集合（cell 数足够多，不是冒烟测试）。

    `n_cells` = 训练房数 × seed 数。全量实验是 12 房 × 2 seed = 24；
    取 20 作阈值以容忍个别房间缺失。低于此值的表（如 2 epoch / 1 房的冒烟）
    精度极不可靠，**不得**作为锚点或参与排序比较。
    """
    out = set()
    for p in glob.glob(os.path.join(_PROJECT, "data", "*", "*_k*_ep*.json")):
        base = os.path.basename(p)
        if base.startswith("summary") or "@" not in base:
            continue
        if "_tval" in base:                    # test 验收表不参与（防泄漏）
            continue
        m = re.match(r"^(.*?)_k\d+_ep\d+", base)
        if not m:
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if len(rec.get("cells") or {}) >= int(min_cells):
            out.add(m.group(1))
    return out


def _hand_config_archs() -> set:
    """把 CONFIGS 里的手工配置也转成架构串，避免重复实测同一架构。"""
    out = set()
    try:
        from nas_ptseg.multibranch_exp import CONFIGS
        for name, cfg in CONFIGS:
            br = str(cfg.get("branches", ""))
            fu = str(cfg.get("fuse", "late"))
            if not br or br == "points":
                continue
            zb = int(cfg.get("z_bins", 16))
            parts = []
            for tok in br.split(","):
                tok = tok.strip()
                if tok.startswith("voxel") and "_z" not in tok:
                    parts.append(f"{tok}_z{zb}")
                else:
                    parts.append(tok)
            a = ",".join(parts) + "@" + fu
            if ARCH.parse(a) is not None:
                out.add(a)
    except Exception:  # noqa: BLE001
        pass
    return out


def measure(archs: List[str], epochs: int, n_train: int, n_val: int,
            out_dir: str, log_file: str, max_rounds: int = 10) -> int:
    """调用 multibranch_exp 真正训练这些架构串。**带重试**。

    ⚠⚠ **必须带重试**（手工拼命令行的方式已经在 2026-09-25 吃过一次亏）：
    本项目训练进程有**间歇性原生崩溃**（`exit -1073740791` = `0xC0000409`，
    无 Python 回溯），曾在第 3 个配置的第 24 个 cell 处崩溃，
    表停在 276/288 且进程静默消失。
    因为结果表**每个 cell 都落盘**，重试只会重跑缺失的那部分，
    所以"多轮重试直到全部完成"是最省成本的正确做法 ——
    否则一次崩溃就白等一整轮（本例 ≈50 分钟）。
    """
    if not archs:
        print("[arch_loop] 没有待测架构（全部已实测）", flush=True)
        return 0
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "1", "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "*"})
    print(f"[arch_loop] 开始实测 {len(archs)} 个架构：", flush=True)
    for a in archs:
        print(f"    {a}", flush=True)

    rc = 0
    for rnd in range(1, max(1, int(max_rounds)) + 1):
        # 每轮只跑"还没落表完成"的架构（每 cell 落盘 → 天然可续跑）
        todo = [a for a in archs if not _table_complete(a, out_dir, epochs)]
        if not todo:
            print(f"[arch_loop] 第 {rnd} 轮：全部完成", flush=True)
            return 0
        print(f"[arch_loop] 第 {rnd} 轮：待跑 {len(todo)} 个", flush=True)
        cmd = [PY_EXE, "-W", "ignore", "-m", "nas_ptseg.multibranch_exp",
               "--k", "48", "--epochs", str(int(epochs)),
               "--seeds", "12345", "23456",
               "--n-train", str(int(n_train)), "--n-val", str(int(n_val)),
               "--only", *todo,
               "--out-dir", out_dir, "--log-file", log_file]
        r = subprocess.run(cmd, cwd=_PROJECT, env=env)
        rc = int(r.returncode)
        # ⚠⚠ **不要只看 returncode 就宣告成功**。
        # 实测：子进程在"第一格写盘"时抛 PermissionError 而退出，
        # returncode 仍可能为 0，于是外层把"什么都没写"当成功。
        # 真正的判据是**表有没有长出来**，所以每一轮都以 `_table_complete` 复核。
        still = [a for a in todo if not _table_complete(a, out_dir, epochs)]
        if still and rc == 0:
            print(f"[arch_loop] ⚠ 第 {rnd} 轮 exit=0 但仍有 {len(still)} 个未完成"
                  f"（子进程可能静默失败）→ 继续重试", flush=True)
            rc = 1
        elif rc != 0:
            print(f"[arch_loop] 第 {rnd} 轮 exit={rc}"
                  f"（原生崩溃？—— 从已落盘的 cell 续跑）", flush=True)
    left = [a for a in archs if not _table_complete(a, out_dir, epochs)]
    if left:
        print(f"[arch_loop] ⚠ 重试 {max_rounds} 轮后仍缺 {len(left)} 个：", flush=True)
        for a in left:
            print(f"      {a}", flush=True)
            print(f"      path={_table_path(a, out_dir, epochs)}", flush=True)
    return rc


def _table_path(arch: str, out_dir: str, epochs: int, tag_suffix: str = "") -> str:
    """结果表的路径。**必须与 `multibranch_exp.run_config` 完全一致**。

    ⚠ 两处文件名规则必须由**同一个函数**产出（`arch_space.safe_filename`），
    否则 arch_loop 会在错误的路径上找表 → 永远认为"没跑完" → 无限重跑。
    实测踩过：`|feat=N` 里的 `|` 是 Windows 非法字符，写盘失败，
    而两处各自的拼名逻辑又不一致，导致死循环。
    """
    fname = ARCH.safe_filename(arch.replace("+", "_plus_"))
    return os.path.join(out_dir, f"{fname}_k48_ep{int(epochs)}{tag_suffix}.json")


def _table_complete(arch: str, out_dir: str, epochs: int, expect: int = 24) -> bool:
    """结果表是否已跑完（cell 数达到 expect）。

    ⚠ 必须显式数 cell：只看文件存在是不够的 —— 崩溃时文件已经存在
    （每 cell 落盘写了前面的），但内容不全（曾出现 276/288）。
    """
    p = _table_path(arch, out_dir, epochs)
    if not os.path.exists(p):
        return False
    try:
        rec = json.load(open(p, "r", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    cells = rec.get("cells") or {}
    return len(cells) >= int(expect)


def main() -> None:
    ap = argparse.ArgumentParser(description="进化闭环：提出 → 实测 → 写回")
    ap.add_argument("--archs", nargs="+", default=None,
                    help="直接指定要实测的架构串（须符合 arch_space 语法）")
    ap.add_argument("--from-evolve", action="store_true",
                    help="从进化输出目录提取策略提出的架构串（LLM 路线）")
    ap.add_argument("--sample", action="store_true",
                    help="**自主采样提出**（不依赖 LLM；用已实测的倒 U 规律做邻域扰动）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evolve-dir",
                    default=os.path.join(_PROJECT, "evolve", "openevolve_output_pareto_gen"))
    ap.add_argument("--budget", type=int, default=3, help="本轮最多实测几个（越少越快）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--out-dir", default=os.path.join(_PROJECT, "data", "multibranch_evolved"))
    ap.add_argument("--log-file", default=os.path.join(_PROJECT, "_arch_loop.log"))
    ap.add_argument("--dry-run", action="store_true", help="只列出将要实测的架构")
    args = ap.parse_args()

    if args.archs:
        archs = []
        for a in args.archs:
            if ARCH.parse(a) is None:
                print(f"[skip] 非法架构串：{a}")
                continue
            archs.append(a)
    elif args.sample:
        done = measured_archs()
        archs = propose_by_sampling(done, budget=args.budget, seed=args.seed)
        if not archs:
            print("[arch_loop] 采样器未能提出新架构（空间已被测完？）")
            return
    elif args.from_evolve:
        done = measured_archs()
        cands = extract_from_evolve(args.evolve_dir)
        archs = [a for a in cands if a not in done][:max(0, int(args.budget))]
        if not archs:
            print("[arch_loop] 进化目录里没有尚未实测的新架构。")
            print("           建议改用 --sample（自主采样提出，不依赖 LLM 是否愿意探索）。")
            return
    else:
        ap.error("必须给 --archs / --from-evolve / --sample 之一")

    print("[arch_loop] 本轮计划实测：")
    for a in archs:
        spec = ARCH.parse(a)
        print(f"    {a}   估成本 {spec['cost_ms']:.1f} ms/step  "
              f"{spec['n_branches']} 分支  fuse={spec['fuse']}")
    if args.dry_run:
        print("[arch_loop] dry-run，未启动训练")
        return

    rc = measure(archs, args.epochs, args.n_train, args.n_val,
                 args.out_dir, args.log_file)
    print(f"[arch_loop] 训练退出码 {rc}")
    done = measured_archs()
    print("[arch_loop] 当前已实测的架构串：")
    for k in sorted(done):
        print(f"    {k:<56s} {done[k]:.4f}")
    print("\n下一步：重跑搜索，让这些**真实精度**进入候选池：")
    print("    powershell -ExecutionPolicy Bypass -File evolve/run.ps1 -Space pareto-gen")


if __name__ == "__main__":
    main()
