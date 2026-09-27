"""GPU 设备硬约束 + 资源占用前置检查（项目级强制，用户裁定）。

约束（用户明确定下，不可绕过）
================================================================
1. **本机所有 GPU 操作必须在 GPU 1 上进行**（物理编号 1）。
2. **如果系统资源被占用，不许挤占其他程序资源**。
   即：不得把别人的作业挤出显存、不得占满算力。

为什么必须在**代码里**强制，而不是只靠启动脚本
================================================================
torch 默认用可见设备的 0 号。若忘记设 `CUDA_VISIBLE_DEVICES=1`，进程会直接抢到
**物理 GPU 0** —— 而本机 GPU 0 常有用户在跑的作业（实测 2026-09-23 占用 17.3 GB /
88% 利用率）。那不是"慢一点"，是直接挤爆别人的显存。启动脚本设一次环境变量
是不够可靠的（换入口就漏），所以这里在**包导入时**就把变量钉死。

实现方式
================================================================
在 `nas_ptseg/__init__.py` 顶部 `from . import gpu`，于是任何 `nas_ptseg.*` 的导入
都会先执行本模块的副作用：把 `CUDA_VISIBLE_DEVICES` 设为 `1`。
必须在 `import torch` **之前**完成（torch 在首次 CUDA 调用时读取该变量）。

可用环境变量（都有安全默认值，正常情况下不需要动）
================================================================
    NAS_LIDAR_GPU               默认 "1"    物理 GPU 编号
    NAS_LIDAR_GPU_MIN_FREE_MIB  默认 8000   启动前要求的最小空闲显存（MiB）
    NAS_LIDAR_GPU_MEM_FRACTION  默认 0.5    本进程显存占用上限（占总显存的比）
    NAS_LIDAR_GPU_IGNORE_GUARD  默认未设    设为 "1" 才跳过空闲检查（会大声警告）
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

# ---------------------------------------------------------------------------
# 1) 导入即生效：把可见设备钉到 GPU 1
# ---------------------------------------------------------------------------
_FORCE_GPU = os.environ.get("NAS_LIDAR_GPU", "1").strip() or "1"
_prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = _FORCE_GPU

# 记录是否覆盖了调用者原本的设置，便于审计
_OVERRIDDEN_FROM: Optional[str] = None
if _prev_cvd not in (None, "", _FORCE_GPU):
    _OVERRIDDEN_FROM = _prev_cvd
    print(
        f"[gpu] 已覆盖 CUDA_VISIBLE_DEVICES: {_prev_cvd!r} -> {_FORCE_GPU!r}"
        f"（项目硬约束：只用 GPU {_FORCE_GPU}）"
    )

# 其它可能让 CUDA 抢资源的开关：显式关掉
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

MIN_FREE_MIB = int(os.environ.get("NAS_LIDAR_GPU_MIN_FREE_MIB", "8000"))
MEM_FRACTION = float(os.environ.get("NAS_LIDAR_GPU_MEM_FRACTION", "0.5"))
IGNORE_GUARD = os.environ.get("NAS_LIDAR_GPU_IGNORE_GUARD", "") == "1"


# ---------------------------------------------------------------------------
# 2) 物理显存查询（用 nvidia-smi 看**真实**空闲量，而不是进程内的可见设备）
# ---------------------------------------------------------------------------
def query_gpu_state() -> list:
    """返回 [{"index","name","used_mib","free_mib","total_mib","util_pct"}, ...]。

    读不到时返回 []（不抛异常 —— 在某些沙箱里 nvidia-smi 可能不可用）。
    """
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            out.append({
                "index": int(parts[0]),
                "name": parts[1],
                "total_mib": int(parts[2]),
                "used_mib": int(parts[3]),
                "free_mib": int(parts[4]),
                "util_pct": int(parts[5]),
            })
        return out
    except Exception:  # noqa: BLE001
        return []


def gpu_state(idx: Optional[int] = None) -> Optional[dict]:
    want = int(_FORCE_GPU) if idx is None else int(idx)
    for g in query_gpu_state():
        if g["index"] == want:
            return g
    return None


def format_state() -> str:
    rows = query_gpu_state()
    if not rows:
        return "  （nvidia-smi 不可用，无法读取显存状态）"
    lines = []
    for g in rows:
        mark = "  <-- 本项目使用" if g["index"] == int(_FORCE_GPU) else ""
        lines.append(
            f"  GPU {g['index']} {g['name']}: "
            f"用于 {g['used_mib']:>6d} MiB / 空闲 {g['free_mib']:>6d} MiB "
            f"/ 共 {g['total_mib']} MiB, 利用率 {g['util_pct']:>3d}%{mark}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3) 启动前的容量检查：资源被占就不许挤占
# ---------------------------------------------------------------------------
class GPUNotAvailable(RuntimeError):
    """目标 GPU 空闲显存不足（用户硬约束：不许挤占其他程序资源）。"""


def assert_capacity(need_mib: Optional[int] = None, verbose: bool = True) -> dict:
    """检查目标 GPU 是否还有足够空闲显存；不足就**拒绝启动**。

    这是"不许挤占其他程序资源"这条约束的落地点：与其事后把别人的作业挤崩，
    不如在启动前就说清楚"现在不能跑，等一会儿"。
    """
    need = int(MIN_FREE_MIB if need_mib is None else need_mib)
    st = gpu_state()
    if verbose:
        print("[gpu] 当前显卡状态：")
        print(format_state())
    if st is None:
        if verbose:
            print(f"[gpu] ⚠ 读不到 GPU {_FORCE_GPU} 的状态，跳过容量检查")
        return {"checked": False}

    if st["free_mib"] < need:
        msg = (
            f"GPU {_FORCE_GPU} 空闲显存 {st['free_mib']} MiB < 要求 {need} MiB。\n"
            f"用户硬约束：不许挤占其他程序资源。请等占用释放后重跑，\n"
            f"或显式降低 NAS_LIDAR_GPU_MIN_FREE_MIB（自行承担挤占风险）。"
        )
        if IGNORE_GUARD:
            print(f"[gpu] ⚠⚠ 已设 NAS_LIDAR_GPU_IGNORE_GUARD=1，强行继续：{msg}")
            return {"checked": True, "forced": True, **st}
        raise GPUNotAvailable(msg)
    if verbose:
        print(f"[gpu] ✅ GPU {_FORCE_GPU} 空闲 {st['free_mib']} MiB ≥ 要求 {need} MiB，可以启动")
    return {"checked": True, **st}


# ---------------------------------------------------------------------------
# 4) 交给 torch：返回 device，并给本进程设显存上限
# ---------------------------------------------------------------------------
def setup_torch_device(need_mib: Optional[int] = None, verbose: bool = True):
    """做容量检查 → 限制本进程显存占用 → 返回 torch.device。

    `set_per_process_memory_fraction` 是"不挤占他人"的第二道保险：
    即使本进程代码把 batch 写大，也不会无界增长到把别人挤掉。

    ⚠ 必须在 `import torch` 之后调用（本函数内部延迟导入 torch）。
    """
    state = assert_capacity(need_mib, verbose=verbose)
    import torch  # 延迟导入：本模块在无 torch 的 CPU 环境里也要能 import

    if not torch.cuda.is_available():
        if verbose:
            print("[gpu] CUDA 不可用，退回 CPU")
        return torch.device("cpu")

    n_visible = torch.cuda.device_count()
    if n_visible != 1:
        # CUDA_VISIBLE_DEVICES 生效时应当只看到 1 张卡；否则说明约束没起作用
        print(f"[gpu] ⚠ 期望只看到 1 张卡（CUDA_VISIBLE_DEVICES={_FORCE_GPU}），"
              f"实际 {n_visible} 张 —— 请检查环境变量是否被后续代码覆盖")

    try:
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
        if verbose:
            print(f"[gpu] 本进程显存上限设为总显存的 {MEM_FRACTION:.0%}"
                  f"（≈{MEM_FRACTION * 24564:.0f} MiB），避免无界增长挤占他人")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu] ⚠ 设置显存上限失败（不影响正确性）：{e}")

    if verbose:
        name = torch.cuda.get_device_name(0)
        print(f"[gpu] 使用物理 GPU {_FORCE_GPU}（进程内可见索引 0）：{name}")
    return torch.device("cuda")


def summary() -> dict:
    """把约束状态打包进结果 JSON，便于事后审计。"""
    return {
        "forced_gpu": _FORCE_GPU,
        "overridden_from": _OVERRIDDEN_FROM,
        "min_free_mib": MIN_FREE_MIB,
        "mem_fraction": MEM_FRACTION,
        "state": gpu_state(),
    }


def main() -> None:
    print("=" * 72)
    print("GPU 硬约束自检")
    print("=" * 72)
    print(f"  强制物理 GPU        : {_FORCE_GPU}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  启动前最小空闲显存  : {MIN_FREE_MIB} MiB")
    print(f"  本进程显存上限      : {MEM_FRACTION:.0%}")
    print(f"  跳过检查            : {IGNORE_GUARD}")
    print("")
    try:
        st = assert_capacity()
        print("")
        print("  → 可以启动训练")
    except GPUNotAvailable as e:
        print("")
        print(f"  → 拒绝启动：{e}")


if __name__ == "__main__":
    main()
