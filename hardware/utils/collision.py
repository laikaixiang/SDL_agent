"""
碰撞检测模块 — 机械臂移动前安全检查

V1 最小集:
  1. Z 低于桌面(安全间隙 5mm)
  2. XY 超出工作空间 (reach ∈ [100, 400]mm)

TODO(V2): 模块 AABB 碰撞(Pipette/spinCoater/slideBox),双臂互撞,轨迹扫描体碰撞
"""
from typing import Tuple


def check_collision(target_pose: dict) -> Tuple[int, str]:
    """
    检查目标位姿是否安全。

    Args:
        target_pose: {"x": float, "y": float, "z": float, "r": float}

    Returns:
        (200, "") if safe
        (400, "描述") if collision detected
    """
    x = target_pose.get("x", 0)
    y = target_pose.get("y", 0)
    z = target_pose.get("z", 0)

    # 1. 末端不能撞桌面 (安全间隙 5mm) — z=高度(J3 升降方向)
    if z < 5:
        return 400, f"末端高度 z={z:.1f}mm 太低(会撞桌面,至少 5mm)"

    # 2. 末端 XY 不能超出工作空间
    reach = (x * x + y * y) ** 0.5
    if reach > 400:
        return 400, f"末端 reach={reach:.1f}mm 超出最大臂展 400mm"
    if reach < 100:
        return 400, f"末端 reach={reach:.1f}mm 小于最小臂展 100mm"

    return 200, ""
