"""
机械臂移动工具
"""

from .registry import register_tool


@register_tool(
    name="move_robot_arm",
    description="移动机械臂到指定坐标",
    params={
        "x": {"type": "float", "description": "X坐标", "required": True},
        "y": {"type": "float", "description": "Y坐标", "required": True},
        "z": {"type": "float", "description": "Z坐标", "required": True}
    }
)
def execute_move_robot_arm(x: float, y: float, z: float) -> str:
    """
    底层同步函数：移动机械臂到指定坐标位置

    当前为模拟实现，实际部署时需取消注释subprocess调用。

    Args:
        x : X轴坐标
        y : Y轴坐标
        z : Z轴坐标

    Returns:
        str: 机械臂移动结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        # res = subprocess.run(
        #     ["python", "arm_ctrl.py", str(x), str(y), str(z)],
        #     capture_output=True, text=True,
        # )
        # return res.stdout.strip()
        return f"机械臂已精准移动至坐标 ({x}, {y}, {z})"
    except Exception as e:
        return f"机械臂移动失败: {str(e)}"
