"""
滴液
"""

from .registry import register_tool


@register_tool(
    name="drop",
    description="滴液",
    params={
        "drop_plate": {"type": "int", "description": "需要滴液基底的目标托盘", "required": True},
        "drop_pos": {"type": "int", "description": "基底在托盘中位置", "required": True, "default": 1},
        "tip": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1},
        "Vol": {"type": "int", "description": "滴液体积(uL)", "required": True, "default": 60},
    }
)
def drop(drop_plate: int, drop_pos: int, tip: int, Vol: int) -> str:
    """
    底层同步函数：滴液

    当前为模拟实现（返回确认消息），实际部署时需取消注释

    Args:
        "drop_plate": {"type": "int", "description": "需要滴液基底的目标托盘", "required": True},
        "drop_pos": {"type": "int", "description": "基底在托盘中位置", "required": True, "default": 1},
        "tip": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1},
        "Vol": {"type": "int", "description": "滴液体积(uL)", "required": True, "default": 60},

    Returns:
        str: 返回结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        print(f"{tip}号泵向{drop_plate}的{drop_pos}位置滴了{Vol}ul溶液")
        return f"{tip}号泵向{drop_plate}的{drop_pos}位置滴了{Vol}ul溶液"
    except Exception as e:
        return f"设置失败，错误：{e}"