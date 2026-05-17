"""
基板转移
"""

from .registry import register_tool


@register_tool(
    name="move_glass",
    description="基板转移",
    params={
        "start_plate": {"type": "int", "description": "基底托盘位置_起始", "required": True},
        "start_pos": {"type": "int", "description": "基底在托盘中位置_起始", "required": True, "default": 1},
        "end_plate": {"type": "int", "description": "基底托盘位置_目标", "required": True},
        "end_pos": {"type": "int", "description": "基底在托盘中位置_目标", "required": True, "default": 1}
    }
)
def move_glass(start_plate: int, start_pos: int, end_plate: int, end_pos: int) -> str:
    """
    底层同步函数：滴液

    当前为模拟实现（返回确认消息），实际部署时需取消注释

    Args:
        "start_plate": {"type": "int", "description": "基底托盘位置_起始", "required": True},
        "start_pos": {"type": "int", "description": "基底在托盘中位置_起始", "required": True, "default": 1},
        "end_plate": {"type": "int", "description": "基底托盘位置_目标", "required": True},
        "end_pos": {"type": "int", "description": "基底在托盘中位置_目标", "required": True, "default": 1}

    Returns:
        str: 返回结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        print(f"成功将基板从{start_plate}号托盘的{start_pos}位置转移到{end_plate}号托盘的{end_pos}位置")
        return f"成功将基板从{start_plate}号托盘的{start_pos}位置转移到{end_plate}号托盘的{end_pos}位置"
    except Exception as e:
        return f"设置失败，错误：{e}"