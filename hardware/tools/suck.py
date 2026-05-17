"""
吸液
"""

from .registry import register_tool


@register_tool(
    name="suck",
    description="吸液",
    params={
         "bottom_box": {"type": "int", "description": "试剂瓶盒编号", "required": True},
         "bottom_pos": {"type": "int", "description": "试剂瓶在盒中位置(从1开始)", "required": True, "default": 1},
         "tip": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1},
         "Vol": {"type": "int", "description": "吸液体积(uL)", "required": True, "default": 60},
    }
)
def suck(bottom_box: int, bottom_pos: int, tip: int, Vol: int) -> str:
    """
    底层同步函数：滴液

    当前为模拟实现（返回确认消息），实际部署时需取消注释

    Args:
        "bottom_box": {"type": "int", "description": "试剂瓶盒编号", "required": True},
         "bottom_pos": {"type": "int", "description": "试剂瓶在盒中位置(从1开始)", "required": True, "default": 1},
         "tip": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1},
         "Vol": {"type": "int", "description": "吸液体积(uL)", "required": True, "default": 60},

    Returns:
        str: 返回结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        print(f"{tip}号泵从{bottom_box}号盘的{bottom_pos}号瓶吸取了{Vol}ul溶液")
        return f"{tip}号泵从{bottom_box}号盘的{bottom_pos}号瓶吸取了{Vol}ul溶液"
    except Exception as e:
        return f"设置失败，错误：{e}"