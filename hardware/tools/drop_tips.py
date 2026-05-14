"""
放枪头
"""

from .registry import register_tool


@register_tool(
    name="drop_tips",
    description="放枪头",
    params={
        "tip_box": {"type": "int", "description": "枪头盒位置", "required": True},
        "tip_pos": {"type": "int", "description": "枪头在枪头盒中位置(从1开始)", "required": False, "default": 1},
        "tips": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1}
    }
)
def drop_tips(tip_box: int, tips: int, tip_pos: int = 1) -> str:
    """
    底层同步函数：放枪头

    当前为模拟实现（返回确认消息），实际部署时需取消注释

    Args:
        "tip_box": 枪头盒位置
        "tip_pos":  枪头在枪头盒中的位置，从1开始
        "tips":     枪头编号，1或者2

    Returns:
        str: 返回结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        tip_pos = 0 # 占位
        print(f"{tips}号泵把枪头扔到了枪头盒{tip_box}中")
        return f"{tips}号泵把枪头扔到了枪头盒{tip_box}中"
    except Exception as e:
        return f"设置失败，错误：{e}"