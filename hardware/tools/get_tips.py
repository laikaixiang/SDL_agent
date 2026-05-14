"""
取枪头
"""

from .registry import register_tool


@register_tool(
    name="get_tips",
    description="取枪头",
    params={
        "tip_box": {"type": "int", "description": "枪头盒位置", "required": True},
        "tip_pos": {"type": "int", "description": "枪头在枪头盒中位置(从1开始)", "required": True, "default": 1},
        "tips": {"type": "int", "description": "空气泵编号(1或者2)", "required": True, "default": 1}
    }
)
def get_tips(tip_box: int, tip_pos: int, tips: int) -> str:
    """
    底层同步函数：取枪头

    当前为模拟实现（返回确认消息），实际部署时需取消注释

    Args:
        t"tip_box": 枪头盒位置
        "tip_pos":  枪头在枪头盒中的位置，从1开始
        "tips":     枪头编号，1或者2

    Returns:
        str: 返回结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        return f"{tips}从枪头盒{tip_box}中成功取到{tip_pos}号枪头"
    except Exception as e:
        return f"设置失败，错误：{e}"