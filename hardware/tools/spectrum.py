"""
光谱采集工具
"""

from .registry import register_tool


@register_tool(
    name="collect_spectrum",
    description="启动光谱仪数据采集",
    params={
        "duration": {"type": "int", "description": "采集时长(秒)", "required": False, "default": 60}
    }
)
def execute_collect_spectrum(duration: int = 60) -> str:
    """
    底层同步函数：启动光谱仪数据采集

    创建SpectrometerClient实例并启动后台采集线程。
    采集的数据会存储在SpectrometerClient内部的DataFrame中，
    可通过spec_client.get_latest_data()获取。

    Args:
        duration : 预计采集时长（秒），默认60秒（仅用于提示，不控制实际停止）

    Returns:
        str: 启动结果消息
    """
    try:
        from ..spec_client import SpectrometerClient
        spec = SpectrometerClient()
        spec.start_collection()
        return f"光谱仪数据采集已启动，预计持续 {duration} 秒"
    except Exception as e:
        return f"光谱仪启动失败: {str(e)}"
