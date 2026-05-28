"""
实验序列控制工具
"""

from .registry import register_tool
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC


@register_tool(
    name="start_experiment",
    description="启动已注册的实验序列",
    params={}
)
def start_experiment() -> str:
    """
    底层同步函数：向自动化平台发送实验序列启动指令"start"

    此函数是start_experiment()异步工具函数的同步版本，
    供core/hardware_controller.py的前缀命令路径调用。

    Returns:
        str: 成功时返回"实验序列启动指令已发送"，失败时返回错误消息
    """
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, "start")
            return "实验序列启动指令已发送"
        else:
            if client.connect(timeout=2):
                client.publish(EXPERIMENT_TOPIC, "start")
                return "实验序列启动指令已发送"
            else:
                return "MQTT连接失败，无法启动实验"
    except Exception as e:
        return f"启动实验失败: {str(e)}"
