"""
旋涂实验工具
"""

import json
from .registry import register_tool
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
# from ..utils.reagent import find_reagent


@register_tool(
    name="spin_coat",
    description="执行旋涂实验，向平台发送旋涂参数",
    params={
        "spin_speed": {"type": "int", "description": "转速(rpm)", "required": True, "default": 2000},
        "spin_acc": {"type": "int", "description": "加速度(rpm/s)", "required": True, "default": 1000},
        "spin_dur": {"type": "int", "description": "持续时间(ms)", "required": True, "default": 5000},
    }
)
def spin_coat(
    spin_speed: int,
    spin_acc: int,
    spin_dur: int,
) -> str:
    """
    底层同步函数：向自动化平台发送旋涂实验MQTT指令

    Args:
        spin_speed : 旋涂转速(rpm)
        spin_acc   : 加速度(rpm/s)
        spin_dur   : 持续时间(ms)

    Returns:
        str: 成功时返回"实验指令下发成功..."消息，失败时返回"指令下发失败: ..."消息
    """
    #"c<spin_speed>,<spin_acc>,<spin_dur>"
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"c{spin_speed},{spin_acc},{spin_dur}")
            return f"旋涂实验指令记录成功。 转速:{spin_speed}rpm, 时长:{spin_dur}ms"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"c{spin_speed},{spin_acc},{spin_dur}")
                return f"旋涂实验指令记录成功。 转速:{spin_speed}rpm, 时长:{spin_dur}ms"
            else:
                f"指令下发失败"
    except Exception as e:
        return f"指令下发失败: {str(e)}"
