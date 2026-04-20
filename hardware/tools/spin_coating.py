"""
旋涂实验工具
"""

import json
from .registry import register_tool
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from ..utils.reagent import find_reagent


@register_tool(
    name="spin_coating",
    description="执行旋涂实验，向平台发送旋涂参数",
    params={
        "spin_speed": {"type": "int", "description": "转速(rpm)", "required": True},
        "spin_acc": {"type": "int", "description": "加速度(rpm/s)", "required": True},
        "spin_dur": {"type": "int", "description": "持续时间(ms)", "required": True},
        "reagent": {"type": "str", "description": "试剂名称", "required": True},
        "volume": {"type": "int", "description": "体积(µl)", "required": True}
    }
)
def execute_spin_coating(
    spin_speed: int,
    spin_acc: int,
    spin_dur: int,
    reagent: str,
    volume: int,
) -> str:
    """
    底层同步函数：向自动化平台发送旋涂实验MQTT指令

    此函数将实验参数打包为JSON格式发送到"do_experiment" MQTT主题。

    Args:
        spin_speed : 旋涂转速(rpm)
        spin_acc   : 加速度(rpm/s)
        spin_dur   : 持续时间(ms)
        reagent    : 试剂名称
        volume     : 体积(µl)

    Returns:
        str: 成功时返回"实验指令下发成功..."消息，失败时返回"指令下发失败: ..."消息
    """
    payload = {
        "action": "do_experiment",
        "params": {
            "spin_speed": spin_speed,
            "spin_acc": spin_acc,
            "spin_dur": spin_dur,
            "reagent": reagent,
            "volume": volume,
        },
    }
    try:
        client = get_mqtt_client()
        if not client.check_connect():
            client.connect(timeout=2)
        client.publish(EXPERIMENT_TOPIC, json.dumps(payload))
        return f"实验指令下发成功。试剂:{reagent}, 转速:{spin_speed}rpm, 时长:{spin_dur}ms"
    except Exception as e:
        return f"指令下发失败: {str(e)}"
