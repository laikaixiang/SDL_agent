"""
滴液
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool
from hardware.pos_config.parse_pos import parse_dispenser


@register_tool(
    name="spit",
    description="滴液",
    params={
        "spit_pos": {"type": "int", "description": "需要滴液的目标基底的位置", "required": True, "default": 0},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},
        "vol": {"type": "int", "description": "滴液体积(uL)", "required": True, "default": 60},
    }
)
def spit(spit_pos: int, tip: int, vol: int) -> str:
    """
    底层同步函数：滴液

    Args:
        "spit_pos": {"type": "int", "description": "需要滴液的目标基底的位置", "required": True, "default": 0},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},
        "vol": {"type": "int", "description": "滴液体积(uL)", "required": True, "default": 60},

    Returns:
        str: 返回结果消息
    """
    # "d<action code>,tip,x,y,z,vol", action code 3: spit
    spit_pos = parse_dispenser(tip, 2, spit_pos)
    x, y, z = spit_pos[0], spit_pos[1], spit_pos[2]
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")# move horizontal
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")# move tip vertical
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"d3,{tip},0,0,0,{vol}")# spit
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")# move tip back
            client.listen_to_message("done")
            return f"滴液机已执行:{tip}号泵向x={spit_pos[0]},y={spit_pos[1]},z={spit_pos[2]}位置滴{vol}ul溶液"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"d3,{tip},0,0,0,{vol}")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")
                client.listen_to_message("done")
                return f"滴液机已执行:{tip}号泵向x={spit_pos[0]},y={spit_pos[1]},z={spit_pos[2]}位置滴{vol}ul溶液"
            else:
                return f"滴液机滴液失败"
    except Exception as e:
        return f"滴液机滴液失败: {str(e)}"
