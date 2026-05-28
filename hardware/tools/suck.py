"""
吸液
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool


@register_tool(
    name="suck",
    description="吸液",
    params={
        "bottle_pos": {"type": "Tuple[int, int, int]", "description": "需要吸液的目标试剂瓶的位置,[x,y,z]", "required": True, "default": [0,0,0]},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},
        "vol": {"type": "int", "description": "吸液体积(uL)", "required": True, "default": 60},
    }
)
def suck(bottle_pos: Tuple[int, int, int], tip: int, vol: int) -> str:
    """
    底层同步函数：吸液

    Args:
        "spit_pos": {"type": "Tuple[int,int,int]", "description": "需要吸液的目标试剂瓶的位置,[x,y,z]", "required": True, "default": [0,0,0]},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},
        "vol": {"type": "int", "description": "吸液体积(uL)", "required": True, "default": 60},

    Returns:
        str: 返回结果消息
    """
    # "d<action code>,tip,x,y,z,vol", action code 2: suck
    x, y, z = bottle_pos[0], bottle_pos[1], bottle_pos[2]
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")#move horizontal
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")# move tip vertical
            client.publish(EXPERIMENT_TOPIC, f"d2,{tip},0,0,0,{vol}")# suck
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")# move tip back
            return f"滴液机已保存:{tip}号泵向x={bottle_pos[0]},y={bottle_pos[1]},z={bottle_pos[2]}位置吸取{vol}ul溶液"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")
                client.publish(EXPERIMENT_TOPIC, f"d2,{tip},0,0,0,{vol}")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")
                return f"滴液机已保存:{tip}号泵向x={bottle_pos[0]},y={bottle_pos[1]},z={bottle_pos[2]}位置滴{vol}ul溶液"
            else:
                return f"滴液机吸液步骤保存失败"
    except Exception as e:
        return f"滴液机吸液步骤保存失败: {str(e)}"
